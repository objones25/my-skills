#!/usr/bin/env python3
"""PQN (Gallici et al., ICLR 2025): Q-learning for many parallel envs, single file.

Q-learning with no replay buffer and no target network. Stability comes from
LayerNorm after every hidden layer, and from training on fresh data from many
parallel environments with lambda-returns. Structure and defaults follow
purejaxql (github.com/mttga/purejaxql, Apache-2.0): pqn_minatar.py and
config/alg/pqn_minatar.yaml.

The whole run (all seeds) is one jax.jit(jax.vmap(train)) call:

  update loop (scan, NUM_UPDATES)
    rollout (scan, NUM_STEPS): eps-greedy act -> vmap(env.step) over NUM_ENVS
    lambda-returns (reverse scan)
    epochs (scan, NUM_EPOCHS)
      minibatches (scan): loss 0.5 * (Q(s,a) - G_lambda)^2

Differences from purejaxql, on purpose:
  * time-limit truncation bootstraps from the pre-reset observation (mprl_common).
  * --minibatch-size is fixed and the number of minibatches follows from
    NUM_ENVS * NUM_STEPS, so raising --num-envs adds optimizer steps instead of
    growing the minibatch (the recipe of Beukman et al. 2026; pass
    --num-minibatches to override).
  * an MLP torso only (gymnax vector envs and symbolic Craftax). MinAtar pixels
    work flattened; use purejaxql's CNN for Atari.

Usage
  python pqn_jax.py --env CartPole-v1 --total-timesteps 500000 --num-seeds 4
  python pqn_jax.py --env Breakout-MinAtar --total-timesteps 10000000 --num-envs 128
  python pqn_jax.py --env CartPole-v1 --flops 3e12 --results-csv iso.csv --eval-csv eval.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from flops_per_frame import forward_flops
from mprl_common import AutoReset, append_csv, evaluate, flat, make_env, warn_if_degenerate_eval, write_eval_row


class QNetwork(nn.Module):
    action_dim: int
    hidden: int = 128
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.layers):
            x = nn.Dense(self.hidden)(x)
            x = nn.LayerNorm()(x)  # PQN's stabilizer: replaces the target network
            x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray   # truncation bootstrap already folded in
    done: jnp.ndarray     # terminated | truncated
    q_val: jnp.ndarray    # Q(obs) under the params that acted
    ep_return: jnp.ndarray
    ep_done: jnp.ndarray


def make_train(cfg: dict):
    env, env_params = make_env(cfg["env"])
    wrapper = AutoReset(env, env_params, cfg["sticky_prob"])
    n_actions = env.action_space(env_params).n
    n, t = cfg["num_envs"], cfg["num_steps"]
    batch = n * t
    n_mb = cfg["num_minibatches"] or max(1, batch // cfg["minibatch_size"])
    if batch % n_mb:
        raise ValueError(f"num_envs*num_steps={batch} not divisible by {n_mb} minibatches")
    num_updates = cfg["total_timesteps"] // batch
    if num_updates < 1:
        raise ValueError("total_timesteps smaller than one rollout batch")
    cfg.update(num_minibatches=n_mb, num_updates=num_updates)
    net = QNetwork(n_actions, cfg["hidden"], cfg["layers"])

    eps_sched = optax.linear_schedule(cfg["eps_start"], cfg["eps_finish"],
                                      int(cfg["eps_decay"] * num_updates))
    lr = (optax.linear_schedule(cfg["lr"], 1e-20, num_updates * cfg["num_epochs"] * n_mb)
          if cfg["lr_linear_decay"] else cfg["lr"])
    tx = optax.chain(optax.clip_by_global_norm(cfg["max_grad_norm"]), optax.radam(learning_rate=lr))

    def train(rng):
        rng, k_init, k_reset = jax.random.split(rng, 3)
        obs, state = jax.vmap(wrapper.reset)(jax.random.split(k_reset, n))
        params = net.init(k_init, flat(obs))
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        def update(carry, update_idx):
            ts, obs, state, rng = carry
            eps = eps_sched(update_idx)

            def rollout_step(c, _):
                obs, state, rng = c
                rng, k_eps, k_rand, k_env = jax.random.split(rng, 4)
                q = net.apply(ts.params, flat(obs))
                greedy = jnp.argmax(q, axis=-1)
                rand = jax.random.randint(k_rand, (n,), 0, n_actions)
                action = jnp.where(jax.random.uniform(k_eps, (n,)) < eps, rand, greedy)
                out = jax.vmap(wrapper.step)(jax.random.split(k_env, n), state, action)
                q_final = jnp.max(net.apply(ts.params, flat(out.final_obs)), axis=-1)
                reward = out.reward + cfg["gamma"] * out.truncated * jax.lax.stop_gradient(q_final)
                done = (out.terminated | out.truncated).astype(jnp.float32)
                tr = Transition(obs, action, reward, done, q, out.episode_return, done)
                return (out.obs, out.state, rng), tr

            (obs, state, rng), trs = jax.lax.scan(rollout_step, (obs, state, rng), None, t)

            # Q(lambda) targets, reverse scan (purejaxql pqn_minatar.py _get_target)
            last_q = jnp.max(net.apply(ts.params, flat(obs)), axis=-1) * (1.0 - trs.done[-1])
            g_last = trs.reward[-1] + cfg["gamma"] * last_q

            def target(carry, tr):
                g, next_q = carry
                boot = tr.reward + cfg["gamma"] * (1 - tr.done) * next_q
                g = boot + cfg["gamma"] * cfg["lam"] * (g - next_q)
                g = (1 - tr.done) * g + tr.done * tr.reward
                return (g, jnp.max(tr.q_val, axis=-1)), g

            _, gs = jax.lax.scan(target, (g_last, last_q),
                                 jax.tree.map(lambda x: x[:-1], trs), reverse=True)
            targets = jnp.concatenate([gs, g_last[None]])

            def epoch(c, _):
                ts, rng = c
                rng, k = jax.random.split(rng)
                perm = jax.random.permutation(k, batch)
                o = flat(trs.obs.reshape(batch, *trs.obs.shape[2:]))[perm].reshape(n_mb, -1, flat(trs.obs[0]).shape[-1])
                a = trs.action.reshape(batch)[perm].reshape(n_mb, -1)
                g = targets.reshape(batch)[perm].reshape(n_mb, -1)

                def mb(ts, xs):
                    o, a, g = xs

                    def loss_fn(p):
                        q = net.apply(p, o)
                        qa = jnp.take_along_axis(q, a[:, None], axis=-1).squeeze(-1)
                        return 0.5 * jnp.mean((qa - g) ** 2)

                    loss, grads = jax.value_and_grad(loss_fn)(ts.params)
                    return ts.apply_gradients(grads=grads), loss

                ts, losses = jax.lax.scan(mb, ts, (o, a, g))
                return (ts, rng), losses.mean()

            (ts, rng), losses = jax.lax.scan(epoch, (ts, rng), None, cfg["num_epochs"])
            finished = trs.ep_done.sum()
            metrics = {"loss": losses.mean(), "eps": eps,
                       "episode_return": jnp.where(finished > 0, (trs.ep_return * trs.ep_done).sum() / jnp.maximum(finished, 1.0), jnp.nan),
                       "episodes": finished}
            return (ts, obs, state, rng), metrics

        (ts, *_), metrics = jax.lax.scan(update, (ts, obs, state, rng), jnp.arange(num_updates))
        return ts.params, metrics

    def greedy(params, obs, key):
        return jnp.argmax(net.apply(params, flat(obs)), axis=-1)

    return train, wrapper, net, greedy, n_actions


def parse() -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--flops", type=float, help="compute budget C; sets total timesteps = C / FPF")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--num-minibatches", type=int, default=0, help="override; 0 = derive from minibatch size")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--no-lr-decay", dest="lr_linear_decay", action="store_false")
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.65)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-finish", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.1, help="fraction of updates to anneal over")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--sticky-prob", type=float, default=0.0,
                   help="repeat the previous action with this probability (MinAtar default 0.1; discrete only)")
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--eval-seed", type=int, default=12345, help="keep fixed across agents for Elo")
    p.add_argument("--results-csv", help="append flops,fpf,batch_size(=N*T),minibatch_size,lr,seed,score,model rows")
    p.add_argument("--eval-csv", help="append per-instance eval returns for elo_league.py")
    p.add_argument("--name", default=None, help="agent name prefix in --eval-csv")
    return vars(p.parse_args())


def main() -> int:
    cfg = parse()
    env, env_params = make_env(cfg["env"])
    obs_dim = int(np.prod(env.observation_space(env_params).shape))
    probe = QNetwork(env.action_space(env_params).n, cfg["hidden"], cfg["layers"])
    p0 = probe.init(jax.random.key(0), jnp.ones((1, obs_dim)))
    f_fwd = forward_flops(lambda x: probe.apply(p0, x), jnp.ones((obs_dim,)))
    fpf = f_fwd * (2 + 3 * cfg["num_epochs"])  # act + truncation bootstrap fwd + epochs x (fwd+bwd)
    if cfg["flops"]:
        cfg["total_timesteps"] = int(cfg["flops"] / fpf)

    train, wrapper, net, greedy, _ = make_train(cfg)
    print(f"env={cfg['env']} N={cfg['num_envs']} T={cfg['num_steps']} batch={cfg['num_envs'] * cfg['num_steps']} "
          f"minibatches={cfg['num_minibatches']} updates={cfg['num_updates']} FPF={fpf:.3g} "
          f"C={fpf * cfg['total_timesteps']:.3g} backend={jax.default_backend()}")
    keys = jax.random.split(jax.random.key(cfg["seed"]), cfg["num_seeds"])
    t0 = time.perf_counter()
    params, metrics = jax.block_until_ready(jax.jit(jax.vmap(train))(keys))
    dt = time.perf_counter() - t0
    steps = cfg["num_updates"] * cfg["num_envs"] * cfg["num_steps"] * cfg["num_seeds"]
    print(f"trained {steps:.3g} env steps in {dt:.1f}s ({steps / dt:.3g} steps/s incl. compile)")

    ret = np.asarray(metrics["episode_return"])  # (seeds, updates)
    for q in (0.1, 0.5, 1.0):
        hi = max(1, int(q * ret.shape[1]))
        window = ret[:, max(0, hi - 10):hi]  # last 10 updates up to this point; NaN = no episode finished
        print(f"  {int(q * 100):3d}% of updates: train return per seed {[round(float(x), 1) for x in np.nanmean(window, 1)]}")

    warn_if_degenerate_eval(wrapper, cfg["eval_episodes"], cfg["eval_seed"])
    ev = jax.jit(jax.vmap(lambda p: evaluate(wrapper, greedy, p, cfg["eval_episodes"], cfg["eval_seed"])))(params)
    ev = np.asarray(ev)
    print(f"greedy eval mean return per seed: {[round(float(x), 1) for x in ev.mean(1)]}")

    name = cfg["name"] or f"pqn_N{cfg['num_envs']}_h{cfg['hidden']}"
    if cfg["results_csv"]:
        c = fpf * cfg["total_timesteps"]
        append_csv(cfg["results_csv"],
                   ["flops", "fpf", "batch_size", "minibatch_size", "lr", "seed", "score", "model"],
                   [[c, fpf, cfg["num_envs"] * cfg["num_steps"], cfg["minibatch_size"], cfg["lr"],
                     cfg["seed"] + s, float(ev[s].mean()), name]
                    for s in range(cfg["num_seeds"])])
    if cfg["eval_csv"]:
        for s in range(cfg["num_seeds"]):
            write_eval_row(cfg["eval_csv"], f"{name}_s{cfg['seed'] + s}", ev[s])
    return 0


if __name__ == "__main__":
    sys.exit(main())
