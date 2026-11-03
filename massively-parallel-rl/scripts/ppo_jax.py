#!/usr/bin/env python3
"""PPO / A2C for many parallel JAX environments, single file (PureJaxRL style).

The whole run, all seeds included, is one jax.jit(jax.vmap(train)) call, so
environment stepping, rollout storage, GAE and SGD never leave the accelerator.
Structure and defaults follow PureJaxRL (github.com/luchris429/purejaxrl,
Apache-2.0), purejaxrl/ppo.py and ppo_continuous_action.py:

  update loop (scan, NUM_UPDATES)
    rollout (scan, NUM_STEPS): pi(obs) -> vmap(env.step) over NUM_ENVS
    GAE (reverse scan)
    epochs (scan, UPDATE_EPOCHS)
      minibatches (scan): clipped policy loss + value loss - entropy bonus

Choices that differ from PureJaxRL, on purpose:
  * Time-limit truncation bootstraps from the pre-reset observation
    (mprl_common.AutoReset). --truncation-as-terminal restores the usual,
    biased behaviour so the difference can be measured.
  * --minibatch-size is fixed and the number of minibatches is derived from
    NUM_ENVS * NUM_STEPS, so adding environments adds optimizer steps rather
    than growing the minibatch (recipe of Beukman et al. 2026). Use
    --num-minibatches to override.
  * --algo a2c gives the setting of Zheng et al. 2026 (Sec. 3.1, App. A.2):
    one pass, one minibatch, no clipping or advantage normalization, learning
    rate decayed linearly to 0. With a single pass the PPO ratio is 1, so the
    clipped objective equals the plain policy gradient.

Usage
  python ppo_jax.py --env CartPole-v1 --total-timesteps 500000 --num-seeds 4
  python ppo_jax.py --env Pendulum-v1 --total-timesteps 2000000 --num-envs 64 --gamma 0.95
  python ppo_jax.py --algo a2c --env Breakout-MinAtar --num-envs 256 --num-steps 16
  python ppo_jax.py --env Craftax-Symbolic-v1 --num-envs 1024 --total-timesteps 100000000
  python ppo_jax.py --env CartPole-v1 --flops 1e12 --results-csv iso.csv --eval-csv eval.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from flops_per_frame import forward_flops
from mprl_common import AutoReset, append_csv, evaluate, flat, make_env, warn_if_degenerate_eval, write_eval_row


class ActorCritic(nn.Module):
    action_dim: int
    discrete: bool
    hidden: int = 64
    layers: int = 2
    activation: str = "tanh"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        act = {"tanh": nn.tanh, "relu": nn.relu, "gelu": nn.gelu}[self.activation]

        def torso(h):
            for _ in range(self.layers):
                h = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
                if self.layer_norm:
                    h = nn.LayerNorm()(h)
                h = act(h)
            return h

        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(torso(x))
        if self.discrete:
            pi = distrax.Categorical(logits=mean)
        else:
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(torso(x))
        return pi, value.squeeze(-1)


class RunningNorm(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @staticmethod
    def create(dim):
        return RunningNorm(jnp.zeros(dim), jnp.ones(dim), jnp.float32(1e-4))

    def update(self, x):
        """Merge the statistics of batch x (B, dim) into the running estimate."""
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + b_n
        mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        return RunningNorm(mean, m2 / tot, tot)

    def __call__(self, x):
        return jnp.clip((x - self.mean) / jnp.sqrt(self.var + 1e-8), -10.0, 10.0)


class Transition(NamedTuple):
    obs: jnp.ndarray       # normalized if --norm-obs: exactly what the policy saw
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray    # truncation bootstrap folded in
    done: jnp.ndarray
    ep_return: jnp.ndarray
    ep_done: jnp.ndarray


def make_train(cfg: dict):
    env, env_params = make_env(cfg["env"])
    wrapper = AutoReset(env, env_params, cfg["sticky_prob"])
    space = env.action_space(env_params)
    discrete = hasattr(space, "n")
    action_dim = space.n if discrete else int(np.prod(space.shape))
    low = None if discrete else jnp.asarray(space.low)
    high = None if discrete else jnp.asarray(space.high)
    obs_dim = int(np.prod(env.observation_space(env_params).shape))

    n, t = cfg["num_envs"], cfg["num_steps"]
    batch = n * t
    if cfg["algo"] == "a2c":
        cfg.update(update_epochs=1, num_minibatches=1, norm_adv=False, clip_vloss=False)
    n_mb = cfg["num_minibatches"] or max(1, batch // cfg["minibatch_size"])
    if batch % n_mb:
        raise ValueError(f"num_envs*num_steps={batch} not divisible by {n_mb} minibatches")
    num_updates = cfg["total_timesteps"] // batch
    if num_updates < 1:
        raise ValueError("total_timesteps smaller than one rollout batch")
    cfg.update(num_minibatches=n_mb, num_updates=num_updates)

    net = ActorCritic(action_dim, discrete, cfg["hidden"], cfg["layers"], cfg["activation"], cfg["layer_norm"])
    steps_total = num_updates * cfg["update_epochs"] * n_mb
    lr = optax.linear_schedule(cfg["lr"], 0.0, steps_total) if cfg["anneal_lr"] else cfg["lr"]
    tx = optax.chain(optax.clip_by_global_norm(cfg["max_grad_norm"]), optax.adam(lr, eps=1e-5))

    def to_env_action(a):
        return a if discrete else jnp.clip(a, low, high)

    def train(rng):
        rng, k_init, k_reset = jax.random.split(rng, 3)
        obs, state = jax.vmap(wrapper.reset)(jax.random.split(k_reset, n))
        rms = RunningNorm.create(obs_dim)
        if cfg["norm_obs"]:
            rms = rms.update(flat(obs))
        params = net.init(k_init, jnp.zeros((1, obs_dim)))
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        def norm(x, rms):
            return rms(flat(x)) if cfg["norm_obs"] else flat(x)

        def update(carry, _):
            ts, obs, state, rms, rng = carry

            def rollout_step(c, _):
                obs, state, rms, rng = c
                rng, k_act, k_env = jax.random.split(rng, 3)
                o = norm(obs, rms)
                pi, value = net.apply(ts.params, o)
                action = pi.sample(seed=k_act)
                log_prob = pi.log_prob(action)
                out = jax.vmap(wrapper.step)(jax.random.split(k_env, n), state, to_env_action(action))
                if cfg["norm_obs"]:
                    rms = rms.update(flat(out.obs))
                reward = out.reward * cfg["reward_scale"]
                if cfg["truncation_as_terminal"]:
                    done = out.terminated | out.truncated
                else:
                    _, v_final = net.apply(ts.params, norm(out.final_obs, rms))
                    reward = reward + cfg["gamma"] * out.truncated * jax.lax.stop_gradient(v_final)
                    done = out.terminated | out.truncated
                done = done.astype(jnp.float32)
                tr = Transition(o, action, log_prob, value, reward, done, out.episode_return, done)
                return (out.obs, out.state, rms, rng), tr

            (obs, state, rms, rng), trs = jax.lax.scan(rollout_step, (obs, state, rms, rng), None, t)
            _, last_val = net.apply(ts.params, norm(obs, rms))

            def gae_step(c, tr):
                gae, next_value = c
                delta = tr.reward + cfg["gamma"] * next_value * (1 - tr.done) - tr.value
                gae = delta + cfg["gamma"] * cfg["gae_lambda"] * (1 - tr.done) * gae
                return (gae, tr.value), gae

            _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_val), last_val), trs, reverse=True, unroll=16)
            targets = adv + trs.value

            def epoch(c, _):
                ts, rng = c
                rng, k = jax.random.split(rng)
                perm = jax.random.permutation(k, batch)
                data = jax.tree.map(lambda x: x.reshape(batch, *x.shape[2:])[perm].reshape(n_mb, -1, *x.shape[2:]),
                                    (trs, adv, targets))

                def mb(ts, d):
                    tr, a, g = d

                    def loss_fn(p):
                        pi, value = net.apply(p, tr.obs)
                        ratio = jnp.exp(pi.log_prob(tr.action) - tr.log_prob)
                        aa = (a - a.mean()) / (a.std() + 1e-8) if cfg["norm_adv"] else a
                        pg = -jnp.minimum(ratio * aa, jnp.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * aa).mean()
                        if cfg["clip_vloss"]:
                            v_clip = tr.value + jnp.clip(value - tr.value, -cfg["clip_eps"], cfg["clip_eps"])
                            vloss = 0.5 * jnp.maximum((value - g) ** 2, (v_clip - g) ** 2).mean()
                        else:
                            vloss = 0.5 * ((value - g) ** 2).mean()
                        ent = pi.entropy().mean()
                        approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
                        return pg + cfg["vf_coef"] * vloss - cfg["ent_coef"] * ent, (pg, vloss, ent, approx_kl)

                    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                    return ts.apply_gradients(grads=grads), aux

                ts, aux = jax.lax.scan(mb, ts, data)
                return (ts, rng), aux

            (ts, rng), (pg, vl, ent, kl) = jax.lax.scan(epoch, (ts, rng), None, cfg["update_epochs"])
            finished = trs.ep_done.sum()
            metrics = {"episode_return": jnp.where(finished > 0, (trs.ep_return * trs.ep_done).sum()
                                                   / jnp.maximum(finished, 1.0), jnp.nan),
                       "pg_loss": pg.mean(), "v_loss": vl.mean(), "entropy": ent.mean(), "approx_kl": kl.mean()}
            return (ts, obs, state, rms, rng), metrics

        (ts, _, _, rms, _), metrics = jax.lax.scan(update, (ts, obs, state, rms, rng), None, num_updates)
        return (ts.params, rms), metrics

    def act_greedy(params_rms, obs, key):
        params, rms = params_rms
        o = rms(flat(obs)) if cfg["norm_obs"] else flat(obs)
        pi, _ = net.apply(params, o)
        a = pi.mode() if discrete else pi.mean()
        return to_env_action(a)

    return train, wrapper, net, act_greedy, obs_dim


def parse() -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--flops", type=float, help="compute budget C; sets total timesteps = C / FPF")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--num-minibatches", type=int, default=0, help="override; 0 = derive from minibatch size")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--activation", choices=["tanh", "relu", "gelu"], default="tanh")
    p.add_argument("--layer-norm", action="store_true")
    p.add_argument("--norm-obs", action="store_true", help="running mean/var observation normalization")
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    p.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
    p.add_argument("--truncation-as-terminal", action="store_true",
                   help="treat time-limit ends as terminal (the common bug); for comparison only")
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
    if cfg["algo"] == "a2c" and cfg["anneal_lr"] is False:
        print("note: Zheng et al. decay the A2C learning rate linearly to 0")
    train, wrapper, net, act_greedy, obs_dim = make_train(cfg)
    p0 = net.init(jax.random.key(0), jnp.ones((1, obs_dim)))
    def fwd(x):
        pi, v = net.apply(p0, x)
        return (pi.logits if net.discrete else pi.loc), v

    f_fwd = forward_flops(fwd, jnp.ones((obs_dim,)))
    fpf = f_fwd * (2 + 3 * cfg["update_epochs"])  # act + truncation bootstrap + epochs x (fwd + bwd)
    if cfg["flops"]:
        cfg["total_timesteps"] = int(cfg["flops"] / fpf)
        train, wrapper, net, act_greedy, obs_dim = make_train(cfg)

    print(f"{cfg['algo']} env={cfg['env']} N={cfg['num_envs']} T={cfg['num_steps']} "
          f"batch={cfg['num_envs'] * cfg['num_steps']} minibatches={cfg['num_minibatches']} "
          f"epochs={cfg['update_epochs']} updates={cfg['num_updates']} FPF={fpf:.3g} "
          f"C={fpf * cfg['total_timesteps']:.3g} backend={jax.default_backend()}")
    keys = jax.random.split(jax.random.key(cfg["seed"]), cfg["num_seeds"])
    t0 = time.perf_counter()
    params_rms, metrics = jax.block_until_ready(jax.jit(jax.vmap(train))(keys))
    dt = time.perf_counter() - t0
    steps = cfg["num_updates"] * cfg["num_envs"] * cfg["num_steps"] * cfg["num_seeds"]
    print(f"trained {steps:.3g} env steps in {dt:.1f}s ({steps / dt:.3g} steps/s incl. compile)")

    ret = np.asarray(metrics["episode_return"])
    for q in (0.1, 0.5, 1.0):
        hi = max(1, int(q * ret.shape[1]))
        window = ret[:, max(0, hi - 10):hi]
        print(f"  {int(q * 100):3d}% of updates: train return per seed "
              f"{[round(float(x), 1) for x in np.nanmean(window, 1)]}")
    print(f"  final approx_kl {float(np.mean(metrics['approx_kl'][:, -1])):.4f}  "
          f"entropy {float(np.mean(metrics['entropy'][:, -1])):.3f}")

    warn_if_degenerate_eval(wrapper, cfg["eval_episodes"], cfg["eval_seed"])
    ev = np.asarray(jax.jit(jax.vmap(
        lambda pr: evaluate(wrapper, act_greedy, pr, cfg["eval_episodes"], cfg["eval_seed"])))(params_rms))
    print(f"deterministic eval mean return per seed: {[round(float(x), 1) for x in ev.mean(1)]}")

    name = cfg["name"] or f"{cfg['algo']}_N{cfg['num_envs']}_h{cfg['hidden']}"
    if cfg["results_csv"]:
        c = fpf * cfg["total_timesteps"]
        append_csv(cfg["results_csv"],
                   ["flops", "fpf", "batch_size", "minibatch_size", "lr", "seed", "score", "model"],
                   [[c, fpf, cfg["num_envs"] * cfg["num_steps"], cfg["num_envs"] * cfg["num_steps"] // cfg["num_minibatches"],
                     cfg["lr"], cfg["seed"] + s, float(ev[s].mean()), name] for s in range(cfg["num_seeds"])])
    if cfg["eval_csv"]:
        for s in range(cfg["num_seeds"]):
            write_eval_row(cfg["eval_csv"], f"{name}_s{cfg['seed'] + s}", ev[s])
    return 0


if __name__ == "__main__":
    sys.exit(main())
