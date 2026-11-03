"""Minimal PPO for continuous control, written to be read next to concepts/policy-gradient.md.

Pendulum-v1 never terminates; every episode is truncated at 200 steps, so it is the smallest
real test of time-out bootstrapping. Measured Nov 2, 2026 (80k steps, gamma 0.99, seeds 0-2,
deterministic eval): correct bootstrapping -135.4 / -135.0 / -122.1 (mean -130.8);
--timeout-as-terminal -124.9 / -167.2 / -148.0 (mean -146.7). A small, noisy edge with 3 seeds,
consistent with Andrychowicz et al. finding little difference at long time limits; the bias is
larger when episodes are short relative to 1/(1-gamma) (scripts/rl_math.py truncation_bias).

    python examples/ppo_pendulum.py                         # about 75 s on 2 CPU cores, eval near -130
    python examples/ppo_pendulum.py --timeout-as-terminal   # the classic bug
    python examples/ppo_pendulum.py --steps 50000 --seed 1  # quick smoke run

Needs torch and gymnasium. GAE, KL, clip fraction and explained variance come from
scripts/rl_math.py so the math lives in one tested place.

Blocks
    1. Policy: Gaussian with state-independent log std; critic: separate MLP.
    2. Rollout: N independent envs stepped in a loop; on truncation we store V(final obs)
       before resetting, which is what lets GAE bootstrap correctly.
    3. Update: clipped surrogate + value loss - entropy bonus, E epochs of minibatch SGD,
       global grad clipping, early stop when the k3 KL estimate exceeds 1.5 * target_kl.
    4. Evaluation: deterministic (mean) actions on fresh seeds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rl_math  # noqa: E402


def mlp(inp: int, out: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, init_log_std: float = 0.0):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim)
        self.v = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        nn.init.orthogonal_(self.mu[-1].weight, gain=0.01)  # small initial actions (37 Details: policy output gain 0.01; Andrychowicz et al. §3.2)
        nn.init.zeros_(self.mu[-1].bias)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mu(obs), self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


def evaluate(model: ActorCritic, episodes: int, seed: int, act_scale: float) -> float:
    env = gym.make("Pendulum-v1")
    total = []
    for e in range(episodes):
        obs, _ = env.reset(seed=seed + e)
        ret, done = 0.0, False
        while not done:
            with torch.no_grad():
                a = model.mu(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            obs, r, term, trunc, _ = env.step(np.clip(a, -1, 1) * act_scale)
            ret += float(r)
            done = term or trunc
        total.append(ret)
    return float(np.mean(total))


def train(args: argparse.Namespace) -> dict[str, float]:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    envs = [gym.make("Pendulum-v1") for _ in range(args.num_envs)]
    act_scale = float(envs[0].action_space.high[0])
    obs = np.stack([e.reset(seed=int(rng.integers(1 << 30)))[0] for e in envs]).astype(np.float32)
    model = ActorCritic(obs.shape[1], envs[0].action_space.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    N, T = args.num_envs, args.rollout
    batch = N * T
    mb = batch // args.minibatches
    iters = max(1, args.steps // batch)
    print(f"batch {N} envs x {T} steps = {batch}; {args.minibatches} minibatches of {mb}; "
          f"{args.epochs} epochs; {iters} iterations; timeout_as_terminal={args.timeout_as_terminal}")
    ep_ret, recent, t0 = np.zeros(N), [], time.time()
    for it in range(iters):
        O = np.zeros((T, N, obs.shape[1]), np.float32)
        A = np.zeros((T, N, 1), np.float32)
        LP, R, V, NV = (np.zeros((T, N), np.float32) for _ in range(4))
        TERM, TRUNC = np.zeros((T, N)), np.zeros((T, N))
        for t in range(T):
            with torch.no_grad():
                ot = torch.as_tensor(obs)
                d = model.dist(ot)
                a = d.sample()
                LP[t] = d.log_prob(a).sum(-1).numpy()
                V[t] = model.value(ot).numpy()
            O[t], A[t] = obs, a.numpy()
            next_obs = np.zeros_like(obs)
            final_obs = np.zeros_like(obs)
            for i, env in enumerate(envs):
                o2, r, term, trunc, _ = env.step(np.clip(A[t, i], -1, 1) * act_scale)
                R[t, i] = r * args.reward_scale
                ep_ret[i] += r
                final_obs[i] = o2
                if args.timeout_as_terminal:
                    term, trunc = term or trunc, False
                TERM[t, i], TRUNC[t, i] = term, trunc
                if term or trunc:
                    recent.append(ep_ret[i])
                    ep_ret[i] = 0.0
                    o2, _ = env.reset()
                next_obs[i] = o2
            with torch.no_grad():
                NV[t] = model.value(torch.as_tensor(final_obs)).numpy()  # V of the true next state
            obs = next_obs.astype(np.float32)
        adv, ret = rl_math.gae(R, V, NV, TERM, TRUNC, args.gamma, args.lam)
        adv = rl_math.normalize(adv)
        flat = lambda x: torch.as_tensor(x.reshape(batch, *x.shape[2:]), dtype=torch.float32)  # noqa: E731
        bO, bA, bLP, bAdv, bRet = map(flat, (O, A, LP, adv, ret))
        stop = False
        for _ in range(args.epochs):
            perm = torch.randperm(batch)
            for k in range(args.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                d = model.dist(bO[idx])
                lp = d.log_prob(bA[idx]).sum(-1)
                ratio = (lp - bLP[idx]).exp()
                pg = -torch.min(ratio * bAdv[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * bAdv[idx]).mean()
                vloss = 0.5 * (model.value(bO[idx]) - bRet[idx]).pow(2).mean()
                loss = pg + args.vf_coef * vloss - args.ent_coef * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
            with torch.no_grad():
                new_lp = model.dist(bO).log_prob(bA).sum(-1).numpy()
            kl = rl_math.kl_estimators(bLP.numpy(), new_lp)["k3"]
            if kl > 1.5 * args.target_kl:
                stop = True
                break
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            ev = rl_math.explained_variance(V.ravel(), ret.ravel())
            cf = rl_math.clip_fraction(bLP.numpy(), new_lp, args.clip)
            mean_ret = np.mean(recent[-20:]) if recent else float("nan")
            print(f"it {it:4d}  steps {batch * (it + 1):7d}  return(last 20 eps) {mean_ret:8.1f}  "
                  f"kl {kl:.4f}{' (early stop)' if stop else ''}  clipfrac {cf:.2f}  "
                  f"explvar {ev if ev is None else round(ev, 2)}  std {model.log_std.exp().mean().item():.2f}  "
                  f"{time.time() - t0:.0f}s")
    score = evaluate(model, 10, seed=10_000 + args.seed, act_scale=act_scale)
    print(f"deterministic eval over 10 episodes: {score:.1f}  (random policy is about -1200; solved is about -200)")
    return {"eval_return": score}


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout-as-terminal", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse())
