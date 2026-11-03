#!/usr/bin/env python3
"""Estimate FLOPs per frame (FPF), the model-size unit of Zheng et al. (2026).

Zheng et al. measure model size as FPF, "the number of FLOPs spent on each
observation during training", so total compute is C = FPF x D (D = frames). This
replaces the C = 6ND rule of dense language models, which does not hold for conv
nets or recurrent cores. The paper does not spell out its exact accounting, so this
script makes the assumption explicit:

    FPF = F_fwd * (1 + 3 * passes)

F_fwd   forward FLOPs for one observation, measured by XLA's cost analysis of the
        compiled forward pass (multiply-add counts as 2 FLOPs).
1       the forward pass that picks the action during the rollout.
3       forward + backward per training pass (backward ~ 2x forward, as in Kaplan
        et al. 2020).
passes  how many times each frame is trained on: 1 for A2C, update_epochs for PPO.

Off-policy methods reuse each frame many times, so for them pass
--passes = updates_per_step * batch_size / num_envs (the replay ratio).

Usage
  python flops_per_frame.py --obs 8268 --hidden 512 512 --actions 43 --passes 1
  python flops_per_frame.py --obs 8268 --hidden 512 512 --actions 43 --passes 4 --flops 1e17

As a library:
  from flops_per_frame import forward_flops
  f = forward_flops(lambda x: model.apply(params, x), example_obs)  # per observation
"""
from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp


def forward_flops(apply_fn, example_obs, batch: int = 64) -> float:
    """XLA-counted forward FLOPs per observation for any jittable apply_fn(obs_batch)."""
    x = jnp.broadcast_to(example_obs, (batch, *jnp.shape(example_obs)))
    compiled = jax.jit(apply_fn).lower(x).compile()
    cost = compiled.cost_analysis()
    if isinstance(cost, (list, tuple)):  # older jax returned a list with one dict
        cost = cost[0]
    if not cost or "flops" not in cost:
        raise RuntimeError("XLA cost analysis returned no FLOP count on this backend")
    return float(cost["flops"]) / batch


def mlp_apply(obs_dim: int, hidden: list[int], out_dim: int):
    sizes = [obs_dim, *hidden, out_dim]
    ws = [jnp.ones((a, b)) for a, b in zip(sizes[:-1], sizes[1:])]

    def apply(x):
        for w in ws[:-1]:
            x = jnp.tanh(x @ w)
        return x @ ws[-1]
    return apply


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obs", type=int, required=True, help="flattened observation size")
    p.add_argument("--hidden", type=int, nargs="+", required=True, help="hidden widths of the torso")
    p.add_argument("--actions", type=int, required=True, help="policy logits (critic head adds 1)")
    p.add_argument("--passes", type=float, default=1.0, help="training passes per frame (A2C 1, PPO epochs)")
    p.add_argument("--flops", type=float, help="budget C: also print frames D = C / FPF")
    args = p.parse_args()

    apply = mlp_apply(args.obs, args.hidden, args.actions + 1)
    f_fwd = forward_flops(apply, jnp.ones((args.obs,)))
    fpf = f_fwd * (1 + 3 * args.passes)
    print(f"forward FLOPs/obs : {f_fwd:.4g}")
    print(f"FPF (1 + 3*{args.passes:g}) : {fpf:.4g}")
    if args.flops:
        print(f"frames at C={args.flops:.3g}: {args.flops / fpf:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
