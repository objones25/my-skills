"""Shared pieces for the JAX scaffolds: a truncation-aware auto-reset wrapper,
a fixed-instance evaluator, and result writers.

Why a custom wrapper: gymnax's Environment.step auto-resets inside the call and
returns a single `done` that is True both when the task really ended (termination)
and when the time limit ran out (truncation). Treating a truncation as a termination
tells the critic the episode's future is worth 0, which biases values in any task
whose episodes end by time limit (Pendulum never terminates, so every gymnax
Pendulum "done" is a truncation). This wrapper calls step_env/reset_env itself so
it can report the two separately and hand back the pre-reset final observation.

Bootstrapping trick used by both scaffolds: on a truncated step, fold the bootstrap
into the reward, r' = r + gamma * V(final_obs), then treat the step as terminal.
The GAE / lambda-return code then needs no special case. (Brax's PPO instead masks
the truncated step out of the loss; both avoid bootstrapping through the reset.)
"""
from __future__ import annotations

import csv
import os
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class WrappedState(NamedTuple):
    env_state: Any
    episode_return: jnp.ndarray
    episode_length: jnp.ndarray
    last_action: jnp.ndarray     # for sticky actions (discrete envs only)


class StepOut(NamedTuple):
    obs: jnp.ndarray             # observation after auto-reset (what the agent acts on next)
    state: WrappedState
    reward: jnp.ndarray
    terminated: jnp.ndarray      # true end of task: do not bootstrap
    truncated: jnp.ndarray       # time limit: bootstrap from final_obs
    final_obs: jnp.ndarray       # observation before the reset (equals obs when not done)
    episode_return: jnp.ndarray  # return of the episode that just ended (valid where done)
    episode_length: jnp.ndarray


def _max_steps(params) -> int | None:
    for name in ("max_steps_in_episode", "max_timesteps"):
        if hasattr(params, name):
            return getattr(params, name)
    return None


class AutoReset:
    """Wrap a gymnax-style env (gymnax, Craftax) with explicit termination/truncation.

    Requires env.step_env / env.reset_env (the non-resetting gymnax methods). The
    time-limit test is state.time >= params.max_steps_in_episode (or max_timesteps).
    Envs without those fields get truncated = False everywhere and a warning.
    Edge case: if the task ends on exactly the last allowed step, the step is
    labelled truncated (the wrapper cannot tell which condition fired).

    sticky_prob > 0 repeats the previous action with that probability, as the
    original MinAtar does (sticky_action_prob=0.1 by default; gymnax drops it).
    gymnax MinAtar games have 1 or 2 start states and key-independent dynamics,
    so without stickiness every "evaluation instance" replays the same game.
    The last action carries over across episodes, as in MinAtar.
    """

    def __init__(self, env, params, sticky_prob: float = 0.0):
        self.env, self.params, self.sticky_prob = env, params, sticky_prob
        self.max_steps = _max_steps(params)
        if self.max_steps is None:
            print("warning: env params expose no time limit; truncation will never be reported")
        if sticky_prob and not hasattr(env.action_space(params), "n"):
            raise ValueError("sticky actions are defined for discrete action spaces only")

    def reset(self, key):
        obs, s = self.env.reset_env(key, self.params)
        return obs, WrappedState(s, jnp.float32(0.0), jnp.int32(0), jnp.int32(0))

    def step(self, key, state: WrappedState, action) -> StepOut:
        k_step, k_reset, k_sticky = jax.random.split(key, 3)
        if self.sticky_prob:
            repeat = jax.random.uniform(k_sticky) < self.sticky_prob
            action = jnp.where(repeat, state.last_action, action).astype(jnp.int32)
        last_action = jnp.asarray(action, jnp.int32) if self.sticky_prob else state.last_action
        obs_st, s_st, reward, done, _ = self.env.step_env(k_step, state.env_state, action, self.params)
        done = jnp.asarray(done, bool)
        reward = jnp.asarray(reward, jnp.float32)
        if self.max_steps is not None and hasattr(s_st, "time"):
            truncated = done & (s_st.time >= self.max_steps)
        else:
            truncated = jnp.zeros_like(done)
        terminated = done & ~truncated
        obs_re, s_re = self.env.reset_env(k_reset, self.params)
        s_next = jax.tree.map(lambda a, b: jax.lax.select(done, a, b), s_re, s_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        ep_ret = state.episode_return + reward
        ep_len = state.episode_length + 1
        new = WrappedState(s_next, jnp.where(done, 0.0, ep_ret), jnp.where(done, 0, ep_len), last_action)
        return StepOut(obs, new, reward, terminated, truncated, obs_st, ep_ret, ep_len)


def instance_diversity(wrapper: AutoReset, n_instances: int, eval_seed: int) -> tuple[int, bool]:
    """(distinct start observations among the eval instances, whether a step depends on its key).

    If both are small/False, a deterministic policy plays at most that many distinct
    games, however many instances you evaluate, and an Elo league built on them has
    almost nothing to compare.
    """
    base = jax.random.key(eval_seed)
    keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(jnp.arange(n_instances))
    obs, state = jax.vmap(wrapper.reset)(keys)
    distinct = len({np.asarray(o).tobytes() for o in np.asarray(obs)})
    s0 = jax.tree.map(lambda x: x[0], state)
    space = wrapper.env.action_space(wrapper.params)
    n_act = getattr(space, "n", None)

    def play(seed):  # 20 steps of a fixed, varying action sequence under different keys
        def body(st, t):
            a = (t % n_act).astype(jnp.int32) if n_act else jnp.zeros(space.shape)
            out = wrapper.step(jax.random.fold_in(jax.random.key(seed), t), st, a)
            return out.state, out.final_obs
        _, obs_seq = jax.lax.scan(body, s0, jnp.arange(20))
        return np.asarray(obs_seq)

    return distinct, bool(not np.array_equal(play(1), play(2)))


def warn_if_degenerate_eval(wrapper: AutoReset, n_instances: int, eval_seed: int) -> None:
    distinct, stochastic = instance_diversity(wrapper, n_instances, eval_seed)
    if not stochastic and distinct < min(n_instances, 32):
        print(f"warning: only {distinct} distinct start states and key-independent steps: a deterministic "
              f"policy plays at most {distinct} distinct games across {n_instances} eval instances. "
              f"Add --sticky-prob 0.1 (MinAtar's own default) or evaluate a stochastic policy.")


def make_env(name: str):
    """Return (env, params) for a gymnax or Craftax environment name."""
    if name.startswith("Craftax"):
        try:
            from craftax.craftax_env import make_craftax_env_from_name
        except ImportError as e:
            raise SystemExit("craftax not installed: pip install craftax") from e
        env = make_craftax_env_from_name(name, auto_reset=False)
        return env, env.default_params
    import gymnax
    return gymnax.make(name)


def flat(x: jnp.ndarray) -> jnp.ndarray:
    """Flatten everything but the leading axis and cast to float32."""
    return x.reshape(x.shape[0], -1).astype(jnp.float32)


def evaluate(wrapper: AutoReset, act_fn: Callable, params, n_instances: int, eval_seed: int,
             max_steps: int | None = None) -> jnp.ndarray:
    """Return one episode return per evaluation instance, shape (n_instances,).

    Instance i always starts from the reset key fold_in(eval_seed, i), so every
    agent sees the same instances: the paired comparison the Elo league needs.
    act_fn(params, obs_batch, key) -> actions.
    """
    horizon = max_steps or wrapper.max_steps
    if horizon is None:
        raise ValueError("pass max_steps for envs without a time limit")
    base = jax.random.key(eval_seed)
    keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(jnp.arange(n_instances))
    obs, state = jax.vmap(wrapper.reset)(keys)

    def body(carry, t):
        obs, state, ret, alive = carry
        k = jax.random.fold_in(base, 10_000_000 + t)
        action = act_fn(params, obs, k)
        out = jax.vmap(wrapper.step)(jax.random.split(k, n_instances), state, action)
        ret = ret + out.reward * alive
        alive = alive * (1.0 - (out.terminated | out.truncated))
        return (out.obs, out.state, ret, alive), None

    init = (obs, state, jnp.zeros(n_instances), jnp.ones(n_instances))
    (_, _, ret, _), _ = jax.lax.scan(body, init, jnp.arange(horizon))
    return ret


def append_csv(path: str, header: list[str], rows: list[list]) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def write_eval_row(path: str, agent: str, returns) -> None:
    """Append one agent's per-instance returns in the wide format elo_league.py reads."""
    returns = np.asarray(returns, dtype=float)
    append_csv(path, ["agent", *[f"ep{i}" for i in range(len(returns))]], [[agent, *returns.tolist()]])
