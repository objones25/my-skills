"""Access-protocol wrappers for reset-based RL experiments (Power of Resets / RVFS project).

Three protocols, from weakest to strongest (Notes 1 §2.3, Notes 4 §2, Mhammedi-Foster-Rakhlin 2024):
    online      reset() only samples s0 ~ mu; trajectories are the only feedback.
    local       online, plus restore(token) to any state the learner has already
                visited (a token can only be minted at the current state). This is
                "local simulator access" (RLLS) in the Power of Resets paper.
    generative  local, plus restore_state(h, s) to an arbitrary state, never visited.

Every step() counts as one query, so sample complexity comparisons across
protocols are honest. Set max_queries to enforce a budget.

Noisy observations: the simulator keeps the TRUE latent state internally and
emits obs = obs_fn(s, rng). A token is a checkpoint of the latent state, so
restore() puts the system back in that latent state even though the learner
only ever saw a noisy observation of it. Choose what restore() emits:
    resample_obs_on_restore=True   fresh noise draw (the physical-simulator reading)
    resample_obs_on_restore=False  the same observation stored at checkpoint time
This choice changes what an algorithm can learn by repeated resets (re-sampling
lets it average noise away), so state it explicitly in any write-up.

Self-check / demo:  python local_sim.py
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

Array = np.ndarray
ObsFn = Callable[[int, np.random.Generator], Any]

PROTOCOLS = ("online", "local", "generative")


class BudgetExceeded(RuntimeError):
    pass


class ProtocolViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class Token:
    """Opaque checkpoint handle. Learners should not read the fields; they exist for logging."""

    uid: int
    h: int
    latent: Any = field(repr=False)
    obs: Any = field(repr=False)


@dataclass
class Counters:
    steps: int = 0
    resets: int = 0
    restores: int = 0
    episodes: int = 0


# ---------------------------------------------------------------------------
# Observation models for tabular latent states
# ---------------------------------------------------------------------------
def identity_obs() -> ObsFn:
    return lambda s, rng: int(s)


def onehot_gaussian_obs(S: int, sigma: float) -> ObsFn:
    """obs = onehot(s) + N(0, sigma^2 I). Decodable in practice only while sigma << 1/2."""
    eye = np.eye(S)
    return lambda s, rng: eye[s] + sigma * rng.standard_normal(S)


def block_obs(S: int, d: int, sigma: float, seed: int) -> tuple[ObsFn, Array]:
    """Rich observation in R^d around a random per-state center (a Block-MDP-style emission).

    With sigma well below half the minimum center distance the latent state is
    decodable from one observation (the Block MDP assumption). Raise sigma to break
    decodability. Returns (obs_fn, centers) so you can measure the separation.
    """
    centers = np.random.default_rng(seed).standard_normal((S, d))
    return (lambda s, rng: centers[s] + sigma * rng.standard_normal(d)), centers


def aliased_obs(S: int, p_alias: float) -> ObsFn:
    """With probability p_alias emit a uniformly random state id instead of s (never decodable for p_alias > 0)."""

    def f(s: int, rng: np.random.Generator) -> int:
        return int(rng.integers(S)) if rng.random() < p_alias else int(s)

    return f


# ---------------------------------------------------------------------------
# Tabular finite-horizon simulator
# ---------------------------------------------------------------------------
class TabularSim:
    """Episodic finite-horizon tabular MDP behind an access protocol.

    P (S, A, S) or (H, S, A, S); r (S, A) or (H, S, A); mu (S,).
    Episode steps are h = 0..H-1. step() returns (obs, reward, done, info) with
    info = {"h": h_next}; latent state is never exposed through step().
    """

    def __init__(
        self, P: Array, r: Array, mu: Array, H: int, protocol: str = "local",
        obs_fn: ObsFn | None = None, resample_obs_on_restore: bool = True,
        max_queries: int | None = None, seed: int = 0,
    ) -> None:
        if protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}")
        self.P = P if P.ndim == 4 else np.broadcast_to(P, (H, *P.shape))
        self.r = r if r.ndim == 3 else np.broadcast_to(r, (H, *r.shape))
        if self.P.shape[0] != H or self.r.shape[0] != H:
            raise ValueError("time-indexed P/r need leading dimension H")
        self.S, self.A = self.r.shape[1:]
        self.mu, self.H, self.protocol = np.asarray(mu, float), H, protocol
        self.obs_fn = obs_fn or identity_obs()
        self.resample = resample_obs_on_restore
        self.max_queries = max_queries
        self.rng = np.random.default_rng(seed)
        self.count = Counters()
        self._uid = 0
        self._h: int | None = None
        self._s: int | None = None
        self._obs: Any = None

    # -- protocol -----------------------------------------------------------
    def reset(self) -> Any:
        self.count.resets += 1
        self.count.episodes += 1
        self._h, self._s = 0, int(self.rng.choice(self.S, p=self.mu))
        self._obs = self.obs_fn(self._s, self.rng)
        return self._obs

    def step(self, a: int) -> tuple[Any, float, bool, dict]:
        if self._h is None:
            raise ProtocolViolation("call reset() or restore() before step()")
        if self._h >= self.H:
            raise ProtocolViolation("episode is over (h == H); reset or restore")
        if self.max_queries is not None and self.count.steps >= self.max_queries:
            raise BudgetExceeded(f"query budget {self.max_queries} used up")
        self.count.steps += 1
        h, s = self._h, self._s
        rew = float(self.r[h, s, a])
        p = self.P[h, s, a]
        self._s = int(min(np.searchsorted(np.cumsum(p), self.rng.random(), side="right"), self.S - 1))
        self._h = h + 1
        self._obs = self.obs_fn(self._s, self.rng)
        return self._obs, rew, self._h >= self.H, {"h": self._h}

    def checkpoint(self) -> Token:
        if self.protocol == "online":
            raise ProtocolViolation("online protocol has no checkpoints")
        if self._h is None:
            raise ProtocolViolation("nothing to checkpoint before reset()")
        self._uid += 1
        return Token(self._uid, self._h, self._s, self._obs)

    def restore(self, token: Token) -> Any:
        if self.protocol == "online":
            raise ProtocolViolation("online protocol cannot restore")
        self.count.restores += 1
        self._h, self._s = token.h, token.latent
        self._obs = self.obs_fn(self._s, self.rng) if self.resample else token.obs
        return self._obs

    def restore_state(self, h: int, s: int) -> Any:
        if self.protocol != "generative":
            raise ProtocolViolation("arbitrary-state restore needs the generative protocol")
        self.count.restores += 1
        self._h, self._s = h, int(s)
        self._obs = self.obs_fn(self._s, self.rng)
        return self._obs

    @property
    def h(self) -> int | None:
        return self._h


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------
class GymLocalSim:
    """Local-simulator access for a Gymnasium env, with optional Gaussian observation noise.

    Snapshot strategy, pick one:
      * default: deepcopy the whole env (includes its np_random). Works for
        classic-control envs; slow for big ones; fails for envs holding
        renderers or C handles.
      * get_state/set_state hooks, e.g. for MuJoCo envs:
            get_state = lambda e: (e.unwrapped.data.qpos.copy(), e.unwrapped.data.qvel.copy())
            set_state = lambda e, st: e.unwrapped.set_state(*st)
        Hooks do NOT capture the env RNG, so stochastic envs diverge after restore.
    Restores return an observation built from the restored env; with hooks the
    observation is recomputed via obs_from_env(env) which you must supply.

    RNG trap: a deepcopy snapshot also copies the env's np_random, so every
    restore would replay the SAME "random" transition. Algorithms that average
    over repeated samples from one state (RVFS's Monte Carlo backups and tests)
    then see zero variance. fresh_rng_on_restore=True (default) reseeds the
    restored env's np_random from this wrapper's RNG; set False only to replay
    a trajectory exactly for debugging.
    """

    def __init__(
        self, env: Any, H: int, protocol: str = "local", obs_noise_std: float = 0.0,
        get_state: Callable | None = None, set_state: Callable | None = None,
        obs_from_env: Callable | None = None, max_queries: int | None = None, seed: int = 0,
        fresh_rng_on_restore: bool = True,
    ) -> None:
        if protocol not in ("online", "local"):
            raise ValueError("GymLocalSim supports 'online' and 'local'")
        if (get_state is None) != (set_state is None):
            raise ValueError("pass both get_state and set_state, or neither")
        if get_state is not None and obs_from_env is None:
            raise ValueError("hook-based snapshots need obs_from_env to rebuild observations")
        self.env, self.H, self.protocol = env, H, protocol
        self.noise = obs_noise_std
        self.get_state, self.set_state, self.obs_from_env = get_state, set_state, obs_from_env
        self.max_queries = max_queries
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.fresh_rng = fresh_rng_on_restore
        self.count = Counters()
        self._uid = 0
        self._h: int | None = None
        self._clean_obs: Any = None

    def _noisy(self, obs: Any) -> Any:
        if self.noise == 0.0:
            return obs
        obs = np.asarray(obs, dtype=float)
        return obs + self.noise * self.rng.standard_normal(obs.shape)

    def reset(self) -> Any:
        self.count.resets += 1
        self.count.episodes += 1
        obs, _ = self.env.reset(seed=self.seed + self.count.resets)
        self._h, self._clean_obs = 0, obs
        return self._noisy(obs)

    def step(self, a: Any) -> tuple[Any, float, bool, dict]:
        if self._h is None or self._h >= self.H:
            raise ProtocolViolation("reset() or restore() first")
        if self.max_queries is not None and self.count.steps >= self.max_queries:
            raise BudgetExceeded(f"query budget {self.max_queries} used up")
        self.count.steps += 1
        obs, rew, term, trunc, info = self.env.step(a)
        self._h += 1
        self._clean_obs = obs
        done = bool(term or trunc or self._h >= self.H)
        return self._noisy(obs), float(rew), done, {"h": self._h, "terminated": bool(term), **info}

    def checkpoint(self) -> Token:
        if self.protocol == "online":
            raise ProtocolViolation("online protocol has no checkpoints")
        self._uid += 1
        snap = self.get_state(self.env) if self.get_state else copy.deepcopy(self.env)
        return Token(self._uid, self._h, snap, self._clean_obs)

    def restore(self, token: Token) -> Any:
        if self.protocol == "online":
            raise ProtocolViolation("online protocol cannot restore")
        self.count.restores += 1
        if self.set_state:
            self.set_state(self.env, token.latent)
            self._clean_obs = self.obs_from_env(self.env)
        else:
            self.env = copy.deepcopy(token.latent)
            self._clean_obs = token.obs
            if self.fresh_rng:
                self.env.unwrapped.np_random = np.random.default_rng(int(self.rng.integers(2**63)))
        self._h = token.h
        return self._noisy(self._clean_obs)


# ---------------------------------------------------------------------------
# Demo: why resets help on a combination lock
# ---------------------------------------------------------------------------
def frontier_search_lock(sim: TabularSim, dead_obs: Any, goal_obs: Any) -> int:
    """Toy reset-based explorer for the noiseless combination lock.

    Keeps a checkpoint at the deepest chain state reached; from it, tries actions
    until one does not lead to the dead state. Uses about H * (A + 1) / 2 steps on
    average instead of about A ** H episodes. Returns steps used to reach the goal.
    """
    sim.reset()
    frontier = sim.checkpoint()
    while True:
        for a in range(sim.A):
            sim.restore(frontier)
            obs, _, _, _ = sim.step(a)
            if obs == goal_obs:
                return sim.count.steps
            if obs != dead_obs:
                frontier = sim.checkpoint()
                break
        else:
            raise RuntimeError("every action from the frontier is dead; is the lock slippery?")


def _self_check() -> None:
    from mdp_tools import combination_lock

    H, A = 10, 4
    P, r, _ = combination_lock(H, A, seed=0)
    mu = np.zeros(H + 2)
    mu[0] = 1.0
    sim = TabularSim(P, r, mu, H=H + 1, protocol="local", seed=0)
    used = frontier_search_lock(sim, dead_obs=H + 1, goal_obs=H)
    assert used <= H * A
    online = TabularSim(P, r, mu, H=H + 1, protocol="online")
    online.reset()
    try:
        online.checkpoint()
        raise AssertionError("online protocol must refuse checkpoints")
    except ProtocolViolation:
        pass
    print(f"ok: local-access frontier search opened a {H}-step, {A}-action lock in {used} steps; "
          f"a uniform random policy succeeds with probability {A ** -H:.1e} per episode")


if __name__ == "__main__":
    _self_check()
