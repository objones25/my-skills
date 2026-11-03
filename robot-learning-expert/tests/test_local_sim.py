import numpy as np
import pytest

import local_sim as ls
import mdp_tools as mt


def _lock_sim(protocol, **kw):
    P, r, secret = mt.combination_lock(H=6, A=3, seed=0)
    mu = np.zeros(8)
    mu[0] = 1.0
    return ls.TabularSim(P, r, mu, H=7, protocol=protocol, **kw), secret


def test_online_refuses_resets_to_states():
    sim, _ = _lock_sim("online")
    sim.reset()
    with pytest.raises(ls.ProtocolViolation):
        sim.checkpoint()


def test_local_refuses_arbitrary_states_but_generative_allows():
    sim, _ = _lock_sim("local")
    with pytest.raises(ls.ProtocolViolation):
        sim.restore_state(3, 3)
    gen, _ = _lock_sim("generative")
    assert gen.restore_state(3, 3) == 3


def test_restore_replays_from_checkpoint():
    sim, secret = _lock_sim("local")
    sim.reset()
    sim.step(int(secret[0]))
    tok = sim.checkpoint()
    o1 = sim.step(int(secret[1]))[0]
    sim.restore(tok)
    assert sim.h == 1
    assert sim.step(int(secret[1]))[0] == o1 == 2


def test_budget_enforced():
    sim, _ = _lock_sim("local", max_queries=3)
    sim.reset()
    for _ in range(3):
        sim.step(0)
    with pytest.raises(ls.BudgetExceeded):
        sim.step(0)


def test_frontier_search_beats_exhaustive():
    sim, _ = _lock_sim("local", seed=1)
    used = ls.frontier_search_lock(sim, dead_obs=7, goal_obs=6)
    assert used <= 6 * 3


def test_noisy_obs_resample_vs_frozen():
    P, r, _ = mt.combination_lock(H=3, A=2, seed=0)
    mu = np.zeros(5)
    mu[0] = 1
    f = ls.onehot_gaussian_obs(5, 0.3)
    for resample in (True, False):
        sim = ls.TabularSim(P, r, mu, H=4, obs_fn=f, resample_obs_on_restore=resample, seed=0)
        o0 = sim.reset()
        tok = sim.checkpoint()
        o1 = sim.restore(tok)
        assert np.array_equal(o0, o1) == (not resample)


def test_gym_deepcopy_restore_is_exact():
    gym = pytest.importorskip("gymnasium")
    env = gym.make("CartPole-v1")
    sim = ls.GymLocalSim(env, H=50)
    sim.reset()
    sim.step(0)
    tok = sim.checkpoint()
    a = [sim.step(x)[0] for x in (1, 1, 0)]
    sim.restore(tok)
    b = [sim.step(x)[0] for x in (1, 1, 0)]
    assert np.allclose(a, b)


def _noisy_walk_env(gym):
    class NoisyWalk(gym.Env):
        """x' = x + a + N(0, 1); stochastic transitions drawn from self.np_random."""

        observation_space = gym.spaces.Box(-np.inf, np.inf, (1,))
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.x = np.zeros(1)
            return self.x.copy(), {}

        def step(self, a):
            self.x = self.x + a + self.np_random.standard_normal(1)
            return self.x.copy(), 0.0, False, False, {}

    return NoisyWalk()


@pytest.mark.parametrize("fresh", [True, False])
def test_gym_restore_rng_policy(fresh):
    gym = pytest.importorskip("gymnasium")
    sim = ls.GymLocalSim(_noisy_walk_env(gym), H=10, fresh_rng_on_restore=fresh)
    sim.reset()
    tok = sim.checkpoint()
    draws = []
    for _ in range(3):
        sim.restore(tok)
        draws.append(float(sim.step(1)[0][0]))
    assert (len(set(draws)) == 3) == fresh
