"""GAE for a PureJaxRL-style rollout from mprl_common.AutoReset outputs."""
import jax
import jax.numpy as jnp


def gae(rewards, values, final_values, terminated, truncated, last_value, gamma=0.99, lam=0.95):
    """rewards, values, terminated, truncated, final_values: [T, N]; last_value: [N].

    values[t] = V(obs_t); final_values[t] = V(final_obs_t), the pre-reset observation of step t;
    last_value = V(obs_T), the observation after the last step of the rollout.
    """
    rewards = rewards + gamma * final_values * truncated
    done = jnp.logical_or(terminated, truncated).astype(rewards.dtype)

    def step(carry, x):
        adv, next_value = carry
        r, v, d = x
        delta = r + gamma * next_value * (1.0 - d) - v
        adv = delta + gamma * lam * (1.0 - d) * adv
        return (adv, v), adv

    _, advantages = jax.lax.scan(
        step, (jnp.zeros_like(last_value), last_value), (rewards, values, done), reverse=True
    )
    return advantages, advantages + values
