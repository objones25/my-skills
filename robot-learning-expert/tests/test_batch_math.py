import pytest

import batch_math as bm


def test_g1_shipped_config_numbers():
    b = bm.PPOBatch(4096)
    assert b.batch == 98_304
    assert b.minibatch == 24_576
    assert b.grad_steps_per_iter == 20
    assert b.sim_seconds_per_iter == pytest.approx(0.48)
    assert bm.PPOBatch(16384).minibatch == 98_304


def test_minibatch_grows_with_envs_when_count_fixed():
    sizes = [bm.PPOBatch(n).minibatch for n in (64, 256, 1024)]
    assert sizes == [384, 1536, 6144]


def test_validation():
    with pytest.raises(ValueError):
        bm.PPOBatch(0)
    with pytest.raises(ValueError):
        bm.PPOBatch(3, steps_per_env=1, num_mini_batches=2)


def test_cost():
    c = bm.cost(iters_per_min=57, minutes=45, price_per_hour=1.79, batch=98_304)
    assert c["iterations"] == pytest.approx(2565)
    assert c["dollars"] == pytest.approx(1.3425)
