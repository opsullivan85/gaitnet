import sys
from pathlib import Path

module_path = Path(__file__).parent.parent
sys.path.append(str(module_path))

import os

import numpy as np

from gaitnet.util import SharedMemoryVectorPool, VectorPool

# uneven split of objects across workers
NUM_INSTANCES = 7
NUM_WORKERS = 3


class Accumulator:
    """Stateful object with numeric, bool, None and non-numeric calls."""

    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.reset()

    def reset(self) -> None:
        self.total = np.zeros(3)

    def add(self, value: np.ndarray, weight: float) -> np.ndarray:
        self.total = self.total + self.scale * weight * value
        return self.total.reshape(3, 1).astype(np.float32)

    def norm(self, value: np.ndarray) -> float:
        return float(np.linalg.norm(self.total) + np.sum(value))

    def in_contact(self) -> np.ndarray:
        return self.total > 0

    def name_length(self, name: str) -> int:
        return len(str(name))


def run_calls(pool: VectorPool) -> list[np.ndarray]:
    """Run the same sequence of calls on a pool, returning every result."""
    rng = np.random.default_rng(0)
    n = NUM_INSTANCES
    results = []
    for _ in range(4):
        results.append(
            pool.call(
                Accumulator.add,
                mask=None,
                value=rng.standard_normal((n, 3)),
                weight=rng.standard_normal(n),
            )
        )
        results.append(pool.call(Accumulator.in_contact, mask=None))
        results.append(pool.call(Accumulator.norm, mask=None, value=rng.standard_normal((n, 3))))

    mask = np.arange(n) % 2 == 0
    # masked call on a signature whose results already go through shared memory
    results.append(pool.call(Accumulator.norm, mask=mask, value=rng.standard_normal((n, 3))))
    results.append(pool.call(Accumulator.reset, mask=mask))
    # a new argument shape gets its own buffers
    results.append(pool.call(Accumulator.norm, mask=None, value=rng.standard_normal((n, 5))))
    # non-numeric arguments go through the pipes
    names = np.array(["a" * i for i in range(n)])
    results.append(pool.call(Accumulator.name_length, mask=None, name=names))
    results.append(pool.call(Accumulator.add, mask=None, value=np.ones((n, 3)), weight=np.ones(n)))
    return results


def test_shared_memory_matches_pipes():
    with VectorPool(NUM_INSTANCES, Accumulator, NUM_WORKERS, scale=2.0) as pool:
        expected = run_calls(pool)
    with SharedMemoryVectorPool(NUM_INSTANCES, Accumulator, NUM_WORKERS, scale=2.0) as pool:
        actual = run_calls(pool)

    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert (a.dtype, a.shape) == (e.dtype, e.shape), f"call {i}"
        if e.dtype == object:
            # nan != nan, so compare object arrays by their repr
            assert repr(a.tolist()) == repr(e.tolist()), f"call {i}"
        else:
            np.testing.assert_array_equal(a, e, err_msg=f"call {i}")


def test_numeric_results_use_shared_memory():
    with SharedMemoryVectorPool(NUM_INSTANCES, Accumulator, NUM_WORKERS, scale=2.0) as pool:
        run_calls(pool)
        shared_output = {
            key[0]: channel.output is not None for key, channel in pool._channels.items()
        }

    assert shared_output["add"]
    assert shared_output["in_contact"]
    assert shared_output["norm"]
    assert not shared_output["reset"]
    assert "name_length" not in shared_output


def test_cleanup_unlinks_shared_memory():
    pool = SharedMemoryVectorPool(NUM_INSTANCES, Accumulator, NUM_WORKERS, scale=2.0)
    run_calls(pool)
    names = [block.name for channel in pool._channels.values() for block in channel.blocks]
    assert names

    pool._cleanup()
    for name in names:
        assert not os.path.exists(f"/dev/shm/{name}")
