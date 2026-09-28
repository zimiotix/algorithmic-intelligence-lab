"""Random streams: every agent gets its own, and a run is fully set by its seed."""

import numpy as np

from ailab.core.rng import SEED_MAX, fresh_seed, step_seed


def _pcg(x: np.ndarray) -> np.ndarray:
    """Warp's rand_pcg (warp/native/rand.h), in numpy uint32 arithmetic."""
    x = x.astype(np.uint32)
    with np.errstate(over="ignore"):
        b = x * np.uint32(747796405) + np.uint32(2891336453)
        c = ((b >> ((b >> np.uint32(28)) + np.uint32(4))) ^ b) * np.uint32(277803737)
    return (c >> np.uint32(22)) ^ c


def _states(seeds: np.ndarray, n_agents: int) -> np.ndarray:
    """rand_init(seed, agent) = pcg(seed + pcg(agent)) for every (step, agent)."""
    agents = _pcg(np.arange(n_agents, dtype=np.uint32))
    with np.errstate(over="ignore"):
        return _pcg(seeds.astype(np.uint32)[:, None] + agents[None, :])


def _shifted_copies(states: np.ndarray) -> int:
    """Pairs of (agent, step) whose random state matches on two consecutive steps: one
    agent replaying another's random choices. Chance collisions of a 32-bit state happen;
    two in a row mean the streams are locked together."""
    pairs = (states[:-1].astype(np.uint64) << np.uint64(32)) | states[1:].astype(np.uint64)
    flat = pairs.ravel()
    return len(flat) - len(np.unique(flat))


def test_hashed_steps_give_every_agent_its_own_stream():
    steps, agents = 1500, 3000
    hashed = np.array([step_seed(42, s) for s in range(steps)])
    assert _shifted_copies(_states(hashed, agents)) == 0


def test_adding_the_step_to_the_seed_locks_agents_together():
    # the old scheme (seed + step): agents whose pcg(id) differ by less than the run
    # length replay each other. This is the bug the hash fixes.
    steps, agents = 1500, 3000
    added = (42 * 1_000_003 + np.arange(steps)) % (2**31 - 1)
    assert _shifted_copies(_states(added, agents)) > 0


def test_step_seeds_fit_warp_and_are_reproducible():
    s = [step_seed(7, k) for k in range(1000)]
    assert all(0 <= x < 2**31 - 1 for x in s)
    assert s == [step_seed(7, k) for k in range(1000)]
    assert len(set(s)) == len(s)


def test_fresh_seeds_are_in_the_lab_range():
    assert all(0 <= fresh_seed() < SEED_MAX for _ in range(100))
