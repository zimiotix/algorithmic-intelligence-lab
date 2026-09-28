"""Seeds for deterministic randomness.

A run is fully determined by one integer seed (shown in the Lab, so any run can be
replayed by typing it back in). Two helpers:

* ``fresh_seed()``: a new, unpredictable seed for a new run (launch, "New seed").
* ``step_seed(seed, step)``: the seed for one simulation step's GPU random numbers.
  Kernels call ``wp.rand_init(step_seed, agent_id)``, which is
  ``pcg(step_seed + pcg(agent_id))``. If the step were simply *added* to the seed, agent
  i at step s would get exactly the random numbers of agent j at step s + pcg(i) - pcg(j):
  two agents replaying each other's "random" choices, shifted in time. Hashing the step
  (splitmix64) scatters consecutive steps across the whole range, so no two agents'
  streams line up.
"""

from __future__ import annotations

import secrets

SEED_MAX = 1_000_000          # seeds shown in the Lab: 0 .. 999,999
_M64 = (1 << 64) - 1


def fresh_seed() -> int:
    return secrets.randbelow(SEED_MAX)


def step_seed(seed: int, step: int) -> int:
    """splitmix64 of (seed, step), folded to a non-negative 31-bit int for Warp."""
    z = (seed * 0x9E3779B97F4A7C15 + step + 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return (z ^ (z >> 31)) % (2**31 - 1)
