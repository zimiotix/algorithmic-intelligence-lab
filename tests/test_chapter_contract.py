"""The contract every chapter must honour. New chapters are covered automatically."""

import numpy as np
import pytest

from ailab.core.catalog import discover, load_sim_class
from ailab.core.sim import ScriptedInput, SimContext
from ailab.render.scene import Scene

CHAPTERS = discover().ordered()
STEPS = 45


def run(info, seed, device="cpu", steps=STEPS):
    sim = load_sim_class(info)(SimContext(device=device, seed=seed))
    x0, y0, x1, y1 = sim.world
    script = ScriptedInput("circle", ((x0 + x1) / 2, (y0 + y1) / 2), (x1 - x0) / 4, 2.0,
                           presses={5: ("C",)})
    for k in range(steps):
        sim.advance(script.at(k, sim.dt))
    return sim


@pytest.mark.parametrize("info", CHAPTERS, ids=lambda c: c.id)
def test_same_seed_same_run(info):
    assert run(info, 7).digest() == run(info, 7).digest()


@pytest.mark.parametrize("info", CHAPTERS, ids=lambda c: c.id)
def test_seed_matters(info):
    if not load_sim_class(info).seeded:
        pytest.skip("chapter is fully deterministic without a seed by default")
    assert run(info, 1).digest() != run(info, 2).digest()


@pytest.mark.parametrize("info", CHAPTERS, ids=lambda c: c.id)
def test_draw_and_hud(info):
    sim = run(info, 3, steps=5)
    for o in sim.OVERLAYS:
        sim.show.set(o.key, True)
    s = Scene()
    sim.draw(s)
    assert s.commands, "draw() produced nothing"
    for cmd in s.commands:
        for v in cmd.data.values():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                assert np.isfinite(v).all(), f"non-finite values in {cmd.kind}"
    assert sim.hud()


@pytest.mark.parametrize("info", CHAPTERS, ids=lambda c: c.id)
def test_params_are_safe_at_their_limits(info):
    sim = run(info, 1, steps=1)
    for p in sim.PARAMS:
        assert p.clamp(p.default) == p.default, f"{p.key}: default outside its range"
        values = list(p.choices) if p.choices else [p.lo, p.hi, p.default]
        if p.kind == "int" and p.key.startswith("n_"):
            values = [p.lo, p.default]          # huge agent counts are slow on CPU
        for v in values:
            sim.set_param(p.key, v)
            sim.advance(ScriptedInput().at(0, sim.dt))
        sim.set_param(p.key, p.default)


def test_gpu_runs_are_deterministic(request):
    if not request.config.getoption("--gpu"):
        pytest.skip("pass --gpu to check CUDA determinism")
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    for info in CHAPTERS:
        if info.compute == "gpu":
            assert run(info, 5, "cuda:0").digest() == run(info, 5, "cuda:0").digest(), info.id
