# Algorithmic Intelligence Lab

**Watch intelligence emerge from simple, exact rules.**

A native Linux app for learning *deterministic* intelligent algorithms (no neural
networks) by playing with them. Every chapter is a live simulation you interact with using
the mouse and keyboard. Overlays show what the algorithm senses and decides, and a three-level
Deep dive (Intuition, The Math, Research) explains it for high-school students and
researchers alike.

![Home](docs/img/home.jpg)

| Fish school (you are the predator) | Ant colony (stigmergy) | Lidar racer (Follow-the-Gap) |
|---|---|---|
| ![fish](docs/img/fish.jpg) | ![ants](docs/img/lab-ants.jpg) | ![racer](docs/img/lab-racer.jpg) |

## Quick start

```bash
uv sync
uv run ailab
```

Needs Linux, Python 3.12+, OpenGL 3.3. An NVIDIA GPU is used automatically when present;
otherwise everything runs on the CPU. Optional: a TeX install (`latex`, `dvisvgm`) for
typeset equations.

```bash
uv run ailab --sysinfo                              # what your machine brings
uv run ailab --list                                 # all chapters
uv run ailab --chapter swarms.fish-school           # open one directly
uv run ailab --device cpu                           # force the CPU backend
uv run ailab --render-gpu nvidia                    # hybrid laptops: draw on the dGPU
./tools/install_desktop.sh                          # add it to your app launcher
```

## System-aware by design

Every launch probes the machine (CPU, RAM, every GPU, NVIDIA driver and CUDA, which GPU
draws the window, and TeX) and shows the results in a startup system check, on the home
page and in the sidebar. Heavy chapters run on the GPU through CUDA, and light ones stay on
the CPU. On a CPU-only machine, agent counts scale down automatically. On hybrid laptops
the iGPU usually draws while the dGPU computes, and the app shows you which is which.

## What is inside

| Layer | Choice | Why |
|---|---|---|
| UI | Qt 6 (PySide6 Essentials), native widgets, Wayland/X11 | Native Linux look and feel |
| Compute | NVIDIA Warp | One kernel source runs on CUDA **or** CPU |
| Rendering | ModernGL: SDF shapes, 4× MSAA, HDR bloom, procedural backgrounds | Crisp at any zoom, and it looks good |
| Maths text | Markdown + LaTeX → SVG (cached) | Real typeset equations, rendered natively |
| Exact maths | SymPy in the test suite | Every equation shown is checked against the code |
| Explainer videos | Manim Community Edition (optional) | Deep-dive animations |

### Determinism

Same seed + same input ⇒ the same run, bit for bit, on a given device. Simulations use a
fixed time step, seeded random numbers, double-buffered GPU kernels and integer atomics
(float atomics would make results depend on thread timing). `tests/test_chapter_contract.py`
enforces this for **every** chapter automatically.

### Memory systems

Algorithms that need memory use one of three shared kinds (`ailab/core/memory.py`):

- **Environmental memory** (`FieldMemory`): written into the world, evaporates and diffuses.
  Ant pheromones, a car's occupancy map.
- **Working memory** (`forget`): a belief held by one agent that fades over time, e.g. a
  fish remembering where it last saw the predator.
- **Episodic trace** (`TraceMemory`): a ring buffer of recent observations, e.g. trails.

## Repository layout

```
ailab/                 the engine (never chapter-specific)
  core/                simulation contract, params, clock, memory, system probe, catalogue
  render/              Scene API (numpy) + ModernGL renderer + shaders
  text/                markdown + LaTeX
  app/                 Qt application (boot check, main window, viewport)
chapters/
  tracks.toml          curriculum tracks (order, colour)
  <track>/<slug>/      one self-contained chapter: chapter.toml, sim.py, intro.md,
                       advanced.md, tests/, media/ (optional Manim scenes)
templates/chapter/     what `tools/new_chapter.py` copies
tests/                 engine tests + the per-chapter contract
docs/                  roadmap, images
```

Chapters are discovered from `chapter.toml` only, and their code is imported the first
time a chapter is opened, so the catalogue scales to hundreds of chapters.

## Contributing a chapter

See [CONTRIBUTING.md](CONTRIBUTING.md). In short:

```bash
uv run tools/new_chapter.py planning a-star "A*: The Shortest Path"
uv run ailab --chapter planning.a-star
uv run pytest chapters/planning/a-star tests
```

## License

MIT. See [LICENSE](LICENSE).
