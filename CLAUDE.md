# Notes for AI assistants working on this repo

- Python via `uv` (`uv sync`, `uv run ...`). Python ≥ 3.12; the venv is pinned to 3.13.
- Run `uv run pytest -q` (Warp CPU backend, ~minutes) and `uv run ruff check ailab chapters tools`.
- `CONTRIBUTING.md` is the chapter-authoring contract; follow it for any new chapter.
- Never let ruff rewrite `float(0.0)` / `int(0)` in Warp kernels (UP018 is disabled on purpose).
- Visual check without a window: `uv run tools/snap.py <id> out.png` (EGL, headless).
  App window check: `AILAB_CAPTURE=out.png,<secs>[,<tab>[,<section>]] uv run ailab --no-boot --chapter <id>`.
- Equations shown in `advanced.md` must match the code and have a test in the chapter's `tests/`.
- Generated artefacts (thumbnails, equation SVGs, videos) go to `~/.cache/ailab` or are gitignored.
- Lab kit per chapter: `TOOLS`, `EXPERIMENTS` (+ `check` methods), `LEGEND` (colour key),
  `LIVE_MATH` (+ `terms` and `live_math()`), `reality` and `[glossary]` in chapter.toml. The contract tests
  enforce them, and every `Param` needs a `help` text (it is the hover card).
- `PARAMS` grouped with `section(...)`, plus `PRESETS`; no magic constants in kernels
  (a behaviour-shaping number is a `Param`). The contract test checks presets.
- Kernel randomness: `wp.rand_init(step_seed(seed, step), id)` (ailab.core.rng), never
  `seed + step`. Keep Warp struct arguments small (split them): a ~35-field struct crashed
  on CUDA only. Check new kernels on the GPU with `wp.config.verify_cuda = True`.
- The app starts each launch with a fresh seed; `--seed N` replays a run.
- Terms before equations: define symbols in words before each display equation.
- Cross-platform: `encoding="utf-8"` on all text IO, app files via `ailab.core.paths`,
  headless GL via `snapshot.standalone_context()`. CI runs Ubuntu, Windows and macOS.
