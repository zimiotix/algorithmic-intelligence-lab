# Writing a chapter

A chapter is one folder. Nothing outside it needs to change.

```bash
uv run tools/new_chapter.py <track> <slug> "Title"
```

## Checklist

1. **`chapter.toml`**: title, one-line summary, difficulty 1–5, tags, `compute`
   (`gpu` only when it needs thousands of agents), `memory` kinds used, `controls`,
   a `reality` note (**model vs. nature**: what the model simplifies, honestly) and a
   `[glossary]` of every non-parameter symbol used in *The Math*, in plain words.
   Keep `status = "draft"` until everything below is done.
2. **`sim.py`**: `class Sim(Simulation)` with
   - `PARAMS`: every tunable number, with a `symbol` that matches the maths text and a
     `help` text (shown when the learner hovers the slider: say what raising it does).
     No magic constants in kernels: if a number shapes behaviour, it is a `Param`.
     Group them with `section("Senses", Param(...), ...)`; groups fold in the Controls
     panel and only the first starts open, so the panel stays calm;
   - `PRESETS`: named starting points (`Preset(key, title, values, tip)`). Parameters a
     preset doesn't mention go back to their defaults, so a preset is always repeatable;
   - `playback`: the speed the Lab starts at, if 1× is too slow to see the idea;
   - `OVERLAYS`: the "see inside" layers (what it senses, what it decides, why);
   - `TOOLS`: what the learner can hold in the Lab (hotbar, keys 1–9). Read the active
     one with `self.tool_of(inp)`; the first tool is the default;
   - `EXPERIMENTS`: guided things to try, each with the idea it reveals (`learn`) and,
     when possible, a `check` method that ticks it off automatically;
   - optionally `follow()`: return `(x, y, heading, zoom)` to get a chase camera that
     turns with an agent (keyboard controls then match the screen);
   - `LEGEND`: a colour key (`Swatch(color, label, shape)`), shown in the Guide, so
     every colour on screen has a meaning the learner can look up;
   - `LIVE_MATH` + `live_math()`: the key equations with the focus agent's numbers
     plugged in. Give each `LiveEq` its `terms` ("v: speed · a_b: braking"), so the
     symbols are defined before the equation is shown;
   - `reset(seed)`: build all state from `seed`, the only randomness allowed;
   - `step(inp)`: advance exactly `self.dt`;
   - `draw(scene)`: pure drawing, no state changes that affect the simulation;
   - `state_arrays()`: everything that defines the state (used for determinism tests).
3. **`intro.md`**: the story, for a curious 15-year-old. No equations. End with
   "Things to try" experiments.
4. **`advanced.md`**: exactly three `#` sections: `# Intuition`, `# The Math`,
   `# Research`. Write the maths **as implemented**, not as in a textbook, if they differ.
   **Define the terms before each equation**, in words. The app also puts a notation
   primer and a symbol table (from `PARAMS` + `[glossary]`) at the top of *The Math*.
   End *Research* with a `## Model vs. nature` section, and `intro.md` too.
5. **`tests/`**: for every equation in *The Math*, a test that checks the code against it
   (SymPy for derivations, a tiny hand-built situation for the kernel).
6. Run `uv run pytest` and `uv run ruff check ailab chapters tools`, then open the chapter
   in the app and try every overlay and parameter extreme.

## Rules that keep hundreds of chapters sane

- **Engine code is generic.** If a chapter needs something new (a sprite, a memory kind,
  an input), add it to `ailab/` so every chapter can use it.
- **Determinism is not optional.** Use `np.random.default_rng(seed)` on the host. In
  kernels use `wp.rand_init(step_seed(seed, step), id)` with `ailab.core.rng.step_seed`:
  never `seed + step`, which makes agents replay each other's random numbers shifted in
  time (`tests/test_rng.py` shows it). On the GPU: double-buffer state, and use integer
  atomics (or `atomic_min` on ids) whenever order could matter.
- **Keep Warp struct arguments small.** A parameter struct that grew past about 35 fields
  crashed the ant kernel on CUDA (illegal memory access) while the CPU and debug builds were
  fine. Split settings into several small structs (the ants use `Colony` and `Marks`).
- **GPU only where it pays.** Hundreds of agents: numpy is fine and easier to read.
  Thousands and up: Warp kernels. The same Warp kernel runs on the CPU fallback.
- **Warp kernels:** declare loop-mutated variables with constructors (`x = float(0.0)`,
  `n = int(0)`). This is why ruff's UP018 rule is disabled.
- **No generated files in git.** Thumbnails, equation SVGs and rendered videos live in
  `~/.cache/ailab` or are gitignored; they are rebuilt from source.
- **Colours carry meaning:** `palette.REPULSE` (danger/away), `ALIGN`, `ATTRACT`
  (goal/toward), `SENSE` (perception), `MEMORY`. Use them so learners can read any chapter.
- **Folder names never carry numbers.** Order lives in `chapter.toml`/`tracks.toml`.
- **Build with the sandbox kit.** Walls and rocks learners paint go through
  `ailab/core/sandbox.py` (`ObstacleGrid`, `obstacle_at`, `obstacle_push`,
  `obstacle_clearance`), so every chapter's building tools feel the same.
- **Cross-platform.** Read and write text files with `encoding="utf-8"`, put app files
  under `ailab.core.paths`, and never assume a Linux-only tool exists.

## Useful commands

```bash
uv run tools/snap.py <chapter-id> out.png --frames 600 --zoom 3 --center focus
uv run tools/render_media.py <chapter-id>        # Manim scenes -> media/*.mp4
uv run pytest --gpu                              # also check CUDA determinism
AILAB_CAPTURE=shot.png,8,1 uv run ailab --chapter <id>   # screenshot the app window
AILAB_CAPTURE=shot.png,8,1,focus uv run ailab --chapter <id>   # ... in focus mode
AILAB_NO_LATEX=1 uv run ailab                    # see equations as users without TeX do
```
