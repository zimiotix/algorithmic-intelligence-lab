# Writing a chapter

A chapter is one folder. Nothing outside it needs to change.

```bash
uv run tools/new_chapter.py <track> <slug> "Title"
```

## Checklist

1. **`chapter.toml`**: title, one-line summary, difficulty 1–5, tags, `compute`
   (`gpu` only when it needs thousands of agents), `memory` kinds used, `controls`.
   Keep `status = "draft"` until everything below is done.
2. **`sim.py`**: `class Sim(Simulation)` with
   - `PARAMS`: every tunable number, with a `symbol` that matches the maths text;
   - `OVERLAYS`: the "see inside" layers (what it senses, what it decides, why);
   - `reset(seed)`: build all state from `seed`, the only randomness allowed;
   - `step(inp)`: advance exactly `self.dt`;
   - `draw(scene)`: pure drawing, no state changes that affect the simulation;
   - `state_arrays()`: everything that defines the state (used for determinism tests).
3. **`intro.md`**: the story, for a curious 15-year-old. No equations. End with
   "Things to try" experiments.
4. **`advanced.md`**: exactly three `#` sections: `# Intuition`, `# The Math`,
   `# Research`. Write the maths **as implemented**, not as in a textbook, if they differ.
5. **`tests/`**: for every equation in *The Math*, a test that checks the code against it
   (SymPy for derivations, a tiny hand-built situation for the kernel).
6. Run `uv run pytest` and `uv run ruff check ailab chapters tools`, then open the chapter
   in the app and try every overlay and parameter extreme.

## Rules that keep hundreds of chapters sane

- **Engine code is generic.** If a chapter needs something new (a sprite, a memory kind,
  an input), add it to `ailab/` so every chapter can use it.
- **Determinism is not optional.** Use `np.random.default_rng(seed)` or
  `wp.rand_init(seed, id)`. On the GPU: double-buffer state, and use integer atomics (or
  `atomic_min` on ids) whenever order could matter.
- **GPU only where it pays.** Hundreds of agents: numpy is fine and easier to read.
  Thousands and up: Warp kernels. The same Warp kernel runs on the CPU fallback.
- **Warp kernels:** declare loop-mutated variables with constructors (`x = float(0.0)`,
  `n = int(0)`). This is why ruff's UP018 rule is disabled.
- **No generated files in git.** Thumbnails, equation SVGs and rendered videos live in
  `~/.cache/ailab` or are gitignored; they are rebuilt from source.
- **Colours carry meaning:** `palette.REPULSE` (danger/away), `ALIGN`, `ATTRACT`
  (goal/toward), `SENSE` (perception), `MEMORY`. Use them so learners can read any chapter.
- **Folder names never carry numbers.** Order lives in `chapter.toml`/`tracks.toml`.

## Useful commands

```bash
uv run tools/snap.py <chapter-id> out.png --frames 600 --zoom 3 --center focus
uv run tools/render_media.py <chapter-id>        # Manim scenes -> media/*.mp4
uv run pytest --gpu                              # also check CUDA determinism
AILAB_CAPTURE=shot.png,8,1 uv run ailab --chapter <id>   # screenshot the app window
```
