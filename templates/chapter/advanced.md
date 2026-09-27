# Intuition

Explain the mechanism with pictures-in-words, still without heavy maths.

# The Math

(The app puts a notation primer and a table of every symbol at the top of this page,
built from PARAMS and the `[glossary]` in chapter.toml.)

Define the terms **before** each equation, in words. For example: each agent is at
position $\mathbf{x}$ and moves with velocity $\mathbf{v}$; $\Delta t$ is one step. Then:

$$
\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \mathbf{v}_t \, \Delta t
$$

State every rule exactly as the code implements it. Every equation here must have a test
in `tests/` that checks the code against it.

# Research

- Original papers (author, year, title, venue).
- Open questions and experiments a researcher could run with this chapter.

## Model vs. nature

What the model simplifies or leaves out, with references to how the real system works.
