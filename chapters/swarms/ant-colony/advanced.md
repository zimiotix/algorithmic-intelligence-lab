# Intuition

## One ant, four steps, sixty times a second

1. **Sense.** Smell at three points ahead: left, front, right.
2. **Decide.** Turn toward the strongest smell, plus a little random wiggle.
3. **Act.** Step forward, or turn around at a wall.
4. **Mark.** Drop pheromone. The longer it has been since the ant left the nest (or found
   food), the weaker the mark, because a long walk means a worse route.

## Two kinds of memory

- **Memory in the world** (environmental memory): the pheromone grid. It is shared by all
  ants and it forgets on its own through evaporation.
- **Memory in the ant** (working memory): a timer since it last saw home or food, and a
  rough sense of the direction home (path integration). Desert ants really do count
  their steps and track their turns.

## Why short paths win

Imagine two paths, one twice as long. An ant on the short path completes a round trip in
half the time, so it lays twice as many fresh marks per minute. Evaporation removes the
same fraction from both. The short path ends up with a stronger smell, more ants choose
it, and it gets stronger still. This is **positive feedback**, balanced by evaporation
(negative feedback).

## Why ants perceive smells on a log scale

If a busy trail smells 1,000 times stronger than a faint one, a straight linear sensor
would be blinded by it. Nothing else would register, not even the smell of the nest
right next to the ant. Animals, humans included, perceive **ratios**, not differences
(the Weber–Fechner law). The sensors here use $\ln(1 + c)$, which keeps both faint and
strong trails readable.

# The Math

## The pheromone field

Pheromone concentration $c(\mathbf{x}, t)$ obeys a reaction–diffusion equation:

$$
\frac{\partial c}{\partial t} = D \nabla^2 c - \frac{c}{\tau_e} + \sum_{k} q_k\, \delta(\mathbf{x} - \mathbf{x}_k)
$$

Diffusion spreads it ($D$), evaporation removes it with lifetime $\tau_e$, and every ant $k$
deposits at its position. On a grid with cell spacing 1 and time step $\Delta t$:

$$
c_{xy} \leftarrow \Big(c_{xy} + D\Delta t\,(c_{x+1,y} + c_{x-1,y} + c_{x,y+1} + c_{x,y-1} - 4c_{xy})\Big)\, e^{-\Delta t/\tau_e} + \text{deposits}
$$

This explicit scheme is only stable when $D\Delta t \le \tfrac14$, so the code clamps it.
Walls use a *no-flux* boundary: a blocked neighbour counts as having the same value.

## What an ant deposits

An ant that left its source $t_a$ seconds ago deposits, per step,

$$
\Delta c = q\, e^{-t_a/\tau_a}\, \Delta t
$$

so a trail's strength is effectively a *countdown of distance*: stronger near the source.

## The steering rule

Each ant has three smell sensors: left, front and right, reading $S_L, S_F, S_R$. The side
sensors point $\theta_s$ to either side, and all three sit $d_s$ ahead of the ant. A reading
adds up the pheromone $c$ in the 3×3 cells under the sensor and takes its logarithm, plus
fixed "cues" for food or the nest nearby. The decision $\sigma_{turn}$ is $+1$ (turn left),
$0$ (straight on) or $-1$ (turn right):

$$
S = \ln\!\Big(1 + \sum_{3\times3} c\Big) + \text{cues},
\qquad
\sigma_{turn} =
\begin{cases}
0 & S_F \ge \max(S_L, S_R)\\
+1 & S_L > S_R\\
-1 & S_R > S_L
\end{cases}
$$

The ant's heading $\theta$ then turns at the turn rate $\omega$. A loaded ant also feels
the direction of the nest $\theta_{nest}$ with strength $h$ ($[\text{loaded}]$ is 1 when it
carries food, else 0). Finally it wanders by $\sigma$ times a seeded random number $\xi$
between $-1$ and $1$:

$$
\theta \leftarrow \theta + \omega\Delta t\,\Big(\sigma_{turn} + h\,[\text{loaded}]\,\sin(\theta_{nest} - \theta)\Big) + \sigma\sqrt{\Delta t}\;\xi,
\qquad \xi \sim U(-1, 1)
$$

The $\sqrt{\Delta t}$ makes the random walk independent of the step size (it is a discrete
Brownian motion), and $h$ is the path-integration weight.

## The binary choice model

Deneubourg and Goss modelled the double bridge with a choice probability. If $A$ and $B$
ants have used branches A and B:

$$
P_A = \frac{(k + A)^n}{(k + A)^n + (k + B)^n}
$$

With $n \approx 2$ the choice is strongly non-linear: a small early lead gets amplified.
That is **symmetry breaking**. Even with two *equal* branches, the colony picks one.

## Determinism on a GPU

Thousands of ants add pheromone to the same cells at the same time. Floating-point
addition is not associative:

$$
(a + b) + c \ne a + (b + c) \quad\text{(in floating point)}
$$

so the result would depend on thread timing. Here deposits are converted to **fixed-point
integers** ($\times 4096$) and added with integer atomics, which are associative, so the
answer is identical every run. When two ants reach the last crumb in the same step, both
write their id with `atomic_min`, and the lowest id wins, whatever the order.

# Research

## Classic papers

- **S. Goss, S. Aron, J.-L. Deneubourg, J. M. Pasteels (1989)**, *Self-organized
  shortcuts in the Argentine ant*, Naturwissenschaften 76. The double-bridge experiment.
- **J.-L. Deneubourg, S. Aron, S. Goss, J. M. Pasteels (1990)**, *The self-organizing
  exploratory pattern of the Argentine ant*, J. Insect Behavior 3.
- **M. Dorigo, V. Maniezzo, A. Colorni (1996)**, *Ant system: optimization by a colony of
  cooperating agents*, IEEE Trans. SMC-B 26. This turned the idea into **Ant Colony
  Optimization**, used for routing and scheduling problems.
- **R. Wehner (2003)**, *Desert ant navigation: how miniature brains solve complex tasks*,
  J. Comp. Physiol. A 189. Path integration: the source of the *homing* term.
- **J. Jones (2010)**, *Characteristics of pattern formation and evolution in approximations
  of Physarum transport networks*, Artificial Life 16. The three-sensor agent design
  used here.
- **T. C. Schneirla (1944)**, *A unique case of circular milling in ants*, American Museum
  Novitates 1253. The ant mill.

## Things to investigate

- **Speed–accuracy trade-off.** Low $\tau_e$ adapts quickly to change (a blocked path) but
  finds worse paths; high $\tau_e$ is accurate but stubborn. Measure *delivered per
  minute* before and after you block the best trail.
- **Exploration vs. exploitation.** The wander $\sigma$ is the colony's "curiosity". Is
  there an optimal value? Does it depend on the scenario?
- **Symmetric bridge.** Make both branches equal (edit `build_world`). How often does each
  branch win across seeds? Compare with $P_A$ above.
- **From ants to algorithms.** Replace the grid with a graph and you have ACO for the
  travelling salesman problem. A future chapter does exactly that.

## Model vs. nature

This chapter uses two attractive pheromones and identical ants. Real colonies do more:

- **Negative pheromones.** Pharaoh's ants mark unrewarding branches with a repellent
  "no entry" signal that works alongside the attractive trail (E. J. H. Robinson et al.,
  *Nature* 438, 2005).
- **Crowding.** When a trail gets congested, ants are pushed onto the alternative route,
  so traffic splits instead of jamming (A. Dussutour et al., *Nature* 428, 2004).
- **Many signals and species.** Trail pheromones with different lifetimes, recruitment by
  tandem running (N. R. Franks & T. Richardson, *Nature* 439, 2006), and visual
  navigation in desert ants (R. Wehner, *Desert Navigator*, 2020). See T. J. Czaczkes et
  al., *Trail pheromones: an integrative view*, Annu. Rev. Entomol. 60 (2015).

A good exercise: add a third, *repellent* channel to the `FieldMemory` and let ants lay it
where they turn back. Does the colony abandon dead ends faster?
