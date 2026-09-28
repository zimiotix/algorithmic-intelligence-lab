# Intuition

## One ant, four steps, sixty times a second

1. **Sense.** Smell at three points ahead: left, front, right.
2. **Decide.** Turn toward the strongest smell, plus a little random wiggle.
3. **Act.** Step forward; at a wall, slide along it and keep it at one side.
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

An ant that left its source $t_a$ seconds ago deposits, per step, an amount set by the
pheromone rate $q$ and, for an ant carrying food, the food's **quality** $Q$ (1 for
ordinary food, $Q_{rich}$ for rich food; searchers always use $Q = 1$):

$$
\Delta c = q\,Q\, e^{-t_a/\tau_a}\, \Delta t
$$

so a trail's strength is effectively a *countdown of distance*: stronger near the source.
And better food gets a stronger trail. *Lasius niger* workers lay more trail for richer
sugar (Beckers, Deneubourg and Goss, 1993). In the **Two foods** scenario the nest's
corridor forks into two equal branches: with equal food the colony still settles on one
branch at random (symmetry breaking, as on the double bridge), and with one richer pile it
picks the rich one. We measured 87% of the food taken from the rich branch, on every one
of eight seeds. No ant ever compares the two piles.

## The steering rule

Each ant has three smell sensors: left, front and right, reading $S_L, S_F, S_R$. The side
sensors point $\theta_s$ to either side, and all three sit $d_s$ ahead of the ant. A reading
adds up the pheromone $c$ in the 3×3 cells under the sensor and takes its logarithm, plus
cues for food or the nest nearby (their strengths are the Food smell and Nest smell sliders). The decision $\sigma_{turn}$ is $+1$ (turn left),
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

The ant's heading $\theta$ then turns at the turn rate $\omega$. How much a smell pulls
depends on the ant's **trail loyalty** $f_i$ (its personality, below). A loaded ant also
feels the direction of the nest $\theta_{nest}$ with strength $h$ ($[\text{loaded}]$ is 1
when it carries food, else 0). Finally it wanders by its own wander strength $\sigma_i$
times a seeded random number $\xi$ between $-1$ and $1$:

$$
\theta \leftarrow \theta + \omega\Delta t\,\Big(f_i\,\sigma_{turn} + h\,[\text{loaded}]\,\sin(\theta_{nest} - \theta)\Big) + \sigma_i\sqrt{\Delta t}\;\xi,
\qquad \xi \sim U(-1, 1)
$$

The $\sqrt{\Delta t}$ makes the random walk independent of the step size (it is a discrete
Brownian motion), and $h$ is the path-integration weight.

## No two ants alike

Real workers differ: some are bold, fast explorers, others stick faithfully to trails.
When the colony is created, each ant $i$ draws three random numbers $z_1, z_2, z_3$ from a
standard normal distribution (average 0, spread 1), once, from the seed. With the
*individual variety* $\kappa$, they set its walking speed $v_i$, its wander $\sigma_i$ and
its trail loyalty $f_i$ around the colony's values $v$ and $\sigma$:

$$
v_i = v\,e^{\kappa z_1}, \qquad \sigma_i = \sigma\,e^{\kappa z_2}, \qquad f_i = e^{\kappa z_3}
$$

The exponential keeps every factor positive and makes "twice as fast" as likely as "half as
fast". With $\kappa = 0$ every factor is 1: identical ants, the classic model. Each ant
also gets a **wall side** $s_i$, $+1$ (left) or $-1$ (right) with equal chance.

## Following walls

An ant that bumps into a wall doesn't turn around. It turns away from its wall side in
small steps $\Delta_w$ until the way along the wall is free, and slides along it. From then
on it keeps the wall at its side (**thigmotaxis**; real ants do this, and in *Lasius niger*
the habit even spreads socially). A side feeler at angle $\alpha_w$ and reach $d_w$ checks
the wall is still there:

$$
\text{wall at } \mathbf{x} + d_w\,\hat{\mathbf{u}}(\theta + s_i\,\alpha_w)
\;\Rightarrow\; \text{keep following};\qquad
\text{otherwise}\;\; \theta \leftarrow \theta + \omega\Delta t\; s_i\, g_w
$$

where $\hat{\mathbf{u}}(\phi)$ is the unit vector pointing at angle $\phi$. If the wall
disappears (an outside corner), the ant turns back toward it with corner pull $g_w$, for up
to $T_w$ seconds. To avoid circling a pillar forever it lets go at random, with rate
$\lambda_w$ per second, so the chance of letting go during one step is
$1 - e^{-\lambda_w \Delta t}$.

**Why this solves mazes.** The Maze scenario is carved by a randomised depth-first search,
which makes a *perfect maze*: exactly one route between any two places, and no loops. Its
free space is one tree-shaped region, so all its walls form one single, unbroken wall. An
ant that never lets go of that wall walks along all of it and therefore passes every
corridor, including the food and, afterwards, the nest. This is the classic *hand-on-wall*
rule, and the chapter's tests check it in its pure form (no letting go, no smells, no
wander). Real ants, and ours by default, let go and follow smells too, so the guarantee
becomes a strong tendency rather than a proof. That mix is what makes the colony both
thorough and fast. Drawing your own walls can add loops, which is where the rule can fail.

## "No entry": the colony learns from failure

A trail can mislead. Ants that took a wrong turn with food mark a side corridor, and
searchers then follow that scent into a dead end, again and again. Pharaoh's ants solve
this with a second, *repellent* pheromone: a "no entry" mark near branches that don't pay
off (Robinson, Jackson, Holcombe and Ratnieks, 2005).

Here, each searching ant counts how long it has been following a food trail (its reading
$\ln(1 + c)$ at its own position is above $S_{trail}$) without finding food. After $T_f$
seconds of this frustration it lays a "no entry" mark $n$ at rate $q_{ne}$ wherever it is
still on the trail, until it finds food or gets home. The marks live in their own layer
of the ground, fade faster than trails (over $\tau_{ne}$), and lower what a searcher's
sensors read:

$$
S = \ln\!\Big(1 + \sum_{3\times3} c\Big) - k_{ne}\,\ln\!\Big(1 + \sum_{3\times3} n\Big) + \text{cues}
$$

with $k_{ne}$ the no-entry strength. A marked stretch of trail smells weaker than the
unmarked way, so searchers stop following it there. In our open field, where piles run out
and old trails keep pointing at empty places, the marks lifted deliveries by about 40%. In
the Maze the difference was within the noise.

## Navigator ants: noticing you are lost

Following a trail has a trap. If loaded ants end up following *each other's* marks in a
circle, every lap refreshes the circle and nobody leaves: an **ant mill**. An ant inside a
mill cannot see its shape. What it *can* notice is its own turning: on a normal trip left
and right turns cancel out, but in a mill it keeps turning the same way.

So each ant keeps a running total $W$ of its recent steering, in radians, where older
turning fades away over the turning memory $\tau_w$. Each step adds only the turn its
*smells* chose. A mill is ants following each other's trail round and round, so the pull
toward home, the random wander and wall following are left out; none of them count as
circling:

$$
W \leftarrow W\,e^{-\Delta t/\tau_w} + \omega\Delta t\; f_i\,\sigma_{turn}
$$

It also counts $t_{lost}$, the time since it last reached home or food, not counting time
spent following a wall. One full circle is $2\pi$ radians. An ant *carrying food* that has
turned more than $n_{lost}$ circles (mills are made of loaded ants), or *any* ant that has
been out longer than $T_{lost}$ (real foragers give up a fruitless search), decides it is
lost:

$$
|W| > 2\pi\,n_{lost} \;\vee\; t_{lost} > T_{lost}
$$

It faces the nest and becomes a **navigator** for up to $T_n$ seconds. A navigator stops
trusting smells: a trail can lead into a dead end (we saw lost ants pace up and down a side
corridor that wrong-way ants had scented). Instead it runs the **Bug algorithm**, a classic
of robot navigation (Lumelsky and Stepanov, 1987):

1. Head for home by compass, turning with strength $g_n$:
$$
\theta \leftarrow \theta + \omega\Delta t\; g_n \sin(\theta_{nest} - \theta) + \sigma_i\sqrt{\Delta t}\;\xi
$$
2. When a wall blocks the way, follow it (keeping it on side $s_i$, never letting go) and
   remember $d_{hit}$, the distance to home at the moment it met the wall.
3. Leave the wall when the way home is clear *and* it is closer than that:
$$
\text{free toward home} \;\wedge\; \lVert \mathbf{x}_{nest} - \mathbf{x} \rVert < d_{hit}
$$

Each wall is left closer to home than it was met, so the ant can't go round in a loop: it
keeps making progress until it arrives. A navigator lays no pheromone, so it stops feeding
a mill. At the nest, or at food, it becomes an ordinary ant again. After $T_n$ seconds it
trusts smells again for another $T_n$ seconds before it may retry.

**What we measured** (1,500 ants, four seeds, two to five minutes per run). In the Maze,
navigators roughly doubled deliveries, and with the Maze preset 99.9% of ants found food or
got home within ten minutes. But in the Two foods fork they cost about a sixth of the
deliveries: ants circling the round food rooms turn navigator and stop laying trail on the
way home. So navigators are **off by default** and switched on by the Maze and Mill rescue
presets, where getting lost is the real problem. Our first version, which backtracked
along the home trail, did worse than no navigators at all, because trails trapped the
navigators too. Real desert ants (*Cataglyphis*) do navigate home by a path-integration
compass; the "am I circling?" and Bug-algorithm rules are this Lab's inventions.

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
