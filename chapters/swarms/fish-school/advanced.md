# Intuition

## Each fish only answers three questions

Every 1/60 of a second, each fish looks at the neighbours it can currently sense and asks:

| Zone (distance $d$ to a neighbour) | Question | Response |
|---|---|---|
| $d < r_r$ (repulsion) | Is anyone too close? | Turn away from them |
| $r_r \le d < r_o$ (orientation) | Which way are my neighbours heading? | Turn to match |
| $r_o \le d < r_a$ (attraction) | Where is the group? | Turn toward it |

Each answer is an **arrow** (a vector). Add the arrows and you get the direction the fish
wants to turn. Switch on **Steering forces** to see the arrows: red, amber and green,
plus white for their sum.

## Why does a shape appear?

Because the rules **chain**. Fish A copies B, B copies C, C copies D... A turn made by the
fish at the edge travels through the school like a wave. No one sees the whole school,
but information flows through it.

## Fear is contagious

A fish that sees the predator gets a fear level near 1. Its neighbours take a fraction
$c$ of that fear (the *alarm contagion*). Their neighbours take a fraction of *that*,
and so on. The alarm fades with each hop, so it spreads a certain distance and then
stops. Move the *Alarm contagion* slider near 1 and watch the panic sweep the tank.

## Memory makes behaviour smarter than perception

If the predator vanishes (move the mouse out of the tank), frightened fish **keep
fleeing** from the spot where they last saw it, and the memory fades over a few seconds.
Without memory, fish would stop fleeing the instant the predator left their vision cone,
which is easy to exploit: sneak up from the blind spot!

# The Math

## Perception

Fish $i$ is at position $\mathbf{x}_i$ with velocity $\mathbf{v}_i$ and heading
$\hat{\mathbf{h}}_i = \mathbf{v}_i / \lVert \mathbf{v}_i \rVert$. For a neighbour $j$, let
$\mathbf{d}_{ij} = \mathbf{x}_j - \mathbf{x}_i$, $d_{ij} = \lVert \mathbf{d}_{ij} \rVert$ and
$\hat{\mathbf{u}}_{ij} = \mathbf{d}_{ij}/d_{ij}$. Fish $i$ can sense $j$ if

$$
d_{ij} < r_a \quad\text{and}\quad \Big( d_{ij} < r_l \;\;\text{or}\;\; \hat{\mathbf{h}}_i \cdot \hat{\mathbf{u}}_{ij} > \cos\tfrac{\phi}{2} \Big)
$$

The first alternative is the lateral line; the second is the vision cone of width $\phi$.

## The three rules

Each sensed neighbour falls in exactly one zone: closer than the repulsion radius $r_r$,
between $r_r$ and the orientation radius $r_o$ (the set $O_i$), or between $r_o$ and the
attraction radius $r_a$ (the set $A_i$). Each rule has a weight $w$ that says how much it
matters: $w_s$ for separation, $w_a$ for alignment, $w_c$ for cohesion.

Repulsion is weighted so that closer fish push harder, and capped so that a crowd can't
produce an unbounded force:

$$
\mathbf{s}_i = -\sum_{d_{ij} < r_r} \hat{\mathbf{u}}_{ij}\left(1 - \frac{d_{ij}}{r_r}\right),
\qquad
\mathbf{F}^{sep}_i = w_s \,\frac{\mathbf{s}_i}{\max(\lVert \mathbf{s}_i \rVert, 1)}
$$

Alignment steers the heading toward the average neighbour heading:

$$
\mathbf{F}^{ali}_i = w_a \left( \frac{\sum_{j \in O_i} \hat{\mathbf{v}}_j}{\lVert \sum_{j \in O_i} \hat{\mathbf{v}}_j \rVert} - \hat{\mathbf{h}}_i \right)
$$

Cohesion points toward the average direction of the far neighbours:

$$
\mathbf{F}^{coh}_i = w_c \, \frac{\sum_{j \in A_i} \hat{\mathbf{u}}_{ij}}{\lVert \sum_{j \in A_i} \hat{\mathbf{u}}_{ij} \rVert}
$$

## Fear and working memory

Fear $f_i$ is a number from 0 (calm) to 1 (terrified). It fades with the time constant
$\tau_f$ (after $\tau_f$ seconds only about a third is left), is copied from the sensed
neighbours $N_i$ with the contagion factor $c$, and jumps to the threat level $\theta$
when the predator is sensed. The predator moves with velocity $\mathbf{v}_p$, and
$v_{attack}$ (40 by default) is the speed of a real strike:

$$
f_i \leftarrow \max\!\Big( f_i\, e^{-\Delta t/\tau_f},\;\; c \max_{j \in N_i} f_j,\;\; \theta \Big),
\qquad
\theta = \tfrac12 + \tfrac12 \min\!\Big(\frac{\lVert \mathbf{v}_p \rVert}{v_{attack}}, 1\Big)
$$

After $n$ hops the alarm is at most $c^n$, so it survives above a level $\epsilon$ for
$n = \ln\epsilon / \ln c$ hops. With $c = 0.85$ and $\epsilon = 0.3$ that is about 7 fish.

The memory of the predator's position $\tilde{\mathbf{p}}_i$ has strength $m_i$ that is
reset to 1 on every sighting and otherwise forgets:

$$
m_i(t + \Delta t) = m_i(t)\, e^{-\Delta t / \tau_m}
$$

## Fleeing sideways

A fish flees away from the remembered position, with a sideways component that takes it
off the predator's line of attack. That component produces the fountain effect:

$$
\mathbf{F}^{flee}_i = w_f\, m_i \,\operatorname{clamp}\!\Big(1 - \frac{\lVert \mathbf{x}_i - \tilde{\mathbf{p}}_i \rVert}{1.3\, r_p}, 0, 1\Big)\;
\widehat{\big(\hat{\mathbf{a}}_i + k\, \hat{\mathbf{s}}_i\big)}
$$

where $\hat{\mathbf{a}}_i$ points away from $\tilde{\mathbf{p}}_i$, and
$\hat{\mathbf{s}}_i = \operatorname{sign}(\hat{\mathbf{v}}_p^{\perp} \cdot (\mathbf{x}_i - \tilde{\mathbf{p}}_i))\,\hat{\mathbf{v}}_p^{\perp}$
is the side of the predator's path the fish is already on.

## Rock: pressure and a look ahead

Every rock cell $k$ within $r_{rock}$ (3 by default) of the fish pushes it away, gently far off and
strongly up close. On top of that the fish looks ahead along its heading for a distance
$L$ (7 by default). If the look hits rock after a free distance $\ell < L$, it compares two more
looks turned left and right by the side-look angle (0.6 rad by default) and turns toward the freer side:

$$
\mathbf{F}^{rock}_i = w_r \sum_{k:\,d_{ik} < r_{rock}} \hat{\mathbf{u}}_{ki}\Big(1 - \frac{d_{ik}}{r_{rock}}\Big)^2
\;+\; w_r \Big(1 - \frac{\ell}{L}\Big)\, s_i\, \hat{\mathbf{h}}_i^{\perp}
$$

where $\hat{\mathbf{u}}_{ki}$ points from the rock cell to the fish, and $s_i = +1$
(turn left) unless the right look is strictly freer ($s_i = -1$). Rock is also solid:
a step that would end inside it slides along it instead.

## Hunger against fear

Hunger $h_i \in [0, 1]$ grows steadily and drops by the bite size $b$ (0.25 by default) with each bite:

$$
h_i \leftarrow \min\!\Big(h_i + \frac{\Delta t}{\tau_h},\; 1\Big)
$$

A fish swims toward the nearest food flake it can sense (same vision cone, or twice the
lateral-line range), and hunger weakens the flee force by a factor $1 - \rho h_i$:

$$
\mathbf{F}^{food}_i = w_h\, h_i\, \hat{\mathbf{u}}_{food},
\qquad
\mathbf{F}^{flee}_i \;\to\; (1 - \rho\, h_i)\,\mathbf{F}^{flee}_i
$$

So a starving fish ($h_i = 1$) with $\rho = 0.6$ flees at only 40% strength while being
pulled toward food. All fish within reach of a flake bite it in the same step; the bites
are counted with integer atomics and removed afterwards, so the order of the GPU threads
never matters.

## Motion

With $\mathbf{F}_i$ the sum of all forces (walls included), the speed target rises with fear:

$$
v^*_i = v_0 + (v_b - v_0) f_i,
\qquad
\mathbf{a}_i = a_{max}\,\operatorname{clip}(\mathbf{F}_i) + k_v (v^*_i - \lVert \mathbf{v}_i \rVert)\,\hat{\mathbf{h}}_i
$$

and the state advances by semi-implicit Euler: $\mathbf{v} \leftarrow \mathbf{v} + \mathbf{a}\Delta t$,
then $\mathbf{x} \leftarrow \mathbf{x} + \mathbf{v}\Delta t$.

## Why it runs fast: spatial hashing

Checking every pair costs $O(N^2)$: 2.25 million checks per step for 1,500 fish. A hash
grid with cell size $r_a$ means each fish only inspects its own cell and the 8 around it,
which costs $O(N k)$ with $k$ the local density. Turn on **Spatial hash grid** to see it.

## Why it is deterministic

The GPU runs all fish at once, in an order that changes from run to run. The kernel reads
only the previous state and writes a new buffer (double buffering), so no fish ever sees a
half-updated neighbour. Every run with the same seed and inputs is identical, and the test
suite checks this.

# Research

## Where the model comes from

- **C. Reynolds (1987)**, *Flocks, herds and schools: a distributed behavioral model*,
  SIGGRAPH. The original "boids": separation, alignment, cohesion.
- **I. D. Couzin et al. (2002)**, *Collective memory and spatial sorting in animal groups*,
  J. Theor. Biol. 218. The three-zone model used here. Varying $r_o$ switches the group
  between **swarm**, **torus (milling)**, **dynamic parallel** and **highly parallel**
  states. Try it: shrink the orientation band and look for milling.
- **T. Vicsek et al. (1995)**, *Novel type of phase transition in a system of
  self-driven particles*, PRL 75. Order appears suddenly as noise falls: a true phase
  transition.

## Escape behaviour

- **Pitcher & Parrish** (in *Behaviour of Teleost Fishes*, 1993) describe the fountain
  effect and flash expansion, the escape manoeuvres this chapter reproduces.
- **S. B. Rosenthal et al. (2015)**, *Revealing the hidden networks of interaction in
  mobile animal groups*, PNAS 112. Real fish respond to visual neighbours, not metric
  ones, and startle cascades follow the visual network. The vision cone here is a first
  step toward that.
- **N. O. Handegård et al. (2012)**, *The dynamics of coordinated group hunting and
  collective information transfer among schooling prey*, Current Biology 22.

## Foraging under risk

- **S. L. Lima & L. M. Dill (1990)**, *Behavioral decisions made under the risk of
  predation: a review and prospectus*, Can. J. Zool. 68. The classic review of how
  animals trade feeding against safety.
- **M. Milinski & R. Heller (1978)**, *Influence of a predator on the optimal foraging
  behaviour of sticklebacks*, Nature 275. Hungry sticklebacks fed in riskier places; the
  factor $1 - \rho h_i$ here is the simplest caricature of that result.

## Things to measure (open questions for you)

- **Polarisation** $P = \lVert \frac{1}{N}\sum_i \hat{\mathbf{h}}_i \rVert$ (1 means
  everyone swims the same way). How does $P$ depend on $w_a$ and $\phi$?
- **Information speed.** Startle one edge of the school: how fast does fear travel, in
  body lengths per second? How does it depend on $c$ and density?
- **Blind-spot hunting.** Is an attack from behind more successful? Count *caught*.
- **Metric vs. topological neighbours.** Starlings react to their ~7 nearest neighbours
  whatever the distance (Ballerini et al., PNAS 2008). What would change here?
- **Refuges.** Build a reef and count catches with and without it. Does the reef protect
  the school, or trap it?

## Model vs. nature

Everything here is simplified on purpose. Real fish integrate vision, the lateral line,
smell and hearing; vision is blocked by rock (here it is not); schools mix individuals of
different size, hunger and boldness, and personality differences matter (Jolles et al.,
*Current Biology* 2017). Fear and hunger are single numbers here, while real animals have
hormones, learning and species-specific escape reflexes (the Mauthner-cell C-start).
Treat the model as a clear statement of *one* mechanism, not a description of any species.
