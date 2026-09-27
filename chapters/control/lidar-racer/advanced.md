# Intuition

## A reactive driver

This car is a **reactive** controller: it never plans a route and never remembers the
track. At every instant it asks only *"where is the open space right now?"* That makes it
simple and robust, and also short-sighted. It can be tricked by dead ends, and it will
happily take a line no racing driver would.

## Why a safety bubble?

The widest gap might pass right next to a wall. Treating everything near the closest
obstacle as "blocked" keeps the car from shaving the corners of the gap. The bubble is
angular: at distance $r$, a bubble of radius $b$ covers the angle $\arctan(b/r)$. A
close obstacle blocks a wide slice of the view, and a far one blocks a thin sliver.

## Steering like a person: pure pursuit

Humans don't steer at the road right under the bumper; they look a bit ahead. **Pure
pursuit** picks a point at a *lookahead distance* $L_d$ and computes the one circular arc
that passes through it. Look far ahead and you get gentle, smooth, but lazy steering.
Look close and you get sharp, twitchy steering.

### A real bug, and what it taught us

The first version of this chapter used the gap's far point (up to 30 units away) as the
lookahead. On a hairpin that asked for about 10° of steering, and the car drove straight
into the wall: **more than a dozen crashes in 90 seconds** on the Hairpins track. The fix
is what real autonomous cars do: **look ahead proportionally to speed**,
$L_d = k v + L_{min}$. Slow into a hairpin gives a short lookahead and a tight turn.
Crashes dropped to **zero** on all three tracks and lap times improved. Play with
*Lookahead gain* and *Min lookahead* to feel the trade-off between smooth and sharp.

# The Math

## Casting a ray

A ray starts at $\mathbf{o}$ with unit direction $\mathbf{d}$; a wall segment runs from
$\mathbf{a}$ with edge $\mathbf{e} = \mathbf{b} - \mathbf{a}$. They meet where

$$
\mathbf{o} + t\,\mathbf{d} = \mathbf{a} + u\,\mathbf{e}.
$$

With the 2D cross product $\mathbf{p} \times \mathbf{q} = p_x q_y - p_y q_x$ and
$\mathbf{w} = \mathbf{a} - \mathbf{o}$, Cramer's rule gives

$$
t = \frac{\mathbf{w} \times \mathbf{e}}{\mathbf{d} \times \mathbf{e}},
\qquad
u = \frac{\mathbf{w} \times \mathbf{d}}{\mathbf{d} \times \mathbf{e}},
\qquad \text{hit iff } t > 0,\; 0 \le u \le 1.
$$

The measured range is the smallest such $t$ over all segments (and all cones), capped
at $R_{max}$. For a cone of radius $\rho$ at $\mathbf{c}$, solving
$\lVert \mathbf{o} + t\mathbf{d} - \mathbf{c} \rVert = \rho$ gives
$t = -\beta - \sqrt{\beta^2 - \gamma}$, with $\beta = \mathbf{d}\cdot(\mathbf{o} - \mathbf{c})$ and
$\gamma = \lVert \mathbf{o} - \mathbf{c} \rVert^2 - \rho^2$.

## Follow the Gap

Ray $k$ points at angle $\alpha_k$ from the car's heading and measures range $r_k$. The
ray with the smallest smoothed range is at angle $\alpha_{min}$ and range $\bar r_{min}$;
$b$ is the safety-bubble size and $r_{gap}$ the gap threshold.

1. Smooth the ranges with a 3-ray moving average: $\bar r_k$.
2. Bubble: blank every ray with $|\alpha_k - \alpha_{min}| < \arctan(b / \bar r_{min})$.
3. Free rays: $\bar r_k > r_{gap}$. The **gap** is the longest run of consecutive free rays.
4. Aim at the deepest ray in the gap, its centre, or halfway between (*Aim point*).

## Pure pursuit

The car steers its front wheels by the angle $\delta$. It has wheelbase $L$ (the distance
between the axles), and the aim point is at angle $\alpha$ from its heading and distance
$L_d$ (the lookahead). The circle through the rear axle and the aim point has curvature
$\kappa$ (one over its radius $R$). The lookahead grows with speed $v$ by the gain $k$,
starts at $L_{min}$, and never reaches past the aim point's range $\bar r_{target}$:

$$
\kappa = \frac{2 \sin\alpha}{L_d}
\quad\Longrightarrow\quad
\delta = \arctan(L\kappa) = \arctan\!\left(\frac{2L\sin\alpha}{L_d}\right),
\qquad
L_d = \operatorname{clip}(k v + L_{min},\; L_{min},\; \bar r_{target}).
$$

The chord of length $L_d$ subtends the angle $2\alpha$ at the centre of a circle of radius
$R$, so $L_d = 2R\sin\alpha$; that is where the factor 2 comes from.

## Choosing a speed

Here $v_{max}$ is the top speed, $k_\delta$ how much the car slows in turns, and $v^*$ the
speed it aims for. Stopping from speed $v$ at deceleration $a_b$ takes distance
$v^2/(2a_b)$. To be able to
stop before the nearest obstacle ahead (distance $d_{front}$, minus a margin $m$):

$$
v \le \sqrt{2 a_b\,(d_{front} - m)}, \qquad
v^* = \min\!\Big(v_{max},\; \frac{v_{max}}{1 + k_\delta |\delta|},\; \sqrt{2 a_b (d_{front} - m)}\Big)
$$

and the throttle is a proportional controller $a = \operatorname{clip}(k_p (v^* - v), -a_b, a_{max})$.
The steering itself is rate-limited: $|\dot\delta| \le \dot\delta_{max}$.

## The kinematic bicycle model

The car is at $(x, y)$ with heading $\theta$ and speed $v$; a dot means "rate of change",
so $\dot x$ is how fast $x$ changes each second:

$$
\dot x = v\cos\theta,\qquad \dot y = v\sin\theta,\qquad \dot\theta = \frac{v}{L}\tan\delta,\qquad \dot v = a
$$

integrated with explicit Euler at $\Delta t = 1/60$ s. The *Predicted path* overlay simply
runs this model forward for 1.5 s with the current $v$ and $\delta$.

## The occupancy memory

Every lidar hit deposits into a grid that fades with lifetime $\tau_m$: the same
`FieldMemory` as the ants' pheromone. Turn on **Occupancy memory** and after one lap the
car has "drawn" the track. The controller doesn't use it yet; building a map from it is
the job of the SLAM and planning chapters.

# Research

## Sources

- **V. Sezer, M. Gokasan (2012)**, *A novel obstacle avoidance algorithm: "Follow the Gap
  Method"*, Robotics and Autonomous Systems 60.
- **R. C. Coulter (1992)**, *Implementation of the Pure Pursuit Path Tracking Algorithm*,
  Carnegie Mellon University Robotics Institute, CMU-RI-TR-92-01.
- **M. O'Kelly et al. (2020)**, *F1TENTH: An open-source evaluation environment for
  continuous control and reinforcement learning*, NeurIPS 2019 Competition Track (PMLR 123).
  Follow-the-Gap is the standard first algorithm on these 1/10-scale race cars.
- **J. Kong, M. Pfeiffer, G. Schildbach, F. Borrelli (2015)**, *Kinematic and dynamic
  vehicle models for autonomous driving control design*, IEEE Intelligent Vehicles.
- **R. Rajamani**, *Vehicle Dynamics and Control*, Springer. Chapter 2 covers the
  bicycle model.

## Known limitations (good project ideas)

- **No tyre slip.** At high speed a real car slides; the kinematic model can't. Swap in a
  dynamic bicycle model with a linear tyre model and watch Follow-the-Gap struggle.
- **Myopia.** A reactive controller can't handle a dead end or a U-shaped trap. Planning
  (A\*, RRT) or model predictive control (MPC) fix this, at a cost.
- **Disparity extender.** A popular F1TENTH variant widens obstacles at *range
  discontinuities* instead of using a bubble. Implement it and race the two.
- **Racing line.** The fastest line hugs apexes; Follow-the-Gap stays central. Use the
  occupancy map to plan a racing line, then track it with pure pursuit.
