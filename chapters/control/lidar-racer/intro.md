# A car that only sees laser beams

This race car has no map, no GPS and no camera. All it has is a **lidar**: a spinning
laser that measures how far away the nearest wall is in many directions. The faint lines
coming out of the car are those laser rays.

Sixty times a second the car runs the same loop:

1. **Sense.** Fire the rays and read the distances.
2. **Find the gap.** Look for the widest stretch of open space ahead.
3. **Steer.** Aim into that gap.
4. **Choose a speed.** Slow down for sharp turns, and never go faster than you could
   stop in the free space in front of you.
5. **Move.**

That's it: no learning, no training data. The same inputs always give the same driving.

## Read the overlays

- **Rays** go from blue (far) to red (close).
- The **red rays** are the *safety bubble*: the directions near the closest obstacle,
  which the car refuses to consider.
- The **green wedge** is the gap the car chose, and the **green ring** is its aim point.
- The **amber bar** in front of the car is its *braking distance*. The red tick is the
  nearest thing straight ahead. If the bar reaches the tick, the car must brake.
- The **bar chart** at the bottom right is the whole scan, one bar per ray.

## Things to try

> **Build a chicane.** Left-click to drop traffic cones on the track. Can you build an
> obstacle course the car can't solve?

> **Shove it.** Hold the arrow keys to push the car off its line and watch it recover.

> **Fewer eyes.** Drop *Lidar rays* to 9. What goes wrong? Now try 361.

> **Speed demon.** Raise *Top speed* and lower *Braking*. The car can now "see" a wall
> but not stop in time. This is exactly why the speed rule exists.

> **Noisy sensors.** Add *Sensor noise*. Real lidars are noisy. How much can the
> algorithm tolerate?
