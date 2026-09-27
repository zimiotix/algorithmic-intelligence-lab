# Roadmap

Chapters are grouped into tracks (`chapters/tracks.toml`). Within a track, each chapter
builds on the previous one. ✅ = shipped.

## Swarms & Emergence
- ✅ Fish school under attack: boids, Couzin zones, senses, fear contagion, memory
- ✅ Ant colony: stigmergy, double bridge, path integration
- Starling murmurations: topological (7 nearest) neighbours in 3D projection
- Vicsek model: the order/disorder phase transition, measured live
- Cooperative transport: ants carrying an object too heavy for one
- Army-ant bridges and self-assembly
- Termite mound building: stigmergic construction
- Slime mould (Physarum) networks: recreating a rail map
- Firefly synchronisation: coupled oscillators (Kuramoto)

## Perception & Control
- ✅ Lidar racer: ray casting, Follow-the-Gap, pure pursuit, bicycle model
- PID from scratch: the balancing robot
- Wall following and bug algorithms
- Disparity extender vs. Follow-the-Gap: a race
- Stanley controller and lateral error
- Model predictive control (MPC) with a dynamic bicycle model
- Braitenberg vehicles: fear, love and curiosity from two wires

## Search & Planning
- BFS, DFS, Dijkstra on a paintable grid
- A* and heuristics (admissible vs. greedy)
- D* Lite: replanning when the world changes
- RRT and RRT*: planning for a car that can't turn in place
- Potential fields and their local minima
- Occupancy mapping + planning: using the racer's memory

## Estimation & Belief
- Bayes filter on a 1D corridor robot
- Kalman filter: tracking a thrown ball
- Particle filter localisation with the lidar racer
- EKF-SLAM and graph SLAM (simplified)

## Optimisation & Evolution
- Hill climbing and simulated annealing: the travelling salesman
- Ant Colony Optimization (from the ant chapter to graphs)
- Genetic algorithms: evolving walkers (deterministic seeds)
- Particle swarm optimisation
- CMA-ES

## Games & Decisions
- Minimax and alpha-beta: tic-tac-toe to Connect Four
- Monte Carlo tree search (seeded rollouts)
- Pursuit–evasion: predator strategies
- Auctions and market-based task allocation for robot teams

## Cellular Automata & Patterns
- Conway's Life and the zoo of patterns
- Langton's ant and emergent highways
- Reaction–diffusion (Gray–Scott): animal coat patterns
- Abelian sandpile: self-organised criticality
- Wave function collapse: procedural levels

## Platform
- Replays: record inputs, scrub the timeline backwards (possible because runs are deterministic)
- Live equations: hover a term in the Deep dive to highlight it in the simulation
- Challenges with scoring per chapter
- Zero-copy CUDA → OpenGL interop for 100k+ agents
- Flatpak packaging
