# The colony knows the way. No ant does.

An ant is nearly blind and has a brain smaller than a pinhead. It has no map. Yet an ant
colony reliably finds food and, astonishingly, the **shortest path** to it.

The trick is that ants **write their memory into the ground**. As they walk they leave a
chemical trail called **pheromone**, and they tend to follow trails that smell strong:

- Ants **looking for food** leave a *home trail* that says "the nest is back this way".
- Ants **carrying food** leave a *food trail* that says "food is back this way".
- Every trail slowly **evaporates**. A path nobody uses fades away.

A short path gets walked more often per minute than a long one, so its trail is refreshed
faster than it evaporates. More ants follow it, which makes it even stronger. The colony
"decides" without anyone deciding. Scientists call this **stigmergy**: coordination through
marks left in the environment.

## Things to try

> **The double bridge.** Choose *Scenario → Double bridge*. There are two routes to the
> food, and the top one is shorter. Watch the colony explore both, then abandon the
> long one. This is a famous real experiment (Goss et al., 1989).

> **Block the highway.** In *Open field*, wait until a strong trail forms, then
> **left-drag** to build a wall across it. How long does the colony take to reroute?

> **Make an ant mill.** Set *Path integration* to 0. Sometimes ants start following each
> other in a circle forever. Real army ants do this too; it's called an *ant mill*.

> **Forgetful world.** Set *Evaporation time* very low. Trails can't survive the trip
> back, and the colony can't organise.

Switch on **Focus ant sensors** (Ctrl+click an ant) to see its three smell sensors:
left, front and right. The brightest green one is where it will turn.
