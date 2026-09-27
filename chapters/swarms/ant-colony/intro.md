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

> **Block the highway.** In *Open field*, wait until a strong trail forms, then pick
> **Wall** (key 1) and drag across it. How long does the colony take to reroute?

> **A new food source.** Pick **Food** (key 2) and drop a pile far from the nest. How
> long until the first scout finds it, and until a trail forms?

> **Make an ant mill.** Set *Path integration* to 0. Sometimes ants start following each
> other in a circle forever. Real army ants do this too; it's called an *ant mill*.

> **Forgetful world.** Set *Evaporation time* very low. Trails can't survive the trip
> back, and the colony can't organise.

Pick **Inspect** (key 3) and click an ant to follow it. The *Focus ant sensors* overlay
shows its three smell sensors (left, front, right), and **Live Math** on the right shows
the numbers it is deciding with, sixty times a second. The **Guide** on the left lists
experiments that tick themselves off when you manage them.

## Model vs. nature

Real ants are far more varied than these. There are over 14,000 species, and they use many
signals: several trail pheromones with different lifetimes, *repellent* "no entry" marks on
dead-end branches (Pharaoh's ants do this), alarm pheromones, touch, sound, and sight of
the sun and landmarks (desert ants navigate mostly by vision). Crowding changes behaviour
too: on a jammed trail, ants get pushed onto other routes. Here every ant is identical and
there are only two attractive smells. The model shows one real mechanism, stigmergy,
clearly; it is not a portrait of any particular species.
