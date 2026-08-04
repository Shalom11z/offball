# Tactical metrics

Every metric, the model behind it, and what it cannot tell you.

## Conventions

All metrics operate in **pitch metres**, origin at the bottom-left corner, `x`
along the length, `y` across the width.

**The attacking team always plays toward +x.** `score_frame` mirrors
coordinates when they do not, so no metric needs to know which way the teams
kicked off. This is verified by a test that scores a scene and its mirror image
and asserts identical results.

**Nothing here is a judgement.** These are geometric measurements. Turning
"held a −0.4m onside margin for 12 minutes" into "times runs well" is a
coach's job, not the model's.

## Pitch control

**Implementation**: [`control.rs`](../rust/offball-core/src/control.rs) ·
**Question**: if the ball arrived here, who reaches it first?

This is the kernel everything else rests on.

### Model

A reduced member of the time-to-intercept family (Spearman et al.):

1. A player holds their current velocity for `reaction_time` seconds.
2. From there they run straight at the target at `max_speed`.
3. Control is a softmax over arrival times.

```
t_arrive(player, q) = t_react + |q − (pos + vel·t_react)| / v_max

control_attacking(q) = 1 / (1 + exp((t_att − t_def) / τ))
```

τ is the softmax temperature: a player arriving 0.1s earlier only slightly
out-controls a rival, while 2s earlier is near-total control.

### Defaults

| Parameter | Default | Source |
| --- | --- | --- |
| `reaction_time` | 0.7 s | League-average from tracking literature |
| `max_speed` | 7.8 m/s | League-average top speed |
| `tau` | 0.45 s | Chosen so territorial edges are ~2-3m wide |

**These are not fitted to any competition.** They are reasonable starting
values. Fitting them per league — and ideally per player — is on the roadmap.

### What the model omits

- **Acceleration limits.** A stationary player cannot reach top speed
  instantly. The reaction-time term partly absorbs this, but the model
  overestimates how far a standing player can get in the first second.
- **Ball flight time.** Real pitch control weights each location by how long
  the ball would take to get there. This treats all locations as equally
  reachable by the pass.
- **Per-player speed profiles.** A centre-back and a winger share `max_speed`.
- **Fatigue.** An 85th-minute sprint is modelled as an 5th-minute one.

Each of these makes the model optimistic about space far from every player.
Treat absolute areas as comparative, not physical.

### Derived quantities

**Space owned** (`space_ownership`) — the area each player reaches before
anyone else. A velocity-aware Voronoi partition; areas sum exactly to the pitch
area. This is the per-player headline: how much ground a player's positioning
actually claims.

**Dangerous space** (`dangerous_space`) — controlled area weighted by threat
value. This separates "we have lots of the pitch" from "we have the parts that
matter". A winger hugging the touchline in their own half owns space this
correctly values at almost nothing.

## Position value

**Implementation**: `pitch_value`

A normalised [0,1] surface combining proximity to goal with the angle subtended
by the goal mouth:

```
value(p) = 0.65 · exp(−distance_to_goal / 25) + 0.35 · (goal_angle / π)
```

> **This is the weakest component in the system.** It is an analytic
> stand-in, not a learned model. The weights and the 25m decay constant are
> plausible, not fitted. It gets the ordering right — central beats wide, near
> beats far — but the magnitudes carry no meaning.
>
> Replacing it with an expected-threat surface fitted to real shot and goal
> data is the highest-value modelling work outstanding. See
> [06 Roadmap](06-roadmap.md).

Because `dangerous_space` is weighted by this surface, that figure inherits the
same caveat.

## Offside

**Implementation**: `offside_line`, `offside_margin`

The offside line is the greater of the **second-last defender's** `x` and the
**ball's** `x` — you cannot be offside from behind the ball. The defender list
must include the goalkeeper, who is usually but not always the last defender.

Reported as a **continuous signed margin**, not a boolean:

- `+2.0` — two metres beyond the line, in an offside position
- `−0.3` — holding the shoulder of the last defender
- `−8.0` — dropping well off

The coaching question is "how finely is this player timing their runs?", and
a striker living at −0.3m is doing something quite different from one at −6m.

### What this is not

**Not an offside decision.** Law 11 requires that the player be *involved in
active play* at the moment the ball is *played*. This measures position only,
continuously. It is a positioning metric, not a VAR replacement.

Returns `None` with fewer than two defenders tracked — a vision failure, which
callers must treat as "cannot score this frame" rather than substituting a
default.

## Passing lanes

**Implementation**: `passing_lane`

A lane is open when no defender sits within `corridor` metres (default 1.2) of
the straight ball → receiver **segment**.

Segment, not infinite line: a defender standing behind the passer does not
block the pass. This is tested explicitly.

Returns the clearance to the nearest defender as well as the boolean, so
callers can distinguish "barely open" from "wide open".

**Limitations**: assumes a straight ground pass. It does not model lofted
passes over a defender, curled passes around one, or the pass's flight time
(during which the defender moves). A "closed" lane may be very much open to a
chipped ball.

## Team shape

**Implementation**: `team_shape`, `defensive_lines`

| Metric | Definition |
| --- | --- |
| Hull area | Convex-hull area of the outfield block, m². Lower is more compact |
| Depth | Spread along `x`, metres |
| Width | Spread across `y`, metres |
| Centroid | Mean position |

**Exclude the goalkeeper before calling this.** Including them inflates depth
by 20-30m and makes the number meaningless.

`defensive_lines` groups defenders into banks (back line, midfield, forwards)
by 1-D k-means on `x`, quantile-initialised so match reports are reproducible.
`lines_broken` then counts how many banks an attacker is positioned beyond —
the core "is this player breaking lines?" signal.

Fitting `k` banks to a team that is not actually in `k` banks produces
meaningless centres. During a transition, a team has no lines at all.

## Marking pressure

**Implementation**: `marking_pressure`

```
pressure = exp(−distance_to_nearest_opponent / scale)
```

`scale` defaults to 5m — roughly the range over which a defender can contest a
first touch. Returns 0 when no opponents are tracked.

Distance only. It does not know whether the defender is facing the player,
goal-side, or engaged elsewhere.

## Availability

A derived per-frame flag: the player has an **open lane**, is **not offside**,
and is under **pressure below 0.6**. `FrameScore.available_options` counts them.

Mean passing options per frame is, in practice, the most useful single
team-level number this produces: it captures whether a side is creating
solutions for the ball carrier or leaving them isolated.

## Aggregation

**Implementation**: [`report.py`](../python/src/offball/tactics/report.py)

Three rules:

**Medians for distance-like quantities.** A handful of frames where tracking
put a player on the wrong side of the pitch will drag a mean arbitrarily far.
The median shrugs them off.

**Rates, not counts.** Players are on the pitch for different lengths of time
and their team has the ball for different shares of it. A raw count of
line-breaking positions mostly measures minutes played.

**Sample size travels with every number.** `frames` is on every summary. Below
roughly 500 scored frames (20s of off-ball time at 25fps) figures are
indicative only; the TS SDK's `MIN_RELIABLE_FRAMES` and the SQL view's
`is_reliable` both encode this.

## The ball carrier is excluded

`score_frame` identifies the attacker nearest the ball and omits them from the
per-player scores. They are on the ball by definition; including them skews
every team aggregate and makes no sense as an off-ball measurement.

## Reading these numbers honestly

1. **Check `coverage` first.** Below 0.6 the vision stage struggled; treat
   everything downstream as provisional and look at calibration.
2. **Check `frames` per player.** A spectacular figure from 40 frames is
   noise.
3. **Compare within a match, not across matches**, until the model parameters
   are fitted. Opponent quality, game state and pitch size all move these
   numbers.
4. **Absolute areas are comparative.** The control model's omissions make it
   systematically optimistic about space; the ordering is more trustworthy than
   the magnitude.
