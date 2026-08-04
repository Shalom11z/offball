//! Pitch control: "if the ball arrived here, who would get to it first?"
//!
//! This is the kernel the whole off-the-ball analysis rests on. It is
//! evaluated on a grid for every frame we score, so a 90-minute match at 5 Hz
//! with a 105x68 grid is ~200 million cell-player evaluations — which is why
//! it lives in Rust rather than in the Python layer.
//!
//! The model follows the time-to-intercept family (Spearman et al.), reduced to
//! its deterministic core:
//!
//!   1. A player keeps their current velocity for `reaction_time` seconds.
//!   2. From there they run straight at the target at `max_speed`.
//!   3. Control is a softmax over arrival times, so a player who gets there
//!      0.1s earlier only slightly out-controls a rival, while 2s earlier is
//!      near-total control.
//!
//! Deliberately omitted, and listed in docs/03-tactical-metrics.md as known
//! limitations: acceleration limits, ball flight time, and per-player fitted
//! speed profiles.

use crate::geom::Vec2;
use crate::pitch::Pitch;

/// Kinematic state of one player on the pitch plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerState {
    pub pos: Vec2,
    pub vel: Vec2,
    /// `true` for the team being scored (the "attacking" team in the output).
    pub is_attacking: bool,
}

impl PlayerState {
    pub fn new(pos: Vec2, vel: Vec2, is_attacking: bool) -> Self {
        Self { pos, vel, is_attacking }
    }

    /// Stationary player, for tests and for frames where velocity is unknown.
    pub fn still(pos: Vec2, is_attacking: bool) -> Self {
        Self::new(pos, Vec2::new(0.0, 0.0), is_attacking)
    }
}

/// Tunables for the control model. Defaults are league-average values from the
/// tracking literature, not fitted to any particular competition.
#[derive(Debug, Clone, Copy)]
pub struct ControlParams {
    /// Seconds before a player can change direction.
    pub reaction_time: f64,
    /// Top running speed in m/s.
    pub max_speed: f64,
    /// Softmax temperature in seconds. Smaller = harder territorial edges.
    pub tau: f64,
}

impl Default for ControlParams {
    fn default() -> Self {
        Self { reaction_time: 0.7, max_speed: 7.8, tau: 0.45 }
    }
}

impl ControlParams {
    /// Time for `p` to arrive at `target`, given the reaction-then-sprint model.
    pub fn time_to_intercept(&self, p: &PlayerState, target: Vec2) -> f64 {
        let drift = p.pos.add(p.vel.scale(self.reaction_time));
        self.reaction_time + drift.dist(target) / self.max_speed.max(1e-6)
    }
}

/// A scalar field sampled on a regular grid over the pitch.
#[derive(Debug, Clone)]
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub cell_area: f64,
    /// Row-major, `ny` rows of `nx` values.
    pub values: Vec<f64>,
}

impl Grid {
    /// Cell centre in pitch metres.
    pub fn cell_centre(pitch: &Pitch, nx: usize, ny: usize, ix: usize, iy: usize) -> Vec2 {
        Vec2::new(
            (ix as f64 + 0.5) * pitch.length / nx as f64,
            (iy as f64 + 0.5) * pitch.width / ny as f64,
        )
    }

    /// Integral of the field over the pitch, in m^2 when values are in [0,1].
    pub fn total(&self) -> f64 {
        self.values.iter().sum::<f64>() * self.cell_area
    }
}

/// Probability that the **attacking** team controls each cell.
///
/// Returns a grid of values in [0,1]. A cell with no players of one team on the
/// pitch resolves to 0 or 1 accordingly; an empty player list gives a uniform
/// 0.5 field.
pub fn pitch_control(
    players: &[PlayerState],
    pitch: &Pitch,
    nx: usize,
    ny: usize,
    params: &ControlParams,
) -> Grid {
    let cell_area = pitch.area() / (nx * ny) as f64;
    let mut values = vec![0.5; nx * ny];

    for iy in 0..ny {
        for ix in 0..nx {
            let q = Grid::cell_centre(pitch, nx, ny, ix, iy);
            let mut t_att = f64::INFINITY;
            let mut t_def = f64::INFINITY;
            for p in players {
                let t = params.time_to_intercept(p, q);
                if p.is_attacking {
                    if t < t_att {
                        t_att = t;
                    }
                } else if t < t_def {
                    t_def = t;
                }
            }
            values[iy * nx + ix] = match (t_att.is_finite(), t_def.is_finite()) {
                (true, true) => 1.0 / (1.0 + ((t_att - t_def) / params.tau).exp()),
                (true, false) => 1.0,
                (false, true) => 0.0,
                (false, false) => 0.5,
            };
        }
    }

    Grid { nx, ny, cell_area, values }
}

/// Area (m^2) each player is the fastest to reach, i.e. a velocity-aware
/// Voronoi cell. Index `i` of the result corresponds to `players[i]`.
///
/// This is the per-player "space ownership" figure reported in the off-ball
/// summary: how much of the pitch a player's positioning actually claims.
pub fn player_space_ownership(
    players: &[PlayerState],
    pitch: &Pitch,
    nx: usize,
    ny: usize,
    params: &ControlParams,
) -> Vec<f64> {
    let mut owned = vec![0.0; players.len()];
    if players.is_empty() {
        return owned;
    }
    let cell_area = pitch.area() / (nx * ny) as f64;

    for iy in 0..ny {
        for ix in 0..nx {
            let q = Grid::cell_centre(pitch, nx, ny, ix, iy);
            let mut best = f64::INFINITY;
            let mut best_i = 0usize;
            for (i, p) in players.iter().enumerate() {
                let t = params.time_to_intercept(p, q);
                if t < best {
                    best = t;
                    best_i = i;
                }
            }
            owned[best_i] += cell_area;
        }
    }
    owned
}

/// A crude analytic stand-in for a learned expected-threat surface: value rises
/// as you near the goal and as the shooting angle widens.
///
/// Normalised to [0,1]. Documented in docs/03-tactical-metrics.md as the piece
/// most worth replacing with a model fitted to real shot/goal data.
pub fn pitch_value(pitch: &Pitch, p: Vec2) -> f64 {
    let goal = pitch.attacking_goal();
    let half_goal = 7.32 * 0.5;
    let post_a = Vec2::new(goal.x, goal.y - half_goal);
    let post_b = Vec2::new(goal.x, goal.y + half_goal);

    // Distance term: decays over roughly the length of the attacking third.
    let d = p.dist(goal);
    let dist_term = (-d / 25.0).exp();

    // Angle subtended by the goal mouth, normalised by its maximum (pi).
    let a = post_a.sub(p);
    let b = post_b.sub(p);
    let angle = a.cross(b).abs().atan2(a.dot(b)).abs();
    let angle_term = angle / std::f64::consts::PI;

    (dist_term * 0.65 + angle_term * 0.35).clamp(0.0, 1.0)
}

/// Space that is both controlled by the attacking team *and* worth having.
///
/// The headline off-the-ball number: a winger hugging the touchline in their
/// own half owns space that this metric correctly values at almost nothing.
pub fn dangerous_space(control: &Grid, pitch: &Pitch) -> f64 {
    let mut acc = 0.0;
    for iy in 0..control.ny {
        for ix in 0..control.nx {
            let q = Grid::cell_centre(pitch, control.nx, control.ny, ix, iy);
            acc += control.values[iy * control.nx + ix] * pitch_value(pitch, q);
        }
    }
    acc * control.cell_area
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn equal_times_give_even_control() {
        let pitch = Pitch::default();
        let players = vec![
            PlayerState::still(Vec2::new(50.0, 34.0), true),
            PlayerState::still(Vec2::new(55.0, 34.0), false),
        ];
        let g = pitch_control(&players, &pitch, 21, 17, &ControlParams::default());
        // The midpoint between the two is equidistant, so control is ~0.5.
        let ix = (52.5 / 105.0 * 21.0) as usize;
        let iy = (34.0 / 68.0 * 17.0) as usize;
        assert!((g.values[iy * 21 + ix] - 0.5).abs() < 0.05);
    }

    #[test]
    fn a_lone_team_controls_everything() {
        let pitch = Pitch::default();
        let players = vec![PlayerState::still(Vec2::new(50.0, 34.0), true)];
        let g = pitch_control(&players, &pitch, 20, 20, &ControlParams::default());
        assert!(g.values.iter().all(|&v| (v - 1.0).abs() < EPS));
        assert!((g.total() - pitch.area()).abs() < 1e-6);
    }

    #[test]
    fn no_players_is_a_uniform_half_field() {
        let pitch = Pitch::default();
        let g = pitch_control(&[], &pitch, 10, 10, &ControlParams::default());
        assert!(g.values.iter().all(|&v| (v - 0.5).abs() < EPS));
    }

    #[test]
    fn control_is_bounded_and_antisymmetric() {
        let pitch = Pitch::default();
        let a = vec![
            PlayerState::new(Vec2::new(30.0, 20.0), Vec2::new(2.0, 0.0), true),
            PlayerState::new(Vec2::new(70.0, 50.0), Vec2::new(-1.0, 1.0), false),
        ];
        // Same players, teams swapped.
        let b: Vec<PlayerState> = a
            .iter()
            .map(|p| PlayerState::new(p.pos, p.vel, !p.is_attacking))
            .collect();
        let ga = pitch_control(&a, &pitch, 16, 16, &ControlParams::default());
        let gb = pitch_control(&b, &pitch, 16, 16, &ControlParams::default());
        for (va, vb) in ga.values.iter().zip(gb.values.iter()) {
            assert!((0.0..=1.0).contains(va));
            assert!((va + vb - 1.0).abs() < 1e-9, "swapping teams must mirror control");
        }
    }

    #[test]
    fn momentum_shifts_control_forward() {
        let params = ControlParams::default();
        let target = Vec2::new(70.0, 34.0);
        let still = PlayerState::still(Vec2::new(50.0, 34.0), true);
        let sprinting = PlayerState::new(Vec2::new(50.0, 34.0), Vec2::new(6.0, 0.0), true);
        assert!(
            params.time_to_intercept(&sprinting, target) < params.time_to_intercept(&still, target),
            "running toward the target must shorten arrival time"
        );
    }

    #[test]
    fn ownership_partitions_the_whole_pitch() {
        let pitch = Pitch::default();
        let players = vec![
            PlayerState::still(Vec2::new(20.0, 20.0), true),
            PlayerState::still(Vec2::new(80.0, 20.0), false),
            PlayerState::still(Vec2::new(50.0, 55.0), true),
        ];
        let owned = player_space_ownership(&players, &pitch, 60, 40, &ControlParams::default());
        let sum: f64 = owned.iter().sum();
        assert!((sum - pitch.area()).abs() < 1e-6, "ownership must tile the pitch");
        assert!(owned.iter().all(|&a| a > 0.0), "each player owns their own neighbourhood");
    }

    #[test]
    fn pitch_value_rises_toward_goal() {
        let pitch = Pitch::default();
        let far = pitch_value(&pitch, Vec2::new(10.0, 34.0));
        let mid = pitch_value(&pitch, Vec2::new(60.0, 34.0));
        let near = pitch_value(&pitch, Vec2::new(98.0, 34.0));
        assert!(far < mid && mid < near);
        // Central is worth more than the byline corner at the same distance.
        let corner = pitch_value(&pitch, Vec2::new(98.0, 2.0));
        assert!(near > corner);
        assert!((0.0..=1.0).contains(&near));
    }

    #[test]
    fn dangerous_space_prefers_the_final_third() {
        let pitch = Pitch::default();
        let params = ControlParams::default();
        let deep = vec![
            PlayerState::still(Vec2::new(15.0, 34.0), true),
            PlayerState::still(Vec2::new(90.0, 34.0), false),
        ];
        let high = vec![
            PlayerState::still(Vec2::new(90.0, 34.0), true),
            PlayerState::still(Vec2::new(15.0, 34.0), false),
        ];
        let gd = pitch_control(&deep, &pitch, 40, 30, &params);
        let gh = pitch_control(&high, &pitch, 40, 30, &params);
        assert!(dangerous_space(&gh, &pitch) > dangerous_space(&gd, &pitch));
    }
}
