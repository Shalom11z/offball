//! Off-the-ball positioning metrics.
//!
//! Every function here assumes the analysed team attacks toward **+x** (see
//! `pitch::Pitch::flip`). Positions are pitch metres.

use crate::geom::{hull_area, point_segment_distance, Vec2};

/// The offside line for a team attacking +x, per Law 11.
///
/// It is the greater of the second-last defender's `x` and the ball's `x` —
/// you cannot be offside from behind the ball. `defenders_x` should include the
/// goalkeeper, who is usually but not always the last defender.
///
/// Returns `None` with fewer than two defenders visible, which is a tracking
/// failure rather than a real match state, and must not be silently scored.
pub fn offside_line(defenders_x: &[f64], ball_x: f64) -> Option<f64> {
    if defenders_x.len() < 2 {
        return None;
    }
    let mut xs = defenders_x.to_vec();
    // Descending: xs[0] is the deepest-lying (last) defender.
    xs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Some(xs[1].max(ball_x))
}

/// Signed distance from the offside line: positive means beyond it (offside
/// position), negative means safely onside.
///
/// Reported as a continuous value rather than a boolean because the coaching
/// question is "how fine is this player timing their runs?", and a striker
/// living at -0.3m is doing something quite different from one at -6m.
pub fn offside_margin(attacker_x: f64, offside_line_x: f64) -> f64 {
    attacker_x - offside_line_x
}

/// A pass is considered available when no defender sits within `corridor`
/// metres of the straight ball -> receiver segment.
///
/// `corridor` around 1.2m approximates an interceptable reach; widen it for
/// slower passes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneVerdict {
    pub open: bool,
    /// Distance from the lane to the closest defender, in metres.
    pub clearance: f64,
    /// How many defenders are inside the corridor.
    pub blockers: usize,
}

pub fn passing_lane(ball: Vec2, receiver: Vec2, defenders: &[Vec2], corridor: f64) -> LaneVerdict {
    let mut clearance = f64::INFINITY;
    let mut blockers = 0;
    for &d in defenders {
        let dist = point_segment_distance(d, ball, receiver);
        if dist < clearance {
            clearance = dist;
        }
        if dist <= corridor {
            blockers += 1;
        }
    }
    LaneVerdict {
        open: blockers == 0,
        clearance,
        blockers,
    }
}

/// Team shape summary for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Convex-hull area of the outfield block, m^2. Lower = more compact.
    pub hull_area: f64,
    /// Spread along the touchline direction (x), metres.
    pub depth: f64,
    /// Spread across the pitch (y), metres.
    pub width: f64,
    /// Team centroid.
    pub centroid: Vec2,
}

/// Compute the shape summary. The goalkeeper should be excluded by the caller —
/// including them inflates depth by 20-30m and makes the metric useless.
pub fn team_shape(players: &[Vec2]) -> Option<Shape> {
    if players.is_empty() {
        return None;
    }
    let n = players.len() as f64;
    let centroid = Vec2::new(
        players.iter().map(|p| p.x).sum::<f64>() / n,
        players.iter().map(|p| p.y).sum::<f64>() / n,
    );
    let (mut minx, mut maxx) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut miny, mut maxy) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in players {
        minx = minx.min(p.x);
        maxx = maxx.max(p.x);
        miny = miny.min(p.y);
        maxy = maxy.max(p.y);
    }
    Some(Shape {
        hull_area: hull_area(players),
        depth: maxx - minx,
        width: maxy - miny,
        centroid,
    })
}

/// Group defenders into horizontal banks (back line, midfield line, ...) by
/// 1-D k-means on `x`.
///
/// Initialised from evenly spaced quantiles rather than randomly, so the result
/// is deterministic — a requirement for reproducible match reports. Returns
/// centres sorted ascending (deepest line first for a team defending at low x).
pub fn defensive_lines(defenders_x: &[f64], k: usize, iterations: usize) -> Vec<f64> {
    if defenders_x.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(defenders_x.len());
    let mut xs = defenders_x.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Quantile initialisation.
    let mut centres: Vec<f64> = (0..k)
        .map(|i| xs[((i as f64 + 0.5) / k as f64 * xs.len() as f64) as usize % xs.len()])
        .collect();

    for _ in 0..iterations {
        let mut sums = vec![0.0; k];
        let mut counts = vec![0usize; k];
        for &x in &xs {
            let mut best = 0;
            let mut bd = f64::INFINITY;
            for (i, &c) in centres.iter().enumerate() {
                let d = (x - c).abs();
                if d < bd {
                    bd = d;
                    best = i;
                }
            }
            sums[best] += x;
            counts[best] += 1;
        }
        let mut moved = false;
        for i in 0..k {
            if counts[i] > 0 {
                let nc = sums[i] / counts[i] as f64;
                if (nc - centres[i]).abs() > 1e-9 {
                    moved = true;
                }
                centres[i] = nc;
            }
        }
        if !moved {
            break;
        }
    }
    centres.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    centres
}

/// How many opposition banks an attacker has positioned themselves beyond.
///
/// The core "is this player breaking lines?" signal: a striker between the
/// centre-backs and the midfield line scores 1, one on the last shoulder
/// scores the full count.
pub fn lines_broken(attacker_x: f64, line_centres: &[f64]) -> usize {
    line_centres.iter().filter(|&&c| attacker_x > c).count()
}

/// Marking pressure on a player: 1.0 means an opponent is on top of them,
/// decaying to 0 as the nearest opponent gets further away.
///
/// `scale` is the distance in metres at which pressure has decayed to 1/e;
/// ~5m matches the range over which a defender can realistically contest a
/// first touch.
pub fn marking_pressure(player: Vec2, opponents: &[Vec2], scale: f64) -> f64 {
    let nearest = opponents
        .iter()
        .map(|o| o.dist(player))
        .fold(f64::INFINITY, f64::min);
    if !nearest.is_finite() {
        return 0.0;
    }
    (-nearest / scale.max(1e-6)).exp().clamp(0.0, 1.0)
}

/// Distance to the nearest opponent, or `None` when there are none tracked.
pub fn nearest_opponent_distance(player: Vec2, opponents: &[Vec2]) -> Option<f64> {
    opponents
        .iter()
        .map(|o| o.dist(player))
        .fold(None, |acc: Option<f64>, d| {
            Some(acc.map_or(d, |a| a.min(d)))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offside_line_uses_the_second_last_defender() {
        // Keeper deep at 2m, back line at 40/42/45.
        let defs = vec![2.0, 40.0, 42.0, 45.0];
        // Second-last (i.e. second highest x) is 42.
        assert_eq!(offside_line(&defs, 30.0), Some(42.0));
    }

    #[test]
    fn ball_ahead_of_the_defence_pushes_the_line_forward() {
        let defs = vec![2.0, 40.0, 42.0, 45.0];
        // You cannot be offside from behind the ball.
        assert_eq!(offside_line(&defs, 60.0), Some(60.0));
    }

    #[test]
    fn offside_line_needs_two_defenders() {
        assert_eq!(offside_line(&[40.0], 10.0), None);
        assert_eq!(offside_line(&[], 10.0), None);
    }

    #[test]
    fn offside_margin_sign_convention() {
        assert!(
            offside_margin(44.0, 42.0) > 0.0,
            "beyond the line is positive"
        );
        assert!(offside_margin(39.0, 42.0) < 0.0, "onside is negative");
    }

    #[test]
    fn passing_lane_detects_a_blocker() {
        let ball = Vec2::new(0.0, 0.0);
        let receiver = Vec2::new(20.0, 0.0);
        // Defender standing right in the lane.
        let blocked = passing_lane(ball, receiver, &[Vec2::new(10.0, 0.5)], 1.2);
        assert!(!blocked.open);
        assert_eq!(blocked.blockers, 1);
        assert!((blocked.clearance - 0.5).abs() < 1e-9);

        // Same defender, but the lane is now well clear of them.
        let open = passing_lane(ball, receiver, &[Vec2::new(10.0, 6.0)], 1.2);
        assert!(open.open);
        assert_eq!(open.blockers, 0);
    }

    #[test]
    fn a_defender_behind_the_passer_does_not_block() {
        let ball = Vec2::new(10.0, 0.0);
        let receiver = Vec2::new(30.0, 0.0);
        // Directly on the infinite line, but behind the ball.
        let v = passing_lane(ball, receiver, &[Vec2::new(0.0, 0.0)], 1.2);
        assert!(
            v.open,
            "only defenders between passer and receiver block the lane"
        );
    }

    #[test]
    fn empty_defence_leaves_the_lane_open() {
        let v = passing_lane(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), &[], 1.2);
        assert!(v.open);
        assert_eq!(v.blockers, 0);
        assert!(v.clearance.is_infinite());
    }

    #[test]
    fn team_shape_measures_a_compact_block() {
        let compact = vec![
            Vec2::new(40.0, 20.0),
            Vec2::new(45.0, 30.0),
            Vec2::new(42.0, 40.0),
            Vec2::new(48.0, 48.0),
        ];
        let stretched = vec![
            Vec2::new(10.0, 5.0),
            Vec2::new(50.0, 30.0),
            Vec2::new(90.0, 60.0),
            Vec2::new(70.0, 10.0),
        ];
        let a = team_shape(&compact).unwrap();
        let b = team_shape(&stretched).unwrap();
        assert!(a.hull_area < b.hull_area);
        assert!(a.depth < b.depth && a.width < b.width);
        assert!((a.centroid.x - 43.75).abs() < 1e-9);
    }

    #[test]
    fn team_shape_of_nobody_is_none() {
        assert!(team_shape(&[]).is_none());
    }

    #[test]
    fn defensive_lines_recover_a_four_four_two() {
        // Back four at ~30, midfield four at ~50, front two at ~70.
        let xs = vec![
            29.0, 30.0, 31.0, 30.5, // back line
            49.0, 50.0, 51.0, 50.5, // midfield
            70.0, 71.0, // forwards
        ];
        let lines = defensive_lines(&xs, 3, 50);
        assert_eq!(lines.len(), 3);
        assert!(
            (lines[0] - 30.125).abs() < 1.0,
            "back line near 30, got {}",
            lines[0]
        );
        assert!(
            (lines[1] - 50.125).abs() < 1.0,
            "midfield near 50, got {}",
            lines[1]
        );
        assert!(
            (lines[2] - 70.5).abs() < 1.0,
            "forwards near 70, got {}",
            lines[2]
        );
        // Sorted ascending.
        assert!(lines.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn defensive_lines_is_deterministic_and_handles_edges() {
        let xs = vec![30.0, 31.0, 50.0, 70.0];
        assert_eq!(defensive_lines(&xs, 2, 50), defensive_lines(&xs, 2, 50));
        assert!(defensive_lines(&[], 3, 10).is_empty());
        assert!(defensive_lines(&xs, 0, 10).is_empty());
        // k larger than the population is clamped.
        assert_eq!(defensive_lines(&xs, 99, 10).len(), 4);
    }

    #[test]
    fn lines_broken_counts_banks_passed() {
        let lines = vec![30.0, 50.0, 70.0];
        assert_eq!(lines_broken(20.0, &lines), 0);
        assert_eq!(lines_broken(40.0, &lines), 1);
        assert_eq!(lines_broken(60.0, &lines), 2);
        assert_eq!(lines_broken(80.0, &lines), 3);
    }

    #[test]
    fn marking_pressure_decays_with_distance() {
        let p = Vec2::new(50.0, 34.0);
        let tight = marking_pressure(p, &[Vec2::new(50.5, 34.0)], 5.0);
        let loose = marking_pressure(p, &[Vec2::new(65.0, 34.0)], 5.0);
        assert!(tight > 0.8, "a defender half a metre away is tight marking");
        assert!(loose < 0.1);
        assert_eq!(
            marking_pressure(p, &[], 5.0),
            0.0,
            "nobody near = no pressure"
        );
    }

    #[test]
    fn nearest_opponent_distance_picks_the_minimum() {
        let p = Vec2::new(0.0, 0.0);
        let d = nearest_opponent_distance(p, &[Vec2::new(10.0, 0.0), Vec2::new(3.0, 4.0)]);
        assert!((d.unwrap() - 5.0).abs() < 1e-9);
        assert!(nearest_opponent_distance(p, &[]).is_none());
    }
}
