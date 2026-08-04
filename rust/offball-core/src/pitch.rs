//! Pitch coordinate system and the template keypoints used to anchor the
//! homography fit.
//!
//! Convention used everywhere in this codebase:
//!   - origin at the bottom-left corner of the pitch,
//!   - `x` runs 0..length along the touchline, `y` runs 0..width,
//!   - the team **attacking +x** is always normalised to do so before metrics
//!     are computed, so "forward" is unambiguously increasing `x`.

use crate::geom::Vec2;

/// Pitch dimensions in metres. IFAB permits 90-120m by 45-90m, so this is a
/// per-venue parameter rather than a constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pitch {
    pub length: f64,
    pub width: f64,
}

impl Default for Pitch {
    /// The UEFA-standard 105x68 pitch.
    fn default() -> Self {
        Self { length: 105.0, width: 68.0 }
    }
}

impl Pitch {
    pub const fn new(length: f64, width: f64) -> Self {
        Self { length, width }
    }

    pub fn area(&self) -> f64 {
        self.length * self.width
    }

    pub fn centre(&self) -> Vec2 {
        Vec2::new(self.length * 0.5, self.width * 0.5)
    }

    /// Centre of the goal a team attacking +x is shooting at.
    pub fn attacking_goal(&self) -> Vec2 {
        Vec2::new(self.length, self.width * 0.5)
    }

    /// Centre of the goal that same team is defending.
    pub fn defending_goal(&self) -> Vec2 {
        Vec2::new(0.0, self.width * 0.5)
    }

    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= 0.0 && p.x <= self.length && p.y >= 0.0 && p.y <= self.width
    }

    /// Clamp a point onto the pitch. Tracking noise regularly puts a player
    /// half a metre outside the touchline; for area metrics we'd rather clamp
    /// than discard the observation.
    pub fn clamp(&self, p: Vec2) -> Vec2 {
        Vec2::new(p.x.clamp(0.0, self.length), p.y.clamp(0.0, self.width))
    }

    /// Named line/arc intersections a pitch-keypoint model is trained to find.
    /// These are the `dst` side of the homography correspondence set.
    ///
    /// Distances (5.5m box, 16.5m box, 9.15m circle, 7.32m goal) are fixed by
    /// the laws of the game and do not scale with pitch size.
    pub fn template_keypoints(&self) -> Vec<(&'static str, Vec2)> {
        let (l, w) = (self.length, self.width);
        let (hw, hl) = (w * 0.5, l * 0.5);
        vec![
            ("corner_bl", Vec2::new(0.0, 0.0)),
            ("corner_tl", Vec2::new(0.0, w)),
            ("corner_br", Vec2::new(l, 0.0)),
            ("corner_tr", Vec2::new(l, w)),
            ("halfway_bottom", Vec2::new(hl, 0.0)),
            ("halfway_top", Vec2::new(hl, w)),
            ("centre_spot", Vec2::new(hl, hw)),
            ("centre_circle_bottom", Vec2::new(hl, hw - 9.15)),
            ("centre_circle_top", Vec2::new(hl, hw + 9.15)),
            // Left penalty area (16.5m deep, 40.32m wide).
            ("pen_l_bl", Vec2::new(0.0, hw - 20.16)),
            ("pen_l_tl", Vec2::new(0.0, hw + 20.16)),
            ("pen_l_br", Vec2::new(16.5, hw - 20.16)),
            ("pen_l_tr", Vec2::new(16.5, hw + 20.16)),
            ("pen_spot_l", Vec2::new(11.0, hw)),
            // Left goal area (5.5m deep, 18.32m wide).
            ("goal_l_br", Vec2::new(5.5, hw - 9.16)),
            ("goal_l_tr", Vec2::new(5.5, hw + 9.16)),
            // Right penalty area.
            ("pen_r_br", Vec2::new(l, hw - 20.16)),
            ("pen_r_tr", Vec2::new(l, hw + 20.16)),
            ("pen_r_bl", Vec2::new(l - 16.5, hw - 20.16)),
            ("pen_r_tl", Vec2::new(l - 16.5, hw + 20.16)),
            ("pen_spot_r", Vec2::new(l - 11.0, hw)),
            // Right goal area.
            ("goal_r_bl", Vec2::new(l - 5.5, hw - 9.16)),
            ("goal_r_tl", Vec2::new(l - 5.5, hw + 9.16)),
        ]
    }

    /// Mirror a point so that a team attacking -x can be analysed with the same
    /// "forward is +x" code. Applied at the start of the metrics stage using
    /// the period's known attacking direction.
    pub fn flip(&self, p: Vec2) -> Vec2 {
        Vec2::new(self.length - p.x, self.width - p.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_uefa_standard() {
        let p = Pitch::default();
        assert_eq!(p.length, 105.0);
        assert_eq!(p.width, 68.0);
        assert!((p.area() - 7140.0).abs() < 1e-9);
    }

    #[test]
    fn template_keypoints_are_all_on_the_pitch() {
        for pitch in [Pitch::default(), Pitch::new(100.0, 64.0), Pitch::new(115.0, 75.0)] {
            for (name, p) in pitch.template_keypoints() {
                assert!(pitch.contains(p), "{name} at {p:?} is off {pitch:?}");
            }
        }
    }

    #[test]
    fn template_keypoint_names_are_unique() {
        let kps = Pitch::default().template_keypoints();
        let mut names: Vec<&str> = kps.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len());
    }

    #[test]
    fn penalty_area_geometry_matches_the_laws() {
        let p = Pitch::default();
        let kps = p.template_keypoints();
        let get = |n: &str| kps.iter().find(|(k, _)| *k == n).unwrap().1;
        // 40.32m wide, 16.5m deep, spot 11m out.
        assert!((get("pen_l_tl").y - get("pen_l_bl").y - 40.32).abs() < 1e-9);
        assert!((get("pen_l_br").x - 16.5).abs() < 1e-9);
        assert!((get("pen_spot_l").x - 11.0).abs() < 1e-9);
        // Both penalty areas are the same size.
        assert!(((get("pen_r_tr").y - get("pen_r_br").y) - 40.32).abs() < 1e-9);
    }

    #[test]
    fn flip_is_an_involution() {
        let p = Pitch::default();
        let q = Vec2::new(20.0, 15.0);
        let back = p.flip(p.flip(q));
        assert!(back.dist(q) < 1e-9);
        // Attacking and defending goals swap under a flip.
        assert!(p.flip(p.attacking_goal()).dist(p.defending_goal()) < 1e-9);
    }

    #[test]
    fn clamp_pulls_stray_tracks_back_inside() {
        let p = Pitch::default();
        assert_eq!(p.clamp(Vec2::new(-2.0, 70.0)), Vec2::new(0.0, 68.0));
        assert_eq!(p.clamp(Vec2::new(50.0, 30.0)), Vec2::new(50.0, 30.0));
    }
}
