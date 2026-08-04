//! Planar geometry primitives shared by every other module.
//!
//! Everything here is `f64` and allocation-light: these functions run once per
//! player per frame, which at 25 fps over a 90-minute match is on the order of
//! 30 million calls.

/// A point in the plane. Used for both image pixels and pitch metres; the
/// coordinate space is always implied by the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

// Inherent `add`/`sub` are deliberate: `Vec2` is a value type used in dense
// geometric expressions, where `a.sub(b).norm()` reads better than the operator
// form and avoids ambiguity with scalar arithmetic on `.x` / `.y`.
#[allow(clippy::should_implement_trait)]
impl Vec2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }

    pub fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }

    pub fn scale(self, k: f64) -> Vec2 {
        Vec2::new(self.x * k, self.y * k)
    }

    pub fn dot(self, o: Vec2) -> f64 {
        self.x * o.x + self.y * o.y
    }

    /// 2D cross product (the z component of the 3D cross product). Positive
    /// when `o` is counter-clockwise from `self`.
    pub fn cross(self, o: Vec2) -> f64 {
        self.x * o.y - self.y * o.x
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn dist(self, o: Vec2) -> f64 {
        self.sub(o).norm()
    }
}

/// Shortest distance from `p` to the finite segment `a`-`b`.
///
/// Used by the passing-lane test, where the segment is ball -> receiver and `p`
/// is a defender. The finite (not infinite-line) version matters: a defender
/// stood behind the passer is not in the lane.
pub fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let ab = b.sub(a);
    let len2 = ab.dot(ab);
    if len2 <= f64::EPSILON {
        return p.dist(a);
    }
    // Projection parameter of p onto ab, clamped to the segment.
    let t = (p.sub(a).dot(ab) / len2).clamp(0.0, 1.0);
    p.dist(a.add(ab.scale(t)))
}

/// Convex hull via Andrew's monotone chain, returned counter-clockwise.
///
/// Returns fewer than 3 points when the input is degenerate (collinear or
/// too small); callers should treat that as zero area.
pub fn convex_hull(points: &[Vec2]) -> Vec<Vec2> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| (a.x - b.x).abs() < 1e-12 && (a.y - b.y).abs() < 1e-12);
    if pts.len() < 3 {
        return pts;
    }

    let build = |iter: &mut dyn Iterator<Item = &Vec2>| -> Vec<Vec2> {
        let mut chain: Vec<Vec2> = Vec::with_capacity(pts.len());
        for &p in iter {
            while chain.len() >= 2 {
                let n = chain.len();
                // Pop while the last turn is clockwise or collinear.
                if chain[n - 1].sub(chain[n - 2]).cross(p.sub(chain[n - 2])) <= 0.0 {
                    chain.pop();
                } else {
                    break;
                }
            }
            chain.push(p);
        }
        chain.pop(); // last point is the first point of the other chain
        chain
    };

    let mut hull = build(&mut pts.iter());
    hull.extend(build(&mut pts.iter().rev()));
    hull
}

/// Area of a simple polygon via the shoelace formula. Sign-independent.
pub fn polygon_area(poly: &[Vec2]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        acc += a.cross(b);
    }
    (acc * 0.5).abs()
}

/// Convex-hull area of a point set. This is the "team shape" area used by the
/// compactness metrics.
pub fn hull_area(points: &[Vec2]) -> f64 {
    polygon_area(&convex_hull(points))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_distance_clamps_to_endpoints() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 0.0);
        // Perpendicular foot lands inside the segment.
        assert!((point_segment_distance(Vec2::new(5.0, 3.0), a, b) - 3.0).abs() < 1e-9);
        // Behind `a`: distance is to the endpoint, not the infinite line.
        assert!((point_segment_distance(Vec2::new(-4.0, 0.0), a, b) - 4.0).abs() < 1e-9);
        // Beyond `b`.
        assert!((point_segment_distance(Vec2::new(13.0, 4.0), a, b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn degenerate_segment_falls_back_to_point_distance() {
        let a = Vec2::new(2.0, 2.0);
        assert!((point_segment_distance(Vec2::new(2.0, 5.0), a, a) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn hull_of_square_with_interior_points() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(0.0, 4.0),
            Vec2::new(2.0, 2.0), // interior, must be dropped
            Vec2::new(1.0, 3.0), // interior, must be dropped
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        assert!((hull_area(&pts) - 16.0).abs() < 1e-9);
    }

    #[test]
    fn collinear_points_have_zero_area() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(3.0, 3.0),
        ];
        assert!(hull_area(&pts) < 1e-9);
    }

    #[test]
    fn triangle_area() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(6.0, 0.0),
            Vec2::new(0.0, 4.0),
        ];
        assert!((hull_area(&pts) - 12.0).abs() < 1e-9);
    }
}
