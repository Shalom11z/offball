//! Image <-> pitch plane homography.
//!
//! Broadcast footage is a moving perspective view of a flat plane, so a single
//! 3x3 homography maps pixels to pitch metres for any given frame. We solve it
//! with a normalised DLT and wrap it in RANSAC because pitch-keypoint detectors
//! produce confident-but-wrong corners often enough that a plain least-squares
//! fit is unusable in practice.

use crate::geom::Vec2;

/// Row-major 3x3 homography. `h[8]` is normalised to 1.0 by the solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography(pub [f64; 9]);

impl Homography {
    pub const IDENTITY: Homography = Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    /// Map a point through the homography. Returns `None` when the point maps
    /// to (or near) the line at infinity — i.e. the horizon, where a pixel
    /// corresponds to no finite pitch location.
    pub fn apply(&self, p: Vec2) -> Option<Vec2> {
        let h = &self.0;
        let w = h[6] * p.x + h[7] * p.y + h[8];
        if w.abs() < 1e-12 {
            return None;
        }
        Some(Vec2::new(
            (h[0] * p.x + h[1] * p.y + h[2]) / w,
            (h[3] * p.x + h[4] * p.y + h[5]) / w,
        ))
    }

    /// Matrix inverse, giving the pitch -> image mapping.
    pub fn inverse(&self) -> Option<Homography> {
        let m = &self.0;
        let c = [
            m[4] * m[8] - m[5] * m[7],
            m[2] * m[7] - m[1] * m[8],
            m[1] * m[5] - m[2] * m[4],
            m[5] * m[6] - m[3] * m[8],
            m[0] * m[8] - m[2] * m[6],
            m[2] * m[3] - m[0] * m[5],
            m[3] * m[7] - m[4] * m[6],
            m[1] * m[6] - m[0] * m[7],
            m[0] * m[4] - m[1] * m[3],
        ];
        let det = m[0] * c[0] + m[1] * c[3] + m[2] * c[6];
        if det.abs() < 1e-12 {
            return None;
        }
        let mut out = [0.0; 9];
        for i in 0..9 {
            out[i] = c[i] / det;
        }
        Some(Homography(out))
    }

    /// Mean reprojection error in destination units over the given pairs.
    pub fn reprojection_error(&self, src: &[Vec2], dst: &[Vec2]) -> f64 {
        if src.is_empty() {
            return f64::INFINITY;
        }
        let mut acc = 0.0;
        for (s, d) in src.iter().zip(dst.iter()) {
            match self.apply(*s) {
                Some(p) => acc += p.dist(*d),
                None => return f64::INFINITY,
            }
        }
        acc / src.len() as f64
    }
}

/// Hartley normalisation: translate the centroid to the origin and scale so the
/// mean distance from it is sqrt(2). Skipping this step makes the DLT system
/// badly conditioned for pixel coordinates in the hundreds.
fn normalise(pts: &[Vec2]) -> (Vec<Vec2>, Homography) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.y).sum::<f64>() / n;
    let mean_d = pts
        .iter()
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let s = if mean_d > 1e-12 {
        std::f64::consts::SQRT_2 / mean_d
    } else {
        1.0
    };
    let t = Homography([s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0]);
    let out = pts.iter().map(|p| Vec2::new(s * (p.x - cx), s * (p.y - cy))).collect();
    (out, t)
}

fn mat_mul3(a: &Homography, b: &Homography) -> Homography {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += a.0[r * 3 + k] * b.0[k * 3 + c];
            }
            out[r * 3 + c] = acc;
        }
    }
    Homography(out)
}

/// Solve a dense linear system by Gaussian elimination with partial pivoting.
/// `a` is `n` rows of `n` columns, row-major.
fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // Partial pivot: pick the row with the largest magnitude in this column.
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-12 {
            return None; // singular
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        let d = a[col * n + col];
        for row in (col + 1)..n {
            let f = a[row * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
    // Back-substitution.
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = b[row];
        for k in (row + 1)..n {
            acc -= a[row * n + k] * x[k];
        }
        x[row] = acc / a[row * n + row];
    }
    Some(x)
}

/// Direct Linear Transform fit over all correspondences (least squares for
/// n > 4). Requires at least 4 pairs, no 3 of which are collinear.
pub fn fit_dlt(src: &[Vec2], dst: &[Vec2]) -> Option<Homography> {
    if src.len() < 4 || src.len() != dst.len() {
        return None;
    }
    let (ns, ts) = normalise(src);
    let (nd, td) = normalise(dst);

    // Each pair contributes two rows to A*h = rhs, with h[8] fixed to 1.
    let n = ns.len();
    let mut a = Vec::with_capacity(2 * n * 8);
    let mut rhs = Vec::with_capacity(2 * n);
    for i in 0..n {
        let (x, y) = (ns[i].x, ns[i].y);
        let (u, v) = (nd[i].x, nd[i].y);
        a.extend_from_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]);
        rhs.push(u);
        a.extend_from_slice(&[0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]);
        rhs.push(v);
    }

    // Normal equations: (A^T A) h = A^T rhs. An 8x8 solve regardless of n.
    let rows = 2 * n;
    let mut ata = vec![0.0; 64];
    let mut atb = vec![0.0; 8];
    for r in 0..rows {
        for i in 0..8 {
            let air = a[r * 8 + i];
            if air == 0.0 {
                continue;
            }
            for j in 0..8 {
                ata[i * 8 + j] += air * a[r * 8 + j];
            }
            atb[i] += air * rhs[r];
        }
    }
    let h = solve_linear(ata, atb, 8)?;

    let hn = Homography([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]);
    // Undo normalisation: H = Td^-1 * Hn * Ts
    let td_inv = td.inverse()?;
    Some(mat_mul3(&td_inv, &mat_mul3(&hn, &ts)))
}

/// Deterministic RANSAC wrapper around [`fit_dlt`].
///
/// Determinism matters here: the same match footage must produce the same
/// tracking data on every re-run, so we drive sampling from a seeded xorshift
/// rather than a thread RNG.
///
/// `threshold` is the inlier cut-off in destination units (metres, when
/// fitting image -> pitch).
pub fn fit_ransac(
    src: &[Vec2],
    dst: &[Vec2],
    threshold: f64,
    iterations: usize,
    seed: u64,
) -> Option<(Homography, Vec<usize>)> {
    let n = src.len();
    if n < 4 || n != dst.len() {
        return None;
    }
    if n == 4 {
        let h = fit_dlt(src, dst)?;
        return Some((h, (0..4).collect()));
    }

    let mut state = seed | 1; // xorshift64 must not be seeded with zero
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut best_inliers: Vec<usize> = Vec::new();
    for _ in 0..iterations {
        // Sample 4 distinct indices.
        let mut idx = [0usize; 4];
        let mut count = 0;
        let mut guard = 0;
        while count < 4 && guard < 64 {
            guard += 1;
            let c = (next() % n as u64) as usize;
            if !idx[..count].contains(&c) {
                idx[count] = c;
                count += 1;
            }
        }
        if count < 4 {
            continue;
        }

        let s: Vec<Vec2> = idx.iter().map(|&i| src[i]).collect();
        let d: Vec<Vec2> = idx.iter().map(|&i| dst[i]).collect();
        let Some(h) = fit_dlt(&s, &d) else { continue };

        let inliers: Vec<usize> = (0..n)
            .filter(|&i| match h.apply(src[i]) {
                Some(p) => p.dist(dst[i]) <= threshold,
                None => false,
            })
            .collect();
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }

    if best_inliers.len() < 4 {
        return None;
    }
    // Refit on the full consensus set for a lower-variance estimate.
    let s: Vec<Vec2> = best_inliers.iter().map(|&i| src[i]).collect();
    let d: Vec<Vec2> = best_inliers.iter().map(|&i| dst[i]).collect();
    let refined = fit_dlt(&s, &d)?;
    Some((refined, best_inliers))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_h() -> Homography {
        // A realistic broadcast-ish projective transform.
        Homography([1.2, 0.35, 40.0, -0.15, 0.9, 25.0, 0.0006, 0.0011, 1.0])
    }

    #[test]
    fn recovers_a_known_homography() {
        let h = sample_h();
        let src = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(105.0, 0.0),
            Vec2::new(105.0, 68.0),
            Vec2::new(0.0, 68.0),
            Vec2::new(52.5, 34.0),
            Vec2::new(16.5, 13.85),
        ];
        let dst: Vec<Vec2> = src.iter().map(|p| h.apply(*p).unwrap()).collect();

        let fitted = fit_dlt(&src, &dst).expect("fit should succeed");
        assert!(fitted.reprojection_error(&src, &dst) < 1e-6);
    }

    #[test]
    fn inverse_round_trips() {
        let h = sample_h();
        let inv = h.inverse().unwrap();
        let p = Vec2::new(30.0, 20.0);
        let back = inv.apply(h.apply(p).unwrap()).unwrap();
        assert!(back.dist(p) < 1e-9);
    }

    #[test]
    fn ransac_rejects_outliers() {
        let h = sample_h();
        let mut src = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(105.0, 0.0),
            Vec2::new(105.0, 68.0),
            Vec2::new(0.0, 68.0),
            Vec2::new(52.5, 34.0),
            Vec2::new(16.5, 13.85),
            Vec2::new(88.5, 54.15),
            Vec2::new(52.5, 0.0),
        ];
        let mut dst: Vec<Vec2> = src.iter().map(|p| h.apply(*p).unwrap()).collect();

        // Two badly mismatched keypoints, the classic failure of line detectors
        // confusing the two penalty boxes.
        src.push(Vec2::new(20.0, 20.0));
        dst.push(Vec2::new(900.0, -400.0));
        src.push(Vec2::new(70.0, 50.0));
        dst.push(Vec2::new(-250.0, 700.0));

        let (fitted, inliers) =
            fit_ransac(&src, &dst, 1.0, 500, 42).expect("ransac should find consensus");
        assert_eq!(inliers.len(), 8, "the 8 clean correspondences are the consensus set");
        assert!(!inliers.contains(&8));
        assert!(!inliers.contains(&9));
        assert!(fitted.reprojection_error(&src[..8], &dst[..8]) < 1e-6);
    }

    #[test]
    fn ransac_is_deterministic() {
        let h = sample_h();
        // Must be in general position: a homography is undefined for a
        // collinear correspondence set.
        let src: Vec<Vec2> = (0..12)
            .map(|i| {
                let f = i as f64;
                Vec2::new(f * 8.0, 34.0 + 30.0 * (f * 1.7).sin())
            })
            .collect();
        let dst: Vec<Vec2> = src.iter().map(|p| h.apply(*p).unwrap()).collect();
        let a = fit_ransac(&src, &dst, 0.5, 200, 7).unwrap();
        let b = fit_ransac(&src, &dst, 0.5, 200, 7).unwrap();
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn too_few_points_is_none() {
        let src = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
        let dst = src.clone();
        assert!(fit_dlt(&src, &dst).is_none());
    }

    #[test]
    fn horizon_points_map_to_none() {
        // Construct a homography whose third row zeroes out at a known point.
        let h = Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -5.0]);
        assert!(h.apply(Vec2::new(5.0, 2.0)).is_none());
    }
}
