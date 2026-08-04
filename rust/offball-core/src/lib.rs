//! `offball-core` — the numeric kernels behind the offball analytics platform.
//!
//! The crate is usable as a plain Rust library; the `python` feature adds a
//! PyO3 extension module (`offball_core`) that the Python package imports when
//! available, falling back to an equivalent pure-Python implementation when it
//! is not. See `python/src/offball/kernels.py`.

pub mod control;
pub mod geom;
pub mod homography;
pub mod metrics;
pub mod pitch;

pub use control::{ControlParams, Grid, PlayerState};
pub use geom::Vec2;
pub use homography::Homography;
pub use pitch::Pitch;

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    type Pt = (f64, f64);

    fn to_vecs(pts: &[Pt]) -> Vec<Vec2> {
        pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect()
    }

    /// Fit an image -> pitch homography with RANSAC.
    ///
    /// Returns `(h, inlier_indices)` where `h` is a flat 9-element row-major
    /// matrix. Raises `ValueError` if no consensus is found.
    #[pyfunction]
    #[pyo3(signature = (src, dst, threshold=1.5, iterations=1000, seed=0))]
    fn fit_homography(
        src: Vec<Pt>,
        dst: Vec<Pt>,
        threshold: f64,
        iterations: usize,
        seed: u64,
    ) -> PyResult<(Vec<f64>, Vec<usize>)> {
        if src.len() != dst.len() {
            return Err(PyValueError::new_err("src and dst must be the same length"));
        }
        homography::fit_ransac(&to_vecs(&src), &to_vecs(&dst), threshold, iterations, seed)
            .map(|(h, inliers)| (h.0.to_vec(), inliers))
            .ok_or_else(|| PyValueError::new_err("homography fit failed: need >=4 non-collinear correspondences"))
    }

    /// Project points through a flat 9-element homography.
    /// Points on the horizon are returned as `None`.
    #[pyfunction]
    fn project(h: Vec<f64>, pts: Vec<Pt>) -> PyResult<Vec<Option<Pt>>> {
        if h.len() != 9 {
            return Err(PyValueError::new_err("homography must have 9 elements"));
        }
        let mut m = [0.0; 9];
        m.copy_from_slice(&h);
        let hm = Homography(m);
        Ok(pts
            .iter()
            .map(|&(x, y)| hm.apply(Vec2::new(x, y)).map(|p| (p.x, p.y)))
            .collect())
    }

    fn build_players(
        positions: &[Pt],
        velocities: &[Pt],
        is_attacking: &[bool],
    ) -> PyResult<Vec<PlayerState>> {
        if positions.len() != velocities.len() || positions.len() != is_attacking.len() {
            return Err(PyValueError::new_err(
                "positions, velocities and is_attacking must be the same length",
            ));
        }
        Ok(positions
            .iter()
            .zip(velocities)
            .zip(is_attacking)
            .map(|((&(px, py), &(vx, vy)), &att)| {
                PlayerState::new(Vec2::new(px, py), Vec2::new(vx, vy), att)
            })
            .collect())
    }

    /// Attacking-team pitch control on an `nx` by `ny` grid.
    /// Returns `(values, nx, ny, cell_area)` with `values` row-major.
    #[pyfunction]
    #[pyo3(signature = (positions, velocities, is_attacking, pitch_length=105.0, pitch_width=68.0,
                        nx=105, ny=68, reaction_time=0.7, max_speed=7.8, tau=0.45))]
    #[allow(clippy::too_many_arguments)]
    fn pitch_control(
        positions: Vec<Pt>,
        velocities: Vec<Pt>,
        is_attacking: Vec<bool>,
        pitch_length: f64,
        pitch_width: f64,
        nx: usize,
        ny: usize,
        reaction_time: f64,
        max_speed: f64,
        tau: f64,
    ) -> PyResult<(Vec<f64>, usize, usize, f64)> {
        if nx == 0 || ny == 0 {
            return Err(PyValueError::new_err("grid dimensions must be positive"));
        }
        let players = build_players(&positions, &velocities, &is_attacking)?;
        let p = Pitch::new(pitch_length, pitch_width);
        let params = ControlParams { reaction_time, max_speed, tau };
        let g = control::pitch_control(&players, &p, nx, ny, &params);
        Ok((g.values, g.nx, g.ny, g.cell_area))
    }

    /// Per-player owned area in m^2, aligned with the input order.
    #[pyfunction]
    #[pyo3(signature = (positions, velocities, is_attacking, pitch_length=105.0, pitch_width=68.0,
                        nx=105, ny=68, reaction_time=0.7, max_speed=7.8, tau=0.45))]
    #[allow(clippy::too_many_arguments)]
    fn space_ownership(
        positions: Vec<Pt>,
        velocities: Vec<Pt>,
        is_attacking: Vec<bool>,
        pitch_length: f64,
        pitch_width: f64,
        nx: usize,
        ny: usize,
        reaction_time: f64,
        max_speed: f64,
        tau: f64,
    ) -> PyResult<Vec<f64>> {
        if nx == 0 || ny == 0 {
            return Err(PyValueError::new_err("grid dimensions must be positive"));
        }
        let players = build_players(&positions, &velocities, &is_attacking)?;
        let p = Pitch::new(pitch_length, pitch_width);
        let params = ControlParams { reaction_time, max_speed, tau };
        Ok(control::player_space_ownership(&players, &p, nx, ny, &params))
    }

    /// Threat-weighted controlled area, in value-weighted m^2.
    #[pyfunction]
    #[pyo3(signature = (values, nx, ny, cell_area, pitch_length=105.0, pitch_width=68.0))]
    fn dangerous_space(
        values: Vec<f64>,
        nx: usize,
        ny: usize,
        cell_area: f64,
        pitch_length: f64,
        pitch_width: f64,
    ) -> PyResult<f64> {
        if values.len() != nx * ny {
            return Err(PyValueError::new_err("values length must equal nx * ny"));
        }
        let g = Grid { nx, ny, cell_area, values };
        Ok(control::dangerous_space(&g, &Pitch::new(pitch_length, pitch_width)))
    }

    #[pyfunction]
    fn offside_line(defenders_x: Vec<f64>, ball_x: f64) -> Option<f64> {
        metrics::offside_line(&defenders_x, ball_x)
    }

    /// `(open, clearance, blockers)` for the ball -> receiver lane.
    #[pyfunction]
    #[pyo3(signature = (ball, receiver, defenders, corridor=1.2))]
    fn passing_lane(
        ball: Pt,
        receiver: Pt,
        defenders: Vec<Pt>,
        corridor: f64,
    ) -> (bool, f64, usize) {
        let v = metrics::passing_lane(
            Vec2::new(ball.0, ball.1),
            Vec2::new(receiver.0, receiver.1),
            &to_vecs(&defenders),
            corridor,
        );
        (v.open, v.clearance, v.blockers)
    }

    /// `(hull_area, depth, width, centroid_x, centroid_y)`.
    #[pyfunction]
    fn team_shape(players: Vec<Pt>) -> Option<(f64, f64, f64, f64, f64)> {
        metrics::team_shape(&to_vecs(&players))
            .map(|s| (s.hull_area, s.depth, s.width, s.centroid.x, s.centroid.y))
    }

    #[pyfunction]
    #[pyo3(signature = (defenders_x, k=3, iterations=50))]
    fn defensive_lines(defenders_x: Vec<f64>, k: usize, iterations: usize) -> Vec<f64> {
        metrics::defensive_lines(&defenders_x, k, iterations)
    }

    #[pyfunction]
    #[pyo3(signature = (player, opponents, scale=5.0))]
    fn marking_pressure(player: Pt, opponents: Vec<Pt>, scale: f64) -> f64 {
        metrics::marking_pressure(Vec2::new(player.0, player.1), &to_vecs(&opponents), scale)
    }

    #[pymodule]
    fn offball_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__doc__", "Rust kernels for the offball soccer analytics platform.")?;
        m.add_function(wrap_pyfunction!(fit_homography, m)?)?;
        m.add_function(wrap_pyfunction!(project, m)?)?;
        m.add_function(wrap_pyfunction!(pitch_control, m)?)?;
        m.add_function(wrap_pyfunction!(space_ownership, m)?)?;
        m.add_function(wrap_pyfunction!(dangerous_space, m)?)?;
        m.add_function(wrap_pyfunction!(offside_line, m)?)?;
        m.add_function(wrap_pyfunction!(passing_lane, m)?)?;
        m.add_function(wrap_pyfunction!(team_shape, m)?)?;
        m.add_function(wrap_pyfunction!(defensive_lines, m)?)?;
        m.add_function(wrap_pyfunction!(marking_pressure, m)?)?;
        Ok(())
    }
}
