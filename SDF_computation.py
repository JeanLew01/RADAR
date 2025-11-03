import jax
import jax.numpy as jnp
from jax import lax

# ----------------------------
# 1) Vectorized FIM (FAST) 2D
# ----------------------------
def fim_eikonal_2d(dist0, frozen, n_iters=80, h=1.0, f=1.0):
    """
    Parallel (Jacobi/FIM) solver for |∇d| = f with Dirichlet 'frozen' nodes.
    Args:
        dist0:   (H, W) float array. 0 on boundary, large elsewhere (e.g., 1e6).
        frozen:  (H, W) bool array. True = Dirichlet (keep value).
        n_iters: number of Jacobi iterations (increase for tighter convergence).
        h, f:    grid spacing and speed (use 1.0 for Euclidean SDF).
    Returns:
        (H, W) float32 distance field.
    """
    d = dist0.astype(jnp.float32)
    frozen = frozen.astype(jnp.bool_)
    fh = jnp.float32(f * h)

    def eikonal_update(d):
        # 4-neighbors via roll (vectorized)
        up    = jnp.roll(d,  1, axis=0)
        down  = jnp.roll(d, -1, axis=0)
        left  = jnp.roll(d,  1, axis=1)
        right = jnp.roll(d, -1, axis=1)

        # domain boundary: use min(self, neighbor) like original code
        up    = up.at[0, :].set(d[0, :])
        down  = down.at[-1, :].set(d[-1, :])
        left  = left.at[:, 0].set(d[:, 0])
        right = right.at[:, -1].set(d[:, -1])

        a = jnp.minimum(left, right)
        b = jnp.minimum(up, down)

        # Discrete Eikonal update (branchless, vectorized)
        cond_two_dim = jnp.abs(a - b) < fh
        radicand = jnp.maximum(0.0, 2.0 * (fh * fh) - (a - b) * (a - b))
        d2 = 0.5 * (a + b + jnp.sqrt(radicand))
        d1 = jnp.minimum(a, b) + fh
        d_new = jnp.where(cond_two_dim, d2, d1)

        # monotone update + keep Dirichlet nodes fixed
        d_upd = jnp.minimum(d, d_new)
        d_upd = jnp.where(frozen, d, d_upd)
        return d_upd

    def body(_, d):
        return eikonal_update(d)

    fim_loop = jax.jit(lambda d: lax.fori_loop(0, n_iters, body, d))
    return fim_loop(d)

# ---------------------------------------------------
# 2) Original fast sweeping (reference, slower in JAX)
# ---------------------------------------------------
def fast_sweep_2d(dist_grid_flat, frozen_flat, height, width, row=None):
    """
    Gauss–Seidel-style fast sweeping solver for |∇d|=1 with Dirichlet (frozen) nodes.
    This is a close JAX port of the C++ code; kept for reference.
    NOTE: Much slower than fim_eikonal_2d in JAX due to many small scatters.

    Args:
        dist_grid_flat: (H*row,) 1D array
        frozen_flat:    (H*row,) bool 1D array
        height, width:  ints
        row:            row stride (defaults to width)
    Returns:
        updated 1D array after 4 directional sweeps
    """
    if row is None:
        row = width

    dirX = jnp.array([
        [0,         width - 1,  1],
        [width - 1, 0,         -1],
        [width - 1, 0,         -1],
        [0,         width - 1,  1],
    ])
    dirY = jnp.array([
        [0,          height - 1,  1],
        [0,          height - 1,  1],
        [height - 1, 0,          -1],
        [height - 1, 0,          -1],
    ])

    h = jnp.float32(1.0)
    f = jnp.float32(1.0)

    def update_one_cell(dist_grid, iy, ix):
        grid_pos = iy * row + ix

        def frozen_branch(_):
            return dist_grid

        def active_branch(_):
            # y-direction candidate (aa1)
            def y_middle_case(_):
                up_val   = dist_grid[(iy - 1) * row + ix]
                down_val = dist_grid[(iy + 1) * row + ix]
                return jnp.minimum(up_val, down_val)

            def y_top_case(_):
                cur_val  = dist_grid[grid_pos]
                down_val = dist_grid[(iy + 1) * row + ix]
                return jnp.minimum(cur_val, down_val)

            def y_bottom_case(_):
                up_val  = dist_grid[(iy - 1) * row + ix]
                cur_val = dist_grid[grid_pos]
                return jnp.minimum(up_val, cur_val)

            aa1 = lax.cond(
                jnp.logical_or(iy == 0, iy == (height - 1)),
                lambda _: lax.cond(iy == 0, y_top_case, y_bottom_case, operand=None),
                y_middle_case,
                operand=None,
            )

            # x-direction candidate (aa0)
            def x_middle_case(_):
                left_val  = dist_grid[iy * row + (ix - 1)]
                right_val = dist_grid[iy * row + (ix + 1)]
                return jnp.minimum(left_val, right_val)

            def x_left_case(_):
                cur_val   = dist_grid[grid_pos]
                right_val = dist_grid[iy * row + (ix + 1)]
                return jnp.minimum(cur_val, right_val)

            def x_right_case(_):
                left_val = dist_grid[iy * row + (ix - 1)]
                cur_val  = dist_grid[grid_pos]
                return jnp.minimum(left_val, cur_val)

            aa0 = lax.cond(
                jnp.logical_or(ix == 0, ix == (width - 1)),
                lambda _: lax.cond(ix == 0, x_left_case, x_right_case, operand=None),
                x_middle_case,
                operand=None,
            )

            a = aa0
            b = aa1

            cond_two_dim = jnp.abs(a - b) < f * h
            rad = jnp.maximum(0.0, 2.0 * (f * h) * (f * h) - (a - b) * (a - b))
            d_new_2d = 0.5 * (a + b + jnp.sqrt(rad))
            d_new_1d = jnp.minimum(a, b) + f * h
            d_new = jnp.where(cond_two_dim, d_new_2d, d_new_1d)

            new_val = jnp.minimum(dist_grid[grid_pos], d_new)
            return dist_grid.at[grid_pos].set(new_val)

        return lax.cond(frozen_flat[grid_pos], frozen_branch, active_branch, operand=None)

    def do_one_sweep(dist_grid, s):
        x0, x1, x_step = dirX[s]
        y0, y1, y_step = dirY[s]

        ny = jnp.abs(y1 - y0) + 1
        nx = jnp.abs(x1 - x0) + 1

        def y_body(iy_idx, dist_grid):
            iy = y0 + iy_idx * y_step

            def x_body(ix_idx, dist_grid):
                ix = x0 + ix_idx * x_step
                return update_one_cell(dist_grid, iy, ix)

            dist_grid = lax.fori_loop(0, nx, x_body, dist_grid)
            return dist_grid

        dist_grid = lax.fori_loop(0, ny, y_body, dist_grid)
        return dist_grid

    def sweeps_body(s, dist_grid):
        return do_one_sweep(dist_grid, s)

    dist_grid_flat = lax.fori_loop(0, 4, sweeps_body, dist_grid_flat)
    return dist_grid_flat

# --------------------------
# Small helpers (used below)
# --------------------------
def init_square_boundary(H, W, ax, bx, ay, by):
    """Returns (dist0, frozen) for a hollow square boundary."""
    on_square = jnp.zeros((H, W), dtype=bool)
    on_square = (on_square
                 .at[ay, ax:bx+1].set(True)
                 .at[by, ax:bx+1].set(True)
                 .at[ay:by+1, ax].set(True)
                 .at[ay:by+1, bx].set(True))
    dist0 = jnp.where(on_square, 0.0, jnp.float32(1e6))
    return dist0, on_square

def to_signed(dist, inside_mask):
    """Signed from unsigned distance with an inside mask (True=inside => negative)."""
    return jnp.where(inside_mask, -dist, dist)

def _rot(x, y, cx, cy, theta):
    c = jnp.cos(theta); s = jnp.sin(theta)
    xr =  (x - cx) * c + (y - cy) * s
    yr = -(x - cx) * s + (y - cy) * c
    return xr, yr

def _sdf_box_axial(x, y, hx, hy):
    dx = jnp.abs(x) - hx
    dy = jnp.abs(y) - hy
    dxp = jnp.maximum(dx, 0.0)
    dyp = jnp.maximum(dy, 0.0)
    outside = jnp.sqrt(dxp * dxp + dyp * dyp)
    inside  = jnp.minimum(jnp.maximum(dx, dy), 0.0)
    return outside + inside

def build_composite_boundary_and_inside(H, W):
    """
    Composite = (rotated-rectangle ∪ circle ∪ annulus) \ small-hole
    Returns:
      dist0  : (H,W) float, 0 on boundary band
      frozen : (H,W) bool, Dirichlet mask
      inside : (H,W) bool, interior for signed distance
    """
    ys = jnp.linspace(-1.0, 1.0, H)
    xs = jnp.linspace(-1.0, 1.0, W)
    X, Y = jnp.meshgrid(xs, ys)

    # rotated rectangle
    rr_cx, rr_cy = 0.35, -0.15
    rr_hx, rr_hy = 0.28, 0.12
    rr_theta = jnp.deg2rad(30.0)
    Xr, Yr = _rot(X, Y, rr_cx, rr_cy, rr_theta)
    sdf_rect = _sdf_box_axial(Xr, Yr, rr_hx, rr_hy)

    # solid circle
    c1_cx, c1_cy, c1_R = -0.3, -0.25, 0.28
    sdf_circ1 = jnp.sqrt((X - c1_cx)**2 + (Y - c1_cy)**2) - c1_R

    # annulus
    ring_cx, ring_cy = -0.1, 0.45
    r_outer, r_inner = 0.35, 0.20
    sdf_ring_outer = jnp.sqrt((X - ring_cx)**2 + (Y - ring_cy)**2) - r_outer
    sdf_ring_inner = jnp.sqrt((X - ring_cx)**2 + (Y - ring_cy)**2) - r_inner

    # small hole
    hole_cx, hole_cy, hole_R = 0.15, 0.15, 0.12
    sdf_hole = jnp.sqrt((X - hole_cx)**2 + (Y - hole_cy)**2) - hole_R

    # inside masks
    inside_rect  = (sdf_rect <= 0.0)
    inside_c1    = (sdf_circ1 <= 0.0)
    inside_ring  = (sdf_ring_outer <= 0.0) & (sdf_ring_inner >= 0.0)
    inside_hole  = (sdf_hole <= 0.0)

    inside = (inside_rect | inside_c1 | inside_ring) & (~inside_hole)

    # boundary band ~ 1~2 像素
    dx = 2.0 / (W - 1); dy = 2.0 / (H - 1)
    eps = 1.5 * jnp.minimum(dx, dy)
    b_rect  = jnp.abs(sdf_rect)      <= eps
    b_c1    = jnp.abs(sdf_circ1)     <= eps
    b_ring  = (jnp.abs(sdf_ring_outer) <= eps) | (jnp.abs(sdf_ring_inner) <= eps)
    b_hole  = jnp.abs(sdf_hole)      <= eps

    boundary = b_rect | b_c1 | b_ring | b_hole
    dist0  = jnp.where(boundary, 0.0, jnp.float32(1e6))
    frozen = boundary
    return dist0, frozen, inside


