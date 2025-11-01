import jax
import jax.numpy as jnp
from jax import lax

def fast_sweep_2d(dist_grid, frozen, height, width, row=None):
    """
    fast_sweeping for 2D SDF computation

    Args:
        dist_grid (_type_): _description_
        frozen (_type_): _description_
        height (_type_): _description_
        width (_type_): _description_
        row (_type_, optional): _description_. Defaults to None.
    
    Returns:
        dist_grid: updated distance grid after one sweep
    """

    if row is None:
        row = width
    nsweeps = 4
    
    # sweep directions
    dirX = jnp.array([
        [0, width-1, 1],   # left to right
        [width-1, 0, -1],  # right to left
        [width-1, 0, -1],
        [0, width-1, 1],
    ])

    dirY = jnp.array([
        [0, height-1, 1],   # top to bottom
        [0, height-1, 1],   # top to bottom
        [height-1, 0, -1],  # bottom to top
        [height-1, 0, -1],  # bottom to top
    ])

    h = 1.0  # grid spacing
    f = 1.0  # speed function
    
    def update_one_cell(dist_grid, iy, ix):
        grid_pos = iy * row + ix
        
        # skip frozen cells
        def frozen_case():
            return dist_grid
        
        def active_branch():
            # --- y-direction min neighbor (aa[1]) ---
            # care for top/bottom boundariess
            def y_middle_case(_):
                up_val = dist_grid[(iy - 1) * row + ix]
                down_val = dist_grid[(iy + 1) * row + ix]
                return jnp.minimum(up_val, down_val)
            def y_top_case(_):
                cur_val = dist_grid[grid_pos]
                down_val = dist_grid[(iy + 1) * row + ix]
                return jnp.minimum(cur_val, down_val)
            def y_bottom_case(_):
                up_val = dist_grid[(iy - 1) * row + ix]
                cur_val = dist_grid[grid_pos]
                return jnp.minimum(up_val, cur_val)
            aa1 = lax.cond(
                jnp.logical_and(iy == 0, iy == height - 1),
                lambda _: lax.cond(
                    iy == 0,
                    y_top_case,
                    y_bottom_case,
                    operand=None,
                ),
                y_middle_case,
                operand=None,
            )
            
            # --- x-direction min neighbor (aa[0]) ---
            def x_middle_case(_):
                left_val = dist_grid[iy * row + (ix - 1)]
                right_val = dist_grid[iy * row + (ix + 1)]
                return jnp.minimum(left_val, right_val)
            def x_left_case(_):
                cur_val = dist_grid[grid_pos]
                right_val = dist_grid[iy * row + (ix + 1)]
                return jnp.minimum(cur_val, right_val)
            def x_right_case(_):
                left_val = dist_grid[iy * row + (ix - 1)]
                cur_val = dist_grid[grid_pos]
                return jnp.minimum(left_val, cur_val)
            aa0 = lax.cond(
                jnp.logical_and(ix == 0, ix == width - 1),
                lambda _: lax.cond(
                    ix == 0,
                    x_left_case,
                    x_right_case,
                    operand=None,
                ),
                x_middle_case,
                operand=None,
            )
            
            a = aa0
            b = aa1
            
            cond_two_dim = jnp.abs(a - b) < f * h
            d_new_2d =0.5*(a+b+jnp.sqrt(2*f*f*h*h-(a-b)*(a-b)))
            d_new_1d = jnp.minimum(a, b) + f * h
            d_new = jnp.where(cond_two_dim, d_new_2d, d_new_1d)
            
            new_val = jnp.minimum(dist_grid[grid_pos], d_new)
            return dist_grid.at[grid_pos].set(new_val)
        
        return lax.cond(frozen[grid_pos], frozen_branch, active_branch)
    
    # one sweep over all iy, ix for a given s
    def do_one_sweep(dist, s):
        """run one directional sweep"""
        x0, y1, x_step = dirX[s]
        y0, y1, y_step = dirY[s]
        
        # because we want JIT-compile, we use lax.fori_loop instead of python for loop
        ny = jnp.abs(y1-y0) + 1
        nx = jnp.abs(x1-x0) + 1
        
        def y_body(iy_idx, dist_grid):
            iy = y0 + iy_idx * y_step
            
            def x_body(ix_idx, dist_grid):
                ix = x0 + ix_idx * x_step
                dist_grid = update_one_cell(dist_grid, iy, ix)
                return dist_grid
            
            dist_grid = lax.fori_loop(0, nx, x_body, dist_grid)
            return dist_grid

        dist_grid = lax.fori_loop(0, ny, y_body, dist_grid)
        return dist_grid
    # multiple sweeps
    def sweep_body(s, dist_grid):
        return do_one_sweep(dist_grid, s)
    
    dist_grid = lax.fori_loop(0, nsweeps, sweep_body, dist_grid)
    return dist_grid