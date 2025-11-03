import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

H, W = 50, 50
row = W

# 1) init: big distance everywhere
dist = jnp.full((H * row,), 1e6)
frozen = jnp.zeros((H * row,), dtype=bool)

# 2) make a square [20:30) x [20:30) as boundary
for iy in range(20, 30):
    for ix in range(20, 30):
        pos = iy * row + ix
        dist = dist.at[pos].set(0.0)
        frozen = frozen.at[pos].set(True)

# 3) run several sweeps
dist = fast_sweep_2d(dist, frozen, H, W, row)
dist = fast_sweep_2d(dist, frozen, H, W, row)
dist = fast_sweep_2d(dist, frozen, H, W, row)

# 4) visualize
dist_img = dist.reshape(H, W)
plt.imshow(dist_img, origin="lower")
plt.colorbar()
plt.title("distance to center square")
plt.show()