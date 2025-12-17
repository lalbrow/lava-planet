import sys
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import imageio
import matplotlib.colors as mcolors


def load_variables(ds, var_names, t_index):
    """Load the given variable names at the specified time index."""
    loaded = {}
    for name in var_names:
        if name in ds.variables:
            loaded[name] = ds.variables[name][t_index]
        else:
            print(f"Warning: variable '{name}' missing in dataset.")
            loaded[name] = None
    return loaded


def interpolate_to_time(ds, var, t_target, time_array):
    """
    Get var(t_target) using linear interpolation between nearest time slices.
    Assumes var is 4D: (time, x1, x2, x3)
    """
    # nearest lower index
    i = np.searchsorted(time_array, t_target) - 1
    i = max(0, min(i, len(time_array) - 2))

    t0, t1 = time_array[i], time_array[i + 1]
    w = (t_target - t0) / (t1 - t0)

    v0 = ds.variables[var][i]
    v1 = ds.variables[var][i + 1]

    return (1 - w) * v0 + w * v1


def compare_sims(s1, s2):
    """Compare netCDF file outputs of two simulations."""

    sim1nc = f"{s1}/bin/lava_planet-test-main.nc"
    sim2nc = f"{s2}/bin/lava_planet-test-main.nc"

    ds1 = Dataset(sim1nc, 'r')
    ds2 = Dataset(sim2nc, 'r')

    # Shared time array (assumed consistent)
    time = ds1.variables['time'][:]

    variables = ['temp', 'rho', 'press', 'vel1', 'vel2', 'vel3', 'SiO', 'SiO(s)']
    n_times = np.min([300, len(time)])
    t_eval_idx = np.linspace(0, len(time)-1, n_times, dtype=int)

    # For GIFs
    gif_kwargs = dict(duration=0.08, loop=0)

    altitude = ds1.variables["x1"][:] / 1e3      
    theta = ds1.variables["x2"][:]  

    for var in variables:
        if var not in ds1.variables or var not in ds2.variables:
            print(f"Skipping {var}, not in both simulations.")
            continue

        print(f"Processing {var}...")

        frames = []

        for t in t_eval_idx:
            # interpolate both sims to time t
            v1 = load_variables(ds1, [var], t)[var]
            v2 = load_variables(ds2, [var], t)[var]

            # compute residual
            res = v1 - v2

            # squeeze 3D → 2D if needed, assuming nx3=1
            if res.ndim == 3:
                res2d = res[:, :, 0]
            else:
                res2d = res

            # --- Plot frame with correct axes ---
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(np.fliplr(res2d),
                        origin='lower',
                        extent=[theta.min(), theta.max(),
                                altitude.min(), altitude.max()],
                        cmap='coolwarm',
                        norm=mcolors.CenteredNorm(),
                        aspect='auto')

            ax.set_xlabel("Latitude (°)")
            ax.set_ylabel("Altitude (km)")
            ax.set_title(f"{var} residual at t={time[t]:.3f}")

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f"{var} residual")


            # Save frame to a numpy array
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)

        # Write GIF
        gif_path = f"sim_compare/{s1}-{s2}_{var}_residual.gif"
        imageio.mimsave(gif_path, frames, **gif_kwargs)
        print(f"Saved: {gif_path}")

    ds1.close()
    ds2.close()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_sim.py sim1 sim2")
        sys.exit(1)

    sim1 = sys.argv[1]
    sim2 = sys.argv[2]
    compare_sims(sim1, sim2)
