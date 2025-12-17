import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl
import sys
from matplotlib import ticker


mpl.rcParams.update(mpl.rcParamsDefault)
cm_to_inch = 0.393701
width=33.87 * cm_to_inch
height=19.05 * cm_to_inch
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.default'] = 'default'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
R_planet = 6500e3
ALT_sub = 400e3
R_eff = R_planet + ALT_sub * 1000.0  # m


# ---------- Helper Functions ----------
def plot_streamfunction_fields(
    altitude, theta, vars, outdir, prefix, t,
    stream_type='velocity'
):
    """
    Plot velocity or mass streamfunction contours overlaid on scalar fields.

    Args:
        altitude: altitude array (km)
        theta: colatitude array (radians)
        vars: dictionary of variables including vel1, vel2, rho
        outdir: output directory
        prefix: filename prefix
        t: current time
        stream_type: 'velocity' or 'mass'
    """
    if 'vel1' not in vars or 'vel2' not in vars:
        print("Warning: vel1 or vel2 not found, skipping streamfunction plots")
        return

    if stream_type == 'mass' and 'rho' not in vars:
        print("Warning: rho not found, cannot compute mass streamfunction")
        return

    # === Average over phi ===
    vel1_avg = average_over_phi(vars['vel1'])  # w (m/s)
    vel2_avg = average_over_phi(vars['vel2'])  # v_theta (m/s)
    z_m = altitude * 1000.0

    if stream_type == 'mass':
        rho_avg = average_over_phi(vars['rho'])
        vel2_avg = rho_avg * vel2_avg  
        column_mass = np.trapz(rho_avg, z_m, axis=0)
        vel2_avg /= column_mass[np.newaxis, :]


    # === Grid ===
    ALT, THETA = np.meshgrid(altitude, theta, indexing='ij')

    # === Scalar fields ===
    scalar_fields = {}
    for name in ['temp', 'rho', 'press', 'vel1', 'vel2', 'SiO', 'SiO(s)']:
        if name in vars and vars[name] is not None:
            field = average_over_phi(vars[name])
            if name in ['rho', 'press']:
                scalar_fields[f'log10 {name}'] = np.log10(field)
            else:
                scalar_fields[name] = field

    vel_mag = np.sqrt(vel1_avg**2 + vel2_avg**2)
    scalar_fields['vel_magnitude'] = vel_mag

    # === Streamfunction ===
    dz = np.gradient(z_m)
    psi = np.zeros_like(vel2_avg)

    for k in range(1, psi.shape[0]):
        psi[k, :] = psi[k-1, :] - vel2_avg[k, :] * dz[k]

    # Remove arbitrary offset
    psi -= np.nanmean(psi)

    # === NORMALIZATION FOR COMPARABILITY ===
    # Normalize by max absolute value so velocity and mass
    # streamfunctions can be compared visually
    psi_norm = psi / np.nanmax(np.abs(psi))

    # === Figure layout ===
    n = len(scalar_fields)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    axes = axes.flat if n > 1 else [axes]

    # Fixed contour levels for comparability
    levels = np.linspace(-1, 1, 21)

    for i, (name, field) in enumerate(scalar_fields.items()):
        ax = axes[i]

        cmap = 'plasma'
        norm = None
        if name in ['vel1', 'vel2']:
            cmap = 'RdBu_r'
            norm = colors.CenteredNorm(vcenter=0)

        im = ax.contourf(
            THETA, ALT, field,
            50, cmap=cmap, norm=norm
        )

        cs = ax.contour(
            THETA, ALT, psi_norm,
            levels=levels,
            colors='white',
            linewidths=1.0
        )

        ax.clabel(cs, levels[::4], fontsize=7, inline=True)

        cbar = plt.colorbar(im, ax=ax, label=name, fraction=0.046, pad=0.04)
        cbar.locator = mpl.ticker.MaxNLocator(nbins=4)
        cbar.formatter = mpl.ticker.FuncFormatter(lambda x, _: f"{x:.3g}")
        cbar.update_ticks()

        label = "Velocity" if stream_type == 'velocity' else "Mass Flux"
        ax.set_title(f"{name} + {label} streamlines")
        ax.set_xlabel("Colatitude (rad)")
        ax.set_ylabel("Altitude (km)")

    for ax in axes[len(scalar_fields):]:
        ax.axis("off")

    plt.suptitle(
        f"{label} Streamfunction (normalized), t={t:.2f}s",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(
        f"{outdir}/{prefix}_{stream_type}_streamfunction_t{t:.2f}.pdf",
        dpi=200
    )
    plt.close()
    print(f"✅ Saved {stream_type} streamfunction plot")


def plot_vector_fields(
    altitude, theta, vars, outdir, prefix, t,
    stride=4, field_type='velocity'
):
    """
    Plot velocity or mass-flux vector fields overlaid on scalar fields.

    Args:
        altitude: altitude array (km)
        theta: latitude array (radians)
        vars: dictionary of variables including vel1, vel2, rho
        outdir: output directory
        prefix: filename prefix
        t: current time
        stride: quiver stride for downsampling
        field_type: 'velocity' or 'mass_flux'
    """
    if 'vel1' not in vars or 'vel2' not in vars:
        print("Warning: vel1 or vel2 not found, skipping vector field plots")
        return

    if field_type == 'mass_flux' and 'rho' not in vars:
        print("Warning: rho not found, cannot plot mass flux")
        return

    # === Average over phi ===
    vel1_avg = average_over_phi(vars['vel1'])  # w (m/s)
    vel2_avg = average_over_phi(vars['vel2'])  # v_theta (m/s)

    if field_type == 'mass_flux':
        rho_avg = average_over_phi(vars['rho'])  # kg/m^3
        vel1_avg = rho_avg * vel1_avg             # kg / (m^2 s)
        vel2_avg = rho_avg * vel2_avg

    # === Grid ===
    ALT, THETA = np.meshgrid(altitude, theta, indexing='ij')

    # === Scalar fields ===
    scalar_fields = {}
    for name in ['temp', 'rho', 'press', 'vel1', 'vel2', 'SiO', 'SiO(s)']:
        if name in vars and vars[name] is not None:
            field = average_over_phi(vars[name])
            if name in ['rho', 'press']:
                scalar_fields[f'log10 {name}'] = np.log10(field)
            else:
                scalar_fields[name] = field

    # Reference magnitude (velocity or mass flux)
    mag = np.sqrt(vel1_avg**2 + vel2_avg**2)
    scalar_fields[f'{field_type}_magnitude'] = mag

    # === Figure layout ===
    n = len(scalar_fields)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    axes = axes.flat if n > 1 else [axes]

    # === Downsample ===
    ALT_sub = ALT[::stride, ::stride]
    THETA_sub = THETA[::stride, ::stride]
    vel1_sub = vel1_avg[::stride, ::stride]
    vel2_sub = vel2_avg[::stride, ::stride]

    # === Geometry ===
    R_eff = R_planet + ALT_sub * 1000.0  # m

    for i, (name, field) in enumerate(scalar_fields.items()):
        ax = axes[i]

        # --- Background scalar ---
        cmap = 'plasma'
        norm = None
        if name in ['vel1', 'vel2']:
            cmap = 'RdBu_r'
            norm = colors.CenteredNorm(vcenter=0)

        im = ax.contourf(
            THETA, ALT, field,
            50, cmap=cmap, norm=norm
        )

        # --- Vector components in plot coordinates ---
        U_phys = vel2_sub / R_eff          # dθ/dt or mass-flux analogue
        V_phys = vel1_sub / 1000.0         # dz/dt or mass-flux analogue

        # --- Visual scaling only ---

        if field_type == 'velocity':
                    vec_vis = 1e2
        elif field_type == 'mass_flux':
                    vec_vis = 1e6

        U = -U_phys * vec_vis              # sign flip preserved
        V =  V_phys * vec_vis 

        ax.quiver(
            THETA_sub, ALT_sub,
            U, V,
            angles='xy',
            scale_units='xy',
            scale=1,
            color='white',
            alpha=0.8,
            width=0.003
        )

        # --- Colorbar ---
        cbar = plt.colorbar(im, ax=ax, label=name, fraction=0.046, pad=0.04)
        cbar.locator = mpl.ticker.MaxNLocator(nbins=4)
        cbar.formatter = mpl.ticker.FuncFormatter(lambda x, _: f"{x:.3g}")
        cbar.update_ticks()

        label = "Velocity" if field_type == 'velocity' else "Mass Flux"
        ax.set_title(f"{name} + {label}")
        ax.set_xlabel("Colatitude (rad)")
        ax.set_ylabel("Altitude (km)")

    for ax in axes[len(scalar_fields):]:
        ax.axis("off")

    plt.suptitle(
        f"{label} Vectors + Scalar Fields (t={t:.2f}s)",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(
        f"{outdir}/{prefix}_{field_type}_fields_t{t:.2f}.pdf",
        dpi=200
    )
    plt.close()
    print(f"✅ Saved {field_type} vector field plot")


def create_vector_field_gifs(
    ds, altitude, theta, outdir, prefix,
    scalar_field='temp', max_frames=100, 
    keep_first_seconds=5.0, stride=5, fixed_cbar=True
):
    """
    Create evolution GIF with velocity vectors overlaid on a scalar field.
    
    Args:
        ds: NetCDF dataset
        altitude: altitude array (km)
        theta: latitude array (radians)
        outdir: output directory
        prefix: filename prefix
        scalar_field: name of scalar field to plot ('temp', 'rho', 'press', or 'vel_mag')
        max_frames: maximum number of frames
        keep_first_seconds: keep all frames in first N seconds
        stride: quiver stride for downsampling
        fixed_cbar: if True, use fixed colorbar limits; if False, adjust per frame
    """
    # Load data
    time = np.array(ds.variables['time'][:])
    vel1 = ds.variables['vel1'][:]  # radial
    vel2 = ds.variables['vel2'][:]  # latitudinal
    
    # Load scalar field
    if scalar_field == 'vel_mag':
        scalar_data = np.sqrt(vel1**2 + vel2**2)
        log_scale = False
        cmap = 'plasma' 
        label = 'Velocity Magnitude (m/s)'
    else:
        scalar_data = ds.variables[scalar_field][:]
        log_scale = scalar_field in ['rho', 'press']
        cmap = 'plasma'
        label = f'log10 {scalar_field}' if log_scale else scalar_field
        if log_scale:
            scalar_data = np.log10(scalar_data)
    
    # Average over phi
    scalar_avg = np.mean(scalar_data, axis=-1)
    vel1_avg = np.mean(vel1, axis=-1)
    vel2_avg = np.mean(vel2, axis=-1)
    vel_mag_avg = np.sqrt(vel1_avg**2 + vel2_avg**2)
    
    # Frame selection
    early_mask = time <= keep_first_seconds
    early_indices = np.where(early_mask)[0]
    later_indices = np.where(~early_mask)[0]
    
    total_frames = len(early_indices) + len(later_indices)
    if total_frames > max_frames:
        n_late_keep = max_frames - len(early_indices)
        later_indices = np.linspace(later_indices[0], later_indices[-1], 
                                   n_late_keep, dtype=int)
    
    frames_to_use = np.concatenate([early_indices, later_indices])
    print(f"Creating vector field GIF with {len(frames_to_use)} frames")
    
    # Color normalization for scalar field
    if fixed_cbar:
        vmin, vmax = np.nanmin(scalar_avg), np.nanmax(scalar_avg)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize()  # Will be updated per frame
    
    # Create meshgrid
    ALT, THETA = np.meshgrid(altitude, theta, indexing='ij')
    ALT_sub = ALT[::stride, ::stride]
    THETA_sub = THETA[::stride, ::stride]
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(0.9*width, 0.8*height))
    
    # Initial frame
    first_frame = frames_to_use[0]
    scalar_first = np.fliplr(scalar_avg[first_frame])
    vel1_first = np.fliplr(vel1_avg[first_frame])[::stride, ::stride]
    vel2_first = np.fliplr(vel2_avg[first_frame])[::stride, ::stride]
    vel_mag_first = np.nanmax(vel_mag_avg[first_frame])
    
    # Plot scalar background
    cont = ax.imshow(
        scalar_first,
        cmap=cmap,
        norm=norm,
        extent=[theta.min(), theta.max(), altitude.min(), altitude.max()],
        aspect='auto',
        origin='lower'
    )

    U_phys = vel2_first / R_eff          # rad/s
    V_phys = vel1_first / 1000.0         # km/s

    # VISUAL scaling only
    theta_vis = 3e4
    z_vis     = 3e2

    U = U_phys * theta_vis
    V = V_phys * z_vis

    angle_phys = np.arctan2(V_phys, U_phys)
    angle_plot = np.arctan2(V, U)

    print(np.nanmax(np.abs(angle_phys - angle_plot)))



    quiv = ax.quiver(
        THETA_sub, ALT_sub,
        -U, V,
        angles='xy',
        scale_units='xy',
        scale=1,
        color='white',
        alpha=0.7,
        width=0.003
    )


    
    cbar = fig.colorbar(cont, ax=ax, label=label)
    ax.set_xlabel("Colatitude (°)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"{label} + velocity  t={time[first_frame]:.2f}s")
    
    def update(i):
        frame = frames_to_use[i]
        scalar_data_frame = np.fliplr(scalar_avg[frame])
        vel1_frame = np.fliplr(vel1_avg[frame])[::stride, ::stride]
        vel2_frame = np.fliplr(vel2_avg[frame])[::stride, ::stride]
        
        # Update scalar field
        cont.set_data(scalar_data_frame)
        
        # Update colorbar limits if not fixed
        if not fixed_cbar:
            frame_vmin = np.nanmin(scalar_data_frame)
            frame_vmax = np.nanmax(scalar_data_frame)
            norm.vmin = frame_vmin
            norm.vmax = frame_vmax
            cont.set_norm(norm)
            cbar.update_normal(cont)
        
        U_phys = vel2_frame / R_eff          # rad/s
        V_phys = vel1_frame / 1000.0         # km/s

        U = -U_phys * theta_vis              # sign + visual scaling
        V =  V_phys * z_vis

        quiv.set_UVC(U, V)

        
        # Update reference arrow
        vel_mag_frame = np.nanmax(vel_mag_avg[frame])
        ax.set_title(f"{label} + velocity  t={time[frame]:.2f}s")
        
        return [cont, quiv]
    
    # Build animation
    anim = FuncAnimation(fig, update, frames=len(frames_to_use), 
                        interval=200, blit=True)
    
    # Save
    outpath = f"{outdir}/{prefix}_vector_{scalar_field}_evolution.gif"
    writer = PillowWriter(fps=5)
    print(f"Saving vector field animation → {outpath}")
    anim.save(outpath, writer=writer, dpi=120)
    
    plt.close(fig)
    print(f"✅ Saved vector field animation to {outpath}")



def average_over_phi(field):
    """Average 3D field over φ."""
    return np.mean(field, axis=2)

def make_mesh(altitude, theta):
    return np.meshgrid(theta, altitude)

def load_variables(ds, var_names, t):
    """Safely load specified variable names at a given time index."""
    loaded = {}
    for name in var_names:
        if name in ds.variables:
            loaded[name] = ds.variables[name][t]
        else:
            print(f"Warning: variable '{name}' not found in dataset.")
            loaded[name] = None
    return loaded


# ---------- Main Driver ----------
def summarize_simulation_spherical_with_vectors(ncfile, params=None, t_index=1, 
                                                make_gif=False, vector_gif_fields=None):
    """
    Extended version that includes vector field plots.
    
    Args:
        vector_gif_fields: list of scalar fields to overlay vectors on 
                          (e.g., ['temp', 'rho', 'vel_mag'])
    """
    ds = Dataset(ncfile, 'r')
    time = ds.variables['time'][:]
    r = ds.variables['x1'][:] / 1e3  # km
    theta = ds.variables['x2'][:]
    phi = ds.variables['x3'][:]

    t = t_index if t_index >= 0 else len(time) - 1
    planet_radius = r.min()
    altitude = r - planet_radius

    if params is None:
        params = ["temp", "rho", "press", "vel1", "vel2", "vel3"]
    
    if vector_gif_fields is None:
        vector_gif_fields = ['temp', 'vel_mag']

    outdir = os.path.dirname(os.path.abspath(ncfile))
    prefix = os.path.splitext(os.path.basename(ncfile))[0]
    vars = load_variables(ds, params, t)

    # Original plots
    print(f"📊 Plotting snapshot at t={time[t]}")
    plot_averaged_fields(altitude, theta, vars, outdir, prefix, time[t])
    plot_statistical_summary(altitude, vars, outdir, prefix, time[t])
    
    # Vector field snapshot
    print(f"🎯 Plotting vector fields")
    plot_vector_fields(altitude, theta, vars, outdir, prefix, time[t], stride=4, field_type='velocity')
    plot_vector_fields(altitude, theta, vars, outdir, prefix, time[t], stride=4, field_type='mass_flux')
    plot_streamfunction_fields(altitude, theta, vars, outdir, prefix, time[t], stream_type='velocity')
    plot_streamfunction_fields(altitude, theta, vars, outdir, prefix, time[t], stream_type='mass')
    
    if len(phi) > 1:
        plot_equatorial_slices(altitude, phi, theta, vars, outdir, prefix, time[t])

    # Create GIFs
    if make_gif:
        for name in params:
            if name in ds.variables:
                print(f"🎞 Creating GIF for {name}...")
                create_evolution_gifs(ds, altitude, theta, outdir, name, prefix, 
                                    cmap='plasma' if 'vel' not in name else 'RdBu_r', fixed_cbar=False)
        
        # Vector field GIFs
        for field in vector_gif_fields:
            print(f"🎯 Creating vector field GIF for {field}...")
            create_vector_field_gifs(ds, altitude, theta, outdir, prefix,
                                   scalar_field=field, stride=5)

    ds.close()
    print(f"✅ All outputs saved to {outdir}")


# ---------- Plotting Functions ----------

def plot_averaged_fields(altitude, theta, vars, outdir, prefix, t):
    """Plot φ-averaged fields in 2x3 grid."""
    ALT, THETA = np.meshgrid(altitude, theta, indexing='ij')
    n = len(vars)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    axes = axes.flat if n > 1 else [axes]

    for i, (name, field) in enumerate(vars.items()):
        if field is None:
            continue
        field_avg = average_over_phi(field)
        if name in ["rho", "press"]:
            field_avg = np.log10(field_avg)
            name = 'log10 ' + name
        cmap = "RdBu_r" if "vel" in name else "plasma"
        vmax = np.abs(field_avg).max() if "vel" in name else None
        im = axes[i].contourf(THETA, ALT, field_avg, 50, cmap=cmap,
                              vmin=-vmax if vmax else None, vmax=vmax)
        cbar = plt.colorbar(im, ax=axes[i], label=name, fraction=0.046, pad=0.04)
        cbar.locator = ticker.MaxNLocator(nbins=4)
        cbar.formatter = ticker.FuncFormatter(lambda x, _: f"{x:.3g}")
        cbar.update_ticks()
        axes[i].set_title(f"{name}")
        axes[i].set_xlabel("Colatitude (°)")
        axes[i].set_ylabel("Altitude (km)")
    for ax in axes[len(vars):]:
        ax.axis("off")

    plt.suptitle(f"φ-Averaged Fields (t={t:.2f})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{prefix}_avg_fields_t{t:.2f}.pdf", dpi=200)
    plt.close()


def plot_equatorial_slices(altitude, phi, theta, vars, outdir, prefix, t):
    """Plot equatorial slices dynamically for any fields."""
    eq_index = np.argmin(np.abs(theta - np.pi / 2))
    PHI, ALT = np.meshgrid(phi, altitude, indexing='ij')
    n = len(vars)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    axes = axes.flat if n > 1 else [axes]

    for i, (name, field) in enumerate(vars.items()):
        if field is None:
            continue
        eq_field = field[:, eq_index, :]
        if name in ["rho", 'press']:
            eq_field = np.log10(eq_field)
            axes[i].set_xlabel('log10 ' + name)
        else:
            axes[i].set_xlabel(name)
        cmap = "RdBu_r" if "vel" in name else "plasma"
        vmax = np.abs(eq_field).max() if "vel" in name else None
        im = axes[i].contourf(
            PHI, ALT, eq_field.T, 50, cmap=cmap,
            vmin=-vmax if vmax else None, vmax=vmax
        )
        plt.colorbar(im, ax=axes[i], label=name)
        axes[i].set_title(f"{name} (Equator)")
        axes[i].set_xlabel("Longitude (°)")
        axes[i].set_ylabel("Altitude (km)")

    for ax in axes[len(vars):]:
        ax.axis("off")

    plt.suptitle(f"Equatorial Slices (t={t})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{prefix}_equator_t{t}.pdf", dpi=200)
    plt.close()


def plot_statistical_summary(altitude, vars, outdir, prefix, t):
    """Vertical mean profiles of all fields."""
    n = len(vars)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    axes = axes.flat if n > 1 else [axes]

    for i, (name, field) in enumerate(vars.items()):
        if field is None:
            continue
        avg = average_over_phi(field)
        if name in ["rho", 'press']:
            avg = np.log10(avg)
            axes[i].set_xlabel('log10 ' + name)
        else:
            axes[i].set_xlabel(name)
        mean_profile = np.mean(avg, axis=1)
        axes[i].plot(mean_profile, altitude, label=name)
        
        axes[i].set_ylabel("Altitude (km)") 
        axes[i].set_ylim(0)

    for ax in axes[len(vars):]:
        ax.axis("off")

    plt.suptitle(f"Vertical Profiles (t={t:.2f})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{prefix}_profiles_t{t:.2f}.pdf", dpi=200)
    plt.close()


# ---------- GIF Creation ----------
def create_evolution_gifs(
    ds, altitude, theta, outdir, prefix, pf,
    max_frames=100, keep_first_seconds=5.0,
    cmap='plasma', fixed_cbar=True
):
    """Create evolution GIFs for fluid simulations with automatic frame reduction and optional fixed color normalization."""

    # --- Load data ---
    time = np.array(ds.variables['time'][:])
    field = ds.variables[prefix][:]

    # Log-scale density
    if prefix in ['rho', 'press']:
        field = np.log10(field)

    # Average over φ
    phi_avg_field = np.mean(field, axis=-1)

    # --- Frame selection ---
    early_mask = time <= keep_first_seconds
    early_indices = np.where(early_mask)[0]
    later_indices = np.where(~early_mask)[0]

    total_frames = len(early_indices) + len(later_indices)
    if total_frames > max_frames:
        n_late_keep = max_frames - len(early_indices)
        later_indices = np.linspace(later_indices[0], later_indices[-1], n_late_keep, dtype=int)

    frames_to_use = np.concatenate([early_indices, later_indices])
    if len(frames_to_use) == 0:
        raise ValueError("No frames selected for animation.")

    print(f"Dataset: {len(time)} frames spanning {time[0]}–{time[-1]} s")
    print(f"Keeping {len(frames_to_use)} frames (≤ {max_frames})")

    # --- Color normalization ---
    if fixed_cbar:
        vmin, vmax = np.nanmin(field), np.nanmax(field)
    else:
        vmin = vmax = None

    if fixed_cbar:
        if 'vel' in prefix:
            try:
                norm = colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
            except ValueError:
                if vmin > 0:
                    norm = colors.TwoSlopeNorm(v_center=0, vmin=-1e-2, vmax=vmax)
                elif vmax < 0:
                    norm = colors.TwoSlopeNorm(v_center=0, vmin=vmin, vmax=1e-2)
                else:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        if 'temp' in prefix:
            norm = colors.Normalize(vmax=3000, vmin=0)

    cmap = mpl.colormaps.get_cmap(cmap)

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(0.9*width, 0.8*height))

    # Rotate 90° counterclockwise for correct orientation
    first_frame = phi_avg_field[frames_to_use[0]]

    cont = ax.imshow(
        np.fliplr(first_frame),
        cmap=cmap,
        norm=norm if fixed_cbar else None,
        extent=[
           theta.min(), theta.max(),
            altitude.min(), altitude.max()],
        aspect='auto',
        origin='lower'
    )

    cbar = fig.colorbar(cont, ax=ax, label=prefix)
    ax.set_xlabel("Colatitude (°)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"{prefix} evolution  t={time[frames_to_use[0]]:.2f}s")

    if not fixed_cbar:
        if 'vel' in prefix:
            norm = colors.CenteredNorm(vcenter=0)
        else:
            # Create a dummy Normalize object that we will update per frame
            norm = colors.Normalize()
        
    

    def update(i):
        frame = frames_to_use[i]
        data = np.fliplr(phi_avg_field[frame])

        if not fixed_cbar:
            # Dynamically adjust colour limits for this frame
            frame_vmin = np.nanmin(data)
            frame_vmax = np.nanmax(data)
            norm.vmin = frame_vmin
            norm.vmax = frame_vmax
            cont.set_norm(norm)
            cbar.update_normal(cont)

        cont.set_data(data)
        ax.set_title(f"{prefix} evolution  t={time[frame]:.2f}s")

        return [cont]


    # --- Build animation ---
    anim = FuncAnimation(fig, update, frames=len(frames_to_use), interval=200, blit=True)

    # --- Save animation ---
    outpath = f"{outdir}/{pf}_{prefix}_evolution.gif"
    writer = PillowWriter(fps=5)
    print(f"Saving animation → {outpath}")
    anim.save(outpath, writer=writer, dpi=120)

    plt.close(fig)
    print(f"✅ Saved animation to {outpath}")


# ---------- Run from Command Line ----------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse_spherical.py FILENAME.nc")
        sys.exit(1)

    ncfile = sys.argv[1]
    params = ['temp', 'rho', 'press', 'vel1', 'vel2', 'vel3', 'SiO', 'SiO(s)']
    summarize_simulation_spherical_with_vectors(
        ncfile, params, make_gif=False, t_index=-1
    )
    
    # Create all GIFs including vector fields
    summarize_simulation_spherical_with_vectors(
        ncfile, params, make_gif=False, 
        vector_gif_fields=['temp', 'rho', 'vel_mag', ]
    )
