"""
Spherical simulation analysis and visualization toolkit.

This module provides tools for analyzing and visualizing spherical fluid dynamics
simulations stored in NetCDF format. It creates static plots, animations, and
vector field visualizations with proper physical coordinate transformations.

Author: Refactored version (Fixed)
Date: 2026-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
from matplotlib import colors, ticker
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl
import sys
from scipy.interpolate import RegularGridInterpolator
import xarray as xr


# ============================================================================
# Configuration
# ============================================================================

class PlotConfig:
    """Central configuration for all plotting parameters."""
    
    # Matplotlib settings
    CM_TO_INCH = 0.393701
    WIDTH = 33.87 * CM_TO_INCH
    HEIGHT = 19.05 * CM_TO_INCH
    
    # Physical constants (all in SI units initially)
    R_PLANET = 6500e3  # m - planetary radius
    ALT_SUB = 400e3    # m - reference altitude
    R_EFF = R_PLANET + ALT_SUB  # m - effective radius for angular conversions
    
    # Plot layout
    NCOLS_FIELDS = 3
    NCOLS_VECTOR = 3
    
    # Animation settings
    MAX_FRAMES = 100
    KEEP_FIRST_SECONDS = 5.0
    FPS = 5
    INTERVAL = 200  # ms
    DPI = 120
    
    # Colormap defaults
    CMAP_SCALAR = 'plasma'
    CMAP_VELOCITY = 'RdBu_r'

    GAMMA = 1.29
    
    @classmethod
    def setup_matplotlib(cls):
        """Apply matplotlib configuration."""
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['mathtext.default'] = 'default'
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['axes.titlesize'] = 10
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10


PlotConfig.setup_matplotlib()


# ============================================================================
# Data Processing Utilities
# ============================================================================

class DataProcessor:
    """Handles data loading and preprocessing operations."""
    
    @staticmethod
    def average_over_phi(field):
        """
        Average 3D field over φ (azimuthal) dimension.
        
        Args:
            field: 3D array with shape (Nz, Nθ, Nphi)
            
        Returns:
            2D array with shape (Nz, Nθ)
        """
        return np.mean(field, axis=2)
    
    @staticmethod
    def load_variables(ds, var_names, t):
        """
        Safely load specified variables at a given time index.
        
        Args:
            ds: NetCDF4 Dataset object
            var_names: List of variable names to load
            t: Time index
            
        Returns:
            Dictionary mapping variable names to arrays
        """
        loaded = {}
        for name in var_names:
            if name in ds.variables:
                loaded[name] = ds.variables[name][t]
            else:
                print(f"Warning: variable '{name}' not found in dataset.")
                loaded[name] = None
        return loaded
    
    @staticmethod
    def compute_velocity_magnitude(vel1, vel2):
        """Compute velocity magnitude from components."""
        return np.sqrt(vel1**2 + vel2**2)
    
    @staticmethod
    def apply_log_scale(field, var_name):
        """Apply logarithmic scaling to density and pressure fields."""
        if var_name in ['rho', 'press', 'SiO_density', 'SiO(s)_density']:
            return np.log10(field), f'log10 {var_name}'
        return field, var_name


class CoordinateTransform:
    """Handles coordinate transformations for vector field plotting."""
    
    @staticmethod
    def velocity_to_plot_coords(v_theta, v_z, altitude, theta, theta_in_degrees=True, r_eff=None):
        """
        Transform physical velocities to plot coordinate rates.
        
        Args:
            v_theta: Velocity in θ direction (m/s)
            v_z: Vertical velocity (m/s)
            altitude: Altitude array (km) - used for meshgrid but not calculations
            theta: Colatitude array (degrees or radians)
            theta_in_degrees: If True, theta is in degrees (default for .nc files)
            r_eff: Effective radius for angular conversion (m). If None, uses PlotConfig.R_EFF
            
        Returns:
            U_plot: Rate in θ direction (degrees/s or rad/s, matching input)
            V_plot: Rate in z direction (km/s)
        """
        if r_eff is None:
            r_eff = PlotConfig.R_EFF
        
        # Convert velocity (m/s) to angular rate
        # v = R * dθ/dt  →  dθ/dt = v/R
        angular_rate_rad_per_s = v_theta / r_eff  # rad/s
        
        if theta_in_degrees:
            # Convert rad/s to deg/s for plotting in degree coordinates
            U_plot = -np.degrees(angular_rate_rad_per_s)  # deg/s
        else:
            U_plot = -angular_rate_rad_per_s  # rad/s
        
        V_plot = v_z / 1000.0  # m/s → km/s
        
        return U_plot, V_plot
    
    @staticmethod
    def ensure_monotonic_coords(x, y, U, V):
        """
        Ensure coordinates are monotonically increasing.
        
        Args:
            x, y: 1D coordinate arrays
            U, V: 2D velocity fields
            
        Returns:
            x, y, U, V with monotonic coordinates
        """
        x = x.copy()
        y = y.copy()
        
        # Flip if not monotonically increasing
        if np.any(np.diff(x) <= 0):
            x = x[::-1]
            U = U[:, ::-1]
            V = V[:, ::-1]
        
        if np.any(np.diff(y) <= 0):
            y = y[::-1]
            U = U[::-1, :]
            V = V[::-1, :]
        
        return x, y, U, V


class FrameSelector:
    """Handles frame selection for animations."""
    
    @staticmethod
    def select_frames(time, max_frames=100, keep_first_seconds=5.0):
        """
        Select frames for animation with more detail in early timesteps.
        
        Args:
            time: Time array
            max_frames: Maximum number of frames
            keep_first_seconds: Duration to keep all frames (seconds)
            
        Returns:
            Array of selected frame indices
        """
        early_mask = time <= keep_first_seconds
        early_indices = np.where(early_mask)[0]
        later_indices = np.where(~early_mask)[0]
        
        total_frames = len(early_indices) + len(later_indices)
        
        if total_frames > max_frames:
            n_late_keep = max_frames - len(early_indices)
            if n_late_keep > 0:
                later_indices = np.linspace(
                    later_indices[0], later_indices[-1], 
                    n_late_keep, dtype=int
                )
            else:
                # Too many early frames
                later_indices = np.array([], dtype=int)
        
        frames = np.concatenate([early_indices, later_indices])
        print(f"Selected {len(frames)} frames out of {len(time)} total")
        return frames


# ============================================================================
# Static Plot Generators
# ============================================================================

class StaticPlotter:
    """Generates static plots for single timesteps."""
    
    def __init__(self, altitude, theta, phi=None, theta_in_degrees=True):
        """
        Initialize plotter with coordinate arrays.
        
        Args:
            altitude: 1D array of altitudes (km)
            theta: 1D array of colatitudes (degrees or radians)
            phi: 1D array of longitudes (degrees or radians), optional
            theta_in_degrees: If True, theta/phi are in degrees
        """
        self.altitude = altitude
        self.theta = theta
        self.phi = phi
        self.theta_in_degrees = theta_in_degrees
        self.ALT, self.THETA = np.meshgrid(altitude, theta, indexing='ij')
        
        # Set axis labels based on units
        self.theta_label = "Colatitude (°)" if theta_in_degrees else "Colatitude (rad)"
        self.phi_label = "Longitude (°)" if theta_in_degrees else "Longitude (rad)"
    
    def plot_averaged_fields(self, vars_dict, outdir, prefix, t):
        """
        Plot φ-averaged fields in grid layout.
        
        Args:
            vars_dict: Dictionary of variables (Nz, Nθ, Nphi)
            outdir: Output directory
            prefix: Filename prefix
            t: Current time
        """
        n = len(vars_dict)
        ncols = PlotConfig.NCOLS_FIELDS
        nrows = int(np.ceil(n / ncols))
        
        fig, axes = plt.subplots(
            nrows, ncols, 
            figsize=(PlotConfig.WIDTH, PlotConfig.HEIGHT)
        )
        axes = axes.flat if n > 1 else [axes]
        
        for i, (name, field) in enumerate(vars_dict.items()):
            if field is None:
                continue
            
            ax = axes[i]
            field_avg = DataProcessor.average_over_phi(field)
            field_avg, label = DataProcessor.apply_log_scale(field_avg, name)
            
            # Choose colormap
            cmap = PlotConfig.CMAP_VELOCITY if ('vel' in name) or ('mach' in name) else PlotConfig.CMAP_SCALAR
            
            # Centered colormap for velocity fields
            if 'vel' in name:
                vmax = np.abs(field_avg).max()
                im = ax.contourf(-self.THETA, self.ALT, field_avg, 50, 
                               cmap=cmap, vmin=-vmax, vmax=vmax)
            elif 'mach' in name:
                norm = mpl.colors.TwoSlopeNorm(vcenter=1)
                im = ax.contourf(-self.THETA, self.ALT, field_avg, 50, 
                               cmap=cmap, norm=norm)
            else:
                im = ax.contourf(-self.THETA, self.ALT, field_avg, 50, cmap=cmap)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
            cbar.locator = ticker.MaxNLocator(nbins=4)
            cbar.formatter = ticker.FuncFormatter(lambda x, _: f"{x:.3g}")
            cbar.update_ticks()
            
            ax.set_title(label)
            ax.set_xlabel(self.theta_label)
            ax.set_ylabel("Altitude (km)")
        
        # Turn off unused axes
        for ax in axes[len(vars_dict):]:
            ax.axis("off")
        
        plt.suptitle(rf"$\phi$-Averaged Fields (t={t:.0f}s)", 
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{prefix}_avg_fields_t{t:.0f}.pdf", dpi=200)
        plt.close()
        print(f"✅ Saved averaged fields plot")
    

    def plot_vertical_profiles(self, vars_dict, outdir, prefix, t, n_profiles=5):
        """
        Plot vertical profiles of all fields at evenly spaced x2 coordinates.
        
        Args:
            vars_dict: Dictionary of variables
            outdir: Output directory
            prefix: Filename prefix
            t: Current time
            n_profiles: Number of evenly spaced profiles to plot (default: 5)
        """
        n = len(vars_dict)
        ncols = PlotConfig.NCOLS_FIELDS
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(PlotConfig.WIDTH, PlotConfig.HEIGHT)
        )
        axes = axes.flat if n > 1 else [axes]

        legend_handles = None
        legend_labels = None
        
        for i, (name, field) in enumerate(vars_dict.items()):
            if field is None:
                continue
            ax = axes[i]
            
            # Average over phi (x3) dimension
            avg = DataProcessor.average_over_phi(field)
            avg, label = DataProcessor.apply_log_scale(avg, name)
            
            # Get the number of x2 points
            nx2 = avg.shape[1]
            
            # Create n_profiles evenly spaced indices across the x2 dimension
            x2_indices = np.linspace(0, nx2-1, n_profiles, dtype=int)
            cmap = mpl.colormaps.get_cmap('Spectral')
            x2_colors = [cmap(l) for l in np.linspace(0, 1, n_profiles)]
            
            # Plot profiles at each selected x2 index
            for idx, c in zip(x2_indices, x2_colors):
                # Extract vertical profile at this x2 location
                profile = avg[:, idx]
                
                # Create label with x2 value (assuming x2 goes from 0 to 2π)
                x2_val = ((idx / nx2) * 180) - 90
                label_str = rf'$\theta$={x2_val:.1f}'
                
                ax.plot(profile, self.altitude, label=label_str, alpha=0.8, c=c)
            
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            
            ax.set_xlabel(label)
            ax.set_ylabel("Altitude (km)")
            ax.set_ylim(bottom=0)
        
        # Turn unused axes into a legend panel
        unused_axes = axes[len(vars_dict):]

        for h in legend_handles:
            h.set_linewidth(2.0)


        if len(unused_axes) > 0 and legend_handles is not None:
            leg_ax = unused_axes[0]
            leg_ax.axis("off")
            leg_ax.legend(
                legend_handles,
                legend_labels,
                loc="center",
                fontsize=9,
                frameon=False
            )
            leg_ax.set_title("Colatitude Profiles", fontsize=10, pad=10)


            # Turn off any remaining unused axes
            for ax in unused_axes[1:]:
                ax.axis("off")
        else:
            for ax in unused_axes:
                ax.axis("off")

        
        plt.suptitle(f"Vertical Profiles at Various Colatitudes (t={t:.0f}s)", 
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{prefix}_profiles_t{t:.0f}.pdf", dpi=200)
        plt.close()
        print(f"✅ Saved vertical profiles plot")
    

    def plot_equatorial_slices(self, vars_dict, outdir, prefix, t):
        """
        Plot equatorial slices (altitude vs longitude).
        
        Args:
            vars_dict: Dictionary of variables
            outdir: Output directory
            prefix: Filename prefix
            t: Current time
        """
        if self.phi is None:
            print("Warning: No phi coordinate, skipping equatorial slices")
            return
        
        # Find equator index - account for degrees vs radians
        if self.theta_in_degrees:
            eq_value = 90.0  # 90 degrees
        else:
            eq_value = np.pi / 2  # π/2 radians
        eq_index = np.argmin(np.abs(self.theta - eq_value))
        PHI, ALT = np.meshgrid(self.phi, self.altitude, indexing='ij')
        
        n = len(vars_dict)
        ncols = PlotConfig.NCOLS_FIELDS
        nrows = int(np.ceil(n / ncols))
        
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(PlotConfig.WIDTH, PlotConfig.HEIGHT)
        )
        axes = axes.flat if n > 1 else [axes]
        
        for i, (name, field) in enumerate(vars_dict.items()):
            if field is None:
                continue
            
            ax = axes[i]
            eq_field = field[:, eq_index, :]
            eq_field, label = DataProcessor.apply_log_scale(eq_field, name)
            
            cmap = PlotConfig.CMAP_VELOCITY if ('vel' in name) or ('mach' in name) else PlotConfig.CMAP_SCALAR
            
            if 'vel' in name:
                vmax = np.abs(eq_field).max()
                im = ax.contourf(PHI, ALT, eq_field.T, 50, 
                               cmap=cmap, vmin=-vmax, vmax=vmax)
            elif 'mach' in name:
                norm = mpl.colors.TwoSlopeNorm(vcenter=1)
                im = ax.contourf(PHI, ALT, eq_field.T, 50, 
                               cmap=cmap, norm=norm)

            else:
                im = ax.contourf(PHI, ALT, eq_field.T, 50, cmap=cmap)
            
            plt.colorbar(im, ax=ax, label=label)
            ax.set_title(f"{label} (Equator)")
            ax.set_xlabel(self.phi_label)
            ax.set_ylabel("Altitude (km)")
        
        for ax in axes[len(vars_dict):]:
            ax.axis("off")
        
        plt.suptitle(f"Equatorial Slices (t={t:.0f}s)", 
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{prefix}_equator_t{t:.0f}.pdf", dpi=200)
        plt.close()
        print(f"✅ Saved equatorial slices plot")


class VectorFieldPlotter:
    """Generates vector field plots with streamlines."""
    
    def __init__(self, altitude, theta, theta_in_degrees=True):
        """
        Initialize with coordinate arrays.
        
        Args:
            altitude: 1D array of altitudes (km)
            theta: 1D array of colatitudes (degrees or radians)
            theta_in_degrees: If True, theta is in degrees
        """
        self.altitude = altitude
        self.theta = theta
        self.theta_in_degrees = theta_in_degrees
        self.ALT, self.THETA = np.meshgrid(altitude, theta, indexing='ij')
        
        # Set axis label based on units
        self.theta_label = "Colatitude (°)" if theta_in_degrees else "Colatitude (rad)"


    def plot_vector_field_quiver(self, vars_dict, outdir, prefix, t, field_type='velocity',
                                 subsample=10, time_scale=100.0):
        """
        Plot velocity or mass-flux quiver plot with arrow length proportional to magnitude.
        
        Args:
            vars_dict: Dictionary of variables
            outdir: Output directory
            prefix: Filename prefix
            t: Current time
            field_type: 'velocity' or 'mass_flux'
            subsample: Spacing between arrows (every Nth point). Higher = fewer arrows
            time_scale: Time interval (seconds) over which to show displacement
        """
        if 'vel1' not in vars_dict or 'vel2' not in vars_dict:
            print("Warning: vel1 or vel2 not found, skipping quiver plots")
            return
        
        if field_type == 'mass_flux' and 'rho' not in vars_dict:
            print("Warning: rho not found, cannot plot mass flux")
            return
        
        # Average over phi
        vel1_avg = DataProcessor.average_over_phi(vars_dict['vel1'])  # vertical (m/s)
        vel2_avg = DataProcessor.average_over_phi(vars_dict['vel2'])  # theta (m/s)
        
        if field_type == 'mass_flux':
            rho_avg = DataProcessor.average_over_phi(vars_dict['rho'])
            vel1_avg = rho_avg * vel1_avg
            vel2_avg = rho_avg * vel2_avg
        
        # Prepare scalar fields
        scalar_fields = self._prepare_scalar_fields(vars_dict)
        
        # Compute magnitude
        mag = DataProcessor.compute_velocity_magnitude(vel1_avg, vel2_avg)
        scalar_fields[f'{field_type}_magnitude'] = mag
        
        # Convert theta to arc length distance (km)
        if self.theta_in_degrees:
            theta_rad = np.deg2rad(self.theta)
        else:
            theta_rad = self.theta
        
        # Use R_EFF for consistency with velocity_to_plot_coords
        reference_radius = PlotConfig.R_EFF
        
        # Arc length distance (km) - convert to km immediately
        arc_distance = (reference_radius * theta_rad) / 1000.0
        arc_distance = arc_distance - arc_distance[0]  # Zero at reference point
        
        # Convert velocities to displacements (km) over time_scale
        # vel1 is vertical velocity (m/s) -> displacement in altitude direction
        # vel2 is horizontal velocity (m/s) -> displacement in arc distance direction
        U_displacement = vel2_avg/1000 #* time_scale / 1000.0  # m/s * s / (1000 m/km) = km
        V_displacement = vel1_avg/1000 #* time_scale / 1000.0  # m/s * s / (1000 m/km) = km
        
        # Convert to xarray DataArrays for easy coarsening
        da_U = xr.DataArray(
            U_displacement,
            dims=['x1', 'x2'],
            coords={'x1': self.altitude, 'x2': arc_distance}
        )
        da_V = xr.DataArray(
            V_displacement,
            dims=['x1', 'x2'],
            coords={'x1': self.altitude, 'x2': arc_distance}
        )
        
        # Coarsen (block average) - boundary='trim' handles non-divisible sizes
        U_coarse = da_U.coarsen(x1=subsample, x2=subsample, boundary='trim').mean()
        V_coarse = da_V.coarsen(x1=subsample, x2=subsample, boundary='trim').mean()
        
        # Extract values and coordinates
        U_sub = U_coarse.values
        V_sub = V_coarse.values
        alt_sub = U_coarse.coords['x1'].values
        dist_sub = U_coarse.coords['x2'].values
        
        # Create meshgrid for subsampled coordinates
        X_sub, Y_sub = np.meshgrid(dist_sub, alt_sub, indexing='ij')

        # Create figure
        n = len(scalar_fields)
        ncols = PlotConfig.NCOLS_VECTOR
        nrows = int(np.ceil(n / ncols))
        width = 8 * ncols
        height = 5 * nrows
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
        axes = axes.flat if n > 1 else [axes]
        
        # Create meshgrid for background scalar fields (full resolution)
        ARC_DIST, ALT_FULL = np.meshgrid(arc_distance, self.altitude, indexing='ij')
        
        for i, (name, field) in enumerate(scalar_fields.items()):
            ax = axes[i]
            
            # Background scalar field
            cmap = PlotConfig.CMAP_VELOCITY if name in ['vel1', 'vel2', 'mach'] else PlotConfig.CMAP_SCALAR
            norm = colors.CenteredNorm(vcenter=0) if name in ['vel1', 'vel2'] else None
            if 'mach' in name:
                norm = colors.TwoSlopeNorm(vcenter=1) 
            
            im = ax.pcolormesh(-ARC_DIST, ALT_FULL, field.T, cmap=cmap, norm=norm, shading='auto')
            
            # Quiver: arrow length represents displacement over time_scale
            # Use white arrows for better visibility on scalar backgrounds
            arrow_color = 'white' if name not in ['vel1', 'vel2'] else 'black'
            ax.quiver(
                -X_sub, Y_sub, U_sub.T, V_sub.T,
                angles='xy',
                scale_units='xy',
                pivot='middle',
                color=arrow_color,
                alpha=0.8,
                width=0.003,
                headwidth=4,
                headlength=5,
                headaxislength=4.5
            )

            # Colorbar for background field
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.locator = ticker.MaxNLocator(nbins=4)
            cbar.formatter = ticker.FuncFormatter(lambda v, _: f"{v:.3g}")
            cbar.update_ticks()
            cbar.set_label(name)
            
            label = "Velocity" if field_type == 'velocity' else "Mass Flux"
            ax.set_title(f"{name} + {label}")
            ax.set_xlabel("Distance from substellar point (km)")
            ax.set_ylabel("Altitude (km)")
            # ax.set_xlim(dist_sub.min(), dist_sub.max())
            # ax.set_ylim(alt_sub.min(), alt_sub.max())
            # ax.set_aspect('equal', adjustable='box')  # Equal aspect ratio since both in km!
        
        # Turn off unused axes
        for ax in axes[len(scalar_fields):]:
            ax.axis("off")
        
        label = "Velocity" if field_type == 'velocity' else "Mass Flux"
        plt.suptitle(
            f"{label} Quiver (Displacement over {time_scale}s) t={t:.0f}s",
            fontsize=14, fontweight="bold"
        )
        
        plt.tight_layout()
        plt.savefig(
            f"{outdir}/{prefix}_{field_type}_quiver_t{t:.0f}.pdf",
            dpi=200
        )
        plt.close()
        print(f"✅ Saved {field_type} quiver plot")
    

    def plot_streamplot(self, vars_dict, outdir, prefix, t):
        """
        Plot velocity or mass-flux streamlines over scalar fields.
        
        Args:
            vars_dict: Dictionary of variables
            outdir: Output directory
            prefix: Filename prefix
            t: Current time
        """
        if 'vel1' not in vars_dict or 'vel2' not in vars_dict:
            print("Warning: vel1 or vel2 not found, skipping vector field plots")
            return
        
        # Average over phi
        vel1_avg = DataProcessor.average_over_phi(vars_dict['vel1'])  # vertical
        vel2_avg = DataProcessor.average_over_phi(vars_dict['vel2'])  # theta
        
        # Prepare scalar fields
        scalar_fields = self._prepare_scalar_fields(vars_dict)
        
        # Add magnitude field
        mag = DataProcessor.compute_velocity_magnitude(vel1_avg, vel2_avg)
        scalar_fields[f'vel_magnitude'] = mag
        
        # Transform velocities to plot coordinates
        U_plot, V_plot = CoordinateTransform.velocity_to_plot_coords(
            vel2_avg, vel1_avg, self.altitude, self.theta, self.theta_in_degrees
        )
        
        # Ensure monotonic coordinates
        x = -self.theta.copy()
        y = self.altitude.copy()
        x, y, U_plot, V_plot = CoordinateTransform.ensure_monotonic_coords(
            x, y, U_plot, V_plot
        )
        
        # Create uniform grid for streamplot
        xx = np.linspace(x.min(), x.max(), len(x))
        yy = np.linspace(y.min(), y.max(), len(y))
        
        # Create figure
        n = len(scalar_fields)
        ncols = PlotConfig.NCOLS_VECTOR
        nrows = int(np.ceil(n / ncols))
        width = 8 * ncols
        height = 5 * nrows
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
        axes = axes.flat if n > 1 else [axes]
        
        for i, (name, field) in enumerate(scalar_fields.items()):
            ax = axes[i]
            
            # Background scalar field
            cmap = PlotConfig.CMAP_VELOCITY if name in ['vel1', 'vel2', 'mach'] else PlotConfig.CMAP_SCALAR
            norm = colors.CenteredNorm(vcenter=0) if name in ['vel1', 'vel2'] else None
            if 'mach' in name:
                norm = colors.TwoSlopeNorm(vcenter=1)

            
            im = ax.contourf(-self.THETA, self.ALT, field, 50, cmap=cmap, norm=norm)
            
            # Streamlines
            ax.streamplot(
                xx, yy, -U_plot, V_plot,
                color='white', density=2, linewidth=0.8, arrowsize=0.8
            )
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.locator = ticker.MaxNLocator(nbins=4)
            cbar.formatter = ticker.FuncFormatter(lambda v, _: f"{v:.3g}")
            cbar.update_ticks()
            cbar.set_label(name)
            
            label = "Velocity"
            ax.set_title(f"{name} + {label}")
            ax.set_xlabel(f"− {self.theta_label}")
            ax.set_ylabel("Altitude (km)")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(self.altitude.min(), self.altitude.max())
        
        # Turn off unused axes
        for ax in axes[len(scalar_fields):]:
            ax.axis("off")
        
        label = "Velocity" 
        plt.suptitle(
            f"{label} Streamlines + Scalar Fields (t={t:.0f}s)",
            fontsize=14, fontweight="bold"
        )
        
        plt.tight_layout()
        plt.savefig(
            f"{outdir}/{prefix}_velocity_streamplot_t{t:.0f}.pdf",
            dpi=200
        )
        plt.close()
        print(f"✅ Saved velocity streamplot")

    
    def _prepare_scalar_fields(self, vars_dict):
        """Prepare scalar fields dictionary with log scaling."""
        scalar_fields = {}
        
        for name in ['temp', 'rho', 'press', 'vel1', 'vel2', 'SiO', 'SiO(s)']:
            if name in vars_dict and vars_dict[name] is not None:
                field = DataProcessor.average_over_phi(vars_dict[name])
                field, label = DataProcessor.apply_log_scale(field, name)
                scalar_fields[label] = field
        
        return scalar_fields


# ============================================================================
# Animation Generators
# ============================================================================

class AnimationGenerator:
    """Generates animated GIFs of simulation evolution."""
    
    def __init__(self, ds, altitude, theta, theta_in_degrees=True):
        """
        Initialize animator with dataset and coordinates.
        
        Args:
            ds: NetCDF4 Dataset object
            altitude: 1D altitude array (km)
            theta: 1D colatitude array (degrees or radians)
            theta_in_degrees: If True, theta is in degrees
        """
        self.ds = ds
        self.altitude = altitude
        self.theta = theta
        self.theta_in_degrees = theta_in_degrees
        self.time = np.array(ds.variables['time'][:])
        
        # Set axis label based on units
        self.theta_label = "Colatitude (°)" if theta_in_degrees else "Colatitude (rad)"
    
    def create_field_evolution_gif(self, var_name, outdir, prefix, 
                                   log_scale=False, fixed_cbar=True):
        """
        Create evolution GIF for a single scalar field.
        
        Args:
            var_name: Variable name in dataset
            outdir: Output directory
            prefix: Filename prefix
            log_scale: Apply log10 scaling
            fixed_cbar: Use fixed colorbar across all frames
        """
        if var_name not in self.ds.variables:
            print(f"Warning: {var_name} not found in dataset")
            return
        
        field = self.ds.variables[var_name][:]
        
        if log_scale:
            field = np.log10(field)
            label = f'log10 {var_name}'
        else:
            label = var_name
        
        # Average over phi
        phi_avg_field = np.mean(field, axis=-1)
        
        # Select frames
        frames = FrameSelector.select_frames(
            self.time, 
            PlotConfig.MAX_FRAMES,
            PlotConfig.KEEP_FIRST_SECONDS
        )
        
        # Colormap and normalization
        cmap = PlotConfig.CMAP_VELOCITY if 'vel' in var_name else PlotConfig.CMAP_SCALAR
        
        if fixed_cbar:
            vmin, vmax = np.nanmin(field), np.nanmax(field)
            if 'vel' in var_name:
                norm = colors.CenteredNorm(vcenter=0, vmin=vmin, vmax=vmax)
            else:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = colors.CenteredNorm(vcenter=0) if 'vel' in var_name else colors.Normalize()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(0.9 * PlotConfig.WIDTH, 0.8 * PlotConfig.HEIGHT))
        
        first_frame = phi_avg_field[frames[0]]
        cont = ax.imshow(
            np.fliplr(first_frame),
            cmap=cmap, norm=norm if fixed_cbar else None,
            extent=[self.theta.min(), self.theta.max(),
                   self.altitude.min(), self.altitude.max()],
            aspect='auto', origin='lower'
        )
        
        cbar = fig.colorbar(cont, ax=ax, label=label)
        ax.set_xlabel(self.theta_label)
        ax.set_ylabel("Altitude (km)")
        ax.set_title(f"{label} evolution  t={self.time[frames[0]]:.0f}s")
        
        def update(i):
            frame = frames[i]
            data = np.fliplr(phi_avg_field[frame])
            
            if not fixed_cbar:
                frame_vmin = np.nanmin(data)
                frame_vmax = np.nanmax(data)
                norm.vmin = frame_vmin
                norm.vmax = frame_vmax
                cont.set_norm(norm)
                cbar.update_normal(cont)
            
            cont.set_data(data)
            ax.set_title(f"{label} evolution  t={self.time[frame]:.0f}s")
            return [cont]
        
        anim = FuncAnimation(fig, update, frames=len(frames), 
                           interval=PlotConfig.INTERVAL, blit=True)
        
        outpath = f"{outdir}/{prefix}_{var_name}_evolution.gif"
        writer = PillowWriter(fps=PlotConfig.FPS)
        anim.save(outpath, writer=writer, dpi=PlotConfig.DPI)
        plt.close(fig)
        print(f"✅ Saved animation to {outpath}")
    
    def create_streamline_evolution_gif(self, scalar_field, outdir, prefix,
                                       fixed_cbar=False):
        """
        Create evolution GIF with velocity streamlines overlaid on scalar field.
        
        Args:
            scalar_field: Scalar field to display ('temp', 'rho', 'vel_mag', etc.)
            outdir: Output directory
            prefix: Filename prefix
            fixed_cbar: Use fixed colorbar across all frames
        """
        vel1 = self.ds.variables['vel1'][:]
        vel2 = self.ds.variables['vel2'][:]
        
        # Prepare scalar data
        if scalar_field == 'vel_mag':
            scalar_data = np.sqrt(vel1**2 + vel2**2)
            log_scale = False
            label = 'Velocity Magnitude (m/s)'
        else:
            if scalar_field not in self.ds.variables:
                print(f"Warning: {scalar_field} not found")
                return
            scalar_data = self.ds.variables[scalar_field][:]
            log_scale = scalar_field in ['rho', 'press']
            label = f'log10 {scalar_field}' if log_scale else scalar_field
            if log_scale:
                scalar_data = np.log10(scalar_data)
        
        # Average over phi
        scalar_avg = np.mean(scalar_data, axis=-1)
        vel1_avg = np.mean(vel1, axis=-1)
        vel2_avg = np.mean(vel2, axis=-1)
        
        # Select frames
        frames = FrameSelector.select_frames(
            self.time,
            PlotConfig.MAX_FRAMES,
            PlotConfig.KEEP_FIRST_SECONDS
        )
        
        # Prepare coordinates
        x = -self.theta.copy()
        y = self.altitude.copy()
        
        # Ensure monotonic
        flip_x = np.any(np.diff(x) <= 0)
        flip_y = np.any(np.diff(y) <= 0)
        if flip_x:
            x = x[::-1]
        if flip_y:
            y = y[::-1]
        
        xx = np.linspace(x.min(), x.max(), len(x))
        yy = np.linspace(y.min(), y.max(), len(y))
        
        # Normalization
        if fixed_cbar:
            vmin, vmax = np.nanmin(scalar_avg), np.nanmax(scalar_avg)
            if vmin >= vmax:
                vmin, vmax = None, None
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        def update(i):
            frame = frames[i]
            ax.clear()
            
            scalar_f = scalar_avg[frame]
            v_theta = vel2_avg[frame]
            v_z = vel1_avg[frame]
            
            if flip_x:
                scalar_f = scalar_f[:, ::-1]
                v_theta = v_theta[:, ::-1]
                v_z = v_z[:, ::-1]
            
            if flip_y:
                scalar_f = scalar_f[::-1, :]
                v_theta = v_theta[::-1, :]
                v_z = v_z[::-1, :]
            
            # Scalar background
            cont = ax.contourf(x, y, scalar_f, 50, 
                             cmap=PlotConfig.CMAP_SCALAR, norm=norm)
            
            if not fixed_cbar:
                if np.nanmin(scalar_f) < np.nanmax(scalar_f):
                    norm.vmin = np.nanmin(scalar_f)
                    norm.vmax = np.nanmax(scalar_f)
                    cont.set_norm(norm)
            
            # Transform velocities to plot coordinates
            # x is -theta, so theta_for_transform is -x (positive theta)
            theta_for_transform = -x
            
            U_trans, V_trans = CoordinateTransform.velocity_to_plot_coords(
                v_theta, v_z, y, theta_for_transform, self.theta_in_degrees
            )
            
            # U_trans is already negative from the transform
            U = U_trans
            V = V_trans
            
            interp_U = RegularGridInterpolator(
                (y, x), U, bounds_error=False, fill_value=np.nan
            )
            interp_V = RegularGridInterpolator(
                (y, x), V, bounds_error=False, fill_value=np.nan
            )
            
            Xg, Yg = np.meshgrid(xx, yy)
            pts = np.stack([Yg.ravel(), Xg.ravel()], axis=-1)
            Uu = interp_U(pts).reshape(Yg.shape)
            Vu = interp_V(pts).reshape(Yg.shape)
            
            ax.streamplot(xx, yy, Uu, Vu, color='white', 
                        density=1.2, linewidth=0.8, arrowsize=0.8)
            
            ax.set_xlabel(f"− {self.theta_label}")
            ax.set_ylabel("Altitude (km)")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_title(f"{label} + velocity  t={self.time[frame]:.0f}s")
            
            return []
        
        anim = FuncAnimation(fig, update, frames=len(frames),
                           interval=PlotConfig.INTERVAL, blit=False)
        
        outpath = f"{outdir}/{prefix}_streamlines_{scalar_field}_evolution.gif"
        anim.save(outpath, writer=PillowWriter(fps=PlotConfig.FPS), 
                 dpi=PlotConfig.DPI)
        plt.close(fig)
        print(f"✅ Saved streamline animation to {outpath}")


# ============================================================================
# Main Analysis Class
# ============================================================================

class SimulationAnalyzer:
    """Main class for analyzing spherical simulations."""
    
    def __init__(self, ncfile):
        """
        Initialize analyzer with NetCDF file.
        
        Args:
            ncfile: Path to NetCDF file
        """
        self.ncfile = ncfile
        self.ds = Dataset(ncfile, 'r')
        
        # Try to get planet radius from NetCDF attributes
        try:
            planet_radius_m = self.ds.getncattr('PlanetRadius')
            print(f'R_P = {planet_radius_m/1e3:.1f} km')
            PlotConfig.R_PLANET = planet_radius_m
        except AttributeError:
            print("Warning: PlanetRadius attribute not found in NetCDF file")
            print(f"Using default R_P = {PlotConfig.R_PLANET/1e3:.1f} km from PlotConfig")

        try:
            PlotConfig.GAMMA = self.ds.getncattr('Gamma')
        except AttributeError:
            print(rf"Can't find gamma in .nc file, using gamma = {PlotConfig.GAMMA}")
        
        # Load coordinates
        self.time = self.ds.variables['time'][:]
        r = self.ds.variables['x1'][:] / 1e3  # convert m to km
        self.theta = self.ds.variables['x2'][:]
        self.phi = self.ds.variables['x3'][:]
        
        # Detect if coordinates are in degrees or radians
        # .nc files typically have x2, x3 in degrees
        self.theta_in_degrees = self._detect_degrees(self.theta)
        self.phi_in_degrees = self._detect_degrees(self.phi)
        
        if self.theta_in_degrees:
            print(f"📐 Detected theta in degrees (range: {self.theta.min():.1f}° to {self.theta.max():.1f}°)")
        else:
            print(f"📐 Detected theta in radians (range: {self.theta.min():.3f} to {self.theta.max():.3f})")
        
        self.altitude = r
        
        # Output paths
        self.outdir = os.path.dirname(os.path.abspath(ncfile))
        self.prefix = os.path.splitext(os.path.basename(ncfile))[0]
    
    @staticmethod
    def _detect_degrees(coord_array):
        """
        Detect if coordinate array is in degrees or radians.
        
        Args:
            coord_array: Coordinate array (theta or phi)
            
        Returns:
            True if degrees, False if radians
        """
        max_val = np.max(np.abs(coord_array))
        # If any value > 2π (≈6.28), assume degrees
        # Typical ranges: degrees [0,360] or [-90,90], radians [0,2π] or [-π,π]
        return max_val > 2 * np.pi
    
    def analyze_snapshot(self, t_index=-1, var_names=None):
        """
        Create all static plots for a single timestep.
        
        Args:
            t_index: Time index (-1 for last)
            var_names: List of variable names to analyze
        """
        if var_names is None:
            var_names = ['temp', 'rho', 'press', 'vel1', 'vel2', 'vel3']
        
        t = t_index if t_index >= 0 else len(self.time) - 1
        vars_dict = DataProcessor.load_variables(self.ds, var_names, t)
        if 'SiO' in var_names:
            vars_dict['SiO_density'] = vars_dict['SiO'] * vars_dict['rho'] 
            vars_dict['SiO(s)_density'] = vars_dict['SiO(s)'] * vars_dict['rho'] 
        sound_speed_field = np.sqrt(PlotConfig.GAMMA*(vars_dict['press'] )/(vars_dict['rho']))
        net_vel_field = np.sqrt(vars_dict['vel1']**2 + vars_dict['vel2']**2 + vars_dict['vel3']**2) 
        vars_dict['mach'] = net_vel_field/sound_speed_field
        
        print(f"📊 Plotting snapshot at t={self.time[t]:.0f}s")
        
        # Create plotters - pass degree information
        static_plotter = StaticPlotter(self.altitude, self.theta, self.phi, self.theta_in_degrees)
        vector_plotter = VectorFieldPlotter(self.altitude, self.theta, self.theta_in_degrees)
        
        # Generate plots
        static_plotter.plot_averaged_fields(
            vars_dict, self.outdir, self.prefix, self.time[t]
        )
        static_plotter.plot_vertical_profiles(
            vars_dict, self.outdir, self.prefix, self.time[t]
        )
        
        if len(self.phi) > 1:
            static_plotter.plot_equatorial_slices(
                vars_dict, self.outdir, self.prefix, self.time[t]
            )
        
        # Vector field plots
        print(f"🎯 Plotting vector fields")
        vector_plotter.plot_streamplot(
            vars_dict, self.outdir, self.prefix, self.time[t], 
        )
        vector_plotter.plot_streamplot(
            vars_dict, self.outdir, self.prefix, self.time[t],
        )

        vector_plotter.plot_vector_field_quiver(
            vars_dict, self.outdir, self.prefix, self.time[t],
            field_type='velocity', subsample=2,
        )

        vector_plotter.plot_vector_field_quiver(
            vars_dict, self.outdir, self.prefix, self.time[t],
            field_type='mass_flux', subsample=2,
        )

    
    def create_animations(self, var_names=None, vector_fields=None):
        """
        Create animated GIFs of simulation evolution.
        
        Args:
            var_names: List of variables to animate
            vector_fields: List of scalar fields to overlay vectors on
        """
        if var_names is None:
            var_names = ['temp', 'rho', 'press']
        
        if vector_fields is None:
            vector_fields = ['temp', 'vel_mag']
        
        animator = AnimationGenerator(self.ds, self.altitude, self.theta, self.theta_in_degrees)
        
        # Create field evolution GIFs
        for name in var_names:
            if name in self.ds.variables:
                print(f"🎞 Creating GIF for {name}...")
                log_scale = name in ['rho', 'press']
                animator.create_field_evolution_gif(
                    name, self.outdir, self.prefix,
                    log_scale=log_scale, fixed_cbar=False
                )
        
        # Create vector field GIFs
        for field in vector_fields:
            print(f"🎯 Creating vector field GIF for {field}...")
            animator.create_streamline_evolution_gif(
                field, self.outdir, self.prefix, fixed_cbar=False
            )

    def analyze_mass_flux_evolution(self, var_names=None, time_indices=None):
        """
        Analyze mass flux evolution over multiple timesteps.
        
        Args:
            var_names: List of variable names needed ['rho', 'vel1', 'vel2']
            time_indices: List of time indices to plot. If None, automatically selects ~8 evenly spaced times
        """
        if var_names is None:
            var_names = ['rho', 'vel1', 'vel2']
        
        # Auto-select time indices if not provided
        if time_indices is None:
            n_times = min(8, len(self.time))  # Max 8 timesteps
            time_indices = np.linspace(0, len(self.time)-1, n_times, dtype=int)
        
        print(f"📊 Computing mass flux evolution over {len(time_indices)} timesteps...")
        
        # Storage for results
        all_net_flux_vs_radius = []
        all_vertical_flux_vs_latitude = []
        all_latitudinal_flux_vs_latitude = []
        times_plotted = []
        
        # Precompute coordinate info (same for all times)
        r = (self.altitude * 1000.0) + PlotConfig.R_PLANET  # meters
        theta = self.theta.copy()
        dphi = 2 * np.pi / len(self.phi)
        
        if self.theta_in_degrees:
            theta_rad = np.deg2rad(theta)
        else:
            theta_rad = theta.copy()
        
        dtheta = np.gradient(theta_rad)
        dr = np.gradient(r)
        sin_theta = np.sin(theta_rad)
        
        # Broadcast metric factors
        r2 = r[:, None, None] ** 2
        r1 = r[:, None, None]
        sin_t = sin_theta[None, :, None]
        dr_ = dr[:, None, None]
        dtheta_ = dtheta[None, :, None]
        
        # Loop over time indices
        for t_idx in time_indices:
            vars_dict = DataProcessor.load_variables(self.ds, var_names, t_idx)
            
            rho = vars_dict["rho"]
            vr = vars_dict["vel1"]
            vt = vars_dict["vel2"]
            
            # Mass flux densities
            flux_r = rho * vr
            flux_t = rho * vt
            
            # 1. Net radial flux through each spherical shell
            net_flux_vs_radius = np.sum(
                flux_r * r2 * sin_t * dtheta_ * dphi,
                axis=(1, 2)
            )
            
            # 2. Vertical mass flux at each latitude
            vertical_flux_vs_latitude = np.sum(
                flux_r * r2 * sin_t * dr_ * dphi,
                axis=(0, 2)
            )
            
            # 3. Latitudinal mass flux
            latitudinal_flux_vs_latitude = np.sum(
                flux_t * r1 * sin_t * dr_ * dphi,
                axis=(0, 2)
            )
            
            all_net_flux_vs_radius.append(net_flux_vs_radius)
            all_vertical_flux_vs_latitude.append(vertical_flux_vs_latitude)
            all_latitudinal_flux_vs_latitude.append(latitudinal_flux_vs_latitude)
            times_plotted.append(self.time[t_idx])
        
        # Convert to arrays
        all_net_flux_vs_radius = np.array(all_net_flux_vs_radius)
        all_vertical_flux_vs_latitude = np.array(all_vertical_flux_vs_latitude)
        all_latitudinal_flux_vs_latitude = np.array(all_latitudinal_flux_vs_latitude)
        
        # Create plots
        latitude_deg = 90.0 - np.rad2deg(theta_rad)
        
        # Use a colormap for time evolution
        cmap = mpl.colormaps.get_cmap('viridis')
        colors = [cmap(i / (len(time_indices) - 1)) for i in range(len(time_indices))]
        
        fig, axes = plt.subplots(
            1, 3,
            figsize=(15, 4),
            constrained_layout=True
        )
        
        # Panel 1: Radial flux vs radius
        for i, (flux, t, c) in enumerate(zip(all_net_flux_vs_radius, times_plotted, colors)):
            label = f't={t:.0f}s' if i % 2 == 0 else None  # Label every other line
            axes[0].plot(self.altitude, flux, lw=1.5, color=c, label=label)
        
        axes[0].axhline(0, color="k", lw=0.8, alpha=0.5, ls='--')
        axes[0].set_title("Net Radial Flux Through Shells")
        axes[0].set_ylabel(r"$\dot{M}_r\ \mathrm{[kg\ s^{-1}]}$")
        axes[0].set_xlabel("Altitude (km)")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8, loc='best')
        
        # Panel 2: Vertical flux vs latitude
        for i, (flux, t, c) in enumerate(zip(all_vertical_flux_vs_latitude, times_plotted, colors)):
            label = f't={t:.0f}s' if i % 2 == 0 else None
            axes[1].plot(latitude_deg, flux, lw=1.5, color=c, label=label)
        
        axes[1].axhline(0, color="k", lw=0.8, alpha=0.5, ls='--')
        axes[1].set_title("Vertical Mass Transport vs Latitude")
        axes[1].set_ylabel(r"$\dot{M}_\mathrm{vertical}\ \mathrm{[kg\ s^{-1}]}$")
        axes[1].set_xlabel("Latitude (deg)")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8, loc='best')
        
        # Panel 3: Latitudinal flux vs latitude
        for i, (flux, t, c) in enumerate(zip(all_latitudinal_flux_vs_latitude, times_plotted, colors)):
            label = f't={t:.0f}s' if i % 2 == 0 else None
            axes[2].plot(latitude_deg, flux, lw=1.5, color=c, label=label)
        
        axes[2].axhline(0, color="k", lw=0.8, alpha=0.5, ls='--')
        axes[2].set_title("Latitudinal Mass Transport")
        axes[2].set_ylabel(r"$\dot{M}_\theta\ \mathrm{[kg\ s^{-1}]}$")
        axes[2].set_xlabel("Latitude (deg)")
        axes[2].grid(alpha=0.3)
        axes[2].legend(fontsize=8, loc='best')
        
        fig.suptitle(
            "Mass Flux Evolution",
            fontsize=14,
            fontweight="bold"
        )
        
        outfile = f"{self.outdir}/{self.prefix}_mass_flux_evolution.pdf"
        plt.savefig(outfile, dpi=200)
        plt.close()
        
        print(f"✅ Saved mass flux evolution plot: {outfile}")
        
        # Print summary statistics
        print(f"\n📈 Mass Flux Summary:")
        print(f"   Times analyzed: {times_plotted[0]:.0f}s to {times_plotted[-1]:.0f}s")
        print(f"   Final radial outflow (top): {all_net_flux_vs_radius[-1, -1]:.2e} kg/s")
        print(f"   Max vertical transport: {np.max(np.abs(all_vertical_flux_vs_latitude)):.2e} kg/s")

    
    def close(self):
        """Close NetCDF dataset."""
        self.ds.close()


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Main entry point for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python analyse_spherical_fixed.py FILENAME.nc [--gifs]")
        print("\nOptions:")
        print("  --gifs    Create animated GIFs (slower)")
        sys.exit(1)
    
    ncfile = sys.argv[1]
    make_gifs = '--gifs' in sys.argv
    
    # Variable list
    var_names = ['temp', 'rho', 'press', 'vel1', 'vel2', 'vel3', 'SiO', 'SiO(s)']
    vector_fields = ['temp', 'rho', 'press', 'SiO(s)', 'SiO']
    
    # Create analyzer
    analyzer = SimulationAnalyzer(ncfile)
    
    try:
        # Generate static plots
        analyzer.analyze_snapshot(t_index=-1, var_names=var_names)
        analyzer.analyze_snapshot(t_index=1, var_names=var_names)
        analyzer.analyze_mass_flux_evolution()
        
        # Generate animations if requested
        if make_gifs:
            print("\n" + "="*60)
            print("Creating animations...")
            print("="*60 + "\n")
            analyzer.create_animations(
                var_names=var_names,
                vector_fields=vector_fields
            )
    
    finally:
        analyzer.close()
    
    print(f"\n✅ All outputs saved to {analyzer.outdir}")


if __name__ == "__main__":
    main()