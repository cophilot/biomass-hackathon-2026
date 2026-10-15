import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvista as pv

from tomo_canopy.processors import TomogramProcessor  # noqa: F401


# --------------------------------------------------------------------------- #
# Central configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    # --- Data selection ---------------------------------------------------- #
    tom_glob: str = "*L2A*"
    expected_level: str = "L2A"
    polarization: str = "HV"
    bounds: tuple = (11.480, -0.080, 11.500, -0.060)  # lon_min, lat_min, lon_max, lat_max
    height_range: tuple = (-10, 60)  # m, vertical crop applied on open
    chunks: str = "auto"
    min_finite_fraction: float = 0.95

    # --- Vertical profile processing --------------------------------------- #
    nominal_vertical_resolution: float = 20.0  # m, rolling-mean window
    # Per-vertical-bin normalization: each voxel is divided by this quantile of
    # the voxels in its height bin (robust to outliers, unlike the max).
    bin_normalization_quantile: float = 0.95
    # Initial state of the "Normalize by height bin" UI toggle. The
    # scattering-height metrics are computed once with this initial setting.
    normalize_bins: bool = True
    # Initial state of the "Column quantile after normalization" UI toggle.
    # False: filter the raw power per column first, then normalize what remains.
    # True: normalize the full volume first, then filter per column.
    filter_after_normalization: bool = False
    sh_median_quantile: float = 0.50
    sh_upper_quantile: float = 0.90

    # Height bands for power-fraction metrics (m above ground)
    lower_band: tuple = (0.0, 10.0)
    middle_band: tuple = (10.0, 30.0)
    upper_band_min: float = 30.0

    # --- Geodesy (degrees -> metres, spherical approximation) -------------- #
    metres_per_degree_lon_equator: float = 111_320.0
    metres_per_degree_lat: float = 111_132.0

    # --- 3D view: initial slider values ------------------------------------ #
    column_quantile: float = 0.80
    vertical_exaggeration: float = 12.0
    point_opacity: float = 0.35
    point_size: float = 15.0
    color_by_level_norm: bool = False  # colour by level-normalised energy instead of height

    # --- 3D view: slider ranges -------------------------------------------- #
    column_quantile_range: tuple = (0.0, 0.99)
    vertical_exaggeration_range: tuple = (1.0, 40.0)
    point_opacity_range: tuple = (0.02, 1.0)
    point_size_range: tuple = (1.0, 40.0)

    # --- 3D view: appearance ----------------------------------------------- #
    window_size: tuple = (1100, 750)
    background: str = "#eeeeee"
    point_cmap: str = "terrain"
    level_norm_cmap: str = "magma"
    level_norm_clim: tuple = (0.0, 1.0)  # values above 1 (top ~5% of a bin) saturate
    ground_color: str = "tan"
    ground_opacity: float = 0.35
    title_font_size: int = 11
    camera_position: str = "iso"

    # --- 3D view: slider layout (normalised viewport coords) --------------- #
    slider_x: tuple = (0.02, 0.30)
    slider_y_start: float = 0.82
    slider_y_step: float = 0.09


CFG = Config()


# --------------------------------------------------------------------------- #
# Load tomogram
# --------------------------------------------------------------------------- #
project_root = next(
    path for path in (Path.cwd(), Path.cwd().parent)
    if (path / "pyproject.toml").exists()
)
tom_source = next((project_root / "data" / "Lope" / "TOM").glob(CFG.tom_glob))

tom_processor = TomogramProcessor()
tom = tom_processor.open(
    tom_source,
    expected_level=CFG.expected_level,
    polarization=CFG.polarization,
    height_range=CFG.height_range,
    bounds=CFG.bounds,
    chunks=CFG.chunks,
)
tom = tom_processor.to_power(tom).load()
power = tom.tomographic_power.sel(polarization=CFG.polarization)

finite_fraction = float(np.isfinite(power).sum() / power.size)
assert finite_fraction > CFG.min_finite_fraction, (
    f"Unexpected TOM coverage: {finite_fraction:.1%}"
)
print(tom.sizes, finite_fraction)


# --------------------------------------------------------------------------- #
# Vertical profile metrics
# --------------------------------------------------------------------------- #
vertical_sampling = float(np.median(np.diff(power.z.values)))
window = max(1, round(CFG.nominal_vertical_resolution / vertical_sampling))
smoothed = power.rolling(z=window, center=True, min_periods=1).mean()
values = np.asarray(smoothed.values, dtype=float)  # (z, y, x)
z = np.asarray(smoothed.z.values, dtype=float)

def bin_scale(volume, quantile):
    """Quantile of each height bin over (y, x), shape (z,); invalid bins -> NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN bins
        scale = np.nanquantile(volume, quantile, axis=(1, 2))
    return np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)


# --------------------------------------------------------------------------- #
# Convert coordinates to metres
# --------------------------------------------------------------------------- #
lon0, lat0 = float(tom.x.mean()), float(tom.y.mean())
metres_per_degree_lon = CFG.metres_per_degree_lon_equator * np.cos(np.deg2rad(lat0))
metres_per_degree_lat = CFG.metres_per_degree_lat


def to_local_x(longitude):
    return (np.asarray(longitude) - lon0) * metres_per_degree_lon


def to_local_y(latitude):
    return (np.asarray(latitude) - lat0) * metres_per_degree_lat


x_m, y_m = to_local_x(tom.x), to_local_y(tom.y)
xx, yy = np.meshgrid(x_m, y_m, indexing="xy")


# --------------------------------------------------------------------------- #
# Interactive 3D scattering structure
# --------------------------------------------------------------------------- #
def scale_for(volume, normalize):
    """Divisor that turns `volume` into normalized energy.

    normalize=True: the configured quantile of each height bin, shape (z, 1, 1).
    normalize=False: one scene-wide value of the same quantile, so the colour
    scale stays comparable while relative differences between height bins are
    preserved.
    """
    if normalize:
        return bin_scale(volume, CFG.bin_normalization_quantile)[:, None, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN volume
        scale = np.nanquantile(volume, CFG.bin_normalization_quantile)
    return scale if np.isfinite(scale) and scale > 0 else np.nan


@lru_cache(maxsize=2)
def full_scale(normalize):
    """Scale computed from the whole (unfiltered) volume."""
    return scale_for(values, normalize)


@lru_cache(maxsize=64)
def column_threshold(quantile, normalize_first=None):
    """Per-column quantile (cached; slider values are rounded).

    normalize_first=None: thresholds the raw smoothed power.
    normalize_first=True/False: thresholds the power after per-bin / scene-wide
    normalization, i.e. the filter runs after normalization.
    """
    source = values if normalize_first is None else values / full_scale(normalize_first)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
        return np.nanquantile(source, quantile, axis=0)


def keep_top_per_column(volume, threshold):
    """Keep voxels at/above their column's threshold; everything else -> NaN."""
    return np.where(volume >= threshold[None, ...], volume, np.nan)


@lru_cache(maxsize=64)
def scale_after_filter(quantile, normalize):
    """Scale computed from the raw power that survives the column filter."""
    filtered = keep_top_per_column(values, column_threshold(quantile))
    return scale_for(filtered, normalize)


def compute_energy(quantile, normalize, filter_after):
    """Normalized energy per voxel; NaN where voxels were filtered out.

    filter_after=False: filter raw power per column, then normalize what remains
    (the normalization scale is computed from the retained voxels only).
    filter_after=True: normalize the full volume, then filter per column.
    """
    if filter_after:
        energy = values / full_scale(normalize)
        return keep_top_per_column(energy, column_threshold(quantile, normalize))
    filtered = keep_top_per_column(values, column_threshold(quantile))
    return filtered / scale_after_filter(quantile, normalize)


# Mutable view state driven by the sliders and toggles.
state = {
    "normalize": CFG.normalize_bins,
    "filter_after": CFG.filter_after_normalization,
    "column_quantile": CFG.column_quantile,
    "exaggeration": CFG.vertical_exaggeration,
    "opacity": CFG.point_opacity,
    "point_size": CFG.point_size,
    "color_by_level_norm": CFG.color_by_level_norm,
    "z_min": float(z.min()),
    "z_max": float(z.max()),
}

plotter = pv.Plotter(notebook=False, window_size=CFG.window_size)
plotter.set_background(CFG.background)
plotter.add_mesh(
    pv.Plane(
        center=(0, 0, 0),
        i_size=np.ptp(x_m), j_size=np.ptp(y_m),
    ),
    color=CFG.ground_color, opacity=CFG.ground_opacity, name="ground",
)

_first_render = True


def update_scene(reset_camera=False):
    """Rebuild the point cloud from the current slider state."""
    quantile = round(state["column_quantile"], 2)
    normalize = state["normalize"]
    filter_after = state["filter_after"]
    z_ok = (z >= state["z_min"]) & (z <= state["z_max"])

    energy = compute_energy(quantile, normalize, filter_after)
    keep = np.isfinite(energy) & z_ok[:, None, None]
    iz, iy, ix = np.where(keep)
    exaggeration = state["exaggeration"]
    pct = f"{CFG.bin_normalization_quantile * 100:g}"
    energy_label = f"energy / {'bin' if normalize else 'scene'} P{pct}"

    if iz.size:
        column_total = np.nansum(energy, axis=0)
        column_share = np.where(column_total > 0, column_total, np.nan)
        points = pv.PolyData(np.column_stack([
            x_m[ix], y_m[iy], z[iz] * exaggeration,
        ]))
        points["height above ground (m)"] = z[iz]
        points["profile contribution"] = energy[iz, iy, ix] / column_share[iy, ix]
        points[energy_label] = energy[iz, iy, ix]

        if state["color_by_level_norm"]:
            scalars, cmap, clim = energy_label, CFG.level_norm_cmap, CFG.level_norm_clim
        else:
            scalars, cmap, clim = (
                "height above ground (m)", CFG.point_cmap,
                (float(z.min()), float(z.max())),
            )
        for title in list(plotter.scalar_bars.keys()):
            plotter.remove_scalar_bar(title)
        plotter.add_mesh(
            points,
            scalars=scalars,
            cmap=cmap,
            clim=clim,
            render_points_as_spheres=True,
            point_size=state["point_size"],
            opacity=state["opacity"],
            name="scatter",
            reset_camera=reset_camera,
            scalar_bar_args={"title": scalars},
        )
    else:
        plotter.remove_actor("scatter")

    plotter.add_text(
        f"Top {(1 - quantile) * 100:.0f}% of power per column "
        f"({'filtered after' if filter_after else 'filtered before'} normalization; "
        f"{'per-bin P' + pct if normalize else 'no bin normalization'}) "
        f"— height exaggerated {exaggeration:.0f}× — {iz.size:,} points",
        font_size=CFG.title_font_size, position="upper_edge", name="title",
    )
    plotter.remove_bounds_axes()
    plotter.show_grid(
        xtitle="east (m)", ytitle="north (m)",
        ztitle=f"display height ({exaggeration:.0f}×)",
        bounds=(
            float(x_m.min()), float(x_m.max()),
            float(y_m.min()), float(y_m.max()),
            float(z.min() * exaggeration), float(z.max() * exaggeration),
        ),
    )
    plotter.render()


def make_slider(key, title, value_range, row, fmt="%.2f", pre=None):
    """Add a slider that writes into `state[key]` and refreshes the scene."""
    def callback(value):
        state[key] = pre(value) if pre else float(value)
        update_scene()

    y = CFG.slider_y_start - row * CFG.slider_y_step
    plotter.add_slider_widget(
        callback,
        rng=value_range,
        value=state[key],
        title=title,
        pointa=(CFG.slider_x[0], y),
        pointb=(CFG.slider_x[1], y),
        fmt=fmt,
        style="modern",
        interaction_event="end",  # update on release, not on every drag tick
    )


z_range = (float(z.min()), float(z.max()))
make_slider("column_quantile", "Column quantile", CFG.column_quantile_range, 0)
make_slider("exaggeration", "Vertical exaggeration", CFG.vertical_exaggeration_range, 1, fmt="%.0f")
make_slider("z_min", "Min height (m)", z_range, 2, fmt="%.0f")
make_slider("z_max", "Max height (m)", z_range, 3, fmt="%.0f")
make_slider("opacity", "Point opacity", CFG.point_opacity_range, 4)
make_slider("point_size", "Point size", CFG.point_size_range, 5, fmt="%.0f")


def add_toggle(key, label, row):
    """Add a checkbox button (bottom-left) that flips `state[key]`."""
    def callback(flag):
        state[key] = bool(flag)
        update_scene()

    y = 15 + row * 40
    plotter.add_checkbox_button_widget(
        callback,
        value=state[key],
        position=(15, y),
        size=28,
        color_on="#e8a33d",
        color_off="#3a4250",
    )
    plotter.add_text(label, position=(55, y + 5), font_size=10, name=f"{key}_label")


add_toggle("color_by_level_norm", "Colour by normalized energy", row=0)
add_toggle(
    "normalize",
    f"Normalize by height bin (P{CFG.bin_normalization_quantile * 100:g})",
    row=1,
)
add_toggle("filter_after", "Column quantile after normalization", row=2)

update_scene(reset_camera=True)
plotter.camera_position = CFG.camera_position
plotter.show()