import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

coordinates_filepath = "data/bio_tp_tom_l2a_20251007t050300_20251022t050313_c_g___m___c___t____f001_lut.nc"

annotations = xr.open_dataset(coordinates_filepath)
print("\nannotations")
print(annotations)          # look at dims, variables, dtypes


data_filepath = "data/bio_tp_tom_l2a_20251007t050300_20251022t050313_c_g___m___c___t____f001_geo_tomcubes.nc"
ds = xr.open_dataset(data_filepath, mask_and_scale=True)

print("\nmeasuremnt")
print(ds)          # overview of dimensions, coordinates, variables
print("Dimension sizes: " + str(ds.dims))            # dimension sizes
print("Variables: " + str(ds.data_vars))      # variables
print("Global metadata: " + str(ds.attrs))           # global metadata
print("Cube datatype: " + str(ds["tomographicCubes"].dtype))
print("Cube attrs: " + str(ds["tomographicCubes"].attrs))  # variable metadata (units, long_name, etc.)

for v in ["software", "imagedescription", "datetime", "crs", "swath", "tile_ids", "basin_ids"]:
    print(v, ":", ds[v].values)


def plot_mean_amplitude(ds):
    cube = ds["tomographicCubes"]
    cube.isel(polarizations=0).sel(relativeAzimuthTime=slice(0, 512))


    pol = 0
    amp = (np.abs(cube.isel(polarizations=pol)) ** 2).mean(dim="tomographicHeight")
    db = 10 * np.log10(amp.clip(min=1e-10))

    db.plot.imshow(cmap="gray", robust=True, size=8, aspect=2)
    plt.title("Mean amplitude over tomographic height")
    plt.show()

def plot_height_bins(ds):
    cube = ds["tomographicCubes"].isel(polarizations=0)  # start with one pol

    # power in dB, lazy
    power = 10 * np.log10((np.abs(cube) ** 2).clip(min=1e-12))

    # pull a manageable block into RAM (e.g. every 4th pixel in azimuth/range, all heights)
    sub = power.isel(relativeAzimuthTime=slice(0, None, 4),
                     slantRangeTime=slice(0, None, 4)).compute()
    print(sub.shape)  # (~816, ~495, 81) → ~130 MB, fine

    n_h = sub.sizes["tomographicHeight"]
    step = max(1, n_h // 16)  # up to ~16 panels
    levels = range(0, n_h, step)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    for ax, h in zip(axes.flat, levels):
        sub.isel(tomographicHeight=h).plot.imshow(
            ax=ax, cmap="viridis", robust=True, add_colorbar=False)
        ax.set_title(f"height bin {h}");
        ax.set_xlabel("");
        ax.set_ylabel("")
    fig.show()

plot_height_bins(ds)