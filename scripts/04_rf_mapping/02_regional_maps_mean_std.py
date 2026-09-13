"""
ISF PREDICTION MAPS BY REGION -- MEAN + STD (UNCERTAINTY)

Produces TWO figures, each 5 rows (ISF_a..ISF_e) x 3 columns (Region):
  1. ISF_by_region_5x3_mean.png/pdf -- predicted mean, fertility palette
  2. ISF_by_region_5x3_std.png/pdf  -- predicted std (uncertainty), sequential palette
"""

import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.transforms as mtransforms
import kagglehub

# ===============================================================================
#  CONFIGURATION
# ===============================================================================

BASE_DIR   = '/kaggle/working'
MAPS_ROOT  = '/kaggle/working/Maps/ISF'

# AOI with a 'Region' column -- adjust to your actual file
AOI_PATH   = '/kaggle/input/datasets/antsasarobidyran/area-of-interest/AOI_dissolved.gpkg'
REGION_COL = 'Region'

RESPONSE_ORDER = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']

# Mean + std prediction rasters -- as written by the FIL bootstrap script
PRED_PATHS_MEAN = {
    r: os.path.join(MAPS_ROOT, r, f'{r}_mean.tif') for r in RESPONSE_ORDER
}
PRED_PATHS_STD = {
    r: os.path.join(MAPS_ROOT, r, f'{r}_std.tif') for r in RESPONSE_ORDER
}

OUTPUT_DIR = BASE_DIR
NODATA_OUT = -9999.0

DPI      = 600
N_BINS   = 3
P_LOW    = 2.0
P_HIGH   = 98.0
ROUND_TO = 0.05   # rounding granularity for clean bin-edge labels
DECIMALS = 2      # decimal places used when formatting bin-edge labels

# ===============================================================================
#  FONT (same loading style as the SOC mapping script)
# ===============================================================================

_font_path = kagglehub.dataset_download("hammaadali/arial-font")
_ttf_files = glob.glob(f"{_font_path}/**/*.ttf", recursive=True)
if not _ttf_files:
    raise FileNotFoundError(f"No .ttf found under {_font_path}")
arial_ttf   = _ttf_files[0]
fm.fontManager.addfont(arial_ttf)
font_family = fm.FontProperties(fname=arial_ttf).get_name()
print(f"Font loaded: '{font_family}' from {arial_ttf}")

BASE_FS = 12
plt.rcParams.update({
    'font.family'   : font_family,
    'font.size'     : BASE_FS,
    'axes.labelsize': BASE_FS,
    'axes.titlesize': BASE_FS,
    'axes.linewidth': 0.8,
})

# ===============================================================================
#  TITLES / LABELS
# ===============================================================================

_UNIT_MEAN = r'$\mathregular{(index)}$'
_UNIT_STD  = r'$\mathregular{(index,\ std)}$'

DEPTH_LABELS = {
    'ISF_a': r'$\mathbf{SFI_{0\mathsf{-}10}}$',
    'ISF_b': r'$\mathbf{SFI_{10\mathsf{-}20}}$',
    'ISF_c': r'$\mathbf{SFI_{20\mathsf{-}30}}$',
    'ISF_d': r'$\mathbf{SFI_{30\mathsf{-}60}}$',
    'ISF_e': r'$\mathbf{SFI_{60\mathsf{-}90}}$',
}

# ===============================================================================
#  COLOUR PALETTES
#  Mean  -- fertility semantics: red/orange = low, yellow = moderate, green = high.
#  Std   -- uncertainty semantics: sequential, light = low uncertainty,
#           dark = high uncertainty. NOT diverging, since low/high std isn't
#           "good/bad" the way low/high fertility is.
# ===============================================================================

MEAN_COLORS = ['#D73027', '#FEE08B', '#1A9850']
STD_COLORS  = ['#FEE6CE', '#FD8D3C', '#B30000']

def _make_cmap(name, hex_list):
    c = mcolors.ListedColormap(hex_list, name=name)
    c.set_bad(color='white')
    return c

# ===============================================================================
#  NORM -- TRUE QUANTILE (equal-count) BINNING
# ===============================================================================

def make_clean_norm(data_masked, hex_colors, n_bins,
                     p_low=P_LOW, p_high=P_HIGH, round_to=ROUND_TO,
                     decimals=DECIMALS):
    """
    Quantile-based (equal-count) bin edges.

    Computes n_bins+1 edges as the percentiles evenly spaced between p_low
    and p_high, so each interior bin holds (roughly) the same NUMBER of
    pixels -- much better visual contrast than equal-width bins on skewed
    environmental data. Edges are rounded to `round_to` for clean legend
    labels, and outer edges are extended to the true min/max so every valid
    pixel still gets a colour.
    """
    assert len(hex_colors) == n_bins
    valid = data_masked.compressed()

    pct_points = np.linspace(p_low, p_high, n_bins + 1)
    raw_bounds = np.percentile(valid, pct_points)

    bounds = np.round(raw_bounds / round_to) * round_to

    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + round_to

    bounds[0]  = min(bounds[0],  np.floor(valid.min() / round_to) * round_to)
    bounds[-1] = max(bounds[-1], np.ceil (valid.max() / round_to) * round_to)

    bounds = np.round(bounds, decimals + 4)

    assert len(bounds) - 1 == n_bins, \
        f"Expected {n_bins} bins, got {len(bounds) - 1}"

    cmap_used = _make_cmap('auto', hex_colors)
    norm      = mcolors.BoundaryNorm(bounds, ncolors=n_bins)

    fmt = f'{{:.{decimals}f}}'
    labels = []
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        if i == 0:
            labels.append(f'< {fmt.format(bounds[1])}')
        elif i == n_bins - 1:
            labels.append(f'> {fmt.format(bounds[-2])}')
        else:
            labels.append(f'{fmt.format(lo)}\u2013{fmt.format(hi)}')
    return norm, cmap_used, labels

# ===============================================================================
#  LOAD AOI, GET REGION LIST
# ===============================================================================

print("Loading AOI ...")
aoi = gpd.read_file(AOI_PATH)
assert REGION_COL in aoi.columns, (
    f"'{REGION_COL}' column not found in AOI. Columns present: {list(aoi.columns)}"
)

aoi_regions = aoi.dissolve(by=REGION_COL, as_index=False)
region_names = sorted(aoi_regions[REGION_COL].unique().tolist())
assert len(region_names) == 3, (
    f"Expected exactly 3 regions for a 3-column layout, found {len(region_names)}: "
    f"{region_names}"
)
print(f"Regions found: {region_names}")

# ===============================================================================
#  CROP A RASTER TO ONE REGION GEOMETRY
# ===============================================================================

def crop_raster_to_region(raster_path, region_geom_gdf):
    with rasterio.open(raster_path) as src:
        geom_proj = region_geom_gdf.to_crs(src.crs)
        geoms     = [g.__geo_interface__ for g in geom_proj.geometry]

        out_arr, out_transform = rio_mask(src, geoms, crop=True, nodata=NODATA_OUT)
        out_arr = out_arr[0].astype(np.float32)
        nodata  = src.nodata if src.nodata is not None else NODATA_OUT

    mask = ~np.isfinite(out_arr)
    mask |= np.isclose(out_arr, nodata)
    mask |= np.isclose(out_arr, NODATA_OUT)

    h, w   = out_arr.shape
    left   = out_transform.c
    top    = out_transform.f
    right  = left + out_transform.a * w
    bottom = top  + out_transform.e * h

    data_masked = np.ma.array(out_arr, mask=mask)
    return data_masked, [left, right, bottom, top]

# ===============================================================================
#  DRAW ONE PANEL (same style as the SOC mapping script)
# ===============================================================================

def draw_panel(ax, data, extent, norm, cmap, letter, title, legend_title, labels):
    ax.set_facecolor('white')
    ax.imshow(
        data,
        cmap=cmap, norm=norm, extent=extent,
        origin='upper', interpolation='nearest',
    )
    ax.set_axis_off()

    orig_bbox      = ax.get_position(original=True)
    cell_transform = mtransforms.BboxTransformTo(orig_bbox) + ax.figure.transFigure

    ax.text(0.01, 1.15, letter,
            transform=cell_transform,
            fontsize=BASE_FS + 9, fontweight='bold',
            color='#111111', ha='left', va='top')

    ax.text(0.12, 1.15, title,
            transform=cell_transform,
            fontsize=BASE_FS + 9, fontweight='bold',
            color='#111111', ha='left', va='top',
            linespacing=1.5)

    legend_patches = [
        mpatches.Patch(
            facecolor=list(cmap.colors)[i], edgecolor='#555555',
            linewidth=0.5, label=labels[i],
        )
        for i in range(len(labels))
    ]
    leg = ax.legend(
        handles=legend_patches,
        title=legend_title,
        title_fontsize=BASE_FS + 1,
        fontsize=BASE_FS + 1,
        loc='lower right',
        bbox_to_anchor=(1.5, 0.03),
        bbox_transform=cell_transform,
        frameon=False,
        handlelength=1.5, handleheight=1.0,
        borderpad=0.65, labelspacing=0.25,
    )
    leg.get_title().set_fontweight('bold')
    leg.get_title().set_multialignment('left')

# ===============================================================================
#  LOAD + CROP ALL RASTERS FOR ONE QUANTITY (mean OR std)
# ===============================================================================

def load_and_crop_all(pred_paths, quantity_label):
    print(f"\nCropping {quantity_label} rasters to each region ...")
    data_by_resp_region   = {r: {} for r in RESPONSE_ORDER}
    extent_by_resp_region = {r: {} for r in RESPONSE_ORDER}

    for r in RESPONSE_ORDER:
        raster_path = pred_paths[r]
        for region in region_names:
            region_gdf = aoi_regions[aoi_regions[REGION_COL] == region]
            data_masked, extent = crop_raster_to_region(raster_path, region_gdf)
            data_by_resp_region[r][region]   = data_masked
            extent_by_resp_region[r][region] = extent
            valid = data_masked.compressed()
            if valid.size:
                print(f"  {r} / {region} [{quantity_label}]: n_valid={valid.size:,}  "
                      f"range=[{valid.min():.2f}, {valid.max():.2f}]")
            else:
                print(f"  {r} / {region} [{quantity_label}]: NO VALID PIXELS")

    return data_by_resp_region, extent_by_resp_region

# ===============================================================================
#  BUILD SHARED PER-REGION NORM/CMAP/LABELS FOR ONE QUANTITY
# ===============================================================================

def build_shared_norms(data_by_resp_region, hex_colors, n_bins, quantity_label):
    print(f"\nBuilding shared quantile-based color scale per region ({quantity_label}) ...")
    norms, cmaps, labels_by_region = {}, {}, {}
    for region in region_names:
        combined = np.ma.concatenate(
            [data_by_resp_region[r][region].compressed() for r in RESPONSE_ORDER]
        )
        combined_masked = np.ma.array(combined, mask=np.zeros_like(combined, dtype=bool))
        n, c, l = make_clean_norm(combined_masked, hex_colors, n_bins)
        norms[region], cmaps[region], labels_by_region[region] = n, c, l
        print(f"  {region} bins [{quantity_label}]: {l}")
    return norms, cmaps, labels_by_region

# ===============================================================================
#  BUILD + SAVE A 5x3 FIGURE FOR ONE QUANTITY
# ===============================================================================

def build_figure(data_by_resp_region, extent_by_resp_region, norms, cmaps,
                  labels_by_region, legend_title_fn, out_name, fig_h_per_row=5.5):
    n_rows = len(RESPONSE_ORDER)
    fig_w  = 18.0
    fig_h  = fig_h_per_row * n_rows

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(fig_w, fig_h),
        facecolor='white',
        squeeze=False,
        gridspec_kw={'wspace': 0.45, 'hspace': 0.15},
    )

    letters = [chr(ord('a') + i) + '.' for i in range(n_rows * 3)]

    for row, r in enumerate(RESPONSE_ORDER):
        lbl = DEPTH_LABELS[r]
        row_letters = letters[row * 3:(row + 1) * 3]
        for col, region in enumerate(region_names):
            draw_panel(
                ax           = axes[row, col],
                data         = data_by_resp_region[r][region],
                extent       = extent_by_resp_region[r][region],
                norm         = norms[region],
                cmap         = cmaps[region],
                letter       = row_letters[col],
                title        = f'{lbl}\n{region}',
                legend_title = legend_title_fn(lbl),
                labels       = labels_by_region[region],
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = os.path.join(OUTPUT_DIR, f'{out_name}.{ext}')
        fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"Saved -> {out}")

    plt.show()
    plt.close(fig)

# ===============================================================================
#  RUN -- MEAN FIGURE
# ===============================================================================

print("\n" + "=" * 70)
print("MEAN PREDICTION MAPS")
print("=" * 70)

mean_data, mean_extent = load_and_crop_all(PRED_PATHS_MEAN, 'mean')
mean_norms, mean_cmaps, mean_labels = build_shared_norms(
    mean_data, MEAN_COLORS, N_BINS, 'mean')

build_figure(
    data_by_resp_region   = mean_data,
    extent_by_resp_region = mean_extent,
    norms                 = mean_norms,
    cmaps                 = mean_cmaps,
    labels_by_region      = mean_labels,
    legend_title_fn        = lambda lbl: f'Predicted {lbl}\n{_UNIT_MEAN}',
    out_name              = 'ISF_by_region_5x3_mean',
)

# ===============================================================================
#  RUN -- STD (UNCERTAINTY) FIGURE
# ===============================================================================

print("\n" + "=" * 70)
print("STD (UNCERTAINTY) PREDICTION MAPS")
print("=" * 70)

std_data, std_extent = load_and_crop_all(PRED_PATHS_STD, 'std')
std_norms, std_cmaps, std_labels = build_shared_norms(
    std_data, STD_COLORS, N_BINS, 'std')

build_figure(
    data_by_resp_region   = std_data,
    extent_by_resp_region = std_extent,
    norms                 = std_norms,
    cmaps                 = std_cmaps,
    labels_by_region      = std_labels,
    legend_title_fn        = lambda lbl: f'Uncertainty {lbl}\n{_UNIT_STD}',
    out_name              = 'ISF_by_region_5x3_std',
)

print("\nDone. Both figures written to:", OUTPUT_DIR)
