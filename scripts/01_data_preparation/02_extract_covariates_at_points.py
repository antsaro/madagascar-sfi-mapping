"""
Build the final covariate table used by every downstream stage.

- Auto-discovers every raster in the covariate directory as a predictor.
- Samples each raster at every soil-profile point (parallelized).
- Assembles SFI responses (ISF_a..ISF_e) + covariates + lat/lon + geometry
  into a single GeoPackage: Data_covariates.gpkg.
"""

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from joblib import Parallel, delayed

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
ROI_PATH    = '/kaggle/input/datasets/antsasarobidyran/area-of-interest/AOI_dissolved.gpkg'
DATA_PATH   = '/kaggle/input/datasets/antsasarobidyran/data-sfi-nirs/Data_SFIv2.gpkg'
CROPPED_DIR = '/kaggle/input/datasets/antsasarobidyran/covariates-sarobidy-taneti'
FINAL_GPKG  = '/kaggle/working/Tanety/Data_covariates.gpkg'

RESULTS_DIR    = '/kaggle/working/Tanety/ML_Results'
PREDICTION_DIR = '/kaggle/working/Tanety/predictions'

os.makedirs(os.path.dirname(FINAL_GPKG), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# COVARIATES: auto-discovered from every raster in CROPPED_DIR
# -----------------------------------------------------------------------
covariate_paths = sorted(
    glob.glob(os.path.join(CROPPED_DIR, '*.tif')) +
    glob.glob(os.path.join(CROPPED_DIR, '*.tiff'))
)
if not covariate_paths:
    raise FileNotFoundError(f"No raster files found in {CROPPED_DIR}")

COVARIATE_PATHS = {os.path.splitext(os.path.basename(p))[0]: p for p in covariate_paths}
COVARIATE_NAMES = list(COVARIATE_PATHS.keys())

# Categorical covariates still need to be flagged manually, auto-discovery
# can't tell a class-code raster from a continuous one by filename alone.
CATEGORICAL_COVARIATES = ['ESA_WorldCover_v100_30m', 'ESA_WorldCover_v200_30m', 'SoilType_30m']

print(f"Discovered {len(COVARIATE_NAMES)} covariates in {CROPPED_DIR}:")
for name in COVARIATE_NAMES:
    print(f"  - {name}")

missing_categorical = [c for c in CATEGORICAL_COVARIATES if c not in COVARIATE_NAMES]
if missing_categorical:
    print(f"Warning: categorical covariate(s) not found among discovered rasters: {missing_categorical}")

# SFI depth layers: a=0-10cm, b=10-20cm, c=20-30cm, d=30-60cm, e=60-90cm
RESPONSES = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']

N_JOBS_CPU    = -1
N_FOLDS_OUTER = 5
N_FOLDS_INNER = 3
OPTUNA_TRIALS_CV    = 50
OPTUNA_TRIALS_FINAL = 50
SEED = 123
TILE_SIZE  = 2048
NODATA_OUT = -9999.0

print("Config ready.")

# -----------------------------------------------------------------------
# LOAD ROI AND SFI DATA POINTS
# -----------------------------------------------------------------------
roi  = gpd.read_file(ROI_PATH)
data = gpd.read_file(DATA_PATH)

print(f"Loaded {len(data)} points from {DATA_PATH}")
print(list(data.columns))
print(data.head())

fig, ax = plt.subplots(figsize=(7, 7))
roi.plot(ax=ax, facecolor='grey', edgecolor='black', linewidth=0.5)
data.plot(ax=ax, color='red', markersize=15, alpha=0.7)
ax.set_title('Data points over ROI')
ax.set_axis_off()
plt.show()

# -----------------------------------------------------------------------
# EXTRACT COVARIATE VALUES AT POINTS (parallel)
# -----------------------------------------------------------------------
def _extract_one_raster(cropped_path, data_gdf):
    name = os.path.splitext(os.path.basename(cropped_path))[0]
    with rasterio.open(cropped_path) as src:
        pts_proj = data_gdf.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts_proj.geometry]
        vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float64)
        nodata = src.nodata
        if nodata is not None:
            vals[np.isclose(vals, nodata)] = np.nan
    return name, vals

print("\nExtracting covariate values at data points (parallel) ...")
extracted = Parallel(n_jobs=N_JOBS_CPU)(
    delayed(_extract_one_raster)(p, data) for p in covariate_paths
)

extracted_df = pd.DataFrame({name: vals for name, vals in extracted})
assert len(extracted_df) == len(data), "Mismatch between extracted rows and input points"

data_extracted = data.reset_index(drop=True).copy()
for col in extracted_df.columns:
    data_extracted[col] = extracted_df[col].values

print(data_extracted.head())

# -----------------------------------------------------------------------
# SUMMARY STATS FOR EXTRACTED COVARIATES
# -----------------------------------------------------------------------
summary_table = (
    extracted_df
    .melt(var_name='variable', value_name='value')
    .groupby('variable')['value']
    .agg(min='min', max='max', mean='mean', median='median', std='std')
    .reset_index()
)
summary_table['cv'] = summary_table['std'] / summary_table['mean'] * 100

print("\nCovariate summary statistics:")
print(summary_table)

# -----------------------------------------------------------------------
# FINAL TABLE: SFI responses + covariates + lat/lon + geometry
# -----------------------------------------------------------------------
keep_cols = RESPONSES + COVARIATE_NAMES + ['lat', 'lon', 'geometry']
missing_final_cols = [c for c in keep_cols if c not in data_extracted.columns]
if missing_final_cols:
    print(f"Warning: columns not found and dropped from final table: {missing_final_cols}")

data_final = data_extracted[[c for c in keep_cols if c in data_extracted.columns]]
print(data_final.head())

data_final.to_file(FINAL_GPKG, driver='GPKG')
print(f"\nSaved final covariate table -> {FINAL_GPKG}")
