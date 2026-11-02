"""
ISF DEPTH MAP PREDICTION -- BOOTSTRAP MEAN + STD (SINGLE GPU, BATCHED STREAMS)

For each depth response (ISF_a..ISF_e):
  - Uses ONLY that depth's VIF-selected numeric rasters (vif_final_{response}.csv)
    plus the two categorical rasters (ESA_WorldCover_v200_30m, SoilType_30m).
  - Re-fits N_BOOTSTRAP RandomForestRegressor models on bootstrap resamples of
    that depth's training matrix (built with the SAME medians/modes/encoder
    stored in the depth's final_model_{response}.joblib bundle), compiles each
    to FIL, and streams tiled, batched GPU prediction across the raster grid.

Output:
  /kaggle/working/Maps/ISF/{response}/{response}_mean.tif
  /kaggle/working/Maps/ISF/{response}/{response}_std.tif
"""

import os, gc, time, csv, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import rasterio
from rasterio.windows import Window
from numba import njit, prange
from tqdm import tqdm
import cupy as cp
import cuml
from cuml.fil import ForestInference
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
print(f"cuML version: {cuml.__version__}")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH   = '/kaggle/working/Tanety/Data_covariates.gpkg'
VIF_DIR     = '/kaggle/working/Tanety/VIF'
RESULTS_DIR = '/kaggle/working/Tanety/ML_Results'
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')
MAPS_ROOT   = '/kaggle/working/Maps/ISF'

# Raster stack directory -- ASSUMES filenames match predictor column names
# exactly, e.g. RASTER_DIR/Elevation_30m.tif, RASTER_DIR/S2_NDVI_30m.tif ...
# Edit raster_path() below if your naming convention differs.
RASTER_DIR = '/kaggle/working/Tanety/Rasters'

RESPONSES = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']
CATEGORICAL_COVARIATES = ['ESA_WorldCover_v200_30m', 'SoilType_30m']

# Full candidate numeric predictor pool (same list the VIF script drew from) --
# used only to build the RASTERS lookup; the *actual* predictors used per
# depth come from that depth's vif_final_{response}.csv.
NUMERIC_PREDICTORS = [
    'CanopyHeight_30m', 'Elevation_30m', 'MAP_30m', 'MAT_30m', 'NPP_30m',
    'S2_B2_30m', 'S2_B3_30m', 'S2_B4_30m', 'S2_B5_30m', 'S2_B6_30m',
    'S2_B8_30m', 'S2_B11_30m', 'S2_B12_30m', 'S2_BSI_30m', 'S2_CIre_30m',
    'S2_ClayIndex_30m', 'S2_GRVI_30m', 'S2_MNDWI_30m', 'S2_NBR_30m',
    'S2_NDVI_30m', 'S2_NIRI_30m', 'S2_SAVI_30m', 'Slope_30m', 'TPI_30m',
    'TWI_30m', 'TreeCover_30m',
]

# Predictors guaranteed present in every depth's VIF-selected set (per the
# VIF script's ALWAYS_KEEP rule) -- used to define the valid-pixel extent.
ANCHOR_BANDS = {'Elevation_30m', 'MAP_30m'}

TILE_SIZE          = 3840
NODATA_OUT         = -9999.0
INPAINT_SKIP_RATIO = 0.001

# -- Bootstrap ensemble -------------------------------------------------------
N_BOOTSTRAP            = 30
BATCH_MODELS           = 30
BOOTSTRAP_N_ESTIMATORS = None   # None = reuse each depth's tuned n_estimators
BOOTSTRAP_RANDOM_SEED  = 0
RF_N_JOBS              = -1

# -- FIL / GPU ----------------------------------------------------------------
FIL_BATCH_SIZE = 500_000
GPU_ID         = 0

os.makedirs(MAPS_ROOT, exist_ok=True)

# Verify GPU
_n_gpus = cp.cuda.runtime.getDeviceCount()
assert GPU_ID < _n_gpus, f"GPU_ID={GPU_ID} but only {_n_gpus} GPU(s) available"
print(f"  GPUs available : {_n_gpus}  |  Using : GPU {GPU_ID}")
with cp.cuda.Device(GPU_ID):
    props = cp.cuda.runtime.getDeviceProperties(GPU_ID)
    free, total = cp.cuda.Device(GPU_ID).mem_info
    print(f"  GPU {GPU_ID}: {props['name'].decode()} | "
          f"free {free/1e9:.1f} / total {total/1e9:.1f} GB")
cp.cuda.Device(GPU_ID).use()

# =============================================================================
# LOAD TRAINING TABLE (shared across all depths)
# =============================================================================

gdf = gpd.read_file(DATA_PATH)
data = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
print(f"Loaded: {data.shape}  -  {DATA_PATH}")

CATEGORICAL_AVAILABLE = [c for c in CATEGORICAL_COVARIATES if c in data.columns]

# =============================================================================
# RASTER LOOKUP
# =============================================================================

def raster_path(name):
    return os.path.join(RASTER_DIR, f'{name}.tif')

ALL_RASTER_NAMES = NUMERIC_PREDICTORS + CATEGORICAL_AVAILABLE

# =============================================================================
# HELPERS
# =============================================================================

def log_mem(label=''):
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e9
        msg = f"  [MEM {label}] CPU {rss:.2f} GB"
        with cp.cuda.Device(GPU_ID):
            free, total = cp.cuda.Device(GPU_ID).mem_info
            msg += f" | GPU{GPU_ID} {(total-free)/1e9:.2f}/{total/1e9:.2f} GB"
        print(msg)
    except Exception:
        pass

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

# =============================================================================
# PER-DEPTH CONFIG + BUNDLE
# =============================================================================

def load_depth_config(response):
    vif_csv = os.path.join(VIF_DIR, f'vif_final_{response}.csv')
    numeric_cols = pd.read_csv(vif_csv)['predictor'].tolist()

    bundle_path = os.path.join(MODELS_DIR, f'final_model_{response}.joblib')
    bundle = joblib.load(bundle_path)

    missing_rasters = [n for n in numeric_cols + CATEGORICAL_AVAILABLE
                        if not os.path.exists(raster_path(n))]
    if missing_rasters:
        print(f"  WARNING ({response}): raster file(s) not found, "
              f"check RASTER_DIR / naming convention: {missing_rasters}")

    return dict(
        numeric_cols=numeric_cols,
        categorical_cols=CATEGORICAL_AVAILABLE,
        bundle=bundle,
    )

# =============================================================================
# BUILD TRAINING MATRIX FOR ONE DEPTH (leakage-safe: uses the bundle's
# already-fitted medians / modes / encoder, exactly as at training time)
# =============================================================================

def build_training_matrix(response, numeric_cols, categorical_cols, bundle):
    sub = data.dropna(subset=[response]).reset_index(drop=True)

    X_num = sub[numeric_cols].fillna(bundle['numeric_medians']).values.astype(np.float32)

    if categorical_cols and bundle.get('encoder') is not None:
        X_cat_raw = sub[categorical_cols].fillna(bundle['categorical_modes'])
        X_cat = bundle['encoder'].transform(X_cat_raw).astype(np.float32)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num

    y = sub[response].values.astype(np.float32)
    print(f"  Training matrix ({response}): X={X.shape}, y={y.shape}")
    return X, y

# =============================================================================
# BUILD + FIL-COMPILE BOOTSTRAP ENSEMBLE (no scaler -- RF trained on raw
# imputed / one-hot-encoded features, same convention as train_final_model)
# =============================================================================

def build_bootstrap_fil_ensemble(bootstrap_params, X_train, y_train):
    n_train = X_train.shape[0]
    fil_models = []

    for b in range(N_BOOTSTRAP):
        t0  = time.time()
        rng = np.random.RandomState(BOOTSTRAP_RANDOM_SEED + b)
        idx = rng.randint(0, n_train, size=n_train)
        X_b, y_b = X_train[idx], y_train[idx]

        rf_b = RandomForestRegressor(
            **bootstrap_params, random_state=BOOTSTRAP_RANDOM_SEED + b,
            n_jobs=RF_N_JOBS
        )
        rf_b.fit(X_b, y_b)

        with cp.cuda.Device(GPU_ID):
            fil_b = ForestInference.load_from_sklearn(rf_b)
        fil_models.append(fil_b)

        del rf_b, X_b, y_b
        gc.collect()

        print(f"  [Bootstrap {b+1}/{N_BOOTSTRAP}] fit + FIL-compiled "
              f"({time.time()-t0:.1f}s)")
        if (b + 1) % 10 == 0:
            log_mem(f'after {b+1} bootstrap models')

    log_mem('all bootstrap models compiled')
    return fil_models

def _fil_predict_raw(fil_model, X_gpu):
    raw = fil_model.predict(X_gpu)
    if hasattr(raw, 'values'):
        raw = raw.values
    if isinstance(raw, np.ndarray):
        raw = cp.asarray(raw)
    return raw.ravel().astype(cp.float32)

# =============================================================================
# BATCHED, STREAM-CONCURRENT PREDICTION FOR A GROUP OF MODELS
# =============================================================================

def _gpu_predict_group(models_group, X_cpu):
    n_v = X_cpu.shape[0]
    n_m = len(models_group)
    out = np.empty((n_m, n_v), dtype=np.float32)

    with cp.cuda.Device(GPU_ID):
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_m)]

        for p0 in range(0, n_v, FIL_BATCH_SIZE):
            p1 = min(p0 + FIL_BATCH_SIZE, n_v)
            X_gpu = cp.asarray(X_cpu[p0:p1])

            results = [None] * n_m
            for mi, (model, stream) in enumerate(zip(models_group, streams)):
                with stream:
                    results[mi] = _fil_predict_raw(model, X_gpu)

            for stream in streams:
                stream.synchronize()

            for mi in range(n_m):
                out[mi, p0:p1] = cp.asnumpy(results[mi])

            del X_gpu, results
            cp.get_default_memory_pool().free_all_blocks()

    return out

# =============================================================================
# NUMBA INPAINTING -- 4-pass nearest-neighbour (numeric bands only)
# =============================================================================

@njit(parallel=True, fastmath=True)
def _inpaint_numba(flat_arr, mask_flat, H, W, fallback):
    out = flat_arr.copy()
    for r in prange(H):
        base = r * W; last = fallback
        for c in range(W):
            i = base + c
            if not mask_flat[i]: last = out[i]
            else: out[i] = last
    for r in prange(H):
        base = r * W; last = fallback
        for c in range(W - 1, -1, -1):
            i = base + c
            if not mask_flat[i]: last = out[i]
            elif out[i] == fallback: out[i] = last
    for c in prange(W):
        last = fallback
        for r in range(H):
            i = r * W + c
            if not mask_flat[i]: last = out[i]
            elif out[i] == fallback: out[i] = last
    for c in prange(W):
        last = fallback
        for r in range(H - 1, -1, -1):
            i = r * W + c
            if not mask_flat[i]: last = out[i]
            elif out[i] == fallback: out[i] = last
    return out

def inpaint_band(arr2d, mask2d, fallback):
    n_miss = int(mask2d.sum())
    if n_miss == 0:
        return arr2d
    fb = np.float32(fallback)
    if n_miss / arr2d.size < INPAINT_SKIP_RATIO:
        arr2d[mask2d] = fb
        return arr2d
    H, W   = arr2d.shape
    flat   = arr2d.ravel()
    flat_m = mask2d.ravel()
    flat[flat_m] = fb
    result = _inpaint_numba(flat, flat_m, H, W, fb)
    np.copyto(arr2d, result.reshape(H, W))
    return arr2d

def warmup_numba():
    print("  Warming up Numba JIT ...", end=' ', flush=True)
    t0 = time.time()
    _a = np.zeros(64 * 64, dtype=np.float32)
    _m = np.zeros(64 * 64, dtype=np.bool_)
    _inpaint_numba(_a, _m, 64, 64, np.float32(0.0))
    print(f"done ({time.time()-t0:.1f}s)")

# =============================================================================
# TILE BUILDER -- numeric (inpainted) + categorical (mode-filled, one-hot)
# =============================================================================

def _build_X_tile(rs, cs, numeric_cols, categorical_cols, raster_handles,
                   src_nodata, H, W, numeric_medians, categorical_modes, encoder):
    re = min(rs + TILE_SIZE, H);  ce = min(cs + TILE_SIZE, W)
    th = re - rs;                  tw = ce - cs
    win = Window(cs, rs, tw, th)

    num_bands, cat_bands = {}, {}
    valid = np.ones((th, tw), dtype=bool)

    for name in numeric_cols:
        src = raster_handles[name]
        arr = src.read(1, window=win).astype(np.float32)
        if arr.shape != (th, tw):
            tmp = np.full((th, tw), np.nan, np.float32)
            tmp[:arr.shape[0], :arr.shape[1]] = arr
            arr = tmp
        nd  = src_nodata[name]
        msk = ~np.isfinite(arr)
        if nd is not None:
            msk |= np.isclose(arr, nd)
        if name in ANCHOR_BANDS:
            valid &= ~msk
        else:
            if msk.any():
                inpaint_band(arr, msk, numeric_medians[name])
        num_bands[name] = arr

    for name in categorical_cols:
        src = raster_handles[name]
        arr = src.read(1, window=win).astype(np.float32)
        if arr.shape != (th, tw):
            tmp = np.full((th, tw), np.nan, np.float32)
            tmp[:arr.shape[0], :arr.shape[1]] = arr
            arr = tmp
        nd  = src_nodata[name]
        msk = ~np.isfinite(arr)
        if nd is not None:
            msk |= np.isclose(arr, nd)
        if msk.any():
            arr[msk] = float(categorical_modes[name])
        cat_bands[name] = arr

    n_v = int(valid.sum())
    if n_v == 0:
        return None, None, th, tw, win

    flat = np.where(valid.ravel())[0].astype(np.int64)

    num_cols = [num_bands[name].ravel()[flat].copy() for name in numeric_cols]
    num_bands.clear()
    X_num = np.column_stack(num_cols).astype(np.float32) if num_cols else \
        np.empty((len(flat), 0), dtype=np.float32)

    if categorical_cols:
        cat_df = pd.DataFrame({
            name: cat_bands[name].ravel()[flat].round().astype(
                categorical_modes[name].__class__ if np.isscalar(categorical_modes[name])
                else np.int64)
            for name in categorical_cols
        })
        cat_bands.clear()
        X_cat = encoder.transform(cat_df).astype(np.float32)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num

    return X, flat, th, tw, win

# =============================================================================
# MAIN PIPELINE FOR ONE DEPTH -- mean + std written together
# =============================================================================

def fil_predict_mean_std_batched(response, fil_models, numeric_cols,
                                  categorical_cols, encoder, numeric_medians,
                                  categorical_modes, output_dir):

    print(f'\n{"="*70}')
    print(f'  FIL BOOTSTRAP MEAN + STD (BATCHED STREAMS) -- {response}')
    print(f'  Bootstrap models : {len(fil_models)}')
    print(f'  Numeric rasters  : {numeric_cols}')
    print(f'  Categorical rast.: {categorical_cols}')
    print(f'{"="*70}')

    all_names = numeric_cols + categorical_cols
    raster_handles = {name: rasterio.open(raster_path(name)) for name in all_names}
    src_nodata = {name: raster_handles[name].nodata for name in all_names}

    ref_name = 'Elevation_30m' if 'Elevation_30m' in numeric_cols else numeric_cols[0]
    ref = raster_handles[ref_name]
    profile, H, W, crs = ref.profile.copy(), ref.height, ref.width, ref.crs

    row_starts = list(range(0, H, TILE_SIZE))
    col_starts = list(range(0, W, TILE_SIZE))
    n_cols_t   = len(col_starts)
    tiles = [(ti * n_cols_t + tj + 1, row_starts[ti], col_starts[tj])
             for ti in range(len(row_starts))
             for tj in range(len(col_starts))]
    print(f'  Grid {H}x{W}  |  {len(tiles)} tiles ({TILE_SIZE}px)  |  CRS: {crs}')

    out_prof = profile.copy()
    out_prof.update(dtype='float32', count=1, nodata=NODATA_OUT,
                     compress='lzw', predictor=2,
                     tiled=True, blockxsize=512, blockysize=512, bigtiff='YES')

    os.makedirs(output_dir, exist_ok=True)
    PATH_MEAN = os.path.join(output_dir, f'{response}_mean.tif')
    PATH_STD  = os.path.join(output_dir, f'{response}_std.tif')

    tracking_csv = os.path.join(output_dir, f'tile_tracking_{response}_mean_std.csv')
    trk_fields = ['tile_idx', 'row_start', 'col_start',
                  'n_valid_pixels', 'infer_s', 'total_s', 'cum_min', 'status']
    trk_file   = open(tracking_csv, 'w', newline='')
    trk_writer = csv.DictWriter(trk_file, fieldnames=trk_fields)
    trk_writer.writeheader()
    trk_file.flush()
    print(f'  Tracking log -> {tracking_csv}')

    log_mem('startup')
    t0 = time.time()
    B  = len(fil_models)
    model_groups = list(chunked(fil_models, BATCH_MODELS))
    print(f'  {len(model_groups)} group(s) of up to {BATCH_MODELS} models each')

    with rasterio.open(PATH_MEAN, 'w', **out_prof) as h_mean, \
         rasterio.open(PATH_STD,  'w', **out_prof) as h_std:

        pbar = tqdm(total=len(tiles), desc=f'{response} tiles (batched)',
                    unit='tile', ncols=95)

        for tidx, rs, cs in tiles:
            t_tile = time.time()
            re = min(rs + TILE_SIZE, H); ce = min(cs + TILE_SIZE, W)
            th = re - rs;                 tw = ce - cs
            win = Window(cs, rs, tw, th)

            mean_tile = np.full((th, tw), NODATA_OUT, np.float32)
            std_tile  = np.full((th, tw), NODATA_OUT, np.float32)
            n_v       = 0
            infer_s   = 0.0
            status    = 'nodata'

            try:
                X, flat, _, _, _ = _build_X_tile(
                    rs, cs, numeric_cols, categorical_cols, raster_handles,
                    src_nodata, H, W, numeric_medians, categorical_modes, encoder)

                if X is not None:
                    n_v = X.shape[0]

                    running_sum   = np.zeros(n_v, dtype=np.float64)
                    running_sumsq = np.zeros(n_v, dtype=np.float64)

                    t_i = time.time()
                    for group in model_groups:
                        preds_group = _gpu_predict_group(group, X)
                        running_sum   += preds_group.sum(axis=0, dtype=np.float64)
                        running_sumsq += np.square(
                            preds_group, dtype=np.float64
                        ).sum(axis=0)
                        del preds_group
                    infer_s = time.time() - t_i
                    del X

                    mean_v   = running_sum / B
                    variance = (running_sumsq - B * mean_v * mean_v) / max(B - 1, 1)
                    std_vals  = np.sqrt(np.clip(variance, 0.0, None)).astype(np.float32)
                    mean_vals = mean_v.astype(np.float32)
                    del running_sum, running_sumsq, mean_v, variance

                    r2, c2 = np.unravel_index(flat, (th, tw))
                    mean_tile[r2, c2] = mean_vals
                    std_tile[r2, c2]  = std_vals
                    del mean_vals, std_vals, r2, c2, flat
                    status = 'ok'

            except Exception as e:
                status = f'error:{e}'

            h_mean.write(mean_tile, 1, window=win)
            h_std.write(std_tile, 1, window=win)
            del mean_tile, std_tile
            gc.collect()

            trk_writer.writerow(dict(
                tile_idx=tidx, row_start=rs, col_start=cs,
                n_valid_pixels=n_v, infer_s=f'{infer_s:.2f}',
                total_s=f'{time.time()-t_tile:.2f}',
                cum_min=f'{(time.time()-t0)/60:.2f}', status=status,
            ))
            trk_file.flush()
            pbar.update(1)
            if status not in ('ok', 'nodata'):
                tqdm.write(f"  ERROR tile {tidx}: {status}")

        pbar.close()

    trk_file.close()
    for h in raster_handles.values():
        h.close()

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed/60:.1f} min')
    log_mem('done')
    for p in (PATH_MEAN, PATH_STD):
        if os.path.exists(p):
            print(f'  {os.path.basename(p):<40s}  {os.path.getsize(p)/1e6:.1f} MB')
    print(f'  Tracking -> {tracking_csv}')

# =============================================================================
# MAIN -- loop over all five depths
# =============================================================================

def main():
    warmup_numba()

    for response in RESPONSES:
        print(f"\n{'='*70}")
        print(f"  DEPTH: {response}")
        print(f"{'='*70}")

        cfg = load_depth_config(response)
        numeric_cols     = cfg['numeric_cols']
        categorical_cols = cfg['categorical_cols']
        bundle           = cfg['bundle']

        best_params = dict(bundle['best_params'])
        for k in ('random_state', 'n_jobs'):
            best_params.pop(k, None)
        if BOOTSTRAP_N_ESTIMATORS is not None:
            best_params['n_estimators'] = BOOTSTRAP_N_ESTIMATORS
        print(f"  Tuned n_estimators: {bundle['best_params'].get('n_estimators')}  "
              f"| Bootstrap n_estimators: {best_params.get('n_estimators')}")

        X_train, y_train = build_training_matrix(
            response, numeric_cols, categorical_cols, bundle)

        print(f"\nFitting + FIL-compiling {N_BOOTSTRAP} bootstrap models "
              f"({response}) ...")
        fil_models = build_bootstrap_fil_ensemble(best_params, X_train, y_train)
        del X_train, y_train
        gc.collect()

        output_dir = os.path.join(MAPS_ROOT, response)
        log_mem(f'before batched prediction ({response})')
        fil_predict_mean_std_batched(
            response, fil_models, numeric_cols, categorical_cols,
            bundle['encoder'], bundle['numeric_medians'],
            bundle['categorical_modes'], output_dir)

        del fil_models, bundle
        with cp.cuda.Device(GPU_ID):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

    print(f"\nAll depth maps written under: {MAPS_ROOT}")


if __name__ == '__main__':
    main()
