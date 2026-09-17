"""
ISF DEPTH MAP PREDICTION -- POINT ESTIMATE (NO BOOTSTRAP MEAN/STD)
CPU-KRIGING VARIANT
(RF trend on GPU via FIL, single model, batched  +  single residual kriging
pass on CPU via PyKrige's native execute())

WHAT THIS IS
-------------
Same point-estimate pipeline as the GPU-kriging script:

    final_map = RF_trend(final model)  +  kriged(residuals of final model)

The only difference from the GPU version: the residual kriging step uses
PyKrige's own CPU implementation (OrdinaryKriging.execute(..., backend='C',
n_closest_points=K)) instead of the custom batched GPU kriging. This is
slower for large rasters but removes the custom linear-algebra code path
entirely, which is useful as a cross-check / fallback when a GPU kriging
implementation is unavailable or needs validating against ground truth.

- If the depth's saved bundle (final_rk_model_{response}.joblib) already
  contains the final fitted RandomForestRegressor under bundle['model'],
  that model is used directly (no re-fit).
- If bundle['model'] is absent, a single RF is fit once on the FULL
  training matrix using bundle['best_rf_params'] (Optuna-tuned), with a
  fixed random_state -- deterministic, no bootstrapping.
- Residuals of that one model on the full training set are kriged ONCE
  with PyKrige (CPU), reusing bundle['best_vgm_params'] (Optuna-tuned
  family/nlags/weight).
- No mean/std bands are produced -- only a single output band per depth:
      {response}.tif

OUTPUT LOCATION
----------------
Written to a SEPARATE directory from the GPU-kriging run, so the two sets
of maps never overwrite each other:
  /kaggle/working/Maps/ISF_RK_point_cpuKrige/{response}/{response}.tif

RF trend still runs on GPU via FIL (this part was already validated and is
not the bottleneck) -- only the kriging backend changed.
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
from pykrige.ok import OrdinaryKriging

warnings.filterwarnings('ignore')
print(f"cuML version: {cuml.__version__}")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH   = '/kaggle/working/Tanety/Data_covariates.gpkg'
VIF_DIR     = '/kaggle/working/Tanety/VIF'
RESULTS_DIR = '/kaggle/working/Tanety/RK_Results'
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')

# Separate output root so this run never collides with the GPU-kriging maps
MAPS_ROOT   = '/kaggle/working/Maps/ISF_RK_point_cpuKrige'

RASTER_DIR = '/kaggle/input/datasets/antsasarobidyran/covariates-sarobidy-taneti'

RESPONSES = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']
CATEGORICAL_COVARIATES = ['ESA_WorldCover_v200_30m', 'SoilType_30m']

# UTM 38S projected coordinates (meters) -- named lon/lat in the source file
# but are NOT geographic degrees. The prediction rasters MUST be in the
# same projected CRS/units, since pixel coordinates are compared directly
# against these training coordinates for kriging.
COORD_COLS = ['lon', 'lat']

# Predictors guaranteed present in every depth's VIF-selected set -- used to
# define the valid-pixel extent.
ANCHOR_BANDS = {'Elevation_30m', 'MAP_30m'}

TILE_SIZE          = 3840
NODATA_OUT         = -9999.0
INPAINT_SKIP_RATIO = 0.001

# -- Single-model fit (used only if bundle['model'] is not already saved) ---
FINAL_RANDOM_SEED = 0
RF_N_JOBS         = -1

# -- FIL / GPU (RF trend only -- kriging is CPU in this variant) -------------
FIL_BATCH_SIZE = 500_000
GPU_ID         = 0

# -- Residual kriging (CPU, PyKrige native) ----------------------------------
KRIGE_N_CLOSEST = 32
# PyKrige's execute() takes the full query array at once; we still chunk it
# per-tile-sized batches to keep progress/tracking granular and memory bounded
# on very large tiles. This is a pure knob, not required by PyKrige itself.
KRIGE_CHUNK     = 20_000

os.makedirs(MAPS_ROOT, exist_ok=True)

# Verify GPU (still used for the RF trend / FIL step)
_n_gpus = cp.cuda.runtime.getDeviceCount()
assert GPU_ID < _n_gpus, f"GPU_ID={GPU_ID} but only {_n_gpus} GPU(s) available"
print(f"  GPUs available : {_n_gpus}  |  Using : GPU {GPU_ID}")
with cp.cuda.Device(GPU_ID):
    props = cp.cuda.runtime.getDeviceProperties(GPU_ID)
    free, total = cp.cuda.Device(GPU_ID).mem_info
    print(f"  GPU {GPU_ID}: {props['name'].decode()} | "
          f"free {free/1e9:.1f} / total {total/1e9:.1f} GB")
cp.cuda.Device(GPU_ID).use()

print("  NOTE: point-estimate mode -- single RF trend model (GPU/FIL) + "
      "single kriging pass per depth on CPU via PyKrige (no bootstrap "
      "mean/std).")

# =============================================================================
# CPU KRIGING (PyKrige native execute())
# =============================================================================

def fit_ordinary_kriging(coords, resid, vgm_params):
    """
    Fit a full PyKrige OrdinaryKriging model on CPU, reusing the depth's
    Optuna-tuned family/nlags/weight. Unlike the GPU variant, we keep the
    fitted PyKrige object itself -- prediction also runs through it.

    Returns None on failure (caller should fall back to trend-only).
    """
    try:
        ok_model = OrdinaryKriging(
            coords[:, 0], coords[:, 1], resid,
            variogram_model=vgm_params['variogram_model'],
            nlags=vgm_params['nlags'],
            weight=vgm_params['weight'],
            enable_plotting=False, verbose=False, pseudo_inv=True,
        )
        print(f"  Variogram fitted (CPU/PyKrige): model={ok_model.variogram_model}, "
              f"params={[float(p) for p in ok_model.variogram_model_parameters]}")
        return ok_model
    except Exception as e:
        print(f"  WARNING: variogram fit failed ({e}); "
              f"this depth's map will be trend-only "
              f"(zero residual correction).")
        return None


def krige_predict_chunked(ok_model, xs, ys, chunk=KRIGE_CHUNK, n_closest=KRIGE_N_CLOSEST):
    """
    CPU kriging prediction via PyKrige's own execute(), chunked to keep
    memory bounded on large tiles. Returns a zero array (trend-only
    fallback) if the model is None.
    """
    n = xs.shape[0]
    if ok_model is None:
        return np.zeros(n, dtype=np.float32)

    out = np.zeros(n, dtype=np.float32)
    for p0 in range(0, n, chunk):
        p1 = min(p0 + chunk, n)
        z_hat, _ss = ok_model.execute(
            'points', xs[p0:p1], ys[p0:p1],
            backend='C', n_closest_points=n_closest,
        )
        out[p0:p1] = np.asarray(z_hat, dtype=np.float32)
    return out


def validate_pykrige_cpu():
    """
    Lightweight self-check that fit_ordinary_kriging + krige_predict_chunked
    round-trip sanely on a small synthetic case (PyKrige against itself,
    chunked vs. unchunked) -- mainly to confirm the chunking logic doesn't
    change results before trusting a full run.
    """
    rng = np.random.RandomState(0)
    coords = rng.uniform(0, 1000, size=(300, 2))
    resid = rng.normal(size=300)
    vgm_params = dict(variogram_model='spherical', nlags=6, weight=True)

    ok_model = fit_ordinary_kriging(coords, resid, vgm_params)
    assert ok_model is not None, "variogram fit failed in validation"

    xs_q = rng.uniform(0, 1000, size=50)
    ys_q = rng.uniform(0, 1000, size=50)

    z_unchunked, _ = ok_model.execute('points', xs_q, ys_q, backend='C',
                                       n_closest_points=32)
    z_chunked = krige_predict_chunked(ok_model, xs_q, ys_q, chunk=17, n_closest=32)

    max_abs_diff = float(np.max(np.abs(np.asarray(z_unchunked) - z_chunked)))
    print(f"  [validate_pykrige_cpu] max |unchunked - chunked| = {max_abs_diff:.6g}")
    return max_abs_diff


# =============================================================================
# LOAD TRAINING TABLE (shared across all depths)
# =============================================================================

gdf = gpd.read_file(DATA_PATH)
data = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
print(f"Loaded: {data.shape}  -  {DATA_PATH}")

missing_coords = [c for c in COORD_COLS if c not in data.columns]
if missing_coords:
    raise KeyError(f"Coordinate column(s) not found in data: {missing_coords}")

CATEGORICAL_AVAILABLE = [c for c in CATEGORICAL_COVARIATES if c in data.columns]

vif_final = pd.read_csv(os.path.join(VIF_DIR, 'vif_final.csv'))
NUMERIC_SELECTED = vif_final['predictor'].tolist()
print(f"Numeric predictors (VIF-selected, shared): {len(NUMERIC_SELECTED)}")
print(f"Categorical predictors: {CATEGORICAL_AVAILABLE}")
print(f"Coordinate columns (UTM, meters): {COORD_COLS}")

# =============================================================================
# RASTER LOOKUP
# =============================================================================

def raster_path(name):
    return os.path.join(RASTER_DIR, f'{name}.tif')

ALL_RASTER_NAMES = NUMERIC_SELECTED + CATEGORICAL_AVAILABLE

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

# =============================================================================
# PER-DEPTH CONFIG + BUNDLE
# =============================================================================

def load_depth_config(response):
    bundle_path = os.path.join(MODELS_DIR, f'final_rk_model_{response}.joblib')
    bundle = joblib.load(bundle_path)

    missing_rasters = [n for n in ALL_RASTER_NAMES
                        if not os.path.exists(raster_path(n))]
    if missing_rasters:
        print(f"  WARNING ({response}): raster file(s) not found, "
              f"check RASTER_DIR / naming convention: {missing_rasters}")

    return dict(
        numeric_cols=NUMERIC_SELECTED,
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

    y = sub[response].values.astype(np.float64)
    coords = sub[COORD_COLS].values.astype(np.float64)
    print(f"  Training matrix ({response}): X={X.shape}, y={y.shape}, coords={coords.shape}")
    return X, y, coords

# =============================================================================
# BUILD THE SINGLE POINT-ESTIMATE MODEL: one RF trend (-> FIL, GPU) + one
# residual kriging pass (-> PyKrige, CPU). No bootstrapping.
# =============================================================================

def build_point_estimate_model(bundle, rf_params, vgm_params, X_train, y_train, coords_train):
    t0 = time.time()

    rf_final = bundle.get('model')
    if rf_final is not None:
        print("  Using bundle['model'] (already-fitted final RF) -- no re-fit.")
    else:
        print("  bundle['model'] not found -- fitting a single RF once on the "
              "full training matrix with the tuned hyperparameters.")
        rf_final = RandomForestRegressor(
            **rf_params, random_state=FINAL_RANDOM_SEED, n_jobs=RF_N_JOBS
        )
        rf_final.fit(X_train, y_train)

    resid = y_train - rf_final.predict(X_train)

    with cp.cuda.Device(GPU_ID):
        fil_model = ForestInference.load_from_sklearn(rf_final)

    ok_model = fit_ordinary_kriging(coords_train, resid, vgm_params)

    print(f"  Point-estimate model ready: RF fit/compiled (GPU) + "
          f"PyKrige variogram fitted (CPU) ({time.time()-t0:.1f}s)")
    log_mem('point-estimate model built')
    return fil_model, ok_model

def _fil_predict_raw(fil_model, X_gpu):
    raw = fil_model.predict(X_gpu)
    if hasattr(raw, 'values'):
        raw = raw.values
    if isinstance(raw, np.ndarray):
        raw = cp.asarray(raw)
    return raw.ravel().astype(cp.float32)

def _gpu_predict_single(fil_model, X_cpu):
    n_v = X_cpu.shape[0]
    out = np.empty(n_v, dtype=np.float32)

    with cp.cuda.Device(GPU_ID):
        for p0 in range(0, n_v, FIL_BATCH_SIZE):
            p1 = min(p0 + FIL_BATCH_SIZE, n_v)
            X_gpu = cp.asarray(X_cpu[p0:p1])
            pred = _fil_predict_raw(fil_model, X_gpu)
            out[p0:p1] = cp.asnumpy(pred)
            del X_gpu, pred
            cp.get_default_memory_pool().free_all_blocks()

    return out

def pixel_to_xy(transform, rows, cols):
    """Vectorized pixel-center -> CRS coordinates from a rasterio Affine."""
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return xs, ys

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
# MAIN PIPELINE FOR ONE DEPTH -- single RF trend (GPU) + single kriged
# residual (CPU, PyKrige). Writes ONE band: {response}.tif (point estimate).
# =============================================================================

def rk_predict_point_estimate(response, fil_model, ok_model, numeric_cols,
                               categorical_cols, encoder, numeric_medians,
                               categorical_modes, output_dir):

    print(f'\n{"="*70}')
    print(f'  REGRESSION-KRIGING POINT ESTIMATE (CPU KRIGING) -- {response}')
    print(f'  Numeric rasters  : {numeric_cols}')
    print(f'  Categorical rast.: {categorical_cols}')
    print(f'  Kriging (CPU/PyKrige): n_closest={KRIGE_N_CLOSEST}')
    print(f'{"="*70}')

    all_names = numeric_cols + categorical_cols
    raster_handles = {name: rasterio.open(raster_path(name)) for name in all_names}
    src_nodata = {name: raster_handles[name].nodata for name in all_names}

    ref_name = 'Elevation_30m' if 'Elevation_30m' in numeric_cols else numeric_cols[0]
    ref = raster_handles[ref_name]
    profile, H, W, crs = ref.profile.copy(), ref.height, ref.width, ref.crs
    ref_transform = ref.transform
    print(f'  Raster CRS: {crs}  (must match the UTM coordinate system used '
          f'for training lon/lat)')

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
    PATH_OUT = os.path.join(output_dir, f'{response}.tif')

    tracking_csv = os.path.join(output_dir, f'tile_tracking_{response}_point_cpuKrige.csv')
    trk_fields = ['tile_idx', 'row_start', 'col_start',
                  'n_valid_pixels', 'trend_s', 'krige_s', 'total_s', 'cum_min', 'status']
    trk_file   = open(tracking_csv, 'w', newline='')
    trk_writer = csv.DictWriter(trk_file, fieldnames=trk_fields)
    trk_writer.writeheader()
    trk_file.flush()
    print(f'  Tracking log -> {tracking_csv}')

    log_mem('startup')
    t0 = time.time()

    with rasterio.open(PATH_OUT, 'w', **out_prof) as h_out:

        pbar = tqdm(total=len(tiles), desc=f'{response} tiles (point est., CPU krige)',
                    unit='tile', ncols=95)

        for tidx, rs, cs in tiles:
            t_tile = time.time()
            re = min(rs + TILE_SIZE, H); ce = min(cs + TILE_SIZE, W)
            th = re - rs;                 tw = ce - cs
            win = Window(cs, rs, tw, th)

            out_tile = np.full((th, tw), NODATA_OUT, np.float32)
            n_v      = 0
            trend_s  = 0.0
            krige_s  = 0.0
            status   = 'nodata'

            try:
                X, flat, _, _, _ = _build_X_tile(
                    rs, cs, numeric_cols, categorical_cols, raster_handles,
                    src_nodata, H, W, numeric_medians, categorical_modes, encoder)

                if X is not None:
                    n_v = X.shape[0]

                    r2, c2 = np.unravel_index(flat, (th, tw))
                    global_rows = (rs + r2).astype(np.float64)
                    global_cols = (cs + c2).astype(np.float64)
                    xs, ys = pixel_to_xy(ref_transform, global_rows, global_cols)

                    t_tr = time.time()
                    trend_vals = _gpu_predict_single(fil_model, X)  # (n_v,)
                    trend_s = time.time() - t_tr

                    t_kr = time.time()
                    krig_resid = krige_predict_chunked(ok_model, xs, ys)
                    krige_s = time.time() - t_kr

                    combined = (trend_vals.astype(np.float64) + krig_resid).astype(np.float32)

                    out_tile[r2, c2] = combined
                    del X, trend_vals, krig_resid, combined, r2, c2, flat, xs, ys
                    status = 'ok'

            except Exception as e:
                status = f'error:{e}'

            h_out.write(out_tile, 1, window=win)
            del out_tile
            gc.collect()

            trk_writer.writerow(dict(
                tile_idx=tidx, row_start=rs, col_start=cs,
                n_valid_pixels=n_v, trend_s=f'{trend_s:.2f}', krige_s=f'{krige_s:.2f}',
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
    if os.path.exists(PATH_OUT):
        print(f'  {os.path.basename(PATH_OUT):<40s}  {os.path.getsize(PATH_OUT)/1e6:.1f} MB')
    print(f'  Tracking -> {tracking_csv}')

# =============================================================================
# MAIN -- loop over all five depths
# =============================================================================

def main():
    warmup_numba()

    # One-time sanity check that chunked CPU kriging agrees with PyKrige's
    # own unchunked execute() on a small synthetic case. Comment out once
    # verified.
    validate_pykrige_cpu()

    for response in RESPONSES:
        print(f"\n{'='*70}")
        print(f"  DEPTH: {response}  (point estimate, CPU kriging)")
        print(f"{'='*70}")

        cfg = load_depth_config(response)
        numeric_cols     = cfg['numeric_cols']
        categorical_cols = cfg['categorical_cols']
        bundle           = cfg['bundle']

        rf_params = dict(bundle['best_rf_params'])
        for k in ('random_state', 'n_jobs'):
            rf_params.pop(k, None)
        print(f"  Tuned n_estimators: {bundle['best_rf_params'].get('n_estimators')}")

        vgm_params = bundle['best_vgm_params']
        print(f"  Reusing tuned variogram: {vgm_params}")

        X_train, y_train, coords_train = build_training_matrix(
            response, numeric_cols, categorical_cols, bundle)

        fil_model, ok_model = build_point_estimate_model(
            bundle, rf_params, vgm_params, X_train, y_train, coords_train)
        del X_train, y_train, coords_train
        gc.collect()

        output_dir = os.path.join(MAPS_ROOT, response)
        log_mem(f'before point-estimate prediction ({response})')
        rk_predict_point_estimate(
            response, fil_model, ok_model, numeric_cols, categorical_cols,
            bundle['encoder'], bundle['numeric_medians'],
            bundle['categorical_modes'], output_dir)

        del fil_model, ok_model, bundle
        with cp.cuda.Device(GPU_ID):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

    print(f"\nAll depth point-estimate maps (CPU kriging) written under: {MAPS_ROOT}")


if __name__ == '__main__':
    main()
