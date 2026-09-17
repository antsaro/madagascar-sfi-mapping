"""
ISF DEPTH MAP PREDICTION -- POINT ESTIMATE (NO BOOTSTRAP MEAN/STD)
(RF trend on GPU via FIL, single model, batched  +  single residual kriging
pass on GPU: custom batched local ordinary kriging, local/moving-window
kriging for scalability)

WHAT THIS IS
-------------
A simplified version of the bootstrap regression-kriging pipeline: instead
of re-fitting N_BOOTSTRAP RF models + N_BOOTSTRAP kriging refits and
averaging, this script produces ONE point-estimate prediction per depth:

    final_map = RF_trend(final model)  +  kriged(residuals of final model)

- If the depth's saved bundle (final_rk_model_{response}.joblib) already
  contains the final fitted RandomForestRegressor under bundle['model'],
  that model is used directly (no re-fit -- this matches whatever was
  reported as the depth's final trained model at training time).
- If bundle['model'] is absent, a single RF is fit once on the FULL
  training matrix using bundle['best_rf_params'] (Optuna-tuned), with a
  fixed random_state -- deterministic, no bootstrapping.
- Residuals of that one model on the full training set are kriged ONCE,
  reusing bundle['best_vgm_params'] (Optuna-tuned family/nlags/weight),
  fit fresh on those residuals (same convention as the bootstrap script,
  just without the resampling).
- No mean/std bands are produced -- only a single output band per depth:
      {response}.tif

GPU KRIGING
------------
Same approach as the bootstrap script: PyKrige only fits the variogram
(nugget/sill/range) on CPU (cheap, done once); the actual per-pixel
prediction (KNN search + batched (K+1)x(K+1) linear solves) runs on GPU via
cuml.neighbors.NearestNeighbors + cupy.linalg.solve. Supported variogram
models: linear, power, gaussian, exponential, spherical (no hole-effect).

VALIDATE FIRST: run `validate_gpu_kriging()` (near the top, called once in
main()) before trusting this on a full run -- it compares the GPU kriging
implementation against PyKrige's CPU execute() on a small synthetic case
and prints the max absolute difference (should be small).

Output:
  /kaggle/working/Maps/ISF_RK_point/{response}/{response}.tif
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
from cuml.neighbors import NearestNeighbors as cuNN
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
MAPS_ROOT   = '/kaggle/working/Maps/ISF_RK_point'

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

# -- FIL / GPU (RF trend) -----------------------------------------------------
FIL_BATCH_SIZE = 500_000
GPU_ID         = 0

# -- Residual kriging (GPU) ---------------------------------------------------
KRIGE_N_CLOSEST = 32
KRIGE_CHUNK     = 20_000   # pixels per GPU batch, memory/latency knob

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

print("  NOTE: point-estimate mode -- single RF trend model + single "
      "kriging pass per depth (no bootstrap mean/std).")

# =============================================================================
# GPU LOCAL ORDINARY KRIGING
# =============================================================================

def _variogram_gpu(model, params, h):
    """
    h: cupy array, any shape, distances >= 0.
    params: tuple of floats, same order PyKrige stores in
            ok_model.variogram_model_parameters for that model.
    Returns semivariance gamma(h), same shape as h.
    """
    if model == 'linear':
        slope, nugget = params
        return slope * h + nugget

    elif model == 'power':
        scale, exponent, nugget = params
        return scale * cp.power(h, exponent) + nugget

    elif model == 'gaussian':
        psill, rng, nugget = params
        return psill * (1.0 - cp.exp(-(h ** 2.0) / (rng * 4.0 / 7.0) ** 2.0)) + nugget

    elif model == 'exponential':
        psill, rng, nugget = params
        return psill * (1.0 - cp.exp(-h / (rng / 3.0))) + nugget

    elif model == 'spherical':
        psill, rng, nugget = params
        inside = psill * ((3.0 * h) / (2.0 * rng) - (h ** 3.0) / (2.0 * rng ** 3.0)) + nugget
        outside = psill + nugget
        return cp.where(h <= rng, inside, outside)

    else:
        raise ValueError(
            f"Unsupported variogram model for GPU kriging: {model!r}. "
            f"Supported: linear, power, gaussian, exponential, spherical."
        )


def fit_variogram_only(coords, resid, vgm_params):
    """
    Fit ONLY the variogram (nugget/sill/range) on CPU via PyKrige, reusing
    the depth's Optuna-tuned family/nlags/weight, then return
    (model_name, params_tuple). We do NOT keep PyKrige's fitted
    OrdinaryKriging object for prediction -- that part runs on GPU.

    Returns None on failure (caller should fall back to trend-only).
    """
    try:
        ok_tmp = OrdinaryKriging(
            coords[:, 0], coords[:, 1], resid,
            variogram_model=vgm_params['variogram_model'],
            nlags=vgm_params['nlags'],
            weight=vgm_params['weight'],
            enable_plotting=False, verbose=False, pseudo_inv=True,
        )
        model_name = ok_tmp.variogram_model
        params = tuple(float(p) for p in ok_tmp.variogram_model_parameters)
        del ok_tmp
        return model_name, params
    except Exception as e:
        print(f"  WARNING: variogram fit failed ({e}); "
              f"this depth's map will be trend-only "
              f"(zero residual correction).")
        return None


class GPULocalOK:
    """
    GPU replacement for PyKrige's
        ok_model.execute('points', xs, ys, backend='C', n_closest_points=K)

    Build once (holds the training coords/residuals on GPU + a cuML KNN
    index), then call .predict(xs, ys) for as many tiles as needed -- the
    KNN index and GPU-resident training data are reused across all of them.
    """

    def __init__(self, coords, resid, variogram_model, variogram_params,
                 n_closest_points=32, gpu_id=0, dtype=cp.float64):
        self.model = variogram_model
        self.params = variogram_params
        self.gpu_id = gpu_id
        self.dtype = dtype

        with cp.cuda.Device(gpu_id):
            self._coords = cp.asarray(coords, dtype=dtype)          # (n_train, 2)
            self._resid = cp.asarray(resid, dtype=dtype)            # (n_train,)
            n_train = self._coords.shape[0]
            self.K = min(int(n_closest_points), n_train)  # can't exceed available points

            self._nn = cuNN(n_neighbors=self.K)
            self._nn.fit(self._coords)

    def predict(self, xs, ys, chunk=KRIGE_CHUNK, eps=1e-9):
        """
        xs, ys: 1D numpy (or cupy) arrays of query coordinates, same CRS/units
                as the training coords (projected meters).
        Returns: numpy float32 array, same length as xs, of kriged residual
                 predictions (z_hat), to be added to the RF trend.
        """
        n = xs.shape[0]
        out = np.zeros(n, dtype=np.float32)

        with cp.cuda.Device(self.gpu_id):
            xs_g_all = cp.asarray(xs, dtype=self.dtype)
            ys_g_all = cp.asarray(ys, dtype=self.dtype)
            K = self.K

            for p0 in range(0, n, chunk):
                p1 = min(p0 + chunk, n)
                q = cp.stack([xs_g_all[p0:p1], ys_g_all[p0:p1]], axis=1)  # (B,2)
                B = q.shape[0]

                # --- 1. KNN search (GPU) -------------------------------------
                dist_0K, idx_0K = self._nn.kneighbors(q, n_neighbors=K)
                dist_0K = cp.asarray(dist_0K, dtype=self.dtype)   # (B, K)
                idx_0K = cp.asarray(idx_0K, dtype=cp.int64)       # (B, K)

                # --- 2. gather neighbour coords / residuals -------------------
                nbr_coords = self._coords[idx_0K]                # (B, K, 2)
                nbr_resid = self._resid[idx_0K]                  # (B, K)

                # --- 3. pairwise distances among the K neighbours -------------
                diff = nbr_coords[:, :, None, :] - nbr_coords[:, None, :, :]  # (B,K,K,2)
                dist_KK = cp.sqrt(cp.sum(diff * diff, axis=-1))               # (B,K,K)
                del diff

                # --- 4. build OK linear system --------------------------------
                gamma_KK = _variogram_gpu(self.model, self.params, dist_KK)   # (B,K,K)
                diag_idx = cp.arange(K)
                gamma_KK[:, diag_idx, diag_idx] = 0.0  # self-semivariance = 0 (PyKrige convention)

                gamma_0K = _variogram_gpu(self.model, self.params, dist_0K)   # (B,K)
                gamma_0K = cp.where(dist_0K < eps, 0.0, gamma_0K)             # query coincides w/ a sample

                A = cp.zeros((B, K + 1, K + 1), dtype=self.dtype)
                A[:, :K, :K] = gamma_KK
                A[:, K, :K] = 1.0
                A[:, :K, K] = 1.0
                # A[:, K, K] stays 0.0

                b = cp.ones((B, K + 1), dtype=self.dtype)
                b[:, :K] = gamma_0K
                b3 = b[:, :, None]  # cupy's batched solve wants b as (..., M, K_rhs)

                # --- 5. batched solve ------------------------------------------
                try:
                    sol = cp.linalg.solve(A, b3)                   # (B, K+1, 1)
                except Exception:
                    A[:, diag_idx, diag_idx] += 1e-8
                    sol = cp.linalg.solve(A, b3)
                sol = sol[:, :, 0]                                  # (B, K+1)

                weights = sol[:, :K]                               # (B, K) ignore Lagrange mult.
                z_hat = cp.sum(weights * nbr_resid, axis=1)        # (B,)

                out[p0:p1] = cp.asnumpy(z_hat).astype(np.float32)

                del q, dist_0K, idx_0K, nbr_coords, nbr_resid, dist_KK
                del gamma_KK, gamma_0K, A, b, b3, sol, weights, z_hat
            cp.get_default_memory_pool().free_all_blocks()

        return out


def krige_predict_chunked(gpu_ok, xs, ys, chunk=KRIGE_CHUNK, **_ignored):
    """Returns a zero array (trend-only fallback) if the model is None."""
    if gpu_ok is None:
        return np.zeros(xs.shape[0], dtype=np.float32)
    return gpu_ok.predict(xs, ys, chunk=chunk)


def validate_gpu_kriging():
    """
    Sanity check: compares GPULocalOK against PyKrige's CPU execute() on a
    small synthetic dataset. Run once before trusting a full run.
    """
    rng = np.random.RandomState(0)
    coords = rng.uniform(0, 1000, size=(300, 2))
    resid = rng.normal(size=300)
    vgm_params = dict(variogram_model='spherical', nlags=6, weight=True)

    ok_cpu = OrdinaryKriging(
        coords[:, 0], coords[:, 1], resid,
        variogram_model=vgm_params['variogram_model'],
        nlags=vgm_params['nlags'], weight=vgm_params['weight'],
        enable_plotting=False, verbose=False, pseudo_inv=True,
    )

    xs_q = rng.uniform(0, 1000, size=50)
    ys_q = rng.uniform(0, 1000, size=50)
    z_cpu, _ = ok_cpu.execute('points', xs_q, ys_q, backend='C', n_closest_points=32)

    fit = fit_variogram_only(coords, resid, vgm_params)
    assert fit is not None, "variogram fit failed in validation"
    gpu_ok = GPULocalOK(coords, resid, fit[0], fit[1], n_closest_points=32, gpu_id=GPU_ID)
    z_gpu = gpu_ok.predict(xs_q, ys_q)

    max_abs_diff = float(np.max(np.abs(np.asarray(z_cpu) - z_gpu)))
    print(f"  [validate_gpu_kriging] max |CPU - GPU| = {max_abs_diff:.6g}")
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
# BUILD THE SINGLE POINT-ESTIMATE MODEL: one RF trend (-> FIL) + one
# residual kriging pass (-> GPU OK). No bootstrapping.
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

    fit = fit_variogram_only(coords_train, resid, vgm_params)
    if fit is None:
        ok_model = None
    else:
        model_name, vgm_p = fit
        try:
            ok_model = GPULocalOK(
                coords_train, resid, model_name, vgm_p,
                n_closest_points=KRIGE_N_CLOSEST, gpu_id=GPU_ID,
            )
        except Exception as e:
            print(f"  WARNING: GPU kriging index build failed ({e}); "
                  f"this depth's map will be trend-only.")
            ok_model = None

    print(f"  Point-estimate model ready: RF fit/compiled + variogram + "
          f"GPU KNN index built ({time.time()-t0:.1f}s)")
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
# residual (GPU). Writes ONE band: {response}.tif (point estimate).
# =============================================================================

def rk_predict_point_estimate(response, fil_model, ok_model, numeric_cols,
                               categorical_cols, encoder, numeric_medians,
                               categorical_modes, output_dir):

    print(f'\n{"="*70}')
    print(f'  REGRESSION-KRIGING POINT ESTIMATE -- {response}')
    print(f'  Numeric rasters  : {numeric_cols}')
    print(f'  Categorical rast.: {categorical_cols}')
    print(f'  Kriging (GPU)    : n_closest={KRIGE_N_CLOSEST}')
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

    tracking_csv = os.path.join(output_dir, f'tile_tracking_{response}_point.csv')
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

        pbar = tqdm(total=len(tiles), desc=f'{response} tiles (point estimate)',
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

    # One-time sanity check that GPU kriging agrees with PyKrige's CPU
    # execute() on a small synthetic case. Comment out once verified.
    validate_gpu_kriging()

    for response in RESPONSES:
        print(f"\n{'='*70}")
        print(f"  DEPTH: {response}  (point estimate)")
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

    print(f"\nAll depth point-estimate maps written under: {MAPS_ROOT}")


if __name__ == '__main__':
    main()
