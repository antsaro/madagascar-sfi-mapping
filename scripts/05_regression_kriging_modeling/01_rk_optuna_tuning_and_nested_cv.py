"""
Regression Kriging (RF + Kriging of residuals) with nested CV and Optuna
tuning of BOTH the RF trend model and the residual variogram model.

pip install pykrige  (GeoStat-Framework's PyKrige: https://github.com/GeoStat-Framework/PyKrige)

Framework mirrors the original RF-only script:
  - VIF-selected numeric predictors + one-hot categorical predictors -> RF "trend"
  - Optuna (sequential trials, RF uses all cores) tunes RF hyperparameters
  - NEW: residuals of the RF trend are kriged in space using UTM coordinates
  - A second Optuna study picks the best variogram model (+ nlags/weight) by
    inner-CV RMSE of kriging the residuals themselves
  - Final prediction = RF trend + kriged residual
  - Nested CV reports metrics for RF-only AND RF+Kriging so you can see the
    kriging gain per response
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

from pykrige.ok import OrdinaryKriging

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

DATA_PATH   = '/kaggle/working/Tanety/Data_covariates.gpkg'
VIF_DIR     = '/kaggle/working/Tanety/VIF'
RESULTS_DIR = '/kaggle/working/Tanety/RK_Results'
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')

RESPONSES = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']
CATEGORICAL_COVARIATES = ['ESA_WorldCover_v200_30m', 'SoilType_30m']

# UTM 38S projected coordinates (meters) -- named lon/lat in the source file
# but are NOT geographic degrees, so plain Euclidean kriging is correct here.
COORD_COLS = ['lon', 'lat']

N_FOLDS_OUTER = 5
N_FOLDS_INNER = 4          # inner CV for RF tuning
N_FOLDS_INNER_VGM = 4      # inner CV for variogram-model tuning

OPTUNA_TRIALS_CV_RF   = 50
OPTUNA_TRIALS_FINAL_RF = 50
OPTUNA_TRIALS_VGM_CV    = 30   # variogram search is much cheaper per trial than RF
OPTUNA_TRIALS_VGM_FINAL = 40

SEED = 123

# Candidate variogram models PyKrige supports out of the box. 'hole-effect'
# is left out by default -- it fits periodic/cyclic spatial structure and is
# prone to unstable fits on small residual samples; add it back in if you
# have reason to expect oscillating spatial autocorrelation.
VARIOGRAM_MODELS = ['linear', 'power', 'gaussian', 'spherical', 'exponential']

# RF fits use all cores. Optuna trials (both RF and variogram searches) run
# one at a time on purpose -- letting Optuna ALSO parallelize trials while
# each trial's RF/kriging grabs every core too just oversubscribes the CPU.
RF_N_JOBS = -1
OPTUNA_N_JOBS = 1

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# LOAD DATA + VIF-SELECTED NUMERIC PREDICTORS
# -----------------------------------------------------------------------

gdf = gpd.read_file(DATA_PATH)
data = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
print(f"Loaded: {data.shape}  -  {DATA_PATH}")

missing_coords = [c for c in COORD_COLS if c not in data.columns]
if missing_coords:
    raise KeyError(f"Coordinate column(s) not found in data: {missing_coords}")

vif_final = pd.read_csv(os.path.join(VIF_DIR, 'vif_final.csv'))
NUMERIC_SELECTED = vif_final['predictor'].tolist()
CATEGORICAL_AVAILABLE = [c for c in CATEGORICAL_COVARIATES if c in data.columns]

print(f"Numeric predictors (VIF-selected): {len(NUMERIC_SELECTED)}")
print(f"Categorical predictors: {CATEGORICAL_AVAILABLE}")
print(f"Coordinate columns (UTM 38S, meters): {COORD_COLS}")

# -----------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------

def calc_metrics(obs, pred):
    obs, pred = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]

    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae = mean_absolute_error(obs, pred)
    r2 = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
    iqr = np.percentile(obs, 75) - np.percentile(obs, 25)
    rpiq = iqr / rmse if rmse > 0 else 0.0

    mx, my = obs.mean(), pred.mean()
    vx, vy = obs.var(ddof=0), pred.var(ddof=0)
    sxy = np.cov(obs, pred, ddof=0)[0, 1]
    ccc_den = vx + vy + (mx - my) ** 2
    ccc = (2 * sxy) / ccc_den if ccc_den > 0 else 0.0

    return dict(R2=r2, RMSE=rmse, MAE=mae, RPIQ=rpiq, CCC=ccc)

# -----------------------------------------------------------------------
# TRAIN-ONLY IMPUTATION AND ENCODING (leakage-safe)
# -----------------------------------------------------------------------

def impute_numeric_train_test(X_tr, X_te, numeric_cols):
    tr_df = pd.DataFrame(X_tr, columns=numeric_cols)
    te_df = pd.DataFrame(X_te, columns=numeric_cols)
    medians = tr_df.median()
    still_na = medians[medians.isna()].index
    if len(still_na):
        medians[still_na] = 0.0
    return tr_df.fillna(medians).values, te_df.fillna(medians).values, medians


def impute_categorical_train_test(X_tr_cat, X_te_cat, categorical_cols):
    tr_df = X_tr_cat.reset_index(drop=True).copy()
    te_df = X_te_cat.reset_index(drop=True).copy()
    modes = tr_df.mode().iloc[0]
    return tr_df.fillna(modes), te_df.fillna(modes), modes


def encode_categorical_train_test(X_tr_cat_df, X_te_cat_df):
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    tr_enc = encoder.fit_transform(X_tr_cat_df)
    te_enc = encoder.transform(X_te_cat_df)
    feature_names = list(encoder.get_feature_names_out(X_tr_cat_df.columns))
    return tr_enc, te_enc, feature_names, encoder

# -----------------------------------------------------------------------
# OPTUNA TUNING - RF TREND MODEL (sequential trials, RF does the parallel work)
# -----------------------------------------------------------------------

def make_rf_objective(X_np, y_np, n_inner=N_FOLDS_INNER):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 600),
            max_depth=trial.suggest_int('max_depth', 5, 30),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 15),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 8),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            random_state=42,
            n_jobs=RF_N_JOBS,
        )
        model = RandomForestRegressor(**params)
        inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=0)
        scores = []
        for tr, va in inner_kf.split(X_np):
            model.fit(X_np[tr], y_np[tr])
            scores.append(np.sqrt(mean_squared_error(y_np[va], model.predict(X_np[va]))))
        return float(np.mean(scores))
    return objective


def make_progress_callback(n_trials, label, log_every=10):
    pbar = tqdm(total=n_trials, desc=label, leave=False)

    def callback(study, trial):
        pbar.update(1)
        if trial.number % log_every == 0 or trial.number == n_trials - 1:
            postfix = {'best_rmse': f"{study.best_value:.4f}"}
            if 'max_depth' in trial.params:
                postfix['depth'] = trial.params['max_depth']
            if 'variogram_model' in trial.params:
                postfix['vgm'] = trial.params['variogram_model']
            pbar.set_postfix(**postfix)
        if trial.number == n_trials - 1:
            pbar.close()

    return callback


def tune_rf(X_np, y_np, n_trials, label='optuna-rf'):
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        make_rf_objective(X_np, y_np),
        n_trials=n_trials,
        n_jobs=OPTUNA_N_JOBS,
        show_progress_bar=False,
        callbacks=[make_progress_callback(n_trials, label)],
    )
    return {**study.best_params, 'random_state': 42, 'n_jobs': RF_N_JOBS}

# -----------------------------------------------------------------------
# OPTUNA TUNING - VARIOGRAM MODEL FOR RESIDUAL KRIGING
# -----------------------------------------------------------------------
# We search over variogram model family + nlags (number of bins used to fit
# the empirical variogram) + weight (whether closer lag bins get more weight
# in the least-squares variogram fit). Model quality is judged by leave-out
# kriging RMSE of the RESIDUALS themselves in an inner CV -- i.e. "which
# variogram model + settings krige the RF residuals best".

def _fit_ok(x, y, z, model, nlags, weight):
    return OrdinaryKriging(
        x, y, z,
        variogram_model=model,
        nlags=nlags,
        weight=weight,
        enable_plotting=False,
        verbose=False,
        pseudo_inv=True,   # robust to near-duplicate / collinear coordinates
    )


def make_variogram_objective(coords, resid, n_inner=N_FOLDS_INNER_VGM):
    def objective(trial):
        model = trial.suggest_categorical('variogram_model', VARIOGRAM_MODELS)
        nlags = trial.suggest_int('nlags', 6, 20)
        weight = trial.suggest_categorical('weight', [True, False])

        inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=1)
        scores = []
        for tr, va in inner_kf.split(coords):
            try:
                ok = _fit_ok(coords[tr, 0], coords[tr, 1], resid[tr], model, nlags, weight)
                z_hat, _ = ok.execute('points', coords[va, 0], coords[va, 1])
                rmse = np.sqrt(mean_squared_error(resid[va], z_hat))
                if not np.isfinite(rmse):
                    rmse = 1e6
            except Exception:
                rmse = 1e6  # penalize model/settings combos that fail to fit
            scores.append(rmse)
        return float(np.mean(scores))
    return objective


def tune_variogram(coords, resid, n_trials, label='optuna-vgm'):
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        make_variogram_objective(coords, resid),
        n_trials=n_trials,
        n_jobs=OPTUNA_N_JOBS,
        show_progress_bar=False,
        callbacks=[make_progress_callback(n_trials, label)],
    )
    return study.best_params


def krige_residuals(train_coords, train_resid, test_coords, vgm_params):
    ok = _fit_ok(
        train_coords[:, 0], train_coords[:, 1], train_resid,
        vgm_params['variogram_model'], vgm_params['nlags'], vgm_params['weight'],
    )
    z_hat, z_var = ok.execute('points', test_coords[:, 0], test_coords[:, 1])
    return np.asarray(z_hat), np.asarray(z_var), ok

# -----------------------------------------------------------------------
# NESTED CV PER RESPONSE (RF trend + kriged residuals)
# -----------------------------------------------------------------------

def run_nested_cv_rk(df, response, numeric_cols, categorical_cols, coord_cols,
                      outer_k=N_FOLDS_OUTER, n_trials_rf=OPTUNA_TRIALS_CV_RF,
                      n_trials_vgm=OPTUNA_TRIALS_VGM_CV, seed=SEED):
    # response NaNs are dropped here, not imputed -- imputing a target
    # manufactures observations that were never measured and shrinks its
    # variance artificially.
    sub = df.dropna(subset=[response]).reset_index(drop=True)

    kf = KFold(n_splits=outer_k, shuffle=True, random_state=seed)
    outer_metrics_rf, outer_metrics_rk = [], []
    fold_predictions, best_rf_params_list, best_vgm_params_list = [], [], []

    X_num_raw = sub[numeric_cols].values
    X_cat_raw = sub[categorical_cols] if categorical_cols else pd.DataFrame(index=sub.index)
    coords_all = sub[coord_cols].values.astype(np.float64)
    y_all = sub[response].values.astype(np.float64)

    for i, (tr_idx, te_idx) in enumerate(kf.split(X_num_raw), 1):
        fold_t0 = time.time()
        X_num_tr, X_num_te, _ = impute_numeric_train_test(
            X_num_raw[tr_idx], X_num_raw[te_idx], numeric_cols)

        if categorical_cols:
            X_cat_tr_df, X_cat_te_df, _ = impute_categorical_train_test(
                X_cat_raw.iloc[tr_idx], X_cat_raw.iloc[te_idx], categorical_cols)
            X_cat_tr_enc, X_cat_te_enc, cat_feature_names, encoder = encode_categorical_train_test(
                X_cat_tr_df, X_cat_te_df)
            X_tr = np.hstack([X_num_tr, X_cat_tr_enc])
            X_te = np.hstack([X_num_te, X_cat_te_enc])
            feature_names = numeric_cols + cat_feature_names
        else:
            X_tr, X_te = X_num_tr, X_num_te
            feature_names = list(numeric_cols)
            encoder = None

        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        coords_tr, coords_te = coords_all[tr_idx], coords_all[te_idx]

        # --- RF trend ---
        best_rf_params = tune_rf(X_tr, y_tr, n_trials_rf, label=f'{response} fold{i} RF')
        rf = RandomForestRegressor(**best_rf_params)
        rf.fit(X_tr, y_tr)
        rf_pred_tr = rf.predict(X_tr)
        rf_pred_te = rf.predict(X_te)

        # --- residual kriging ---
        resid_tr = y_tr - rf_pred_tr
        best_vgm_params = tune_variogram(coords_tr, resid_tr, n_trials_vgm,
                                          label=f'{response} fold{i} VGM')
        krig_pred_te, krig_var_te, ok_model = krige_residuals(
            coords_tr, resid_tr, coords_te, best_vgm_params)

        rk_pred_te = rf_pred_te + krig_pred_te

        m_rf = calc_metrics(y_te, rf_pred_te)
        m_rk = calc_metrics(y_te, rk_pred_te)
        m_rf['fold'] = m_rk['fold'] = i
        outer_metrics_rf.append(m_rf)
        outer_metrics_rk.append(m_rk)

        fold_predictions.append(pd.DataFrame({
            'fold': i, 'obs': y_te,
            'pred_rf': rf_pred_te, 'pred_rk': rk_pred_te,
            'krig_resid': krig_pred_te, 'krig_var': krig_var_te,
        }))
        best_rf_params_list.append({**best_rf_params, 'fold': i})
        best_vgm_params_list.append({**best_vgm_params, 'fold': i})

        joblib.dump(
            {'rf': rf, 'encoder': encoder, 'feature_names': feature_names,
             'variogram_params': best_vgm_params, 'ok_model': ok_model},
            os.path.join(MODELS_DIR, f'rk_{response}_fold{i}.joblib')
        )

        print(f"[{response}] fold {i}/{outer_k} done in {time.time() - fold_t0:.1f}s "
              f"-> RF RMSE={m_rf['RMSE']:.3f} R2={m_rf['R2']:.3f}  |  "
              f"RK RMSE={m_rk['RMSE']:.3f} R2={m_rk['R2']:.3f}  "
              f"(vgm={best_vgm_params['variogram_model']})")

    return dict(
        metrics_rf=pd.DataFrame(outer_metrics_rf),
        metrics_rk=pd.DataFrame(outer_metrics_rk),
        predictions=pd.concat(fold_predictions, ignore_index=True),
        best_rf_params=pd.DataFrame(best_rf_params_list),
        best_vgm_params=pd.DataFrame(best_vgm_params_list),
    )

# -----------------------------------------------------------------------
# FINAL MODEL PER RESPONSE (trained on all available data)
# -----------------------------------------------------------------------

def train_final_model_rk(df, response, numeric_cols, categorical_cols, coord_cols,
                          n_trials_rf=OPTUNA_TRIALS_FINAL_RF,
                          n_trials_vgm=OPTUNA_TRIALS_VGM_FINAL, seed=SEED):
    sub = df.dropna(subset=[response]).reset_index(drop=True)

    X_num_raw = sub[numeric_cols]
    medians = X_num_raw.median()
    still_na = medians[medians.isna()].index
    if len(still_na):
        medians[still_na] = 0.0
    X_num = X_num_raw.fillna(medians).values

    if categorical_cols:
        X_cat_raw = sub[categorical_cols].copy()
        modes = X_cat_raw.mode().iloc[0]
        X_cat_raw = X_cat_raw.fillna(modes)
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        X_cat = encoder.fit_transform(X_cat_raw)
        feature_names = numeric_cols + list(encoder.get_feature_names_out(categorical_cols))
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num
        feature_names = list(numeric_cols)
        encoder = None
        modes = None

    y = sub[response].values.astype(np.float64)
    coords = sub[coord_cols].values.astype(np.float64)

    best_rf_params = tune_rf(X, y, n_trials_rf, label=f'{response} final RF')
    rf = RandomForestRegressor(**best_rf_params)
    rf.fit(X, y)
    resid = y - rf.predict(X)

    best_vgm_params = tune_variogram(coords, resid, n_trials_vgm, label=f'{response} final VGM')
    ok_model = OrdinaryKriging(
        coords[:, 0], coords[:, 1], resid,
        variogram_model=best_vgm_params['variogram_model'],
        nlags=best_vgm_params['nlags'],
        weight=best_vgm_params['weight'],
        enable_plotting=False, verbose=False, pseudo_inv=True,
    )

    bundle = {
        'rf': rf,
        'encoder': encoder,
        'numeric_medians': medians,
        'categorical_modes': modes,
        'feature_names': feature_names,
        'best_rf_params': best_rf_params,
        'best_vgm_params': best_vgm_params,
        'ok_model': ok_model,             # holds training coords/residuals for prediction
        'n_obs': len(sub),
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, f'final_rk_model_{response}.joblib'))
    print(f"[{response}] final RF max_depth={best_rf_params['max_depth']}, "
          f"n_estimators={best_rf_params['n_estimators']}  |  "
          f"variogram={best_vgm_params['variogram_model']} (nlags={best_vgm_params['nlags']})")
    return bundle


def predict_rk(bundle, new_data, numeric_cols, categorical_cols, coord_cols):
    """Predict with a saved RF+Kriging bundle on new covariate rows."""
    X_num = new_data[numeric_cols].fillna(bundle['numeric_medians']).values
    if categorical_cols:
        X_cat = new_data[categorical_cols].fillna(bundle['categorical_modes'])
        X_cat_enc = bundle['encoder'].transform(X_cat)
        X = np.hstack([X_num, X_cat_enc])
    else:
        X = X_num

    trend = bundle['rf'].predict(X)
    coords_new = new_data[coord_cols].values.astype(np.float64)
    krig_resid, krig_var = bundle['ok_model'].execute('points', coords_new[:, 0], coords_new[:, 1])
    pred = trend + np.asarray(krig_resid)
    return pred, np.asarray(krig_var)

# -----------------------------------------------------------------------
# VARIOGRAM PLOT (empirical vs. fitted, for the final model per response)
# -----------------------------------------------------------------------

def plot_variogram(ok_model, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(ok_model.lags, ok_model.semivariance, color='steelblue', label='Empirical')
    lags_fine = np.linspace(0, ok_model.lags.max(), 200)
    fitted = ok_model.variogram_function(ok_model.variogram_model_parameters, lags_fine)
    ax.plot(lags_fine, fitted, color='red', label=f"Fitted ({ok_model.variogram_model})")
    ax.set_xlabel('Lag distance (m)')
    ax.set_ylabel('Semivariance')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------

t0 = time.time()
cv_results = {}
for r in tqdm(RESPONSES, desc='Nested CV over responses'):
    print(f"\n=== Nested CV (RF + Kriging): {r} ===")
    cv_results[r] = run_nested_cv_rk(data, r, NUMERIC_SELECTED, CATEGORICAL_AVAILABLE, COORD_COLS)
print(f"Nested CV finished in {(time.time() - t0) / 60:.2f} min")

final_models = {}
t1 = time.time()
for r in tqdm(RESPONSES, desc='Final models'):
    print(f"\nTraining final RF+Kriging model: {r} ...")
    final_models[r] = train_final_model_rk(data, r, NUMERIC_SELECTED, CATEGORICAL_AVAILABLE, COORD_COLS)
    plot_variogram(
        final_models[r]['ok_model'], f'{r} residual variogram',
        os.path.join(RESULTS_DIR, f'variogram_{r}.png'),
    )
print(f"Final models trained in {(time.time() - t1) / 60:.2f} min")

# -----------------------------------------------------------------------
# SUMMARY TABLE (RF-only vs. RF+Kriging, side by side)
# -----------------------------------------------------------------------

rows = []
for r in RESPONSES:
    row = {'response': r, 'n_obs': final_models[r]['n_obs'],
           'best_variogram': final_models[r]['best_vgm_params']['variogram_model']}
    for tag, m in [('rf', cv_results[r]['metrics_rf']), ('rk', cv_results[r]['metrics_rk'])]:
        for k in ['R2', 'RMSE', 'MAE', 'RPIQ', 'CCC']:
            row[f'{k}_{tag}_mean'] = m[k].mean()
            row[f'{k}_{tag}_sd'] = m[k].std()
    rows.append(row)
summary_table_cv = pd.DataFrame(rows)
print("\nNested CV summary (RF vs RF+Kriging):")
print(summary_table_cv)
summary_table_cv.to_csv(os.path.join(RESULTS_DIR, 'cv_summary_rf_vs_rk.csv'), index=False)

obs_pred_all = pd.concat(
    [cv_results[r]['predictions'].assign(response=r) for r in RESPONSES], ignore_index=True
)
obs_pred_all.to_csv(os.path.join(RESULTS_DIR, 'obs_pred_all_folds.csv'), index=False)

best_rf_params_table = pd.concat(
    [cv_results[r]['best_rf_params'].assign(response=r) for r in RESPONSES], ignore_index=True
).sort_values(['response', 'fold'])
best_rf_params_table.to_csv(os.path.join(RESULTS_DIR, 'best_rf_params_per_fold.csv'), index=False)

best_vgm_params_table = pd.concat(
    [cv_results[r]['best_vgm_params'].assign(response=r) for r in RESPONSES], ignore_index=True
).sort_values(['response', 'fold'])
best_vgm_params_table.to_csv(os.path.join(RESULTS_DIR, 'best_vgm_params_per_fold.csv'), index=False)

# -----------------------------------------------------------------------
# OBSERVED VS PREDICTED PLOT (RF vs RF+Kriging)
# -----------------------------------------------------------------------

fig, axes = plt.subplots(2, len(RESPONSES), figsize=(4 * len(RESPONSES), 8.2))
for col, r in enumerate(RESPONSES):
    sub = obs_pred_all[obs_pred_all['response'] == r]
    for row_i, (pred_col, label) in enumerate([('pred_rf', 'RF only'), ('pred_rk', 'RF + Kriging')]):
        ax = axes[row_i, col]
        ax.scatter(sub['obs'], sub[pred_col], alpha=0.6, color='steelblue')
        lims = [sub[['obs', pred_col]].min().min(), sub[['obs', pred_col]].max().max()]
        ax.plot(lims, lims, linestyle='--', color='red')
        ax.set_title(f'{r} ({label})')
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
fig.suptitle('Observed vs Predicted (out-of-fold, nested CV): RF vs RF+Kriging')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'obs_vs_pred_rf_vs_rk.png'), dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------

print(f"\n{'=' * 70}\nFINAL SUMMARY (RF+Kriging)\n{'=' * 70}")
for r in RESPONSES:
    row = summary_table_cv[summary_table_cv['response'] == r].iloc[0]
    print(f"{r}  (n={int(row['n_obs'])}, best variogram={row['best_variogram']})")
    print(f"   R2   : RF={row['R2_rf_mean']:.3f}+/-{row['R2_rf_sd']:.3f}   "
          f"RK={row['R2_rk_mean']:.3f}+/-{row['R2_rk_sd']:.3f}")
    print(f"   RMSE : RF={row['RMSE_rf_mean']:.3f}+/-{row['RMSE_rf_sd']:.3f}   "
          f"RK={row['RMSE_rk_mean']:.3f}+/-{row['RMSE_rk_sd']:.3f}")
    print(f"   MAE  : RF={row['MAE_rf_mean']:.3f}+/-{row['MAE_rf_sd']:.3f}   "
          f"RK={row['MAE_rk_mean']:.3f}+/-{row['MAE_rk_sd']:.3f}")
    print(f"   RPIQ : RF={row['RPIQ_rf_mean']:.3f}+/-{row['RPIQ_rf_sd']:.3f}   "
          f"RK={row['RPIQ_rk_mean']:.3f}+/-{row['RPIQ_rk_sd']:.3f}")
    print(f"   CCC  : RF={row['CCC_rf_mean']:.3f}+/-{row['CCC_rf_sd']:.3f}   "
          f"RK={row['CCC_rk_mean']:.3f}+/-{row['CCC_rk_sd']:.3f}")
    print(f"   Final model saved -> {os.path.join(MODELS_DIR, f'final_rk_model_{r}.joblib')}")

print(f"\nAll outputs: {RESULTS_DIR}")
print(f"All models: {MODELS_DIR}")
