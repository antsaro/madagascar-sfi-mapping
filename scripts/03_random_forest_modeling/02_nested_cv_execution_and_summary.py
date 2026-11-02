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

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

DATA_PATH   = '/kaggle/working/Tanety/Data_covariates.gpkg'
VIF_DIR     = '/kaggle/working/Tanety/VIF'
RESULTS_DIR = '/kaggle/working/Tanety/ML_Results'
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')

RESPONSES = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']
CATEGORICAL_COVARIATES = ['ESA_WorldCover_v200_30m', 'SoilType_30m']

N_FOLDS_OUTER = 5
N_FOLDS_INNER = 4
OPTUNA_TRIALS_CV = 50
OPTUNA_TRIALS_FINAL = 50
SEED = 123

# RF fits use all cores. Optuna trials run one at a time on purpose --
# letting Optuna ALSO parallelize trials (n_jobs=-1) while each trial's RF
# grabs every core too just oversubscribes the CPU and makes everything
# slower, not faster. This was the main bottleneck.
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

vif_final = pd.read_csv(os.path.join(VIF_DIR, 'vif_final.csv'))
NUMERIC_SELECTED = vif_final['predictor'].tolist()
CATEGORICAL_AVAILABLE = [c for c in CATEGORICAL_COVARIATES if c in data.columns]

print(f"Numeric predictors (VIF-selected): {len(NUMERIC_SELECTED)}")
print(f"Categorical predictors: {CATEGORICAL_AVAILABLE}")

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
# OPTUNA TUNING (sequential trials, RF does the parallel work)
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
            pbar.set_postfix(best_rmse=f"{study.best_value:.4f}", depth=trial.params.get('max_depth'))
        if trial.number == n_trials - 1:
            pbar.close()

    return callback


def tune_rf(X_np, y_np, n_trials, label='optuna'):
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
# NESTED CV PER RESPONSE
# -----------------------------------------------------------------------

def run_nested_cv(df, response, numeric_cols, categorical_cols, outer_k=N_FOLDS_OUTER,
                   n_trials=OPTUNA_TRIALS_CV, seed=SEED):
    # response NaNs are dropped here, not imputed -- imputing a target
    # manufactures observations that were never measured and shrinks its
    # variance artificially.
    sub = df.dropna(subset=[response]).reset_index(drop=True)

    kf = KFold(n_splits=outer_k, shuffle=True, random_state=seed)
    outer_metrics, fold_predictions, best_params_list = [], [], []

    X_num_raw = sub[numeric_cols].values
    X_cat_raw = sub[categorical_cols] if categorical_cols else pd.DataFrame(index=sub.index)
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

        best_params = tune_rf(X_tr, y_tr, n_trials, label=f'{response} fold {i}/{outer_k}')
        rf = RandomForestRegressor(**best_params)
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_te)

        m = calc_metrics(y_te, preds)
        m['fold'] = i
        outer_metrics.append(m)
        fold_predictions.append(pd.DataFrame({'fold': i, 'obs': y_te, 'pred': preds}))
        best_params_list.append({**best_params, 'fold': i})

        joblib.dump(
            {'model': rf, 'encoder': encoder, 'feature_names': feature_names},
            os.path.join(MODELS_DIR, f'rf_{response}_fold{i}.joblib')
        )

        print(f"[{response}] fold {i}/{outer_k} done in {time.time() - fold_t0:.1f}s "
              f"-> RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}, max_depth={best_params['max_depth']}")

    return dict(
        metrics=pd.DataFrame(outer_metrics),
        predictions=pd.concat(fold_predictions, ignore_index=True),
        best_params=pd.DataFrame(best_params_list),
    )

# -----------------------------------------------------------------------
# FINAL MODEL PER RESPONSE (trained on all available data)
# -----------------------------------------------------------------------

def train_final_model(df, response, numeric_cols, categorical_cols,
                       n_trials=OPTUNA_TRIALS_FINAL, seed=SEED):
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

    best_params = tune_rf(X, y, n_trials, label=f'{response} final')
    rf = RandomForestRegressor(**best_params)
    rf.fit(X, y)

    bundle = {
        'model': rf,
        'encoder': encoder,
        'numeric_medians': medians,
        'categorical_modes': modes,
        'feature_names': feature_names,
        'best_params': best_params,
        'n_obs': len(sub),
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, f'final_model_{response}.joblib'))
    print(f"[{response}] final model max_depth={best_params['max_depth']}, "
          f"n_estimators={best_params['n_estimators']}")
    return bundle

# -----------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------

t0 = time.time()
cv_results = {}
for r in tqdm(RESPONSES, desc='Nested CV over responses'):
    print(f"\n=== Nested CV: {r} ===")
    cv_results[r] = run_nested_cv(data, r, NUMERIC_SELECTED, CATEGORICAL_AVAILABLE)
print(f"Nested CV finished in {(time.time() - t0) / 60:.2f} min")

final_models = {}
t1 = time.time()
for r in tqdm(RESPONSES, desc='Final models'):
    print(f"\nTraining final model: {r} ...")
    final_models[r] = train_final_model(data, r, NUMERIC_SELECTED, CATEGORICAL_AVAILABLE)
print(f"Final models trained in {(time.time() - t1) / 60:.2f} min")

# -----------------------------------------------------------------------
# SUMMARY TABLE
# -----------------------------------------------------------------------

summary_table_cv = pd.DataFrame([
    {'response': r, 'n_obs': final_models[r]['n_obs'],
     **{f'{k}_mean': cv_results[r]['metrics'][k].mean() for k in ['R2', 'RMSE', 'MAE', 'RPIQ', 'CCC']},
     **{f'{k}_sd': cv_results[r]['metrics'][k].std() for k in ['R2', 'RMSE', 'MAE', 'RPIQ', 'CCC']}}
    for r in RESPONSES
])
print("\nNested CV summary:")
print(summary_table_cv)
summary_table_cv.to_csv(os.path.join(RESULTS_DIR, 'cv_summary.csv'), index=False)

obs_pred_all = pd.concat(
    [cv_results[r]['predictions'].assign(response=r) for r in RESPONSES], ignore_index=True
)
obs_pred_all.to_csv(os.path.join(RESULTS_DIR, 'obs_pred_all_folds.csv'), index=False)

best_params_table = pd.concat(
    [cv_results[r]['best_params'].assign(response=r) for r in RESPONSES], ignore_index=True
).sort_values(['response', 'fold'])
best_params_table.to_csv(os.path.join(RESULTS_DIR, 'best_params_per_fold.csv'), index=False)

# -----------------------------------------------------------------------
# OBSERVED VS PREDICTED PLOT
# -----------------------------------------------------------------------

fig, axes = plt.subplots(1, len(RESPONSES), figsize=(4 * len(RESPONSES), 4.2))
for ax, r in zip(axes, RESPONSES):
    sub = obs_pred_all[obs_pred_all['response'] == r]
    ax.scatter(sub['obs'], sub['pred'], alpha=0.6, color='steelblue')
    lims = [sub[['obs', 'pred']].min().min(), sub[['obs', 'pred']].max().max()]
    ax.plot(lims, lims, linestyle='--', color='red')
    ax.set_title(r)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
fig.suptitle('Observed vs Predicted (out-of-fold, nested CV)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'obs_vs_pred.png'), dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------

print(f"\n{'=' * 60}\nFINAL SUMMARY\n{'=' * 60}")
for r in RESPONSES:
    row = summary_table_cv[summary_table_cv['response'] == r].iloc[0]
    print(f"{r}  (n={int(row['n_obs'])})")
    print(f"   R2   = {row['R2_mean']:.3f} +/- {row['R2_sd']:.3f}")
    print(f"   RMSE = {row['RMSE_mean']:.3f} +/- {row['RMSE_sd']:.3f}")
    print(f"   MAE  = {row['MAE_mean']:.3f} +/- {row['MAE_sd']:.3f}")
    print(f"   RPIQ = {row['RPIQ_mean']:.3f} +/- {row['RPIQ_sd']:.3f}")
    print(f"   CCC  = {row['CCC_mean']:.3f} +/- {row['CCC_sd']:.3f}")
    print(f"   Final model saved -> {os.path.join(MODELS_DIR, f'final_model_{r}.joblib')}")

print(f"\nAll outputs: {RESULTS_DIR}")
print(f"All models: {MODELS_DIR}")
