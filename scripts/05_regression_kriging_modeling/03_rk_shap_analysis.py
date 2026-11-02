import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from joblib import Parallel, delayed
from scipy.stats import spearmanr

# =========================================================================
# CONFIGURATION
# =========================================================================

ARIAL_PATH = '/kaggle/input/datasets/hammaadali/arial-font/arial.ttf'

DATA_PATH   = '/kaggle/working/Tanety/Data_covariates.gpkg'
VIF_DIR     = '/kaggle/working/Tanety/VIF'

RESULTS_DIR = '/kaggle/working/Tanety/RK_Results'
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')
SHAP_DIR    = os.path.join(RESULTS_DIR, 'SHAP')
CACHE_DIR   = os.path.join(SHAP_DIR, 'cache')
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

RESPONSES = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']
CATEGORICAL_COVARIATES = ['ESA_WorldCover_v200_30m', 'SoilType_30m']
PLOT_EXCLUDE_PREFIXES = ['SoilType_30m']

TOP_N        = 10      # features shown per panel (bar + beeswarm + heatmap)
MAX_SAMPLES  = 500      # rows sampled per response for SHAP computation
SEED         = 123

# --- performance controls -------------------------------------------------
N_JOBS           = -1      # parallel workers across the 5 responses (-1 = all cores)
USE_CACHE        = True     # skip recomputation if a cached SHAP result exists on disk
CHECK_ADDITIVITY = False    # skip shap's internal additivity check (pure overhead once trusted)

# --- added depth controls --------------------------------------------------
BOOTSTRAP_N        = 200    # resamples used to build CIs on mean |SHAP|
BOOTSTRAP_CI        = 90    # percentile width of the CI band (e.g. 90 -> 5th/95th pct)
INTERACTION_TOP_K   = 8     # features considered when computing pairwise interactions
INTERACTION_SAMPLES = 150   # subsample size for the (more expensive) interaction values
WATERFALL_TOP_N     = 12    # features shown in the representative waterfall panels

PANEL_LABELS = {'ISF_a': 'a.', 'ISF_b': 'b.', 'ISF_c': 'c.', 'ISF_d': 'd.', 'ISF_e': 'e.'}

RESP_DISPLAY = {
    'ISF_a': r'SFI$_{0-10}$',
    'ISF_b': r'SFI$_{10-20}$',
    'ISF_c': r'SFI$_{20-30}$',
    'ISF_d': r'SFI$_{30-60}$',
    'ISF_e': r'SFI$_{60-90}$',
}

# Depth in cm represented by each response, used for the cross-depth heatmap axis
RESP_DEPTH_CM = {
    'ISF_a': '0-10',
    'ISF_b': '10-20',
    'ISF_c': '20-30',
    'ISF_d': '30-60',
    'ISF_e': '60-90',
}

PRED_LABELS = {
    'CanopyHeight_30m': r'$H_{canopy}$',
    'Elevation_30m':    r'$Elev$',
    'MAP_30m':          r'$MAP$',
    'MAT_30m':          r'$MAT$',
    'NPP_30m':          r'$NPP$',
    'S2_B2_30m':        r'$B_2$',
    'S2_B3_30m':        r'$B_3$',
    'S2_B4_30m':        r'$B_4$',
    'S2_B5_30m':        r'$B_5$',
    'S2_B6_30m':        r'$B_6$',
    'S2_B8_30m':        r'$B_8$',
    'S2_B11_30m':       r'$B_{11}$',
    'S2_B12_30m':       r'$B_{12}$',
    'S2_BSI_30m':       r'$BSI$',
    'S2_CIre_30m':      r'$CI_{re}$',
    'S2_ClayIndex_30m': r'$Clay_{idx}$',
    'S2_GRVI_30m':      r'$GRVI$',
    'S2_MNDWI_30m':     r'$MNDWI$',
    'S2_NBR_30m':       r'$NBR$',
    'S2_NDVI_30m':      r'$NDVI$',
    'S2_NIRI_30m':      r'$NIRI$',
    'S2_SAVI_30m':      r'$SAVI$',
    'Slope_30m':        r'$Slope$',
    'TPI_30m':          r'$TPI$',
    'TWI_30m':          r'$TWI$',
    'TreeCover_30m':    r'$TC$',
}

CATEGORICAL_LABELS = {
    'ESA_WorldCover_v200_30m': 'Land cover',
    'SoilType_30m':            'Soil type',
}

BAR_COLOR = '#4C72B0'

# =========================================================================
# FONT
# =========================================================================

if os.path.exists(ARIAL_PATH):
    fm.fontManager.addfont(ARIAL_PATH)
    font_family = fm.FontProperties(fname=ARIAL_PATH).get_name()
    print(f"Arial loaded: {font_family}")
else:
    print(f"Arial not found at {ARIAL_PATH}, falling back to sans-serif")
    font_family = 'sans-serif'

plt.rcParams.update({
    'font.family':       font_family,
    'font.size':          23,
    'axes.labelsize':     25,
    'axes.titlesize':     26,
    'xtick.labelsize':    22,
    'ytick.labelsize':    23,
    'axes.linewidth':     2.2,
    'xtick.major.width':  2.2,
    'ytick.major.width':  2.2,
    'xtick.major.size':   7,
    'ytick.major.size':   7,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})


def spine_cleanup(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('black')
        ax.spines[sp].set_linewidth(2.0)
    ax.tick_params(axis='both', colors='black', direction='out', length=7, width=2.0)
    ax.grid(False)


def pretty_label(feature_name):
    if feature_name in PRED_LABELS:
        return PRED_LABELS[feature_name]
    for cat_col, cat_label in CATEGORICAL_LABELS.items():
        prefix = f'{cat_col}_'
        if feature_name.startswith(prefix):
            value = feature_name[len(prefix):]
            return f'{cat_label} = {value}'
    return feature_name.replace('_30m', '').replace('_', ' ')


def plot_visible_indices(feature_names, exclude_prefixes=PLOT_EXCLUDE_PREFIXES):
    return np.array([
        i for i, f in enumerate(feature_names)
        if not any(f.startswith(p) for p in exclude_prefixes)
    ])


def zero_clean_formatter(x, pos):
    if abs(x) < 1e-9:
        return '0'
    s = f'{x:.3f}'.rstrip('0').rstrip('.')
    return s

# =========================================================================
# LOAD COVARIATE TABLE + VIF-SELECTED PREDICTORS
# =========================================================================

gdf = gpd.read_file(DATA_PATH)
data = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
print(f"Loaded: {data.shape}  -  {DATA_PATH}")

vif_final = pd.read_csv(os.path.join(VIF_DIR, 'vif_final.csv'))
NUMERIC_SELECTED = vif_final['predictor'].tolist()
CATEGORICAL_AVAILABLE = [c for c in CATEGORICAL_COVARIATES if c in data.columns]


def build_matrix(df, bundle, numeric_cols, categorical_cols):
    X_num = df[numeric_cols].fillna(bundle['numeric_medians']).values
    if categorical_cols and bundle.get('encoder') is not None:
        X_cat_raw = df[categorical_cols].fillna(bundle['categorical_modes'])
        X_cat = bundle['encoder'].transform(X_cat_raw)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num
    return X

# =========================================================================
# CORE SHAP COMPUTATION (parallel-friendly, cached, with interaction depth)
# =========================================================================

def _cache_path(response):
    return os.path.join(CACHE_DIR, f'shap_cache_{response}.joblib')


def compute_shap_for_response(response, max_samples=MAX_SAMPLES, seed=SEED):
    """
    Computes:
      - SHAP values for the RF trend component (TreeExplainer, tree_path_dependent
        perturbation, so no background dataset is needed and no calls to model.predict
        are required -> this is the fast, exact path through the trees).
      - Bootstrap CI band on mean |SHAP| per feature (adds uncertainty depth to the
        barplots instead of reporting a single point estimate).
      - Pairwise SHAP interaction values, restricted to the INTERACTION_TOP_K most
        important features and a smaller subsample, since interaction values are
        O(features^2) and would be wasteful to compute on the full feature set/sample.
      - A sanity check against the RF's own impurity-based feature_importances_
        (Spearman correlation), as a lightweight validation that SHAP ranking and the
        model's internal importance broadly agree.
    Results are cached to disk; reruns with USE_CACHE=True skip all of the above.
    """
    cache_file = _cache_path(response)
    if USE_CACHE and os.path.exists(cache_file):
        return joblib.load(cache_file)

    bundle_path = os.path.join(MODELS_DIR, f'final_rk_model_{response}.joblib')
    bundle = joblib.load(bundle_path)
    model = bundle['rf']
    feature_names = bundle['feature_names']

    sub = data.dropna(subset=[response]).reset_index(drop=True)
    X = build_matrix(sub, bundle, NUMERIC_SELECTED, CATEGORICAL_AVAILABLE)

    rng = np.random.default_rng(seed)
    n = min(max_samples, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    X_sample = X[idx]

    # tree_path_dependent is the default and fastest TreeExplainer mode: it needs
    # no background dataset and no additional calls to the underlying trees.
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X_sample, check_additivity=CHECK_ADDITIVITY)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs = np.abs(shap_values).mean(axis=0)

    # --- bootstrap CI on mean |SHAP| ------------------------------------
    boot_rng = np.random.default_rng(seed + 1)
    n_rows = shap_values.shape[0]
    boot_means = np.empty((BOOTSTRAP_N, shap_values.shape[1]))
    for b in range(BOOTSTRAP_N):
        rows = boot_rng.integers(0, n_rows, size=n_rows)
        boot_means[b] = np.abs(shap_values[rows]).mean(axis=0)
    lo_pct = (100 - BOOTSTRAP_CI) / 2
    hi_pct = 100 - lo_pct
    ci_lo = np.percentile(boot_means, lo_pct, axis=0)
    ci_hi = np.percentile(boot_means, hi_pct, axis=0)

    # --- validation against RF's own impurity-based importances ---------
    rf_importances = getattr(model, 'feature_importances_', None)
    rank_corr = np.nan
    if rf_importances is not None:
        rank_corr, _ = spearmanr(mean_abs, rf_importances)

    # --- pairwise SHAP interaction values (top-K features, smaller sample) ---
    top_k_idx = np.argsort(mean_abs)[-INTERACTION_TOP_K:][::-1]
    n_int = min(INTERACTION_SAMPLES, X_sample.shape[0])
    X_int = X_sample[:n_int]
    try:
        interaction_values = explainer.shap_interaction_values(X_int)
        interaction_values = np.asarray(interaction_values)
        if interaction_values.ndim == 4:
            interaction_values = interaction_values[:, :, :, 0]
        mean_abs_interaction = np.abs(interaction_values).mean(axis=0)
        # keep only the top-K x top-K block to report/plot
        sub_matrix = mean_abs_interaction[np.ix_(top_k_idx, top_k_idx)]
    except Exception as exc:
        print(f"  [{response}] interaction values skipped ({exc})")
        sub_matrix = None

    result = dict(
        shap_values=shap_values,
        X_sample=X_sample,
        feature_names=feature_names,
        mean_abs=mean_abs,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        rank_corr=rank_corr,
        top_k_idx=top_k_idx,
        interaction_matrix=sub_matrix,
    )
    joblib.dump(result, cache_file)
    return result


print("\nComputing SHAP values per response (RF trend component of RF+Kriging) ...")
print(f"  parallel workers: {N_JOBS} | cache: {USE_CACHE} | bootstrap: {BOOTSTRAP_N} reps")

parallel_out = Parallel(n_jobs=N_JOBS, prefer='processes')(
    delayed(compute_shap_for_response)(r) for r in RESPONSES
)
shap_results = {r: res for r, res in zip(RESPONSES, parallel_out)}

for r, res in shap_results.items():
    summary_df = pd.DataFrame({
        'feature': res['feature_names'],
        'label': [pretty_label(f) for f in res['feature_names']],
        'mean_abs_shap': res['mean_abs'],
        'ci_lo': res['ci_lo'],
        'ci_hi': res['ci_hi'],
    }).sort_values('mean_abs_shap', ascending=False)
    summary_df.to_csv(os.path.join(SHAP_DIR, f'shap_summary_{r}.csv'), index=False)

    print(f"  {r}: SHAP-vs-RF-importance Spearman rho = {res['rank_corr']:.3f}")

    if res['interaction_matrix'] is not None:
        top_labels = [pretty_label(res['feature_names'][i]) for i in res['top_k_idx']]
        inter_df = pd.DataFrame(res['interaction_matrix'], index=top_labels, columns=top_labels)
        inter_df.to_csv(os.path.join(SHAP_DIR, f'shap_interactions_{r}.csv'))

# consolidated validation table across all responses/depths
validation_df = pd.DataFrame({
    'response': list(shap_results.keys()),
    'depth_cm': [RESP_DEPTH_CM.get(r, r) for r in shap_results.keys()],
    'shap_vs_rf_importance_spearman': [res['rank_corr'] for res in shap_results.values()],
})
validation_df.to_csv(os.path.join(SHAP_DIR, 'shap_validation_summary.csv'), index=False)
print(f"Per-response SHAP summaries + validation saved to: {SHAP_DIR}")

# =========================================================================
# GRID GEOMETRY
# =========================================================================

resp_keys = [r for r in RESPONSES if r in shap_results]
n_panels = len(resp_keys)
ncols = 2
nrows = int(np.ceil(n_panels / ncols))
GRID_FIGSIZE = (17, 16.5 * nrows / 2)


def is_bottom_of_column(idx, ncols, n_panels):
    return (idx + ncols) >= n_panels

# =========================================================================
# FIGURE 1 - MEAN |SHAP| BARPLOT GRID (now with bootstrap CI whiskers)
# =========================================================================

def draw_shap_barplot(ax, feature_names, mean_abs_shap, ci_lo, ci_hi, top_n=TOP_N):
    visible = plot_visible_indices(feature_names)
    df_panel = pd.DataFrame({
        'label': [pretty_label(feature_names[i]) for i in visible],
        'value': mean_abs_shap[visible],
        'lo': ci_lo[visible],
        'hi': ci_hi[visible],
    }).sort_values('value', ascending=False).head(top_n)
    df_panel = df_panel.sort_values('value', ascending=True).reset_index(drop=True)

    vals = df_panel['value'].values
    lo = df_panel['lo'].values
    hi = df_panel['hi'].values
    labels = df_panel['label'].values
    n = len(df_panel)

    y_pos = np.arange(n)
    ax.barh(y_pos, vals, color=BAR_COLOR, alpha=0.9, height=0.6,
            edgecolor='white', linewidth=0.8)

    err_lo = np.clip(vals - lo, 0, None)
    err_hi = np.clip(hi - vals, 0, None)
    ax.errorbar(vals, y_pos, xerr=[err_lo, err_hi], fmt='none',
                ecolor='#1a1a2e', elinewidth=1.4, capsize=3, capthick=1.4, alpha=0.75)

    for y, val, h in zip(y_pos, vals, hi):
        ax.text(h + vals.max() * 0.03, y, f'{val:.3f}',
                ha='left', va='center', fontsize=19, fontfamily=font_family)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=21, fontfamily=font_family)
    ax.set_xlabel('Mean |SHAP value|', fontsize=24, fontweight='bold',
                  fontfamily=font_family, labelpad=12)
    ax.set_xlim(0, hi.max() * 1.3)
    ax.set_ylim(-0.6, n - 0.4)
    spine_cleanup(ax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(zero_clean_formatter))
    ax.tick_params(axis='x', labelsize=22)


fig1, axes1 = plt.subplots(
    nrows, ncols,
    figsize=GRID_FIGSIZE,
    gridspec_kw=dict(wspace=0.75, hspace=0.35),
)
axes1_flat = axes1.flatten()

for idx, resp in enumerate(resp_keys):
    ax = axes1_flat[idx]
    res = shap_results[resp]
    draw_shap_barplot(ax, res['feature_names'], res['mean_abs'], res['ci_lo'], res['ci_hi'])

    ax.set_title(RESP_DISPLAY.get(resp, resp), fontweight='bold', fontsize=26,
                 loc='left', pad=14, fontfamily=font_family)
    ax.text(-0.52, 1.09, PANEL_LABELS.get(resp, ''),
            transform=ax.transAxes,
            fontsize=38, fontweight='bold',
            va='bottom', ha='left',
            fontfamily=font_family, color='#1a1a2e')

for idx in range(n_panels, nrows * ncols):
    axes1_flat[idx].set_visible(False)

fig1.subplots_adjust(left=0.24, right=0.94, top=0.92, bottom=0.08)

out_png1 = os.path.join(SHAP_DIR, 'shap_barplot_grid.png')
out_pdf1 = os.path.join(SHAP_DIR, 'shap_barplot_grid.pdf')
fig1.savefig(out_png1, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
fig1.savefig(out_pdf1, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
print(f"\nSaved: {out_png1}")
print(f"Saved: {out_pdf1}")
plt.close(fig1)

# =========================================================================
# FIGURE 2 - BEESWARM GRID
# =========================================================================

fig2, axes2 = plt.subplots(
    nrows, ncols,
    figsize=GRID_FIGSIZE,
    gridspec_kw=dict(wspace=0.9, hspace=0.25),
)
axes2_flat = axes2.flatten()

for idx, resp in enumerate(resp_keys):
    ax = axes2_flat[idx]
    res = shap_results[resp]

    shap_values = res['shap_values']
    X_sample = res['X_sample']
    feature_names = res['feature_names']

    visible = plot_visible_indices(feature_names)
    mean_abs = np.abs(shap_values[:, visible]).mean(axis=0)
    top_local = np.argsort(mean_abs)[-TOP_N:][::-1]
    top_idx = visible[top_local]

    explanation = shap.Explanation(
        values=shap_values[:, top_idx],
        data=X_sample[:, top_idx],
        feature_names=[pretty_label(feature_names[i]) for i in top_idx],
    )

    plt.sca(ax)
    shap.plots.beeswarm(explanation, max_display=TOP_N, show=False, color_bar=False, s=60)

    ax.set_title(RESP_DISPLAY.get(resp, resp), fontweight='bold', fontsize=26,
                 loc='left', pad=14, fontfamily=font_family)
    ax.text(-0.52, 1.09, PANEL_LABELS.get(resp, ''),
            transform=ax.transAxes,
            fontsize=38, fontweight='bold',
            va='bottom', ha='left',
            fontfamily=font_family, color='#1a1a2e')

    if is_bottom_of_column(idx, ncols, n_panels):
        ax.set_xlabel('SHAP value', fontsize=24, fontweight='bold', fontfamily=font_family, labelpad=12)
    else:
        ax.set_xlabel('')

    ax.axvline(x=0, color='#333333', linestyle='--', alpha=0.5, linewidth=1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(zero_clean_formatter))
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=21)

for idx in range(n_panels, nrows * ncols):
    axes2_flat[idx].set_visible(False)

fig2.set_size_inches(*GRID_FIGSIZE)

try:
    from shap.plots._utils import red_blue_cmap
    cmap_shap = red_blue_cmap()
except Exception:
    from shap.plots.colors import red_blue
    cmap_shap = red_blue

all_vals = np.concatenate([shap_results[r]['X_sample'].ravel() for r in resp_keys])
norm = Normalize(vmin=np.nanmin(all_vals), vmax=np.nanmax(all_vals))
sm = ScalarMappable(cmap=cmap_shap, norm=norm)
sm.set_array([])

fig2.subplots_adjust(left=0.24, right=0.94, top=0.92, bottom=0.10)
cbar_width = 0.25
cbar_left = 0.5 - cbar_width / 2
cbar_ax = fig2.add_axes([cbar_left, 0.03, cbar_width, 0.015])
cbar = fig2.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Feature value', fontsize=22, fontweight='bold',
               fontfamily=font_family, labelpad=10)
cbar.set_ticks([])
fig2.text(cbar_left - 0.01, 0.03, 'Low', ha='right', va='center', fontsize=21, fontfamily=font_family)
fig2.text(cbar_left + cbar_width + 0.01, 0.03, 'High', ha='left', va='center', fontsize=21, fontfamily=font_family)

out_png2 = os.path.join(SHAP_DIR, 'shap_beeswarm_grid.png')
out_pdf2 = os.path.join(SHAP_DIR, 'shap_beeswarm_grid.pdf')
fig2.savefig(out_png2, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
fig2.savefig(out_pdf2, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
print(f"\nSaved: {out_png2}")
print(f"Saved: {out_pdf2}")
plt.close(fig2)

# =========================================================================
# FIGURE 3 - CROSS-DEPTH IMPORTANCE HEATMAP (new: depth-wise pattern)
# =========================================================================
# Which covariates matter at which soil depth is exactly the kind of
# cross-response pattern the per-panel grids above cannot show at a glance.
# This aggregates the top features (by their best rank across all depths)
# into a single feature x depth matrix of normalized mean |SHAP|.

all_feature_names = shap_results[resp_keys[0]]['feature_names']
visible_master = plot_visible_indices(all_feature_names)

# rank features by their best (max) normalized importance across responses
importance_matrix = np.zeros((len(visible_master), len(resp_keys)))
for j, resp in enumerate(resp_keys):
    res = shap_results[resp]
    vals = res['mean_abs'][visible_master]
    importance_matrix[:, j] = vals / vals.max() if vals.max() > 0 else vals

best_per_feature = importance_matrix.max(axis=1)
order = np.argsort(best_per_feature)[::-1][:TOP_N]
heat_labels = [pretty_label(all_feature_names[visible_master[i]]) for i in order]
heat_matrix = importance_matrix[order]

fig3, ax3 = plt.subplots(figsize=(2.2 * len(resp_keys) + 4, 1.1 * TOP_N + 3))
im = ax3.imshow(heat_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

ax3.set_xticks(np.arange(len(resp_keys)))
ax3.set_xticklabels([RESP_DEPTH_CM.get(r, r) for r in resp_keys], fontsize=22, fontfamily=font_family)
ax3.set_xlabel('Soil depth (cm)', fontsize=24, fontweight='bold', fontfamily=font_family, labelpad=12)
ax3.set_yticks(np.arange(len(heat_labels)))
ax3.set_yticklabels(heat_labels, fontsize=21, fontfamily=font_family)

for i in range(heat_matrix.shape[0]):
    for j in range(heat_matrix.shape[1]):
        val = heat_matrix[i, j]
        color = 'white' if val > 0.6 else 'black'
        ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=17, color=color, fontfamily=font_family)

for sp in ax3.spines.values():
    sp.set_visible(False)
ax3.set_xticks(np.arange(-.5, len(resp_keys), 1), minor=True)
ax3.set_yticks(np.arange(-.5, len(heat_labels), 1), minor=True)
ax3.grid(which='minor', color='white', linewidth=2)
ax3.tick_params(which='minor', bottom=False, left=False)

cbar3 = fig3.colorbar(im, ax=ax3, fraction=0.04, pad=0.03)
cbar3.set_label('Normalized mean |SHAP|', fontsize=22, fontweight='bold', fontfamily=font_family, labelpad=10)

fig3.tight_layout()
out_png3 = os.path.join(SHAP_DIR, 'shap_cross_depth_heatmap.png')
out_pdf3 = os.path.join(SHAP_DIR, 'shap_cross_depth_heatmap.pdf')
fig3.savefig(out_png3, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
fig3.savefig(out_pdf3, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
print(f"\nSaved: {out_png3}")
print(f"Saved: {out_pdf3}")
plt.close(fig3)

# =========================================================================
# FIGURE 4 - TOP-FEATURE INTERACTION HEATMAP GRID (new: interaction depth)
# =========================================================================

fig4, axes4 = plt.subplots(
    nrows, ncols,
    figsize=GRID_FIGSIZE,
    gridspec_kw=dict(wspace=0.6, hspace=0.45),
)
axes4_flat = axes4.flatten()

for idx, resp in enumerate(resp_keys):
    ax = axes4_flat[idx]
    res = shap_results[resp]
    matrix = res['interaction_matrix']
    if matrix is None:
        ax.set_visible(False)
        continue

    labels = [pretty_label(res['feature_names'][i]) for i in res['top_k_idx']]
    im4 = ax.imshow(matrix, cmap='viridis')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14, fontfamily=font_family)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=14, fontfamily=font_family)
    ax.set_title(RESP_DISPLAY.get(resp, resp), fontweight='bold', fontsize=24,
                 loc='left', pad=10, fontfamily=font_family)
    ax.text(-0.35, 1.12, PANEL_LABELS.get(resp, ''),
            transform=ax.transAxes, fontsize=32, fontweight='bold',
            va='bottom', ha='left', fontfamily=font_family, color='#1a1a2e')
    fig4.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

for idx in range(n_panels, nrows * ncols):
    axes4_flat[idx].set_visible(False)

fig4.suptitle('Mean |SHAP interaction value| (top features)', fontsize=24, fontweight='bold',
              fontfamily=font_family, y=1.01)
fig4.tight_layout()

out_png4 = os.path.join(SHAP_DIR, 'shap_interaction_heatmap_grid.png')
out_pdf4 = os.path.join(SHAP_DIR, 'shap_interaction_heatmap_grid.pdf')
fig4.savefig(out_png4, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
fig4.savefig(out_pdf4, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
print(f"\nSaved: {out_png4}")
print(f"Saved: {out_pdf4}")
plt.close(fig4)

print(f"\nAll SHAP outputs: {SHAP_DIR}")
print("Note: these SHAP plots explain the RF trend component of the RF+Kriging "
      "model only; the kriged spatial residual correction has no SHAP decomposition.")
print("Validation: shap_validation_summary.csv reports the Spearman correlation between "
      "SHAP-based ranking and the RF's built-in impurity importance for each depth.")
