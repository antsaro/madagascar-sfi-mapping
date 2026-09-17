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

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

ARIAL_PATH = '/kaggle/input/datasets/hammaadali/arial-font/arial.ttf'

DATA_PATH   = '/kaggle/working/Tanety/Data_covariates.gpkg'
VIF_DIR     = '/kaggle/working/Tanety/VIF'
RESULTS_DIR = '/kaggle/working/Tanety/ML_Results'
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')
SHAP_DIR    = os.path.join(RESULTS_DIR, 'SHAP')
os.makedirs(SHAP_DIR, exist_ok=True)

RESPONSES = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']
CATEGORICAL_COVARIATES = ['ESA_WorldCover_v200_30m', 'SoilType_30m']

# Categorical columns whose dummy features are dropped from the plots only;
# the model itself is still trained and explained with them.
PLOT_EXCLUDE_PREFIXES = ['SoilType_30m']

TOP_N       = 10     # features shown per panel (bar + beeswarm)
MAX_SAMPLES = 500     # rows sampled per response for SHAP computation
SEED        = 123

PANEL_LABELS = {'ISF_a': 'a.', 'ISF_b': 'b.', 'ISF_c': 'c.', 'ISF_d': 'd.', 'ISF_e': 'e.'}

RESP_DISPLAY = {
    'ISF_a': r'SFI$_{0-10}$',
    'ISF_b': r'SFI$_{10-20}$',
    'ISF_c': r'SFI$_{20-30}$',
    'ISF_d': r'SFI$_{30-60}$',
    'ISF_e': r'SFI$_{60-90}$',
}

# Same numeric-predictor labels as the VIF script, so panels read consistently
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

BAR_COLOR = '#4C72B0'   # single flat color for the SHAP barplots, VIF-style

# -----------------------------------------------------------------------
# FONT (identical setup to the VIF script)
# -----------------------------------------------------------------------

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
    """Map a raw (possibly one-hot-encoded) feature name to a display label."""
    if feature_name in PRED_LABELS:
        return PRED_LABELS[feature_name]
    for cat_col, cat_label in CATEGORICAL_LABELS.items():
        prefix = f'{cat_col}_'
        if feature_name.startswith(prefix):
            value = feature_name[len(prefix):]
            return f'{cat_label} = {value}'
    return feature_name.replace('_30m', '').replace('_', ' ')


def plot_visible_indices(feature_names, exclude_prefixes=PLOT_EXCLUDE_PREFIXES):
    """Indices of features allowed to appear in the plots (soil type excluded)."""
    return np.array([
        i for i, f in enumerate(feature_names)
        if not any(f.startswith(p) for p in exclude_prefixes)
    ])


def zero_clean_formatter(x, pos):
    """Format x-axis ticks with up to 3 decimals, dropping trailing zeros
    (0.060 -> 0.06) and showing plain '0' at zero."""
    if abs(x) < 1e-9:
        return '0'
    s = f'{x:.3f}'.rstrip('0').rstrip('.')
    return s

# -----------------------------------------------------------------------
# LOAD COVARIATE TABLE + VIF-SELECTED PREDICTORS (same source as training)
# -----------------------------------------------------------------------

gdf = gpd.read_file(DATA_PATH)
data = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
print(f"Loaded: {data.shape}  -  {DATA_PATH}")

vif_final = pd.read_csv(os.path.join(VIF_DIR, 'vif_final.csv'))
NUMERIC_SELECTED = vif_final['predictor'].tolist()
CATEGORICAL_AVAILABLE = [c for c in CATEGORICAL_COVARIATES if c in data.columns]

# -----------------------------------------------------------------------
# REBUILD THE FEATURE MATRIX EXACTLY AS AT TRAINING TIME, USING THE
# FITTED MEDIANS / MODES / ENCODER STORED IN EACH FINAL MODEL BUNDLE
# -----------------------------------------------------------------------

def build_matrix(df, bundle, numeric_cols, categorical_cols):
    X_num = df[numeric_cols].fillna(bundle['numeric_medians']).values

    if categorical_cols and bundle.get('encoder') is not None:
        X_cat_raw = df[categorical_cols].fillna(bundle['categorical_modes'])
        X_cat = bundle['encoder'].transform(X_cat_raw)
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num

    return X


def compute_shap_for_response(response, max_samples=MAX_SAMPLES, seed=SEED):
    bundle_path = os.path.join(MODELS_DIR, f'final_model_{response}.joblib')
    bundle = joblib.load(bundle_path)
    model = bundle['model']
    feature_names = bundle['feature_names']

    sub = data.dropna(subset=[response]).reset_index(drop=True)
    X = build_matrix(sub, bundle, NUMERIC_SELECTED, CATEGORICAL_AVAILABLE)

    rng = np.random.default_rng(seed)
    n = min(max_samples, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:      # some shap/sklearn combos return (n, features, 1)
        shap_values = shap_values[:, :, 0]

    return shap_values, X_sample, feature_names


print("\nComputing SHAP values per response ...")
shap_results = {}
for r in RESPONSES:
    print(f"  {r} ...")
    shap_values, X_sample, feature_names = compute_shap_for_response(r)
    shap_results[r] = dict(shap_values=shap_values, X_sample=X_sample, feature_names=feature_names)

    mean_abs = np.abs(shap_values).mean(axis=0)
    summary_df = pd.DataFrame({
        'feature': feature_names,
        'label': [pretty_label(f) for f in feature_names],
        'mean_abs_shap': mean_abs,
    }).sort_values('mean_abs_shap', ascending=False)
    summary_df.to_csv(os.path.join(SHAP_DIR, f'shap_summary_{r}.csv'), index=False)

print(f"Per-response SHAP summaries saved to: {SHAP_DIR}")

# -----------------------------------------------------------------------
# GRID GEOMETRY (2 columns x 3 rows, 5 panels, last slot hidden)
# -----------------------------------------------------------------------

resp_keys = [r for r in RESPONSES if r in shap_results]
n_panels = len(resp_keys)
ncols = 2
nrows = int(np.ceil(n_panels / ncols))
GRID_FIGSIZE = (17, 16.5 * nrows / 2)


def is_bottom_of_column(idx, ncols, n_panels):
    return (idx + ncols) >= n_panels

# -----------------------------------------------------------------------
# FIGURE 1 - MEAN |SHAP| BARPLOT GRID
# -----------------------------------------------------------------------

def draw_shap_barplot(ax, feature_names, mean_abs_shap, top_n=TOP_N):
    visible = plot_visible_indices(feature_names)
    df_panel = pd.DataFrame({
        'label': [pretty_label(feature_names[i]) for i in visible],
        'value': mean_abs_shap[visible],
    }).sort_values('value', ascending=False).head(top_n)
    df_panel = df_panel.sort_values('value', ascending=True).reset_index(drop=True)

    vals = df_panel['value'].values
    labels = df_panel['label'].values
    n = len(df_panel)

    y_pos = np.arange(n)
    ax.barh(y_pos, vals, color=BAR_COLOR, alpha=0.9, height=0.6,
            edgecolor='white', linewidth=0.8)

    for y, val in zip(y_pos, vals):
        ax.text(val + vals.max() * 0.02, y, f'{val:.3f}',
                ha='left', va='center', fontsize=19, fontfamily=font_family)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=21, fontfamily=font_family)
    ax.set_xlabel('Mean |SHAP value|', fontsize=24, fontweight='bold',
                  fontfamily=font_family, labelpad=12)
    ax.set_xlim(0, vals.max() * 1.25)
    ax.set_ylim(-0.6, n - 0.4)
    spine_cleanup(ax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(zero_clean_formatter))
    ax.tick_params(axis='x', labelsize=22)


fig1, axes1 = plt.subplots(
    nrows, ncols,
    figsize=GRID_FIGSIZE,
    gridspec_kw=dict(wspace=1.1, hspace=0.35),
)
axes1_flat = axes1.flatten()

for idx, resp in enumerate(resp_keys):
    ax = axes1_flat[idx]
    res = shap_results[resp]
    draw_shap_barplot(ax, res['feature_names'], np.abs(res['shap_values']).mean(axis=0))

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
plt.show()

# -----------------------------------------------------------------------
# FIGURE 2 - BEESWARM GRID
# -----------------------------------------------------------------------

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

# shap.plots.beeswarm resizes the host figure per call; force it back so
# the beeswarm grid matches the barplot grid dimensions exactly.
fig2.set_size_inches(*GRID_FIGSIZE)

# Shared horizontal colorbar (feature-value scale) beneath the grid
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
cbar_ax = fig2.add_axes([0.30, 0.03, 0.4, 0.015])
cbar = fig2.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Feature value', fontsize=22, fontweight='bold',
               fontfamily=font_family, labelpad=10)
cbar.set_ticks([])
fig2.text(0.29, 0.03, 'Low', ha='right', va='center', fontsize=21, fontfamily=font_family)
fig2.text(0.71, 0.03, 'High', ha='left', va='center', fontsize=21, fontfamily=font_family)

out_png2 = os.path.join(SHAP_DIR, 'shap_beeswarm_grid.png')
out_pdf2 = os.path.join(SHAP_DIR, 'shap_beeswarm_grid.pdf')
fig2.savefig(out_png2, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
fig2.savefig(out_pdf2, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
print(f"\nSaved: {out_png2}")
print(f"Saved: {out_pdf2}")
plt.show()

print(f"\nAll SHAP outputs: {SHAP_DIR}")
