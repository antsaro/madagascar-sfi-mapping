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
from scipy.stats import gaussian_kde

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False

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

TOP_N_PDP   = 3       # top predictors shown per depth (one-hot columns excluded)
MAX_SAMPLES = 500      # rows sampled per response for SHAP computation
SEED        = 123

PANEL_LABELS = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.',
                'j.', 'k.', 'l.', 'm.', 'n.', 'o.']

RESP_DISPLAY = {
    'ISF_a': r'SFI$_{0-10}$',
    'ISF_b': r'SFI$_{10-20}$',
    'ISF_c': r'SFI$_{20-30}$',
    'ISF_d': r'SFI$_{30-60}$',
    'ISF_e': r'SFI$_{60-90}$',
}

# Same numeric-predictor labels as the VIF / SHAP scripts, for consistent panels
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

COLORMAP    = 'viridis'
POINT_SIZE  = 100
POINT_ALPHA = 0.75
TREND_COLOR = '#D62728'
RUG_COLOR   = '#333333'

# -----------------------------------------------------------------------
# FONT (identical setup to the other Tanety scripts)
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
    'font.size':          29,
    'axes.labelsize':     31,
    'axes.titlesize':     32,
    'xtick.labelsize':    27,
    'ytick.labelsize':    27,
    'axes.linewidth':     2.4,
    'xtick.major.width':  2.4,
    'ytick.major.width':  2.4,
    'xtick.major.size':   9,
    'ytick.major.size':   9,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})


def zero_clean_formatter(x, pos):
    """Format ticks with up to 3 decimals, dropping trailing zeros, plain '0' at zero."""
    if abs(x) < 1e-9:
        return '0'
    return f'{x:.3f}'.rstrip('0').rstrip('.')


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
    return feature_name.replace('_30m', '').replace('_', ' ')


def point_density(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    xy = np.vstack([x[mask], y[mask]])
    try:
        kde = gaussian_kde(xy)
        density = kde(xy)
    except Exception:
        density = np.ones(mask.sum())
    density = (density - density.min()) / (density.max() - density.min() + 1e-12)
    return density, mask


def smooth_trend(x, y, frac=0.4):
    """PDP-style smoothed trend of SHAP value vs feature value."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    if HAS_LOWESS:
        fit = _lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
        return fit[:, 0], fit[:, 1]
    # Fallback: rolling mean over a fixed window if statsmodels unavailable
    window = max(5, len(x_sorted) // 15)
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_sorted, kernel, mode='same')
    return x_sorted, y_smooth


def add_top_rug(ax, x_data, color=RUG_COLOR, frac_height=0.05, gap=0.02):
    """Marginal rug plot drawn just above the top of the panel."""
    ylim = ax.get_ylim()
    yr = ylim[1] - ylim[0]
    y0 = ylim[1] + gap * yr
    y1 = y0 + frac_height * yr
    ax.vlines(x_data, y0, y1, color=color, alpha=0.5, linewidth=0.9, clip_on=False)
    ax.set_ylim(ylim[0], y1 + gap * yr)

# -----------------------------------------------------------------------
# LOAD COVARIATE TABLE + VIF-SELECTED PREDICTORS (same source as training)
# -----------------------------------------------------------------------

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
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    return shap_values, X_sample, feature_names


print("\nComputing SHAP values per response ...")
shap_results = {}
for r in RESPONSES:
    print(f"  {r} ...")
    shap_values, X_sample, feature_names = compute_shap_for_response(r)
    shap_results[r] = dict(shap_values=shap_values, X_sample=X_sample, feature_names=feature_names)

resp_keys = [r for r in RESPONSES if r in shap_results]

# -----------------------------------------------------------------------
# TOP-3 NUMERIC (NON ONE-HOT) PREDICTORS PER RESPONSE
# -----------------------------------------------------------------------

def numeric_indices(feature_names):
    """Indices of predictors that are NOT one-hot-encoded categorical dummies."""
    return np.array([
        i for i, f in enumerate(feature_names)
        if not any(f.startswith(c + '_') for c in CATEGORICAL_COVARIATES)
    ])


top_features = {}
for r in resp_keys:
    res = shap_results[r]
    num_idx = numeric_indices(res['feature_names'])
    mean_abs = np.abs(res['shap_values'][:, num_idx]).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:TOP_N_PDP]
    top_features[r] = num_idx[order]
    chosen = [res['feature_names'][i] for i in top_features[r]]
    print(f"{r}: top {TOP_N_PDP} predictors -> {chosen}")

# -----------------------------------------------------------------------
# 5 x 3 GRID: ONE ROW PER DEPTH, ONE COLUMN PER RANKED PREDICTOR
# -----------------------------------------------------------------------

nrows, ncols = len(resp_keys), TOP_N_PDP
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(6.2 * ncols, 5.3 * nrows),
    gridspec_kw=dict(wspace=0.42, hspace=0.55),
)

panel_i = 0
for row, r in enumerate(resp_keys):
    res = shap_results[r]
    feature_names = res['feature_names']
    shap_values = res['shap_values']
    X_sample = res['X_sample']

    for col, feat_idx in enumerate(top_features[r]):
        ax = axes[row, col] if nrows > 1 else axes[col]

        x = X_sample[:, feat_idx].astype(float)
        y = shap_values[:, feat_idx].astype(float)

        dens, mask = point_density(x, y)
        xs, ys = x[mask], y[mask]
        sort_idx = dens.argsort()
        xs_s, ys_s, dens_s = xs[sort_idx], ys[sort_idx], dens[sort_idx]

        ax.scatter(xs_s, ys_s, c=dens_s, cmap=COLORMAP, s=POINT_SIZE,
                   alpha=POINT_ALPHA, edgecolors='none', rasterized=True)

        x_fit, y_fit = smooth_trend(xs, ys)
        ax.plot(x_fit, y_fit, color=TREND_COLOR, linewidth=2.4, zorder=3)

        ax.axhline(0, color='#666666', linestyle='--', linewidth=1.1, alpha=0.6, zorder=1)

        feat_name = feature_names[feat_idx]
        ax.set_xlabel(pretty_label(feat_name), fontsize=30, fontweight='bold',
                      fontfamily=font_family, labelpad=12)
        if col == 0:
            ax.set_ylabel('SHAP value', fontsize=30, fontweight='bold',
                          fontfamily=font_family, labelpad=14)

        ax.text(-0.30, 1.14, PANEL_LABELS[panel_i],
                transform=ax.transAxes,
                fontsize=36, fontweight='bold',
                va='bottom', ha='left',
                fontfamily=font_family, color='#999999')

        spine_cleanup(ax)
        ax.set_xticks([xs.min(), xs.max()])
        ax.set_yticks([ys.min(), ys.max()])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(zero_clean_formatter))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(zero_clean_formatter))

        add_top_rug(ax, xs)

        if col == ncols - 1:
            ax.text(1.06, 0.5, RESP_DISPLAY.get(r, r),
                    transform=ax.transAxes, rotation=270,
                    fontsize=30, fontweight='bold', va='center', ha='left',
                    fontfamily=font_family, color='#1a1a2e')

        panel_i += 1

fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.06)

out_png = os.path.join(SHAP_DIR, 'shap_pdp_grid.png')
out_pdf = os.path.join(SHAP_DIR, 'shap_pdp_grid.pdf')
fig.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
fig.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
plt.show()

print(f"\nAll PDP outputs: {SHAP_DIR}")
