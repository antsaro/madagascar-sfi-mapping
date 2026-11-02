import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

ARIAL_PATH = '/kaggle/input/datasets/hammaadali/arial-font/arial.ttf'

DATA_PATH  = '/kaggle/working/Tanety/Data_covariates.gpkg'
OUTPUT_DIR = '/kaggle/working/Tanety/VIF'
VIF_THRESH = 10

# Predictors always retained regardless of VIF.
ALWAYS_KEEP = ['MAP_30m', 'Elevation_30m']

CATEGORICAL_COVARIATES = ['ESA_WorldCover_v100_30m', 'ESA_WorldCover_v200_30m', 'SoilType_30m']

NUMERIC_PREDICTORS = [
    'CanopyHeight_30m', 'Elevation_30m', 'MAP_30m', 'MAT_30m', 'NPP_30m',
    'S2_B2_30m', 'S2_B3_30m', 'S2_B4_30m', 'S2_B5_30m', 'S2_B6_30m',
    'S2_B8_30m', 'S2_B11_30m', 'S2_B12_30m', 'S2_BSI_30m', 'S2_CIre_30m',
    'S2_ClayIndex_30m', 'S2_GRVI_30m', 'S2_MNDWI_30m', 'S2_NBR_30m',
    'S2_NDVI_30m', 'S2_NIRI_30m', 'S2_SAVI_30m', 'Slope_30m', 'TPI_30m',
    'TWI_30m', 'TreeCover_30m'
]

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

VIF_COLORS = {
    'low':      '#2E7D32',
    'moderate': '#F9A825',
    'high':     '#C62828',
    'perfect':  '#4A148C',
}

# -----------------------------------------------------------------------
# FONT
# -----------------------------------------------------------------------

if os.path.exists(ARIAL_PATH):
    fm.fontManager.addfont(ARIAL_PATH)
    font_family = fm.FontProperties(fname=ARIAL_PATH).get_name()
    print(f"Arial loaded: {font_family}")
else:
    print(f"Arial not found at {ARIAL_PATH}, falling back to sans-serif")
    font_family = 'sans-serif'

plt.rcParams.update({
    'font.family':      font_family,
    'font.size':         17,
    'axes.labelsize':    19,
    'xtick.labelsize':   15,
    'ytick.labelsize':   16,
    'axes.linewidth':    1.1,
    'xtick.major.width': 1.1,
    'ytick.major.width': 1.1,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
})


def spine_cleanup(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('black')
        ax.spines[sp].set_linewidth(1.1)
    ax.tick_params(axis='both', colors='black', direction='out', length=4, width=1.1)
    ax.grid(False)


def bar_color(vif_value):
    if np.isinf(vif_value):
        return VIF_COLORS['perfect']
    elif vif_value >= 10:
        return VIF_COLORS['high']
    elif vif_value >= 5:
        return VIF_COLORS['moderate']
    return VIF_COLORS['low']

# -----------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

gdf = gpd.read_file(DATA_PATH)
df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
print(f"Loaded: {df.shape}  -  {DATA_PATH}")

available_predictors = [c for c in NUMERIC_PREDICTORS if c in df.columns]
missing_predictors = [c for c in NUMERIC_PREDICTORS if c not in df.columns]
if missing_predictors:
    print(f"Warning: predictors not found in dataset: {missing_predictors}")

priority_active = [p for p in ALWAYS_KEEP if p in available_predictors]

# -----------------------------------------------------------------------
# CORE VIF FUNCTIONS
# -----------------------------------------------------------------------

def compute_vif(data, cols):
    X = data[cols].dropna()
    Xs = StandardScaler().fit_transform(X)
    return pd.DataFrame({
        'predictor': cols,
        'label':     [PRED_LABELS.get(c, c) for c in cols],
        'VIF':       [variance_inflation_factor(Xs, i) for i in range(len(cols))],
    })


def run_vif_elimination(data, predictors, priority):
    """
    Stepwise VIF elimination on the shared covariate set. Point locations
    are the same across depths, so this only needs to run once.

    Rules:
      - Priority predictors are always retained, no matter their VIF.
      - Each step: among non-priority predictors only, drop the one
        with the highest VIF (if >= VIF_THRESH).
      - If no non-priority predictor has VIF >= threshold, stop, even
        if priority predictors still have high VIF among themselves.
    """
    df_r = data.copy()
    for col in predictors:
        if df_r[col].isnull().any():
            df_r[col] = df_r[col].fillna(df_r[col].median())

    current = list(predictors)
    for p in priority:
        if p not in current:
            current.append(p)

    log, step = [], 0

    for _ in range(1000):
        if len(current) < 2:
            break

        vif_df = compute_vif(df_r, current)

        nonprio_high = vif_df[
            ~vif_df['predictor'].isin(priority) &
             (vif_df['VIF'] >= VIF_THRESH)
        ].sort_values('VIF', ascending=False)

        if nonprio_high.empty:
            prio_high = vif_df[
                vif_df['predictor'].isin(priority) &
                (vif_df['VIF'] >= VIF_THRESH)
            ]
            if not prio_high.empty:
                print(f"[INFO] Stopping: priority predictor(s) "
                      f"{prio_high['predictor'].tolist()} have "
                      f"VIF >= {VIF_THRESH} but are always retained. "
                      f"All {len(current)} remaining predictors kept.")
            break

        to_drop = nonprio_high.iloc[0]['predictor']
        drop_vif = nonprio_high.iloc[0]['VIF']

        log.append({
            'step': step,
            'dropped': to_drop,
            'VIF_at_drop': round(float(drop_vif), 2),
            'reason': 'high_vif_nonpriority',
        })
        print(f"Step {step}: drop '{to_drop}'  VIF={drop_vif:.2f}")
        current.remove(to_drop)
        step += 1

    return compute_vif(df_r, current), log, len(df_r)

# -----------------------------------------------------------------------
# RUN VIF (once, shared across all ISF depths)
# -----------------------------------------------------------------------

print("\nRunning VIF ...")
vif_final, vif_log, n_obs = run_vif_elimination(df, available_predictors, priority_active)
print(f"Done: {len(vif_final)} retained, {len(vif_log)} dropped, n={n_obs}")

vif_final.sort_values('VIF', ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, 'vif_final.csv'), index=False)
pd.DataFrame(vif_log).to_csv(
    os.path.join(OUTPUT_DIR, 'vif_log.csv'), index=False)

# -----------------------------------------------------------------------
# VERTICAL BARPLOT
# -----------------------------------------------------------------------

vif_sorted = vif_final.sort_values('VIF', ascending=False).reset_index(drop=True)
vals = vif_sorted['VIF'].values
labels = vif_sorted['label'].values
n = len(vif_sorted)

finite_vals = vals[~np.isinf(vals)]
max_finite = finite_vals.max() if len(finite_vals) > 0 else 20
plot_vals = np.where(np.isinf(vals), max_finite * 1.2, vals)
colors = [bar_color(v) for v in vals]

fig, ax = plt.subplots(figsize=(11, 3.2))

x_pos = np.arange(n)
ax.bar(x_pos, plot_vals, color=colors, alpha=0.9, width=0.62,
       edgecolor='white', linewidth=0.6)

for x, raw_val, plot_val in zip(x_pos, vals, plot_vals):
    label = '\u221e' if np.isinf(raw_val) else f'{raw_val:.1f}'
    ax.text(x, plot_val + max_finite * 0.03, label,
            ha='center', va='bottom', fontsize=13, fontfamily=font_family)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=14, fontfamily=font_family, rotation=45, ha='right')
ax.set_ylabel('VIF', fontsize=18, fontweight='bold', fontfamily=font_family, labelpad=8)
ax.set_ylim(0, max_finite * 1.4)
ax.set_xlim(-0.6, n - 0.4)
spine_cleanup(ax)

ax.text(0.99, 0.95, f'Retained covariates = {n}',
        transform=ax.transAxes,
        fontsize=15, fontweight='bold',
        va='top', ha='right',
        fontfamily=font_family, color='#222222')

plt.tight_layout(pad=0.5)

out_png = os.path.join(OUTPUT_DIR, 'vif_barplot_vertical.png')
out_pdf = os.path.join(OUTPUT_DIR, 'vif_barplot_vertical.pdf')
plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.08)
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
plt.show()

# -----------------------------------------------------------------------
# CONSOLE SUMMARY
# -----------------------------------------------------------------------

def vif_category(v):
    return 'High (>=10)' if v >= 10 else ('Moderate (5-10)' if v >= 5 else 'Low (<5)')

print(f"\n{'='*55}\nVIF results (n={n_obs})\n{'='*55}")
for _, row in vif_sorted.iterrows():
    print(f"   {row['predictor']:<20}  VIF = {row['VIF']:6.2f}  {vif_category(row['VIF'])}")
print(f"\nEliminated ({len(vif_log)}):")
for e in vif_log:
    print(f"   step {e['step']}  {e['dropped']:<20}  VIF = {e['VIF_at_drop']:.2f}")

print(f"\nAll outputs: {OUTPUT_DIR}")
