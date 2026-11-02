import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

ARIAL_PATH = '/kaggle/input/datasets/hammaadali/arial-font/arial.ttf'

DATA_PATH  = '/kaggle/working/Tanety/Data_covariates.gpkg'
OUTPUT_DIR = '/kaggle/working/Tanety/VIF'
VIF_THRESH = 10

# ISF depth layers: a=0-10cm, b=10-20cm, c=20-30cm, d=30-60cm, e=60-90cm
RESPONSES = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']

# --- Which response layer to feature in the a./b. two-panel figure ---
RESPONSE_FOR_FIGURE = 'ISF_a'   # <- change this to feature a different depth layer

# Predictors always retained regardless of VIF, per response.
ALWAYS_KEEP = ['MAP_30m', 'Elevation_30m']
PRIORITY_PREDICTORS = {r: list(ALWAYS_KEEP) for r in RESPONSES}

CATEGORICAL_COVARIATES = ['ESA_WorldCover_v100_30m', 'ESA_WorldCover_v200_30m', 'SoilType_30m']

NUMERIC_PREDICTORS = [
    'CanopyHeight_30m', 'Elevation_30m', 'MAP_30m', 'MAT_30m', 'NPP_30m',
    'S2_B2_30m', 'S2_B3_30m', 'S2_B4_30m', 'S2_B5_30m', 'S2_B6_30m',
    'S2_B8_30m', 'S2_B11_30m', 'S2_B12_30m', 'S2_BSI_30m', 'S2_CIre_30m',
    'S2_ClayIndex_30m', 'S2_GRVI_30m', 'S2_MNDWI_30m', 'S2_NBR_30m',
    'S2_NDVI_30m', 'S2_NIRI_30m', 'S2_SAVI_30m', 'Slope_30m', 'TPI_30m',
    'TWI_30m', 'TreeCover_30m'
]

PANEL_LABELS = {'ISF_a': 'a.', 'ISF_b': 'b.', 'ISF_c': 'c.', 'ISF_d': 'd.', 'ISF_e': 'e.'}

RESP_DISPLAY = {
    'ISF_a': r'SFI$_{0-10}$',
    'ISF_b': r'SFI$_{10-20}$',
    'ISF_c': r'SFI$_{20-30}$',
    'ISF_d': r'SFI$_{30-60}$',
    'ISF_e': r'SFI$_{60-90}$',
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

VIF_COLORS = {
    'low':      '#32CD32',
    'moderate': '#FFD700',
    'high':     '#FF6B6B',
    'perfect':  '#8B0000',
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
    'font.family':       font_family,
    'font.size':          17,
    'axes.labelsize':     19,
    'axes.titlesize':     19,
    'xtick.labelsize':    16,
    'ytick.labelsize':    16,
    'axes.linewidth':     1.8,
    'xtick.major.width':  1.8,
    'ytick.major.width':  1.8,
    'xtick.major.size':   6,
    'ytick.major.size':   6,
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
        ax.spines[sp].set_linewidth(1.8)
    ax.tick_params(axis='both', colors='black', direction='out', length=6, width=1.8)
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


def run_vif_elimination(data, response, priority):
    """
    Stepwise VIF elimination.

    Rules:
      - Priority predictors are always retained, no matter their VIF.
      - Each step: among non-priority predictors only, drop the one
        with the highest VIF (if >= VIF_THRESH).
      - If no non-priority predictor has VIF >= threshold, stop, even
        if priority predictors still have high VIF among themselves.
    """
    priority_active = [p for p in priority if p in available_predictors]

    df_r = data.dropna(subset=[response]).copy()
    for col in available_predictors:
        if df_r[col].isnull().any():
            df_r[col] = df_r[col].fillna(df_r[col].median())

    current = list(available_predictors)
    for p in priority_active:
        if p not in current:
            current.append(p)

    log, step = [], 0

    for _ in range(1000):
        if len(current) < 2:
            break

        vif_df = compute_vif(df_r, current)

        nonprio_high = vif_df[
            ~vif_df['predictor'].isin(priority_active) &
             (vif_df['VIF'] >= VIF_THRESH)
        ].sort_values('VIF', ascending=False)

        if nonprio_high.empty:
            prio_high = vif_df[
                vif_df['predictor'].isin(priority_active) &
                (vif_df['VIF'] >= VIF_THRESH)
            ]
            if not prio_high.empty:
                print(f"  [INFO] ({response}) Stopping: priority predictor(s) "
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
        print(f"  [{response}] Step {step}: drop '{to_drop}'  VIF={drop_vif:.2f}")
        current.remove(to_drop)
        step += 1

    return compute_vif(df_r, current), log, len(df_r), priority_active, df_r

# -----------------------------------------------------------------------
# RUN VIF PER DEPTH
# -----------------------------------------------------------------------

results = {}
for resp in RESPONSES:
    if resp not in df.columns:
        print(f"[SKIP] Response column not found: {resp}")
        continue
    print(f"\nRunning VIF: {resp} ...")
    fvif, log, n_obs, prio, df_r = run_vif_elimination(df, resp, PRIORITY_PREDICTORS.get(resp, []))
    results[resp] = {'vif': fvif, 'log': log, 'n_obs': n_obs, 'priority': prio, 'data': df_r}
    fvif.sort_values('VIF', ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, f'vif_final_{resp}.csv'), index=False)
    pd.DataFrame(log).to_csv(
        os.path.join(OUTPUT_DIR, f'vif_log_{resp}.csv'), index=False)
    print(f"  Done: {len(fvif)} retained, {len(log)} dropped, n={n_obs}")

# -----------------------------------------------------------------------
# PANEL (a): VIF BARPLOT
# -----------------------------------------------------------------------

def draw_vif_barplot(ax, vif_df, n_retained):
    df_sorted = vif_df.sort_values('VIF', ascending=True).reset_index(drop=True)
    vals = df_sorted['VIF'].values
    labels = df_sorted['label'].values
    n = len(df_sorted)

    finite_vals = vals[~np.isinf(vals)]
    max_finite = finite_vals.max() if len(finite_vals) > 0 else 20
    plot_vals = np.where(np.isinf(vals), max_finite * 1.2, vals)
    colors = [bar_color(v) for v in vals]

    y_pos = np.arange(n)
    ax.barh(y_pos, plot_vals, color=colors, alpha=0.85, height=0.62,
            edgecolor='white', linewidth=0.8)

    for y, raw_val, plot_val in zip(y_pos, vals, plot_vals):
        label = '\u221e' if np.isinf(raw_val) else f'{raw_val:.1f}'
        ax.text(plot_val + max_finite * 0.02, y, label,
                ha='left', va='center', fontsize=13, fontfamily=font_family)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=16, fontfamily=font_family)
    ax.set_xlabel('VIF', fontsize=18, fontweight='bold', fontfamily=font_family, labelpad=8)
    ax.set_xlim(0, max_finite * 1.4)
    ax.set_ylim(-0.6, n - 0.4)
    spine_cleanup(ax)

    ax.tick_params(axis='x', labelsize=15)
    ax.text(0.97, 0.03, f'Retained\ncovariates = {n_retained}',
            transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            va='bottom', ha='right',
            fontfamily=font_family, color='#222222')

# -----------------------------------------------------------------------
# PANEL (b): CORRELATION MATRIX
#   - only the lower triangle + diagonal are colored (semi-matrix); upper
#     triangle left void/white
#   - significance stars overlaid on the colored lower-triangle cells only
#     ( *** p<0.001, ** p<0.01, * p<0.05, blank if not significant )
# -----------------------------------------------------------------------

def stars_from_p(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def draw_correlation_heatmap(ax, data, cols, labels):
    n = len(cols)
    corr = np.eye(n)
    pvals = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
                pvals[i, j] = 0.0
            else:
                r, p = pearsonr(data[cols[i]], data[cols[j]])
                corr[i, j] = r
                pvals[i, j] = p

    # color only the lower triangle + diagonal; upper triangle left void/white
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_fill = np.ma.array(corr, mask=mask_upper)

    cmap = LinearSegmentedColormap.from_list(
        'corr_diverge', ['#2166AC', '#F7F7F7', '#B2182B'], N=256
    )
    cmap.set_bad(color='white')
    im = ax.imshow(corr_fill, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

    # significance stars, lower triangle only; diagonal and upper triangle left blank
    for i in range(n):
        for j in range(i):
            stars = stars_from_p(pvals[i, j])
            if stars:
                text_color = 'black'
                # small vertical nudge: asterisk glyphs sit high in their line-box,
                # so a plain va='center' reads as slightly-too-high in the cell
                ax.text(j, i + 0.08, stars, ha='center', va='center',
                        fontsize=15, fontweight='bold', color=text_color,
                        fontfamily=font_family)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=14, fontfamily=font_family, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=14, fontfamily=font_family)

    # custom grid: only along the boundaries of the colored (lower-triangle +
    # diagonal) cells, including the diagonal "staircase" edge — none drawn
    # over the void upper triangle
    grid_color = '#DDDDDD'
    grid_lw = 1.0
    for r in range(n + 1):
        y = r - 0.5
        if r == 0:
            x_end = 0.5
        elif r == n:
            x_end = (n - 1) + 0.5
        else:
            x_end = r + 0.5
        ax.plot([-0.5, x_end], [y, y], color=grid_color, linewidth=grid_lw, zorder=3)
    for c in range(n + 1):
        x = c - 0.5
        y_start = -0.5 if c == 0 else (c - 1) - 0.5
        y_end = (n - 1) + 0.5
        ax.plot([x, x], [y_start, y_end], color=grid_color, linewidth=grid_lw, zorder=3)

    ax.tick_params(which='both', bottom=False, left=False, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return im

# -----------------------------------------------------------------------
# BUILD THE a./b. TWO-PANEL FIGURE FOR ONE RESPONSE LAYER
# -----------------------------------------------------------------------

if RESPONSE_FOR_FIGURE not in results:
    raise ValueError(f"'{RESPONSE_FOR_FIGURE}' not found among computed results: {list(results)}")

feat = results[RESPONSE_FOR_FIGURE]
# order the heatmap's rows/columns to match panel a's y-axis order exactly:
# the barplot lists predictors ascending bottom-to-top, so the top-to-bottom
# reading order is descending VIF — apply that same order to both axes here.
vif_ordered = feat['vif'].sort_values('VIF', ascending=False).reset_index(drop=True)
retained_cols = vif_ordered['predictor'].tolist()
retained_labels = vif_ordered['label'].tolist()

fig, axes = plt.subplots(
    1, 2, figsize=(11.5, 5.2),
    gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1.05]),
)

# Panel a: VIF barplot (no response/SFI title, no VIF=10 reference line)
draw_vif_barplot(axes[0], feat['vif'], n_retained=len(feat['vif']))
axes[0].text(-0.55, 1.06, 'a.', transform=axes[0].transAxes,
             fontsize=26, fontweight='bold', va='bottom', ha='left',
             fontfamily=font_family, color='#1a1a2e')

# Panel b: correlation matrix of the retained predictors
im_b = draw_correlation_heatmap(axes[1], feat['data'], retained_cols, retained_labels)
axes[1].text(-0.30, 1.06, 'b.', transform=axes[1].transAxes,
             fontsize=26, fontweight='bold', va='bottom', ha='left',
             fontfamily=font_family, color='#1a1a2e')

# Force panel b's matrix to be perfectly square in *physical* size while
# matching panel a's height exactly, so the two panels align on the same row.
fig.canvas.draw()
pos_a = axes[0].get_position()
pos_b = axes[1].get_position()

fig_w_in, fig_h_in = fig.get_size_inches()
panel_height_in = pos_a.height * fig_h_in     # physical height of panel a
square_width_frac = panel_height_in / fig_w_in  # width (fig-fraction) that makes b square

axes[1].set_position([pos_b.x0, pos_a.y0, square_width_frac, pos_a.height])

# manual colorbar, attached right next to the now-resized square panel b —
# half the length of panel b's height, but still vertically centered on it
cbar_pad_frac = 0.015
cbar_width_frac = 0.02
cbar_height_frac = pos_a.height / 2
cbar_y0 = pos_a.y0 + (pos_a.height - cbar_height_frac) / 2  # keep centered
cbar_ax = fig.add_axes([
    pos_b.x0 + square_width_frac + cbar_pad_frac,
    cbar_y0,
    cbar_width_frac,
    cbar_height_frac,
])
cbar = fig.colorbar(im_b, cax=cbar_ax)
cbar.set_label('Pearson r', fontsize=14, fontfamily=font_family, labelpad=8)
cbar.ax.tick_params(labelsize=12)

out_png = os.path.join(OUTPUT_DIR, f'vif_and_corr_{RESPONSE_FOR_FIGURE}.png')
out_pdf = os.path.join(OUTPUT_DIR, f'vif_and_corr_{RESPONSE_FOR_FIGURE}.pdf')
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

for resp in results:
    res = results[resp]
    print(f"\n{'='*55}\n{PANEL_LABELS.get(resp, '')} {resp}  (n={res['n_obs']})\n{'='*55}")
    for _, row in res['vif'].sort_values('VIF', ascending=False).iterrows():
        print(f"   {row['predictor']:<20}  VIF = {row['VIF']:6.2f}  {vif_category(row['VIF'])}")
    print(f"\n  Eliminated ({len(res['log'])}):")
    for e in res['log']:
        print(f"   step {e['step']}  {e['dropped']:<20}  VIF = {e['VIF_at_drop']:.2f}")

print(f"\nAll outputs: {OUTPUT_DIR}")
