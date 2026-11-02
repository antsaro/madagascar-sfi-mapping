import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from scipy.stats import gaussian_kde

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

ARIAL_PATH = '/kaggle/input/datasets/hammaadali/arial-font/arial.ttf'

# Points at the RF+Kriging pipeline's output directory/file
# (obs_pred_all_folds.csv now has columns: fold, obs, pred_rf, pred_rk,
#  krig_resid, krig_var, response  -- instead of a single 'pred' column)
RESULTS_DIR  = '/kaggle/working/Tanety/RK_Results'
OBS_PRED_CSV = os.path.join(RESULTS_DIR, 'obs_pred_all_folds.csv')
OUTPUT_DIR   = RESULTS_DIR

RESPONSES = ['ISF_a', 'ISF_b', 'ISF_c', 'ISF_d', 'ISF_e']
PANEL_LABELS = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.', 'j.']

RESP_DISPLAY = {
    'ISF_a': r'SFI$_{0-10}$',
    'ISF_b': r'SFI$_{10-20}$',
    'ISF_c': r'SFI$_{20-30}$',
    'ISF_d': r'SFI$_{30-60}$',
    'ISF_e': r'SFI$_{60-90}$',
}

RESP_DISPLAY_PLAIN = {
    'ISF_a': 'SFI 0-10 cm',
    'ISF_b': 'SFI 10-20 cm',
    'ISF_c': 'SFI 20-30 cm',
    'ISF_d': 'SFI 30-60 cm',
    'ISF_e': 'SFI 60-90 cm',
}

# Two prediction columns to compare side by side: RF-only trend vs RF+Kriging
MODEL_COLUMNS = [('pred_rf', 'RF'), ('pred_rk', 'RF + Kriging')]

COLORMAP = 'viridis'

FS_BASE       = 22
FS_TICK       = 20
FS_AXLABEL    = 22
FS_TITLE      = 23
FS_PANEL      = 32
FS_STATS      = 19
FS_CBAR_LABEL = 19
FS_CBAR_TICK  = 18

POINT_SIZE  = 70   # increased from 20
POINT_ALPHA = 0.75

if os.path.exists(ARIAL_PATH):
    fm.fontManager.addfont(ARIAL_PATH)
    font_family = fm.FontProperties(fname=ARIAL_PATH).get_name()
    print(f"Arial loaded: {font_family}")
else:
    print(f"Arial not found at {ARIAL_PATH}, falling back to sans-serif")
    font_family = 'sans-serif'

plt.rcParams.update({
    'font.family':       font_family,
    'font.size':          FS_BASE,
    'axes.labelsize':     FS_AXLABEL,
    'axes.titlesize':     FS_TITLE,
    'xtick.labelsize':    FS_TICK,
    'ytick.labelsize':    FS_TICK,
    'axes.linewidth':     1.4,
    'xtick.major.width':  1.4,
    'ytick.major.width':  1.4,
    'xtick.major.size':   6,
    'ytick.major.size':   6,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})

# -----------------------------------------------------------------------
# LOAD OUT-OF-FOLD PREDICTIONS
# -----------------------------------------------------------------------

obs_pred_all = pd.read_csv(OBS_PRED_CSV)
print(f"Loaded: {obs_pred_all.shape}  -  {OBS_PRED_CSV}")

missing_cols = [c for c, _ in MODEL_COLUMNS if c not in obs_pred_all.columns]
if missing_cols:
    raise KeyError(
        f"Expected column(s) {missing_cols} not found in {OBS_PRED_CSV}. "
        f"Available columns: {list(obs_pred_all.columns)}. "
        f"If you're pointing this at the old RF-only 'obs_pred_all_folds.csv' "
        f"(single 'pred' column), use the RF-only version of this script instead."
    )


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


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    sd = np.std(y_true, ddof=1)
    rpd = sd / rmse if rmse > 0 else np.nan
    iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
    rpiq = iqr / rmse if rmse > 0 else np.nan
    return r2, rmse, mae, rpd, rpiq

# -----------------------------------------------------------------------
# BUILD PANEL DATA + SUMMARY TABLE (both RF and RF+Kriging, per response)
# -----------------------------------------------------------------------

panel_data, table_rows = [], []
for r in RESPONSES:
    sub = obs_pred_all[obs_pred_all['response'] == r]
    if sub.empty:
        print(f"[SKIP] No predictions found for '{r}'")
        continue

    y_true = sub['obs'].values.astype(np.float64)
    row_entry = {'tag': r, 'y_true': y_true, 'n': len(sub)}
    table_row = {'Variable': RESP_DISPLAY_PLAIN.get(r, r), 'n': len(sub)}

    for col, label in MODEL_COLUMNS:
        y_pred = sub[col].values.astype(np.float64)
        r2, rmse, mae, rpd, rpiq = compute_metrics(y_true, y_pred)
        row_entry[col] = dict(y_pred=y_pred, r2=r2, rmse=rmse, mae=mae, rpd=rpd, rpiq=rpiq)
        suffix = 'RF' if col == 'pred_rf' else 'RK'
        table_row.update({
            f'R2_{suffix}': round(r2, 3), f'RMSE_{suffix}': round(rmse, 3),
            f'MAE_{suffix}': round(mae, 3), f'RPD_{suffix}': round(rpd, 2),
            f'RPIQ_{suffix}': round(rpiq, 2),
        })

    panel_data.append(row_entry)
    table_rows.append(table_row)

summary_table = pd.DataFrame(table_rows)
print("\n" + summary_table.to_string(index=False))
os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_table.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics_table_nested_rf_vs_rk.csv'), index=False)

# -----------------------------------------------------------------------
# SCATTER GRIDS -- one figure per model (RF only, RK only), 3 rows x 2 cols
# -----------------------------------------------------------------------

def make_scatter_grid(model_col, model_label, out_stem):
    ncols, nrows = 2, 3
    fig = plt.figure(figsize=(7 * ncols, 6.5 * nrows), dpi=300)
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        left=0.09, right=0.95,
        top=0.96, bottom=0.05,
        wspace=0.38, hspace=0.5,
    )

    for idx in range(nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row_i, col_i])

        if idx >= len(panel_data):
            ax.axis('off')
            continue

        pd_ = panel_data[idx]
        resp_short = RESP_DISPLAY.get(pd_['tag'], pd_['tag'])
        y_true = pd_['y_true']
        stats = pd_[model_col]
        y_pred = stats['y_pred']

        dens, mask = point_density(y_true, y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        sort_idx = dens.argsort()
        yt_s, yp_s, dens_s = yt[sort_idx], yp[sort_idx], dens[sort_idx]

        sc = ax.scatter(yt_s, yp_s, c=dens_s, cmap=COLORMAP,
                         s=POINT_SIZE, alpha=POINT_ALPHA, edgecolors='none', rasterized=True)

        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.4, alpha=0.65, zorder=1)

        z = np.polyfit(yt, yp, 1)
        x_fit = np.linspace(yt.min(), yt.max(), 300)
        ax.plot(x_fit, np.poly1d(z)(x_fit), color='red', linewidth=1.8, alpha=0.85, zorder=2)

        ax.set_xlabel(f'Observed {resp_short}', fontsize=FS_AXLABEL, fontfamily=font_family)
        ax.set_ylabel(f'Predicted {resp_short}', fontsize=FS_AXLABEL, fontfamily=font_family)
        ax.set_title(f'{resp_short}  ({model_label})', fontweight='bold', fontsize=FS_TITLE,
                     loc='left', pad=10, fontfamily=font_family)

        ax.text(-0.17, 1.06, PANEL_LABELS[idx],
                transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold',
                va='bottom', ha='left',
                fontfamily=font_family, color='#1a1a2e')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(1.4)
        ax.tick_params(axis='both', colors='black', direction='out',
                        length=6, width=1.4, labelsize=FS_TICK)
        ax.grid(False)

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=20, shrink=0.78)
        cbar.set_label('Point density', rotation=270, labelpad=22,
                        fontsize=FS_CBAR_LABEL, fontfamily=font_family)
        cbar.ax.tick_params(labelsize=FS_CBAR_TICK, width=0.8, length=3, colors='black')
        cbar.outline.set_linewidth(0.8)
        cbar.outline.set_edgecolor('black')

        stats_txt = (
            f"$R^2$    = {stats['r2']:.3f}\n"
            f"RMSE = {stats['rmse']:.2f}\n"
            f"RPIQ  = {stats['rpiq']:.2f}"
        )
        ax.text(0.05, 0.95, stats_txt,
                transform=ax.transAxes,
                fontsize=FS_STATS, verticalalignment='top',
                fontfamily=font_family, color='black',
                linespacing=1.7)

        ax.set_aspect('equal', adjustable='box')

    out_png = os.path.join(OUTPUT_DIR, f'{out_stem}.png')
    out_pdf = os.path.join(OUTPUT_DIR, f'{out_stem}.pdf')
    plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.05)
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.05)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.show()


make_scatter_grid('pred_rf', 'RF', 'sfi_scatter_grid_RF_only')
make_scatter_grid('pred_rk', 'RF + Kriging', 'sfi_scatter_grid_RK_only')

print("\n" + "=" * 90)
print("FINAL SUMMARY TABLE - ALL METRICS (NESTED CV, POOLED OOF, RF vs RF+KRIGING)")
print("=" * 90)
print(summary_table.to_string(index=False))
