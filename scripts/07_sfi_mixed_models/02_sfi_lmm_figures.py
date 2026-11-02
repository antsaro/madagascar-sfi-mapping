#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_sfi_lmm_figures.py
=====================
Publication figures for the Soil Fertility Index (SFI) analysis.

Letters, p-values, estimated marginal means and confidence intervals are NOT computed here: they are read
from the tables written by 01_sfi_lmm_analysis.py (linear mixed model, plot as grouping unit, Holm-adjusted
contrasts), so that every figure is consistent with the reported statistics. Only the raw data
(boxplots) are read from DATA_LM.csv.

  A. Boxplots (comparisons without depth)
       fig_box_01_global_site            landscapes (other factors pooled)
       fig_box_02_global_landcover       land covers (other factors pooled)
       fig_box_03_site_by_landcover      land covers within each landscape
       fig_box_04_landcover_by_site      landscapes within each land cover
  B. Depth profiles (every comparison involving depth)
       fig_prof_01_global_depth          depth, all landscapes and land covers pooled
       fig_prof_02_by_landscape_*        depth by landscape (vertical / horizontal)
       fig_prof_03_by_landcover_*        depth by land cover, landscapes pooled (vertical / horizontal; offset / facets)
       fig_prof_04_landcover_within_landscape_*   land cover within each landscape; * = differs from grassland

Usage (after 01_sfi_lmm_analysis.py, from the repository root):
    python scripts/07_sfi_mixed_models/02_sfi_lmm_figures.py [--data data/DATA_LM.csv] [--out outputs/lmm]
Requirements: numpy, pandas, matplotlib.   Outputs: <out>/figures/*.png (300 dpi) and *.pdf
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------
ADD_CAPTION = True             # explanatory caption under the figures (False if given in the manuscript)
DODGE_STEP = 0.20              # spacing between land covers inside a depth class (fig_prof_03 / 04)

LC = ["Forest", "Shrubland", "Grassland", "Reforestation"]
DL = ["0-10", "10-20", "20-30", "30-60", "60-90"]
DEPTH_CM = dict(zip(DL, [10, 20, 30, 60, 90]))
SITE_ORDER = ["NW", "CH", "SE"]                               # order of landscapes on boxplots
SITE_LABEL = {"NW": "North-West (NW)", "CH": "Central Highlands (CH)", "SE": "South-East (SE)"}
SITE_MAP = {"AF": "NW", "AT": "CH", "IV": "SE"}

FILL = {"Forest": "#2E8B57", "Shrubland": "#98FB98", "Grassland": "#DAA520", "Reforestation": "#A0522D"}
DARK = {"Forest": "#1E5E3B", "Shrubland": "#3F9F5F", "Grassland": "#A9780F", "Reforestation": "#6B3410"}
C_SITE = {"CH": "#2E8B3E", "NW": "#A0522D", "SE": "#1F3FD0"}
D_SITE = {"CH": "#1B5E26", "NW": "#6B3410", "SE": "#14288F"}
C_DEPTH = dict(zip(DL, ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]))
D_DEPTH = dict(zip(DL, ["#5B7FA6", "#4A7BA8", "#3A6FA3", "#1F5A94", "#08306B"]))
PAL = {"site": C_SITE, "lc": FILL, "depth": C_DEPTH}
PAL_DARK = {"site": D_SITE, "lc": DARK, "depth": D_DEPTH}
LEVELS = {"site": SITE_ORDER, "lc": LC, "depth": DL}
LEG_TITLE = {"site": "Landscape", "lc": "Land cover", "depth": "Depth (cm)"}
FACTOR = {"site": "landscape", "lc": "land cover", "depth": "depth"}
PLURAL = {"site": "landscapes", "lc": "land covers", "depth": "depths"}
MARKER = [4, 6, 8, 10, 12]                                    # marker size (points) increases with depth

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.8})


def wrap(s, w):
    return "\n".join(textwrap.wrap(s, w))


def fmt_p(p, digits=2):
    return f"{p:.{digits}e}" if p < 0.001 else f"{p:.3f}"


def save(fig, name, fig_dir, title=None):
    if title:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(title, y=0.985, fontweight="bold", fontsize=14)
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def caption(fig, text, y=-0.02):
    if ADD_CAPTION:
        fig.text(0.5, y, text, ha="center", va="top", fontsize=8, color="#444444")


def style_axes(ax):
    ax.grid(True, color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------------------
# A. Boxplots
# ---------------------------------------------------------------------------------------------
Y_LETTER, Y_N, Y_P, Y_LIM = 0.79, 0.17, 0.125, (0.10, 0.82)


def make_box(d, ann, fig_id, x_var, facet_var, fig_dir, name):
    a = ann[ann.figure == fig_id]
    levels = LEVELS[x_var]
    panels = [None] if facet_var is None else LEVELS[facet_var]
    n_panel = len(panels)
    width = 2.9 * n_panel + 2.2 if facet_var else 5.8
    fig, axs = plt.subplots(1, n_panel, figsize=(width, 5), sharey=True, squeeze=False)
    for ax, panel in zip(axs[0], panels):
        sub = d if panel is None else d[d[facet_var] == panel]
        ap = a if panel is None else a[a.panel == panel]
        data = [sub.loc[sub[x_var] == lv, "SFI"].values for lv in levels]
        pos = np.arange(1, len(levels) + 1)
        bp = ax.boxplot(data, positions=pos, widths=0.65, patch_artist=True, showfliers=False, whis=1.5,
                        medianprops=dict(color="#404040", lw=1.2), boxprops=dict(color="#404040", lw=0.9),
                        whiskerprops=dict(color="#404040", lw=0.9), capprops=dict(color="#404040", lw=0.9))
        for patch, lv in zip(bp["boxes"], levels):
            patch.set_facecolor(matplotlib.colors.to_rgba(PAL[x_var][lv], 0.55))
        for i, cap in enumerate(bp["caps"]):                  # whisker end bars: narrow
            cx = pos[i // 2]
            cap.set_xdata([cx - 0.15, cx + 0.15])
        rng = np.random.default_rng(1)
        for x, lv, vals in zip(pos, levels, data):
            ax.scatter(x + rng.uniform(-0.17, 0.17, len(vals)), vals, s=5, color=PAL[x_var][lv], alpha=0.45,
                       linewidths=0, zorder=2)
            ax.plot(x, vals.mean(), marker="+", color="black", ms=9, mew=1.4, zorder=5)
            r = ap[ap.x_level == lv].iloc[0]
            ax.text(x, Y_LETTER, r.letter, ha="center", va="center", fontsize=14, fontweight="bold", color=PAL_DARK[x_var][lv])
            ax.text(x, Y_N, f"n={int(r.n_samples)}", ha="center", va="center", fontsize=8.5, style="italic", color="#595959")
        ax.text((len(levels) + 1) / 2, Y_P, f"p = {fmt_p(ap.p_panel.iloc[0])}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", style="italic")
        ax.set_xlim(0.4, len(levels) + 0.6)
        ax.set_ylim(*Y_LIM)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_xticks([])
        ax.grid(True, axis="y", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
        if panel is not None:
            ax.set_title(SITE_LABEL[panel] if facet_var == "site" else panel, fontweight="bold", fontsize=11,
                         backgroundcolor="#f2f2f2")
    axs[0, 0].set_ylabel("Soil Fertility Index", fontweight="bold")
    handles = [Patch(facecolor=matplotlib.colors.to_rgba(PAL[x_var][lv], 0.55), edgecolor="#404040",
                     label=(f"{lv} cm" if x_var == "depth" else SITE_LABEL[lv] if x_var == "site" else lv)) for lv in levels]
    leg = fig.legend(handles=handles, title=LEG_TITLE[x_var], loc="center left", bbox_to_anchor=(0.985, 0.5), frameon=False,
                     alignment="left")
    leg.get_title().set_fontweight("bold")
    pooled = [PLURAL[v] for v in ("site", "lc", "depth") if v not in (x_var, facet_var)]
    cap = (f"Boxes: median and interquartile range; whiskers: 1.5 x IQR; + = mean; points = individual samples ("
           f"{' and '.join(pooled)} pooled); n = number of samples. Letters: Holm-adjusted pairwise comparisons of "
           f"{FACTOR[x_var]} levels from the linear mixed model" + (f", within each {FACTOR[facet_var]}" if facet_var else "") +
           f"; the same letter indicates no significant difference. p: Wald F-test of {FACTOR[x_var]}" +
           (f" within each {FACTOR[facet_var]}." if facet_var else
            " (other factors averaged with equal weights; descriptive when interactions are significant)."))
    fig.tight_layout()
    caption(fig, wrap(cap, int(max(60, width * 12.5))))
    save(fig, name, fig_dir)


# ---------------------------------------------------------------------------------------------
# B1. Pooled depth profile and profiles by landscape (depth in true cm, as the sampled layers)
# ---------------------------------------------------------------------------------------------
def cap_v(ax, x, y, lo, hi, color, vertical=True, c=1.6):
    """CI bar with end caps; vertical=True: depth on y axis."""
    if vertical:
        ax.hlines(y, lo, hi, color=color, lw=1.1, alpha=0.85)
        ax.vlines([lo, hi], y - c, y + c, color=color, lw=1.1, alpha=0.85)
    else:
        ax.vlines(x, lo, hi, color=color, lw=1.1, alpha=0.85)
        ax.hlines([lo, hi], x - c, x + c, color=color, lw=1.1, alpha=0.85)


def depth_size_legend(fig, y=0.40, title="Depth (cm)", anchor_x=0.985):
    h = [Line2D([], [], marker="o", ls="", mfc="#999999", mec="#333333", ms=MARKER[i] * 0.8 + 1, label=l)
         for i, l in enumerate(DL)]
    leg = fig.legend(handles=h, title=title, alignment="left", loc="upper left", bbox_to_anchor=(anchor_x, y), frameon=False)
    leg.get_title().set_fontweight("bold")


def fig_prof_01(ann, n_obs, fig_dir):
    g = ann[ann.figure == "global_depth"].copy()
    g["cm"] = g.x_level.map(DEPTH_CM)
    p = g.p_panel.iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 6.8))
    ax.plot(g.emm, g.cm, color="#404040", lw=1.8, zorder=2)
    for r in g.itertuples():
        cap_v(ax, r.emm, r.cm, r.ci_low, r.ci_high, "#404040")
        ax.plot(r.emm, r.cm, "o", ms=MARKER[DL.index(r.x_level)], mfc="#999999", mec="#333333", mew=0.7, zorder=3)
        ax.text(r.ci_high + 0.004, r.cm, r.letter, color="#404040", fontweight="bold", fontsize=12, va="center")
    ax.set_ylim(98, 2); ax.set_xlim(0.40, 0.55)
    ax.set_yticks(list(DEPTH_CM.values())); ax.set_xticks(np.arange(0.40, 0.551, 0.05))
    ax.set_xlabel("Soil Fertility Index", fontweight="bold"); ax.set_ylabel("Depth (cm)", fontweight="bold")
    style_axes(ax)
    depth_size_legend(fig, y=0.62)
    fig.tight_layout()
    caption(fig, f"n={n_obs}\nDepth effect (Wald F-test, mixed model): p={fmt_p(p, 1)}\n" +
            wrap("Points: estimated marginal means (landscapes and land covers pooled with equal weights; descriptive when "
                 "interactions are significant) \u00b1 95% CI. Depths sharing a letter do not differ (Holm).", 75))
    save(fig, "fig_prof_01_global_depth", fig_dir)


def fig_prof_02(a, s, fig_dir):
    a = a.copy()
    site_order = ["CH", "NW", "SE"]
    a["label"] = a.letter_site + a.letter_depth
    s = s.set_index("site")
    by_site = lambda f: " | ".join(f"{k}: {f(s.loc[k])}" for k in ["NW", "CH", "SE"])
    cap = (by_site(lambda r: f"n={int(r.n_samples)}") + "\n"
           "Depth effect within landscape (Wald F-test, mixed model): " + by_site(lambda r: f"p={fmt_p(r.p_depth, 1)}") + "\n" +
           wrap("Points: estimated marginal means \u00b1 95% CI. Letters: capital = landscapes compared within a depth; "
                "lowercase = depths compared within a landscape (Holm); the same letter indicates no significant difference.", 95))
    for layout, size in (("vertical", (7.5, 7.2)), ("horizontal", (9.5, 5.8))):
        vert = layout == "vertical"
        fig, ax = plt.subplots(figsize=size)
        for st in site_order:
            g = a[a.site == st].sort_values("depth_cm"); c = C_SITE[st]
            x, y = (g.emm, g.depth_cm) if vert else (g.depth_cm, g.emm)
            ax.plot(x, y, color=c, lw=1.8, zorder=2)
            for r in g.itertuples():
                cap_v(ax, r.depth_cm, r.depth_cm, r.ci_low, r.ci_high, c, vertical=vert)
                px, py = (r.emm, r.depth_cm) if vert else (r.depth_cm, r.emm)
                ax.plot(px, py, "o", ms=MARKER[DL.index(r.depth)], color=c, mec=c, zorder=3)
                if vert:
                    kw = {"CH": (r.ci_high + 0.004, r.depth_cm, "left", "center"), "NW": (r.ci_low - 0.004, r.depth_cm, "right", "center"),
                          "SE": (r.emm, r.depth_cm - 5, "center", "center")}[st]
                else:
                    kw = {"CH": (r.depth_cm, r.ci_high + 0.007, "center", "bottom"), "NW": (r.depth_cm, r.ci_low - 0.007, "center", "top"),
                          "SE": (r.depth_cm + 6, r.emm + 0.006, "center", "center")}[st]
                ax.text(kw[0], kw[1], r.label, ha=kw[2], va=kw[3], color=c, fontweight="bold", fontsize=12)
        if vert:
            ax.set_ylim(95, 3); ax.set_xlim(0.33, 0.64); ax.set_yticks(list(DEPTH_CM.values())); ax.set_xticks(np.arange(0.35, 0.601, 0.05))
            ax.set_xlabel("Soil Fertility Index", fontweight="bold"); ax.set_ylabel("Depth (cm)", fontweight="bold")
        else:
            ax.set_xlim(3, 97); ax.set_ylim(0.35, 0.62); ax.set_xticks(list(DEPTH_CM.values())); ax.set_yticks(np.arange(0.35, 0.601, 0.05))
            ax.set_xlabel("Depth (cm)", fontweight="bold"); ax.set_ylabel("Soil Fertility Index", fontweight="bold")
        style_axes(ax)
        h = [Line2D([], [], marker="o", ls="", color=C_SITE[k], ms=8, label=k) for k in site_order]
        l1 = fig.legend(handles=h, title="Landscape", alignment="left", loc="upper left", bbox_to_anchor=(0.985, 0.62), frameon=False)
        l1.get_title().set_fontweight("bold")
        depth_size_legend(fig, y=0.45)
        fig.tight_layout()
        caption(fig, cap)
        save(fig, f"fig_prof_02_by_landscape_{layout}", fig_dir)


# ---------------------------------------------------------------------------------------------
# B2. Depth classes equally spaced, land covers offset inside each class (fig_prof_03 and fig_prof_04)
# ---------------------------------------------------------------------------------------------
def draw_equidistant(ax, df, layout, dodge, lim, label=True, star=False, groups=LC):
    vert = layout == "vertical"
    off = {g: (i - (len(groups) - 1) / 2) * DODGE_STEP for i, g in enumerate(groups)}
    pad, cap = 0.012 * (lim[1] - lim[0]), 0.07
    for g in groups:
        s = df[df.lc == g].copy()
        s["di"] = s.depth.map({k: i + 1 for i, k in enumerate(DL)})
        s = s.sort_values("di")
        pos = s.di + (off[g] if dodge else 0)
        c, dk = FILL[g], DARK[g]
        if vert:
            ax.plot(s.emm, pos, color=dk, lw=1.8, zorder=2)
            ax.hlines(pos, s.ci_low, s.ci_high, color=dk, lw=1.1, alpha=0.85, zorder=2)
            ax.vlines(s.ci_low, pos - cap, pos + cap, color=dk, lw=1.1, alpha=0.85)
            ax.vlines(s.ci_high, pos - cap, pos + cap, color=dk, lw=1.1, alpha=0.85)
        else:
            ax.plot(pos, s.emm, color=dk, lw=1.8, zorder=2)
            ax.vlines(pos, s.ci_low, s.ci_high, color=dk, lw=1.1, alpha=0.85, zorder=2)
            ax.hlines(s.ci_low, pos - cap, pos + cap, color=dk, lw=1.1, alpha=0.85)
            ax.hlines(s.ci_high, pos - cap, pos + cap, color=dk, lw=1.1, alpha=0.85)
        for r, pp in zip(s.itertuples(), pos):
            ax.plot(*((r.emm, pp) if vert else (pp, r.emm)), "o", ms=MARKER[r.di - 1], mfc=c, mec="#333333", mew=0.7, zorder=3)
            txt = []
            if label:
                txt.append((r.label, 11))
            if star and r.star:
                txt.append(("*", 20))
            for t, fs in txt:
                if vert:
                    ax.text(r.ci_high + pad, pp, t, color=dk, fontweight="bold", fontsize=fs, va="center", ha="left")
                else:
                    ax.text(pp, r.ci_high + pad, t, color=dk, fontweight="bold", fontsize=fs, va="bottom", ha="center")
    ticks = np.arange(1, 6)
    if vert:
        ax.set_ylim(5.6, 0.4); ax.set_yticks(ticks); ax.set_yticklabels(DL); ax.set_xlim(*lim)
        ax.set_xlabel("Soil Fertility Index", fontweight="bold"); ax.set_ylabel("Depth (cm)", fontweight="bold")
    else:
        ax.set_xlim(0.4, 5.6); ax.set_xticks(ticks); ax.set_xticklabels(DL); ax.set_ylim(*lim)
        ax.set_ylabel("Soil Fertility Index", fontweight="bold"); ax.set_xlabel("Depth (cm)", fontweight="bold")
    style_axes(ax)


def auto_lim(df, vert, extra=0.0):
    lo, hi = df.ci_low.min(), df.ci_high.max()
    sp = hi - lo
    return lo - 0.03 * sp, hi + (0.15 if vert else 0.12) * sp + extra


def lc_legend(fig, facet=False):
    if not facet:
        h = [Line2D([], [], marker="o", ls="", mfc=FILL[g], mec="#333333", ms=8, label=g) for g in LC]
        l1 = fig.legend(handles=h, title="Land cover", alignment="left", loc="upper left", bbox_to_anchor=(0.985, 0.62), frameon=False)
        l1.get_title().set_fontweight("bold")
        depth_size_legend(fig, y=0.40)
    else:
        depth_size_legend(fig, y=0.62)


def fig_prof_03(ann, d, fig_dir):
    low = ann[ann.figure == "landcover_by_depth"][["panel", "x_level", "letter", "emm", "ci_low", "ci_high"]]
    low = low.rename(columns={"panel": "lc", "x_level": "depth", "letter": "lower"})
    up = ann[ann.figure == "depth_by_landcover"][["x_level", "panel", "letter"]].rename(
        columns={"x_level": "lc", "panel": "depth", "letter": "upper"})
    g3 = low.merge(up, on=["lc", "depth"])
    g3["label"] = g3.upper.str.upper() + g3.lower
    p3 = ann[ann.figure == "landcover_by_depth"].drop_duplicates("panel").set_index("panel").p_panel
    n3 = d.lc.value_counts()

    def cap3(w, facet):
        t1 = " | ".join(f"{g}: n={n3[g]}" for g in LC)
        t2 = "Depth effect within land cover (Wald F-test): " + " | ".join(f"{g} p={fmt_p(p3[g], 1)}" for g in LC)
        t3 = ("Points: estimated marginal means (landscapes pooled with equal weights; model including the land cover x depth "
              "interaction) \u00b1 95% CI. Depth classes are equally spaced (not to scale)."
              + ("" if facet else " Land covers are offset within each class for readability.")
              + " Letters: capital = land covers compared within a depth; lowercase = depths compared within a land cover (Holm); "
                "the same letter indicates no significant difference.")
        return "\n".join([wrap(t1, w), wrap(t2, w), wrap(t3, w)])

    specs = [("vertical", "dodge", 8.5, 9.0), ("vertical", "facet", 14.0, 7.2),
             ("horizontal", "dodge", 11.5, 6.8), ("horizontal", "facet", 10.5, 8.5)]
    for layout, mode, w, h in specs:
        vert = layout == "vertical"
        lim = auto_lim(g3, vert)
        if mode == "dodge":
            fig, ax = plt.subplots(figsize=(w, h))
            draw_equidistant(ax, g3, layout, True, lim)
            lc_legend(fig)
        else:
            fig, axs = plt.subplots(*((1, 4) if vert else (2, 2)), figsize=(w, h), sharex=True, sharey=True)
            axs = np.ravel(axs)
            for ax, g in zip(axs, LC):
                draw_equidistant(ax, g3[g3.lc == g], layout, False, lim, groups=[g])
                ax.set_title(g, fontweight="bold", fontsize=11, backgroundcolor="#f2f2f2")
            if vert:
                for ax in axs[1:]:
                    ax.set_ylabel("")
            else:
                for ax in axs[[1, 3]]:
                    ax.set_ylabel("")
                for ax in axs[[0, 1]]:
                    ax.set_xlabel("")
            lc_legend(fig, facet=True)
        caption(fig, cap3(int(w * 11), mode == "facet"))
        title = "Soil Fertility Index (SFI) by soil depth within each land cover type" if (layout, mode) == ("vertical", "facet") else None
        fig.tight_layout()
        save(fig, f"fig_prof_03_by_landcover_{layout}_{mode}", fig_dir, title)


def fig_prof_04(lp, fig_dir):
    lp = lp.copy()
    lp["lc"] = lp["lc"].astype(str)
    lp["star"] = lp.significant.fillna(False).astype(bool)
    sites = ["CH", "SE", "NW"]
    cap = wrap("Points: estimated means \u00b1 95% CI from the model including the landscape x land cover x depth interaction. "
               "Depth classes are equally spaced (not to scale); land covers are offset within each class for readability. "
               "* = land cover differs from grassland (reference) at that depth within the landscape "
               "(Holm-adjusted across depths, p < 0.05).", 150)
    lim_v, lim_h = auto_lim(lp, True), auto_lim(lp, False, extra=0.05)       # extra room above CIs for the asterisks
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 8), sharey=True, sharex=True)
    for ax, s in zip(axs, sites):
        draw_equidistant(ax, lp[lp.site == s], "vertical", True, lim_v, label=False, star=True)
        ax.set_title(SITE_LABEL[s], fontweight="bold", fontsize=11, backgroundcolor="#f2f2f2")
        if s != "CH":
            ax.set_ylabel("")
    lc_legend(fig)
    caption(fig, cap)
    fig.tight_layout()
    save(fig, "fig_prof_04_landcover_within_landscape_vertical", fig_dir)

    fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharey=True, sharex=True)
    for ax, s in zip(axs, sites):
        draw_equidistant(ax, lp[lp.site == s], "horizontal", True, lim_h, label=False, star=True)
        ax.set_title(SITE_LABEL[s], fontweight="bold", fontsize=11, backgroundcolor="#f2f2f2")
    for ax in axs[:2]:
        ax.set_xlabel("")
    lc_legend(fig)
    caption(fig, wrap(cap, 120).replace("\n", "\n"), y=-0.01)
    fig.tight_layout()
    save(fig, "fig_prof_04_landcover_within_landscape_horizontal", fig_dir,
         "Soil Fertility Index (SFI) by soil depth for the four land cover types within each landscape")


# ---------------------------------------------------------------------------------------------
def main(data_file: Path, out_dir: Path) -> None:
    tab_dir, fig_dir = out_dir / "tables", out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(data_file, sep=";")
    d["site"] = d.Site.map(SITE_MAP)
    d["lc"] = d.Landuse.astype(str)
    d["depth"] = d.Depth.astype(int).map(dict(enumerate(DL, 1)))
    assert d[["site", "lc", "depth", "SFI"]].notna().all().all(), "unmapped codes in the data"
    assert d.SFI.between(*Y_LIM).all(), "SFI values outside the boxplot y-axis limits"
    ann = pd.read_csv(tab_dir / "fig_all_annotations.csv")
    print("Boxplots")
    for i, (fid, x, f) in enumerate([("global_site", "site", None), ("global_landcover", "lc", None),
                                     ("site_by_landcover", "lc", "site"), ("landcover_by_site", "site", "lc")], 1):
        make_box(d, ann, fid, x, f, fig_dir, f"fig_box_{i:02d}_{fid}")
    print("Depth profiles")
    fig_prof_01(ann, len(d), fig_dir)
    fig_prof_02(pd.read_csv(tab_dir / "fig_depth_annotations.csv"), pd.read_csv(tab_dir / "fig_depth_site_annotations.csv"), fig_dir)
    fig_prof_03(ann, d, fig_dir)
    fig_prof_04(pd.read_csv(tab_dir / "fig_lc_depth_site_profiles.csv"), fig_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/DATA_LM.csv"), help="input CSV (semicolon-separated)")
    ap.add_argument("--out", type=Path, default=Path("outputs/lmm"), help="output directory (same as 01_sfi_lmm_analysis.py)")
    args, _ = ap.parse_known_args()
    main(args.data, args.out)
