#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_sfi_lmm_analysis.py
======================
Linear mixed-effects analysis of the Soil Fertility Index (SFI) across three
Malagasy landscapes (NW, CH, SE), four land-cover types and five soil depths.

Purpose
-------
1. Account for the non-independence of depth increments sampled in the same plot
   (plot = grouping unit; random intercept and random depth slope).
2. Test landscape x land cover and landscape x depth interactions
   (likelihood-ratio tests) and report planned contrasts with 95% CIs.
3. Check residual assumptions (diagnostic plots and residual variability by group).
4. Report model fit (marginal / conditional R2, ICC; RMSE, RPD and RPIQ of the fitted values).
5. Export the results of every analysis (with p-values, letters last) to Excel,
   and the annotation tables (letters, n, p-values) used by 02_sfi_lmm_figures.py.

Statistical model (final)
-------------------------
    SFI ~ landscape * land_cover + landscape * depth
          + (random intercept + random linear depth slope | plot)
    with intercept and slope variances estimated separately for each landscape
    (intercept and slope assumed uncorrelated within a landscape); the residual
    variance is common to all observations.

Input
-----
    DATA_LM.csv  (semicolon-separated; columns: ID_LABO, Site, Samples_ID, Depth,
                  Landuse, PH, P, K, N, C, CEC, SABLE, ARGILE, LIMON, TEXTURE,
                  SFI, SFI_class). Plot identity is derived from Samples_ID
                  (e.g. "Iv_4_Comp12_3" -> plot "Iv_4_Comp12", depth code 3).

Usage
-----
    python scripts/07_sfi_mixed_models/01_sfi_lmm_analysis.py --data data/DATA_LM.csv --out outputs/lmm
    In Jupyter: paste the script in a cell and run it (default paths, relative to the repository root: data/DATA_LM.csv,
    results in outputs/lmm), or call main(Path("data/DATA_LM.csv"), Path("outputs/lmm")) directly.

Requirements
------------
    Python >= 3.10: numpy, pandas, scipy, statsmodels, patsy, matplotlib, openpyxl
    (exact versions are written to <out>/logs/environment.txt).

References
----------
    Holm S (1979) Scand J Stat 6:65-70.
    Nakagawa S, Schielzeth H (2013) Methods Ecol Evol 4:133-142.
    Nakagawa S, Johnson PCD, Schielzeth H (2017) J R Soc Interface 14:20170213.
    Gelman A, Stern H (2006) Am Stat 60:328-331.
    Seabold S, Perktold J (2010) Proc 9th Python in Science Conf.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 0. Settings
# ----------------------------------------------------------------------------
ALPHA = 0.05                                # significance level (letters, CIs)
SITE_MAP = {"AF": "NW", "AT": "CH", "IV": "SE"}
# The first level of each factor is the reference level of the treatment coding.
SITES = ["CH", "SE", "NW"]
LANDCOVERS = ["Grassland", "Forest", "Shrubland", "Reforestation"]
DEPTHS = ["0-10", "10-20", "20-30", "30-60", "60-90"]
DEPTH_CODE = {1: "0-10", 2: "10-20", 3: "20-30", 4: "30-60", 5: "60-90"}
DEPTH_CM = {"0-10": 10, "10-20": 20, "20-30": 30, "30-60": 60, "60-90": 90}  # lower limit
FIXED_FORMULA = "SFI ~ site*lc + site*depth"      # final fixed-effect structure
# Degrees of freedom of the t-based Wald tests = n_plots - N_BETWEEN_PARAMS, where N_BETWEEN_PARAMS counts the
# between-plot fixed effects: intercept (1) + site (2) + land cover (3) + site:land cover (6).
# Conservative approximation (statsmodels has no Kenward-Roger / Satterthwaite correction).
N_BETWEEN_PARAMS = 12
MAIN_LOG = logging.getLogger("SFI")


# ----------------------------------------------------------------------------
# 1. Data loading and validation
# ----------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Read DATA_LM.csv, derive plot/site/depth variables and run sanity checks."""
    d = pd.read_csv(path, sep=";")
    required = {"Site", "Samples_ID", "Depth", "Landuse", "SFI"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    d["SFI"] = pd.to_numeric(d["SFI"], errors="raise")

    d["plot"] = d["Samples_ID"].str.replace(r"_\d+$", "", regex=True)
    d["site"] = pd.Categorical(d["Site"].map(SITE_MAP), categories=SITES)
    d["lc"] = pd.Categorical(d["Landuse"], categories=LANDCOVERS)
    d["depth"] = pd.Categorical(d["Depth"].astype(int).map(DEPTH_CODE), categories=DEPTHS)
    d["dc"] = d["Depth"].astype(int) - 3     # centred depth rank (-2 ... +2) for random slope
    for s in SITES:                          # site indicators (site-specific variance components)
        d[f"is{s}"] = (d["site"] == s).astype(float)

    # --- sanity checks -------------------------------------------------------
    assert d["Samples_ID"].is_unique, "Duplicated Samples_ID"
    assert d[["SFI", "site", "lc", "depth"]].notna().all().all(), "Missing values / unmapped codes"
    assert (d.groupby("plot", observed=True)[["site", "lc"]].nunique() == 1).all().all(), \
        "A plot is assigned to several sites or land covers"
    return d.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Mixed-model helpers
# ----------------------------------------------------------------------------
def variance_components(slope: bool = True) -> dict:
    """Site-specific variance components: one intercept (and slope) variance per site."""
    vc = {}
    for s in SITES:
        vc[f"i{s}"] = f"0 + is{s}"
        if slope:
            vc[f"s{s}"] = f"0 + is{s}:dc"
    return vc


def fit_mixed(formula: str, d: pd.DataFrame, reml: bool = True,
              structure: str = "D"):
    """
    Fit a linear mixed model with plot as grouping unit.

    structure:
      A  random intercept, homogeneous variance
      B  random intercept + depth slope (correlated), homogeneous variance
      C  random intercept, landscape-specific variance
      D  random intercept + depth slope, landscape-specific variances (final)
    """
    kw = dict(method=["bfgs", "lbfgs", "powell"], maxiter=3000)   # optimisers tried in sequence
    if structure == "A":
        m = smf.mixedlm(formula, d, groups=d["plot"])
    elif structure == "B":
        m = smf.mixedlm(formula, d, groups=d["plot"], re_formula="~dc")
    elif structure in ("C", "D"):
        m = smf.mixedlm(formula, d, groups=d["plot"], re_formula="0",
                        vc_formula=variance_components(slope=(structure == "D")))
    else:
        raise ValueError(structure)
    res = m.fit(reml=reml, **kw)
    if not res.converged:
        MAIN_LOG.warning("Model did not converge: %s (structure %s)", formula, structure)
    return res


def n_params(res) -> int:
    """Number of estimated parameters (fixed + random (co)variances + residual variance)."""
    return len(res.params) + 1


def information_criteria(res, n_obs: int, n_plots: int) -> dict:
    k, ll = n_params(res), res.llf
    return {"k": k, "logLik_ML": ll, "AIC": -2 * ll + 2 * k,
            "BIC_nobs": -2 * ll + np.log(n_obs) * k,
            "BIC_nplots": -2 * ll + np.log(n_plots) * k}


def lr_test(full_formula: str, reduced_formula: str, d: pd.DataFrame,
            structure: str = "D", full_res=None) -> dict:
    """Likelihood-ratio test of nested fixed-effect structures (ML fits)."""
    full = full_res or fit_mixed(full_formula, d, reml=False, structure=structure)
    red = fit_mixed(reduced_formula, d, reml=False, structure=structure)
    chi2 = 2 * (full.llf - red.llf)
    df = len(full.fe_params) - len(red.fe_params)
    return {"chi2": chi2, "df": df, "p": stats.chi2.sf(chi2, df)}


def nakagawa_r2(res, d: pd.DataFrame) -> dict:
    """
    Marginal and conditional R2 (Nakagawa & Schielzeth 2013; Nakagawa et al. 2017)
    for the site-specific variance-component model (random intercept + depth slope).
    """
    k = len(res.fe_params)
    scale = res.scale
    vc = {n.replace(" Var", ""): v * scale for n, v in zip(res.params.index[k:], res.params.values[k:])}
    var_f = np.var(res.model.exog @ res.fe_params.values)
    var_r = np.mean([vc[f"i{s}"] + dc**2 * vc[f"s{s}"]
                     for s, dc in zip(d["site"].astype(str), d["dc"])])
    tot = var_f + var_r + scale
    return {"var_fixed": var_f, "var_random": var_r, "var_resid": scale,
            "R2_marginal": var_f / tot, "R2_conditional": (var_f + var_r) / tot}


# ----------------------------------------------------------------------------
# 3. Multiple-comparison tools
# ----------------------------------------------------------------------------
def holm_adjust(p) -> np.ndarray:
    """Holm (1979) step-down adjustment of a vector of p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def compact_letters(levels, means: dict, p_adj: dict, alpha: float = ALPHA) -> dict:
    """
    Compact letter display from adjusted pairwise p-values.
    Levels are ordered by decreasing mean; letter 'a' = highest group.
    p_adj: {frozenset({level1, level2}): adjusted p-value}.
    Each letter is a maximal set of levels with no significant pairwise difference; letters that are
    not needed to link any pair of non-different levels are dropped.
    """
    order = sorted(levels, key=lambda x: -means[x])
    differs = lambda a, b: p_adj.get(frozenset((a, b)), 1.0) < alpha
    cliques = []
    for r in range(len(order), 0, -1):                      # largest sets first
        for c in itertools.combinations(order, r):
            if all(not differs(a, b) for a, b in itertools.combinations(c, 2)) \
                    and not any(set(c) <= set(o) for o in cliques):
                cliques.append(c)
    links = lambda cs: {frozenset(p) for c in cs for p in itertools.combinations(c, 2)} | {frozenset((x,)) for c in cs for x in c}
    for c in sorted(cliques, key=len):                      # drop redundant letters, smallest first
        rest = [o for o in cliques if o is not c]
        if links(rest) >= links([c]):
            cliques = rest
    cliques.sort(key=lambda c: [order.index(x) for x in c])
    letters = {lv: "" for lv in levels}
    for i, c in enumerate(cliques):
        for lv in c:
            letters[lv] += chr(97 + i)
    return letters


# ----------------------------------------------------------------------------
# 4. Contrast engine (estimated marginal means and linear contrasts)
# ----------------------------------------------------------------------------
class ContrastEngine:
    """
    Linear contrasts of fixed effects with t-based inference.

    Estimated marginal means (EMMs) average model predictions with equal weights
    over the levels of the factors not shown. Degrees of freedom are fixed at
    (number of plots - number of between-plot parameters); this is a conservative
    approximation because Kenward-Roger / Satterthwaite corrections are not
    available in statsmodels.
    """

    def __init__(self, res, formula_rhs: str, n_plots: int, n_between_params: int = N_BETWEEN_PARAMS):
        self.k = len(res.fe_params)
        self.names = list(res.fe_params.index)
        self.beta = res.fe_params.values
        cov = res.cov_params()
        assert list(cov.index[: self.k]) == list(res.fe_params.index)
        self.V = cov.iloc[: self.k, : self.k].values
        self.df = n_plots - n_between_params
        self.grid = pd.DataFrame(list(itertools.product(SITES, LANDCOVERS, DEPTHS)),
                                 columns=["site", "lc", "depth"])
        for col, cats in (("site", SITES), ("lc", LANDCOVERS), ("depth", DEPTHS)):
            self.grid[col] = pd.Categorical(self.grid[col], categories=cats)
        X = patsy.dmatrix(formula_rhs, self.grid, return_type="dataframe")
        assert list(X.columns) == list(res.fe_params.index), "Design/column mismatch"
        self.X = X.values

    def mean_vector(self, site=None, lc=None, depth=None) -> np.ndarray:
        mask = np.ones(len(self.grid), bool)
        for col, val in (("site", site), ("lc", lc), ("depth", depth)):
            if val is not None:
                mask &= (self.grid[col] == val).values
        w = mask / mask.sum()
        return w @ self.X

    def contrast(self, L: np.ndarray, L2: np.ndarray | None = None) -> dict:
        L = L if L2 is None else L - L2
        est, se = L @ self.beta, np.sqrt(L @ self.V @ L)
        t_crit = stats.t.ppf(1 - ALPHA / 2, self.df)
        return {"estimate": est, "SE": se, "CI_low": est - t_crit * se,
                "CI_high": est + t_crit * se, "p_raw": 2 * stats.t.sf(abs(est / se), self.df)}

    def wald_F(self, L: np.ndarray) -> dict:
        """Joint Wald F-test of q linear contrasts (rows of L) equal to zero."""
        b, C = L @ self.beta, L @ self.V @ L.T
        F = float(b @ np.linalg.solve(C, b)) / L.shape[0]
        return {"F": F, "df1": L.shape[0], "df2": self.df, "p": stats.f.sf(F, L.shape[0], self.df)}

    def pairwise(self, levels, getter, family: str) -> pd.DataFrame:
        rows = []
        for a, b in itertools.combinations(levels, 2):
            r = self.contrast(getter(a), getter(b))
            rows.append({"family": family, "contrast": f"{a} - {b}", **r})
        t = pd.DataFrame(rows)
        t["p_holm"] = holm_adjust(t["p_raw"])
        return t


def all_pairwise_families(eng: ContrastEngine) -> pd.DataFrame:
    """Holm-adjusted pairwise contrasts, one family per comparison set."""
    fams = [
        eng.pairwise(SITES, lambda s: eng.mean_vector(site=s), "Site (all depths, all land covers)"),
        eng.pairwise(LANDCOVERS, lambda l: eng.mean_vector(lc=l), "Land cover (all sites, all depths)"),
        eng.pairwise(DEPTHS, lambda x: eng.mean_vector(depth=x), "Depth (all sites)"),
    ]
    for s in SITES:
        fams.append(eng.pairwise(LANDCOVERS, lambda l, s=s: eng.mean_vector(site=s, lc=l), f"Land cover within {s}"))
        fams.append(eng.pairwise(DEPTHS, lambda x, s=s: eng.mean_vector(site=s, depth=x), f"Depth within {s}"))
    for l in LANDCOVERS:
        fams.append(eng.pairwise(SITES, lambda s, l=l: eng.mean_vector(site=s, lc=l), f"Site within {l}"))
    for dp in DEPTHS:
        fams.append(eng.pairwise(SITES, lambda s, dp=dp: eng.mean_vector(site=s, depth=dp), f"Site within depth {dp}"))
    return pd.concat(fams, ignore_index=True)


def estimated_marginal_means(eng: ContrastEngine) -> pd.DataFrame:
    rows = []
    def add(effect, a, b, vec):
        r = eng.contrast(vec)
        rows.append({"effect": effect, "level1": a, "level2": b, "EMM": r["estimate"], "SE": r["SE"],
                     "CI_low": r["CI_low"], "CI_high": r["CI_high"]})
    for s in SITES: add("Site", s, "", eng.mean_vector(site=s))
    for l in LANDCOVERS: add("Land cover", l, "", eng.mean_vector(lc=l))
    for x in DEPTHS: add("Depth", x, "", eng.mean_vector(depth=x))
    for s in SITES:
        for l in LANDCOVERS: add("Site x Land cover", s, l, eng.mean_vector(site=s, lc=l))
    for s in SITES:
        for x in DEPTHS: add("Site x Depth", s, x, eng.mean_vector(site=s, depth=x))
    return pd.DataFrame(rows)


def global_wald_tests(eng: ContrastEngine) -> pd.DataFrame:
    """Global Wald F-tests: land cover within each site (3 df) and depth within each site (4 df)."""
    rows = []
    for s in SITES:
        L = np.array([eng.mean_vector(site=s, lc=l) - eng.mean_vector(site=s, lc="Grassland")
                      for l in LANDCOVERS[1:]])
        rows.append({"test": f"Land cover within {s}", **eng.wald_F(L)})
        L = np.array([eng.mean_vector(site=s, depth=x) - eng.mean_vector(site=s, depth=DEPTHS[0])
                      for x in DEPTHS[1:]])
        rows.append({"test": f"Depth within {s}", **eng.wald_F(L)})
    return pd.DataFrame(rows)


def letters_from_families(pw: pd.DataFrame, emm: pd.DataFrame) -> dict:
    """Compact letters for the four families used in the figures."""
    def pdict(family):
        sub = pw[pw["family"] == family]
        return {frozenset(c.split(" - ")): p for c, p in zip(sub["contrast"], sub["p_holm"])}
    out = {}
    e_lc = emm[emm["effect"] == "Site x Land cover"].set_index(["level1", "level2"])["EMM"]
    e_dp = emm[emm["effect"] == "Site x Depth"].set_index(["level1", "level2"])["EMM"]
    for s in SITES:
        out[("LC", s)] = compact_letters(LANDCOVERS, {l: e_lc[(s, l)] for l in LANDCOVERS}, pdict(f"Land cover within {s}"))
        out[("Depth", s)] = compact_letters(DEPTHS, {x: e_dp[(s, x)] for x in DEPTHS}, pdict(f"Depth within {s}"))
    for l in LANDCOVERS:
        out[("Site|LC", l)] = compact_letters(SITES, {s: e_lc[(s, l)] for s in SITES}, pdict(f"Site within {l}"))
    for x in DEPTHS:
        out[("Site|Depth", x)] = compact_letters(SITES, {s: e_dp[(s, x)] for s in SITES}, pdict(f"Site within depth {x}"))
    return out


# ----------------------------------------------------------------------------
# 5. Supplementary analyses
# ----------------------------------------------------------------------------
def landcover_by_depth(d: pd.DataFrame, n_plots: int):
    """Land cover - grassland contrasts at each depth within each site (three-way interaction model).
    Returns (contrast table, table of estimated means per landscape x land cover x depth)."""
    res = fit_mixed("SFI ~ site*lc*depth", d, reml=True)
    k = len(res.fe_params)
    beta, V = res.fe_params.values, res.cov_params().iloc[:k, :k].values
    df = n_plots - N_BETWEEN_PARAMS
    grid = pd.DataFrame(list(itertools.product(SITES, LANDCOVERS, DEPTHS)), columns=["site", "lc", "depth"])
    for col, cats in (("site", SITES), ("lc", LANDCOVERS), ("depth", DEPTHS)):
        grid[col] = pd.Categorical(grid[col], categories=cats)
    X = patsy.dmatrix("site*lc*depth", grid, return_type="dataframe")
    assert list(X.columns) == list(res.fe_params.index)
    X = X.values
    cell = lambda s, l, x: X[((grid.site == s) & (grid.lc == l) & (grid.depth == x)).values][0]
    t_crit = stats.t.ppf(1 - ALPHA / 2, df)
    rows = []
    for s in SITES:
        for l in LANDCOVERS[1:]:
            for x in DEPTHS:
                L = cell(s, l, x) - cell(s, "Grassland", x)
                est, se = L @ beta, np.sqrt(L @ V @ L)
                rows.append({"site": s, "contrast": f"{l} - Grassland", "depth": x, "estimate": est, "SE": se,
                             "CI_low": est - t_crit * se, "CI_high": est + t_crit * se,
                             "p_raw": 2 * stats.t.sf(abs(est / se), df)})
    t = pd.DataFrame(rows)
    t["p_holm_over_depths"] = t.groupby(["site", "contrast"])["p_raw"].transform(lambda x: holm_adjust(x.values))
    # estimated means of each landscape x land cover x depth cell (for the profile figure)
    emm_rows = []
    for s in SITES:
        for l in LANDCOVERS:
            for x in DEPTHS:
                L = cell(s, l, x)
                est, se = L @ beta, np.sqrt(L @ V @ L)
                emm_rows.append({"site": s, "lc": l, "depth": x, "emm": est,
                                 "ci_low": est - t_crit * se, "ci_high": est + t_crit * se})
    return t, pd.DataFrame(emm_rows)


def fit_performance(res, d: pd.DataFrame) -> pd.DataFrame:
    """
    Error-based fit statistics of the fitted values (same data as used for estimation).
      Marginal    : predictions from the fixed effects only
      Conditional : predictions from fixed effects + predicted plot random effects
    RMSE = root mean squared error; RPD = SD(SFI) / RMSE; RPIQ = IQR(SFI) / RMSE.
    """
    y = d["SFI"].values
    sd, iqr = y.std(ddof=1), np.subtract(*np.percentile(y, [75, 25]))
    preds = {"Marginal (fixed effects only)": res.model.exog @ res.fe_params.values,
             "Conditional (fixed + random effects)": res.fittedvalues.values}
    rows = []
    for label, pred in preds.items():
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        rows.append({"predictions": label, "n": len(y), "SD_SFI": sd, "IQR_SFI": iqr,
                     "RMSE": rmse, "RPD": sd / rmse, "RPIQ": iqr / rmse})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 6. Residual diagnostics (figure + numbers)
# ----------------------------------------------------------------------------
def plot_random_intercepts(res, d: pd.DataFrame) -> np.ndarray:
    """Predicted plot random intercepts, each from its own landscape's component, scaled by the landscape SD."""
    site_of = d.drop_duplicates("plot").set_index("plot")["site"].astype(str)
    out = {s: [] for s in SITES}
    for plot, eff in res.random_effects.items():
        s = site_of[plot]
        out[s].append(eff[eff.index.str.startswith(f"i{s}[")].iloc[0])
    return np.concatenate([np.asarray(v) / np.std(v, ddof=1) for v in out.values()])


def residual_diagnostics(res, d: pd.DataFrame, fig_path: Path) -> pd.DataFrame:
    d = d.copy()
    d["fitted"] = res.fittedvalues.values
    d["std_resid"] = (d["SFI"] - d["fitted"]) / np.sqrt(res.scale)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    sm.qqplot(d["std_resid"], line="45", ax=ax[0, 0]); ax[0, 0].set_title("QQ-plot of standardised residuals")
    ax[0, 1].scatter(d["fitted"], d["std_resid"], s=5, alpha=0.35); ax[0, 1].axhline(0, c="red")
    ax[0, 1].set(xlabel="Fitted values", ylabel="Standardised residuals", title="Residuals vs fitted")
    for a, col, title in ((ax[0, 2], "depth", "Residuals by depth"),
                          (ax[1, 0], "site", "Residuals by landscape"),
                          (ax[1, 1], "lc", "Residuals by land cover")):
        d.boxplot(column="std_resid", by=col, ax=a, grid=False); a.set_title(title); a.set_xlabel(""); a.set_ylabel("")
    sm.qqplot(plot_random_intercepts(res, d), line="45", ax=ax[1, 2])
    ax[1, 2].set_title("QQ-plot of plot random intercepts")
    plt.suptitle(""); plt.tight_layout(); plt.savefig(fig_path, dpi=300); plt.close(fig)

    raw = d["SFI"] - d["fitted"]
    rows = [{"metric": "Residual skewness", "value": stats.skew(raw)},
            {"metric": "Residual excess kurtosis", "value": stats.kurtosis(raw)},
            {"metric": "Levene p (residuals by landscape)",
             "value": stats.levene(*[g.values for _, g in raw.groupby(d["site"], observed=True)])[1]},
            {"metric": "Levene p (residuals by depth)",
             "value": stats.levene(*[g.values for _, g in raw.groupby(d["depth"], observed=True)])[1]}]
    for col in ("site", "depth", "lc"):
        for lv, sd in raw.groupby(d[col], observed=True).std().items():
            rows.append({"metric": f"Residual SD, {col} = {lv}", "value": sd})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 7. Annotation tables for the figures
# ----------------------------------------------------------------------------
def build_figure_tables(d, eng, emm, pw, glob, letters) -> dict:
    wald = glob.set_index("test")
    e_lc = emm[emm["effect"] == "Site x Land cover"].set_index(["level1", "level2"])
    e_dp = emm[emm["effect"] == "Site x Depth"].set_index(["level1", "level2"])

    lc_rows = []
    for s in SITES:
        w = wald.loc[f"Land cover within {s}"]
        for l in LANDCOVERS:
            sub = d[(d["site"] == s) & (d["lc"] == l)]
            lc_rows.append({"site": s, "land_cover": l, "letter": letters[("LC", s)][l],
                            "n_samples": len(sub), "n_plots": sub["plot"].nunique(),
                            "mean_raw": sub["SFI"].mean(), "emm": e_lc.loc[(s, l), "EMM"],
                            "ci_low": e_lc.loc[(s, l), "CI_low"], "ci_high": e_lc.loc[(s, l), "CI_high"],
                            "p_site": w["p"], "F": w["F"]})
    dp_rows, st_rows = [], []
    for s in SITES:
        for x in DEPTHS:
            dp_rows.append({"site": s, "depth": x, "depth_cm": DEPTH_CM[x],
                            "emm": e_dp.loc[(s, x), "EMM"], "ci_low": e_dp.loc[(s, x), "CI_low"],
                            "ci_high": e_dp.loc[(s, x), "CI_high"],
                            "letter_site": letters[("Site|Depth", x)][s].upper(),
                            "letter_depth": letters[("Depth", s)][x],
                            "n_samples": int(((d["site"] == s) & (d["depth"] == x)).sum())})
        w = wald.loc[f"Depth within {s}"]
        st_rows.append({"site": s, "n_samples": int((d["site"] == s).sum()),
                        "n_plots": int(d.loc[d["site"] == s, "plot"].nunique()),
                        "F": w["F"], "p_depth": w["p"]})
    return {"fig_landcover_annotations.csv": pd.DataFrame(lc_rows),
            "fig_depth_annotations.csv": pd.DataFrame(dp_rows),
            "fig_depth_site_annotations.csv": pd.DataFrame(st_rows)}


# ----------------------------------------------------------------------------
# 7b. Annotations for the full set of boxplot figures
# ----------------------------------------------------------------------------
FIGURE_CSVS = ["fig_all_annotations.csv", "fig_depth_annotations.csv",
               "fig_depth_site_annotations.csv", "fig_lc_depth_site_profiles.csv"]
FIG_LEVELS = {"site": SITES, "lc": LANDCOVERS, "depth": DEPTHS}
# (figure id, facet variable, x variable, model)
#   M3 = retained model (landscape x land cover + landscape x depth)
#   M4 = M3 + land cover x depth (needed for any comparison involving land cover AND depth)
FIGURE_SPECS = [
    ("global_site", None, "site", "M3"),
    ("global_landcover", None, "lc", "M3"),
    ("global_depth", None, "depth", "M3"),
    ("site_by_landcover", "site", "lc", "M3"),
    ("landcover_by_site", "lc", "site", "M3"),
    ("landcover_by_depth", "lc", "depth", "M4"),
    ("depth_by_landcover", "depth", "lc", "M4"),
]


def all_figure_annotations(d: pd.DataFrame, engines: dict) -> pd.DataFrame:
    """
    For every boxplot figure: letters (Holm-adjusted pairwise comparisons of the x-axis levels
    within each panel), the global Wald F-test of the x factor within each panel, and n.
    Panels of the 'global_*' figures pool the other factors with equal weights (descriptive when
    the corresponding interactions are significant).
    """
    rows = []
    for fig_id, facet, xvar, model in FIGURE_SPECS:
        eng = engines[model]
        levels = FIG_LEVELS[xvar]
        for panel in ([None] if facet is None else FIG_LEVELS[facet]):
            def getter(level, panel=panel):
                kw = {xvar: level}
                if facet is not None:
                    kw[facet] = panel
                return eng.mean_vector(**kw)

            fam = eng.pairwise(levels, getter, "x")
            p_adj = {frozenset(c.split(" - ")): p for c, p in zip(fam["contrast"], fam["p_holm"])}
            est = {lv: eng.contrast(getter(lv)) for lv in levels}
            letters = compact_letters(levels, {lv: est[lv]["estimate"] for lv in levels}, p_adj)
            wald = eng.wald_F(np.array([getter(lv) - getter(levels[0]) for lv in levels[1:]]))
            for lv in levels:
                mask = (d[xvar] == lv).values
                if facet is not None:
                    mask = mask & (d[facet] == panel).values
                rows.append({"figure": fig_id, "model": model, "facet_var": facet or "", "x_var": xvar,
                             "panel": "All" if panel is None else panel, "x_level": lv, "letter": letters[lv],
                             "n_samples": int(mask.sum()), "n_plots": int(d.loc[mask, "plot"].nunique()),
                             "emm": est[lv]["estimate"], "ci_low": est[lv]["CI_low"], "ci_high": est[lv]["CI_high"],
                             "F": wald["F"], "df1": wald["df1"], "df2": wald["df2"], "p_panel": wald["p"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 8. Results report (Excel): one section per analysis, letters last
# ----------------------------------------------------------------------------
@dataclass
class Block:
    """One table of the report, with a title and an explanatory note."""
    title: str
    table: pd.DataFrame
    note: str = ""


@dataclass
class Section:
    """One Excel sheet, made of one or several blocks."""
    sheet: str          # Excel sheet name (31 characters maximum)
    title: str
    blocks: list


def fmt_p(p) -> str:
    """p-values: 3 decimals, scientific notation below 0.001."""
    if p is None or pd.isna(p):
        return ""
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def _round(df: pd.DataFrame, nd: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes("number").columns:
        out[c] = out[c].round(nd)
    return out


def contrast_block(pw: pd.DataFrame, families: list, title: str, note: str = "") -> Block:
    sub = pw[pw["family"].isin(families)]
    t = pd.DataFrame({"Comparison set": sub["family"].values, "Contrast": sub["contrast"].values,
                      "Difference": sub["estimate"].values, "SE": sub["SE"].values,
                      "CI low": sub["CI_low"].values, "CI high": sub["CI_high"].values,
                      "p": sub["p_raw"].map(fmt_p).values, "p (Holm)": sub["p_holm"].map(fmt_p).values})
    return Block(title, _round(t), note)


def emm_block(emm: pd.DataFrame, effect: str, cols: tuple, title: str) -> Block:
    sub = emm[emm["effect"] == effect]
    t = pd.DataFrame({cols[0]: sub["level1"].values, cols[1]: sub["level2"].values,
                      "Estimated mean": sub["EMM"].values, "SE": sub["SE"].values,
                      "CI low": sub["CI_low"].values, "CI high": sub["CI_high"].values})
    return Block(title, _round(t), "Estimated marginal means with 95% confidence intervals.")


def wald_block(glob: pd.DataFrame, prefix: str, title: str) -> Block:
    sub = glob[glob["test"].str.startswith(prefix)]
    t = pd.DataFrame({"Test": sub["test"].values, "F": sub["F"].values, "df1": sub["df1"].values,
                      "df2": sub["df2"].values, "p": sub["p"].map(fmt_p).values})
    return Block(title, _round(t, 3), "Global Wald F-tests from the mixed model.")


def build_report_sections(d, struct, aic_tab, lrt, fixed_tab, vc_tab, r2, perf, emm, pw,
                          glob, lc_depth, diag, letters, fig_tables) -> list:
    holm_note = "p: unadjusted; p (Holm): adjusted within each comparison set; 95% confidence intervals."
    sections = []

    # 1. Sampling design -------------------------------------------------------------------
    counts = (d.groupby(["site", "lc"], observed=True)
                .agg(plots=("plot", "nunique"), samples=("SFI", "size")).reset_index())
    counts.columns = ["Landscape", "Land cover", "Plots (n)", "Samples (n)"]
    sections.append(Section("1_Design", "1. Sampling design",
                            [Block("Plots and samples per landscape and land cover", counts)]))

    # 2. Model selection -------------------------------------------------------------------
    st = struct.rename(columns={"structure": "Random-effect structure", "k": "Parameters", "logLik_ML": "logLik (ML)",
                                "BIC_nobs": "BIC (n = samples)", "BIC_nplots": "BIC (n = plots)", "delta_AIC": "Delta AIC"})
    ac = aic_tab.rename(columns={"model": "Fixed-effect structure", "k": "Parameters", "logLik_ML": "logLik (ML)",
                                 "BIC_nobs": "BIC (n = samples)", "BIC_nplots": "BIC (n = plots)",
                                 "delta_AIC": "Delta AIC", "delta_BIC_nobs": "Delta BIC (samples)",
                                 "delta_BIC_nplots": "Delta BIC (plots)"})
    sections.append(Section("2_Model_selection", "2. Model selection", [
        Block("Random-effect structures (fixed effects: landscape x land cover + landscape x depth)", _round(st, 2),
              "Lower AIC is better; the retained structure has landscape-specific intercept and slope variances."),
        Block("Pre-specified candidate fixed-effect structures", _round(ac, 2),
              "Candidates were compared, not selected stepwise; the retained model is M3.")]))

    # 3. Interactions ---------------------------------------------------------------------------
    lr = pd.DataFrame({"Term": lrt["term"], "Chi-square": lrt["chi2"].round(2), "df": lrt["df"], "p": lrt["p"].map(fmt_p)})
    sections.append(Section("3_Interactions", "3. Tests of interactions",
                            [Block("Likelihood-ratio tests (maximum-likelihood fits)", lr,
                                   "Landscape x land cover and landscape x depth are the interactions of interest; the others are supplementary.")]))

    # 4. Land cover -------------------------------------------------------------------------------
    sections.append(Section("4_Land_cover", "4. Land cover contrasts", [
        wald_block(glob, "Land cover within", "Global test of land cover within each landscape"),
        contrast_block(pw, [f"Land cover within {s}" for s in SITES], "Pairwise land cover contrasts within each landscape", holm_note),
        contrast_block(pw, ["Land cover (all sites, all depths)"], "Land cover contrasts pooled across landscapes (descriptive)",
                       "Interpret descriptively when the landscape x land cover interaction is significant."),
        emm_block(emm, "Site x Land cover", ("Landscape", "Land cover"), "Estimated means by landscape and land cover")]))

    # 5. Depth ---------------------------------------------------------------------------------------
    change = pw[pw["contrast"].eq("0-10 - 60-90") & pw["family"].str.startswith("Depth within")]
    ch_tab = pd.DataFrame({"Landscape": change["family"].str.replace("Depth within ", "").values,
                           "Change 0-10 minus 60-90 cm": change["estimate"].values, "SE": change["SE"].values,
                           "CI low": change["CI_low"].values, "CI high": change["CI_high"].values,
                           "p (Holm)": change["p_holm"].map(fmt_p).values})
    sections.append(Section("5_Depth", "5. Variation with soil depth", [
        wald_block(glob, "Depth within", "Global test of depth within each landscape"),
        contrast_block(pw, [f"Depth within {s}" for s in SITES], "Pairwise depth contrasts within each landscape", holm_note),
        Block("Total change in SFI between 0-10 and 60-90 cm", _round(ch_tab), "Positive values indicate a decrease with depth."),
        contrast_block(pw, ["Depth (all sites)"], "Pairwise depth contrasts pooled across landscapes", holm_note),
        contrast_block(pw, [f"Site within depth {x}" for x in DEPTHS], "Landscape contrasts within each depth", holm_note),
        emm_block(emm, "Site x Depth", ("Landscape", "Depth (cm)"), "Estimated means by landscape and depth")]))

    # 6. Landscapes ----------------------------------------------------------------------------------------
    sections.append(Section("6_Landscapes", "6. Landscape comparisons", [
        contrast_block(pw, ["Site (all depths, all land covers)"], "Landscape contrasts (all land covers and depths)", holm_note),
        contrast_block(pw, [f"Site within {l}" for l in LANDCOVERS], "Landscape contrasts within each land cover", holm_note)]))

    # 7. Supplementary: land cover by depth ---------------------------------------------------------------
    sup = pd.DataFrame({"Landscape": lc_depth["site"], "Contrast": lc_depth["contrast"], "Depth (cm)": lc_depth["depth"],
                        "Difference": lc_depth["estimate"], "SE": lc_depth["SE"], "CI low": lc_depth["CI_low"],
                        "CI high": lc_depth["CI_high"], "p": lc_depth["p_raw"].map(fmt_p),
                        "p (Holm, across depths)": lc_depth["p_holm_over_depths"].map(fmt_p)})
    sections.append(Section("7_Suppl_LC_by_depth", "7. Supplementary: land cover contrasts by depth",
                            [Block("Land cover - grassland contrasts at each depth within each landscape", _round(sup),
                                   "From the model including the three-way interaction.")]))

    # 8. Model fit and diagnostics -----------------------------------------------------------------------------
    fx = fixed_tab.reset_index().rename(columns={"index": "Term", "coef": "Coefficient"})
    vc = vc_tab.rename(columns={"component": "Component", "variance": "Variance"})
    r2_tab = pd.DataFrame({"Metric": list(r2), "Value": list(r2.values())})
    pf = perf.rename(columns={"predictions": "Predictions", "SD_SFI": "SD of SFI", "IQR_SFI": "IQR of SFI"})
    dg = diag.rename(columns={"metric": "Metric", "value": "Value"})
    sections.append(Section("8_Model_fit", "8. Model fit and diagnostics", [
        Block("Marginal and conditional R2, intraclass correlation", _round(r2_tab, 3), "ICC from a null model with plot as random effect."),
        Block("RMSE, RPD and RPIQ of the fitted values", _round(pf, 4), "Marginal: fixed effects only; conditional: fixed + random effects."),
        Block("Variance components", _round(vc, 6)),
        Block("Fixed-effect coefficients (reference: CH, grassland, 0-10 cm)", _round(fx, 4)),
        Block("Residual diagnostics", _round(dg, 4))]))

    # 9. Letters (last) ---------------------------------------------------------------------------------------------
    lc = fig_tables["fig_landcover_annotations.csv"]
    lc_t = pd.DataFrame({"Landscape": lc["site"], "Land cover": lc["land_cover"], "Samples (n)": lc["n_samples"],
                         "Plots (n)": lc["n_plots"], "Raw mean": lc["mean_raw"], "Estimated mean": lc["emm"],
                         "CI low": lc["ci_low"], "CI high": lc["ci_high"],
                         "p (land cover in landscape)": lc["p_site"].map(fmt_p), "Letter": lc["letter"]})
    dp = fig_tables["fig_depth_annotations.csv"]
    dp_t = pd.DataFrame({"Landscape": dp["site"], "Depth (cm)": dp["depth"], "Samples (n)": dp["n_samples"],
                         "Estimated mean": dp["emm"], "CI low": dp["ci_low"], "CI high": dp["ci_high"],
                         "Letter: landscapes within depth": dp["letter_site"],
                         "Letter: depths within landscape": dp["letter_depth"]})
    dps = fig_tables["fig_depth_site_annotations.csv"]
    dps_t = pd.DataFrame({"Landscape": dps["site"], "Samples (n)": dps["n_samples"], "Plots (n)": dps["n_plots"],
                          "F": dps["F"], "p (depth in landscape)": dps["p_depth"].map(fmt_p)})
    e_lc = emm[emm["effect"] == "Site x Land cover"].set_index(["level1", "level2"])
    rows = [{"Land cover": l, "Landscape": s, "Estimated mean": e_lc.loc[(s, l), "EMM"],
             "CI low": e_lc.loc[(s, l), "CI_low"], "CI high": e_lc.loc[(s, l), "CI_high"],
             "Letter": letters[("Site|LC", l)][s]} for l in LANDCOVERS for s in SITES]
    fa = fig_tables["fig_all_annotations.csv"]
    fa_t = pd.DataFrame({"Figure": fa["figure"], "Panel": fa["panel"], "x-axis level": fa["x_level"], "Samples (n)": fa["n_samples"],
                         "Estimated mean": fa["emm"], "CI low": fa["ci_low"], "CI high": fa["ci_high"],
                         "p (panel, Wald F)": fa["p_panel"].map(fmt_p), "Letter": fa["letter"], "Model": fa["model"]})
    all_fig_block = Block("All boxplot figures: letters and panel tests", _round(fa_t),
                          "Figures with land cover AND depth use model M4 (retained model + land cover x depth); the others use the retained model. "
                          "'All' = factors pooled with equal weights (descriptive when interactions are significant).")
    sections.append(Section("9_Letters", "9. Compact letters (groups sharing a letter do not differ significantly)", [
        Block("Land cover within each landscape", _round(lc_t),
              "Letters from Holm-adjusted pairwise comparisons within a landscape; used in the land cover figure."),
        Block("Depth profiles: landscapes within depth (capital) and depths within landscape (lowercase)", _round(dp_t),
              "Used in the depth-profile figures."),
        Block("Depth effect within each landscape (global test)", _round(dps_t, 3), ""),
        Block("Landscapes within each land cover", _round(pd.DataFrame(rows)), "Letters from Holm-adjusted comparisons among landscapes within a land cover."),
        all_fig_block]))
    return sections


def _py(v):
    """Convert numpy scalars / NaN to plain Python values for spreadsheet cells."""
    if isinstance(v, float) and np.isnan(v):
        return None
    return v.item() if hasattr(v, "item") else v


def write_excel_report(sections: list, path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    fill = PatternFill("solid", fgColor="D9E2F3")
    line = Side(style="thin", color="999999")
    for sec in sections:
        ws = wb.create_sheet(sec.sheet[:31])
        widths, r = {}, 1
        ws.cell(r, 1, sec.title).font = Font(bold=True, size=14)
        r += 2
        for blk in sec.blocks:
            ws.cell(r, 1, blk.title).font = Font(bold=True)
            r += 1
            for j, col in enumerate(blk.table.columns, 1):
                c = ws.cell(r, j, str(col))
                c.font, c.fill = Font(bold=True), fill
                c.border = Border(top=line, bottom=line)
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                widths[j] = max(widths.get(j, 0), min(len(str(col)), 22))
            r += 1
            for row in blk.table.itertuples(index=False):
                for j, v in enumerate(row, 1):
                    ws.cell(r, j, _py(v))
                    widths[j] = max(widths.get(j, 0), len(str(_py(v))))
                r += 1
            if blk.note:
                ws.cell(r, 1, blk.note).font = Font(italic=True, size=9)
                r += 1
            r += 1
        for j, w in widths.items():
            ws.column_dimensions[get_column_letter(j)].width = min(max(w + 2, 8), 60)
    wb.save(path)


# ----------------------------------------------------------------------------
# 9. Main workflow
# ----------------------------------------------------------------------------
def main(data_file: Path, out_dir: Path) -> None:
    tab_dir, fig_dir, log_dir = out_dir / "tables", out_dir / "figures", out_dir / "logs"
    for p in (tab_dir, fig_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        handlers=[logging.FileHandler(log_dir / "analysis_log.txt", mode="w"), logging.StreamHandler()],
                        force=True)
    (log_dir / "environment.txt").write_text(
        "\n".join([f"Python {platform.python_version()} ({platform.platform()})", f"numpy {np.__version__}",
                   f"pandas {pd.__version__}", f"scipy {scipy.__version__}", f"statsmodels {statsmodels.__version__}",
                   f"patsy {patsy.__version__}", f"matplotlib {matplotlib.__version__}"]))
    t0 = time.time()
    MAIN_LOG.info("Results folder: %s", out_dir.resolve())
    out_dir = out_dir.resolve()

    def step(msg: str) -> None:
        """Progress message (print + flush so that it is visible in Jupyter and in a terminal)."""
        print(f"[{time.time() - t0:5.0f} s] {msg}", flush=True)

    step(f"Start. Working folder: {Path.cwd()}")
    step(f"Data file: {data_file.resolve()}  (exists: {data_file.exists()})")
    step(f"Results will be written to: {out_dir}")

    # --- 1. data ------------------------------------------------------------
    step("1/7 Loading and checking the data")
    d = load_data(data_file)
    n_obs, n_plots = len(d), d["plot"].nunique()
    MAIN_LOG.info("Data: %d samples, %d plots", n_obs, n_plots)

    # --- 2. random-effect / variance structure ---------------------------------
    step("2/7 Comparing random-effect structures (about 1 min)")
    labels = {"A": "A. Random intercept, homogeneous variance",
              "B": "B. Random intercept + depth slope (correlated), homogeneous variance",
              "C": "C. Random intercept, landscape-specific variance",
              "D": "D. Random intercept + depth slope, landscape-specific variances (retained)"}
    struct = pd.DataFrame([{"structure": labels[s], **information_criteria(fit_mixed(FIXED_FORMULA, d, False, s), n_obs, n_plots)}
                           for s in "ABCD"])
    struct["delta_AIC"] = struct["AIC"] - struct["AIC"].min()

    # --- 3. fixed-effect structure: LRTs and AIC/BIC of pre-specified candidates --------------
    step("3/7 Testing interactions and comparing fixed-effect structures (about 1 min)")
    ml_final = fit_mixed(FIXED_FORMULA, d, reml=False)
    lrt = pd.DataFrame([
        {"term": "Landscape x land cover", **lr_test(FIXED_FORMULA, "SFI ~ site + lc + site*depth", d, full_res=ml_final)},
        {"term": "Landscape x depth", **lr_test(FIXED_FORMULA, "SFI ~ site*lc + depth", d, full_res=ml_final)},
        {"term": "Land cover x depth (supplementary)", **lr_test(FIXED_FORMULA + " + lc:depth", FIXED_FORMULA, d)},
        {"term": "Landscape x land cover x depth (supplementary)", **lr_test("SFI ~ site*lc*depth", FIXED_FORMULA, d)},
    ])
    candidates = [("M0  site + lc + depth", "SFI ~ site + lc + depth"),
                  ("M1  M0 + site:lc", "SFI ~ site*lc + depth"),
                  ("M2  M0 + site:depth", "SFI ~ site + lc + site*depth"),
                  ("M3  M0 + site:lc + site:depth (retained)", FIXED_FORMULA),
                  ("M4  M3 + lc:depth", FIXED_FORMULA + " + lc:depth"),
                  ("M5  site*lc*depth", "SFI ~ site*lc*depth")]
    aic_tab = pd.DataFrame([{"model": nm, **information_criteria(fit_mixed(f, d, False), n_obs, n_plots)} for nm, f in candidates])
    for c in ("AIC", "BIC_nobs", "BIC_nplots"):
        aic_tab[f"delta_{c}"] = aic_tab[c] - aic_tab[c].min()

    # --- 4. final model (REML) -------------------------------------------------
    step("4/7 Fitting the final model, R2 and ICC")
    final = fit_mixed(FIXED_FORMULA, d, reml=True)
    k = len(final.fe_params)
    fixed_tab = pd.DataFrame({"coef": final.fe_params, "SE": final.bse.iloc[:k]})
    vc_tab = pd.DataFrame({"component": final.params.index[k:], "variance": final.params.values[k:] * final.scale})
    vc_tab.loc[len(vc_tab)] = ["Residual", final.scale]
    r2 = nakagawa_r2(final, d)
    null = smf.mixedlm("SFI ~ 1", d, groups=d["plot"]).fit(reml=True)
    r2["ICC_null_model"] = null.cov_re.iloc[0, 0] / (null.cov_re.iloc[0, 0] + null.scale)
    MAIN_LOG.info("R2 marginal %.3f | conditional %.3f | ICC %.3f",
                  r2["R2_marginal"], r2["R2_conditional"], r2["ICC_null_model"])

    # --- 5. contrasts, tests and letters ---------------------------------------
    step("5/7 Computing contrasts, tests and letters")
    eng = ContrastEngine(final, "site*lc + site*depth", n_plots)
    emm, pw = estimated_marginal_means(eng), all_pairwise_families(eng)
    glob = global_wald_tests(eng)
    letters = letters_from_families(pw, emm)

    # --- 6. supplementary analysis, diagnostics, fit performance ------------------
    step("6/7 Supplementary analysis and residual diagnostics (about 1 min)")
    lc_depth, lc_depth_emm = landcover_by_depth(d, n_plots)
    diag = residual_diagnostics(final, d, fig_dir / "fig_S_residual_diagnostics.png")
    perf = fit_performance(final, d)

    # --- 7. export: results report (Excel) and annotation tables for the figures ----------------
    step("7/7 Writing the Excel report and the figure tables")
    fig_tables = build_figure_tables(d, eng, emm, pw, glob, letters)
    step("7/7 (a) Annotations for all boxplot figures (fits the model with land cover x depth)")
    final_m4 = fit_mixed(FIXED_FORMULA + " + lc:depth", d, reml=True)
    eng_m4 = ContrastEngine(final_m4, "site*lc + site*depth + lc:depth", n_plots)
    fig_tables["fig_all_annotations.csv"] = all_figure_annotations(d, {"M3": eng, "M4": eng_m4})
    # profile figure: land cover within each landscape, with land cover - grassland contrasts by depth
    prof = lc_depth_emm.merge(
        lc_depth.assign(lc=lc_depth["contrast"].str.replace(" - Grassland", "", regex=False))
                [["site", "lc", "depth", "p_holm_over_depths"]],
        on=["site", "lc", "depth"], how="left")
    prof["depth_cm"] = prof["depth"].map(DEPTH_CM)
    prof["significant"] = prof["p_holm_over_depths"] < ALPHA      # False/NaN for the reference (grassland)
    fig_tables["fig_lc_depth_site_profiles.csv"] = prof
    sections = build_report_sections(d, struct, aic_tab, lrt, fixed_tab, vc_tab, r2, perf, emm, pw,
                                     glob, lc_depth, diag, letters, fig_tables)
    write_excel_report(sections, tab_dir / "SFI_results.xlsx")
    for name in FIGURE_CSVS:                       # tables read by 02_sfi_lmm_figures.py
        fig_tables[name].to_csv(tab_dir / name, index=False)
    MAIN_LOG.info("Finished in %.0f s. Files written to: %s | Figures: run 02_sfi_lmm_figures.py", time.time() - t0, tab_dir.resolve())
    step("DONE. Files written:")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            print("   ", f, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/DATA_LM.csv"), help="input CSV (semicolon-separated)")
    ap.add_argument("--out", type=Path, default=Path("outputs/lmm"), help="output directory")
    args, _ = ap.parse_known_args()      # ignores extra arguments added by Jupyter/IPython (-f kernel.json)
    main(args.data, args.out)
