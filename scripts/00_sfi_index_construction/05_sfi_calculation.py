# ============================================================
# STEP 5 — SFI CALCULATION
# SFIi = Sum_j Wj . Sij . p   (equation 2)
# (the p = 1/n factor is already embedded in the standardized scores
# produced in step 4; it is NOT reapplied here)
# ============================================================
"""
Purpose
-------
Calculate the Soil Fertility Index (SFI) for each observation from the
standardized scores (step 4) and the Wj weights (step 3), accepting
only observations with all 6 MSFI indicators available (no "without K"
variant, per the methodology).

Input  : - outputs/data_scored.csv    (04_indicator_scoring.py)
         - outputs/pca_weights_Wj.csv (03_pca_weights.py)
Output : - outputs/sfi_summary.csv (Site, LandUse, Depth_class, SFI, SFI_method)
         - outputs/data_sfi.csv    (full dataset + SFI, SFI_method)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs")
SCORED_DATA_FILE = OUTPUT_DIR / "data_scored.csv"
WEIGHTS_FILE = OUTPUT_DIR / "pca_weights_Wj.csv"

SFI_SUMMARY_FILE = OUTPUT_DIR / "sfi_summary.csv"
SFI_FULL_FILE = OUTPUT_DIR / "data_sfi.csv"

# The 6 MSFI indicators, ALL required (no "without K" variant)
VARS_MSFI = ["pH_score", "CEC_score", "texture_score", "K_score", "P_score", "C_score"]

# Keep in sync with VAR_TO_SCORE_NAME in 03_pca_weights.py
VAR_TO_SCORE_NAME = {
    "pH": "pH_score",
    "CEC": "CEC_score",
    "clay": "texture_score",
    "K": "K_score",
    "P": "P_score",
    "C": "C_score",
}

SUMMARY_COLUMNS = ["Site", "LandUse", "Depth_class", "SFI", "SFI_method"]


# ============================================================
# FUNCTIONS
# ============================================================

def load_scored_data(path: Path) -> pd.DataFrame:
    """Load the scored data produced by 04_indicator_scoring.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found -> {path}. Did you run 04_indicator_scoring.py?"
        )
    return pd.read_csv(path)


def load_weights(path: Path, var_to_score: dict) -> pd.Series:
    """
    Load the Wj_normalized weights produced by 03_pca_weights.py and
    rename them to score column names (see VAR_TO_SCORE_NAME).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File not found -> {path}. Did you run 03_pca_weights.py?"
        )
    weights_table = pd.read_csv(path, index_col=0)
    return weights_table["Wj_normalized"].rename(index=var_to_score)


def align_weights(weights_scores: pd.Series, vars_msfi: list) -> pd.Series:
    """Reorder the weights to match vars_msfi exactly, and validate none are missing."""
    weights_aligned = weights_scores.reindex(vars_msfi)
    missing = weights_aligned[weights_aligned.isna()].index.tolist()
    if missing:
        raise ValueError(
            f"Missing weights after alignment for: {missing}. "
            f"weights_scores index: {weights_scores.index.tolist()}"
        )
    return weights_aligned


def compute_sfi(df: pd.DataFrame, vars_msfi: list, weights_aligned: pd.Series) -> pd.DataFrame:
    """
    Compute SFIi = Sum_j (Wj . Sij_standardized). p=1/n is already
    embedded in the scores (step 4) — do NOT multiply by p again here.
    """
    has_all = df[vars_msfi].notna().all(axis=1)

    df["SFI"] = np.nan
    df["SFI_method"] = "insufficient"

    df.loc[has_all, "SFI"] = (
        df.loc[has_all, vars_msfi].mul(weights_aligned, axis=1).sum(axis=1)
    )
    df.loc[has_all, "SFI_method"] = "complete_6_indicators"
    return df


def report_results(df: pd.DataFrame, weights_aligned: pd.Series, vars_msfi: list) -> None:
    """Print consistency checks (weights, scores, distribution, SFI)."""
    print("=== Aligned Wj weights (should sum to ~1) ===")
    print(weights_aligned)
    print("Sum of Wj weights:", weights_aligned.sum())

    print("\n=== Standardized scores Sij (before weighting) ===")
    print(df[vars_msfi].describe())

    print("\n=== Distribution of calculation methods ===")
    print(df["SFI_method"].value_counts(dropna=False))

    print("\n=== SFI summary ===")
    print(df["SFI"].describe())


def save_outputs(df: pd.DataFrame, summary_cols: list, summary_path: Path, full_path: Path) -> None:
    """Save a summary export and the full dataset, as two distinct files."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df[summary_cols].to_csv(summary_path, index=False)
    df.to_csv(full_path, index=False)
    print(f"\nSaved SFI summary -> {summary_path}")
    print(f"Saved full dataset -> {full_path}")


def run_sfi_calculation(
    scored_data_file: Path = SCORED_DATA_FILE,
    weights_file: Path = WEIGHTS_FILE,
    vars_msfi: list = VARS_MSFI,
    var_to_score: dict = VAR_TO_SCORE_NAME,
    summary_cols: list = SUMMARY_COLUMNS,
    summary_path: Path = SFI_SUMMARY_FILE,
    full_path: Path = SFI_FULL_FILE,
) -> pd.DataFrame:
    """Full pipeline for step 5."""
    data = load_scored_data(scored_data_file)

    missing_cols = [v for v in vars_msfi if v not in data.columns]
    if missing_cols:
        raise KeyError(f"Score columns missing from `data`: {missing_cols}. Check 04_indicator_scoring.py.")

    weights_scores = load_weights(weights_file, var_to_score)
    weights_aligned = align_weights(weights_scores, vars_msfi)

    data = compute_sfi(data, vars_msfi, weights_aligned)
    report_results(data, weights_aligned, vars_msfi)

    missing_summary_cols = [c for c in summary_cols if c not in data.columns]
    if missing_summary_cols:
        raise KeyError(f"Expected summary columns missing: {missing_summary_cols}.")

    save_outputs(data, summary_cols, summary_path, full_path)
    return data


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    data = run_sfi_calculation()
