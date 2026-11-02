# ============================================================
# STEP 4 — INDICATOR SCORING (Table 3, Sulaeman et al., 2021)
# (Python equivalent of cut() in R)
# ============================================================
"""
Purpose
-------
Assign each observation a fertility score (1 to 5, per Table 3) for
each of the 6 MSFI indicators (pH, CEC, K, P, C, texture), then
standardize these scores (Sij . p, with p = 1/n, n = 5 classes;
equation (7) of the methodology).

Input  : outputs/sfi_index/data_prepared.csv, produced by 01_data_preparation.py
         (expected columns: pH, CEC, K, P, C, Texture_code)
Output : outputs/sfi_index/data_scored.csv, reused by 05_sfi_calculation.py

Note on class gaps
--------------------
Table 3 has small gaps between some classes (e.g. Ex-K: 0.3-0.4 and
0.5-0.6; CEC: 16-17 and 24-25). Convention: pd.cut bins are contiguous,
so each gap is absorbed into the adjacent higher class.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs/sfi_index")
PREPARED_DATA_FILE = OUTPUT_DIR / "data_prepared.csv"
SCORED_DATA_FILE = OUTPUT_DIR / "data_scored.csv"

# Number of SFI classes (n in equation 7), so that p = 1/n
N_CLASSES = 5

# --- Scoring thresholds (Table 3, Sulaeman et al., 2021) ---
SCORE_BINS = {
    # Soil pH is scored on a two-sided (non-monotonic) scale: the optimum range
    # (> 6 - 7) receives the highest score (5), and scores decrease progressively
    # towards both acidic (<= 5) and alkaline (> 8.2) conditions, which receive
    # the lowest score (1). The nine intervals are scored with PH_LABELS (below).
    "pH": [-np.inf, 5, 5.4, 5.8, 6, 7, 7.4, 7.8, 8.2, np.inf],
    "CEC": [-np.inf, 5, 16, 24, 40, np.inf],
    "K": [-np.inf, 0.1, 0.3, 0.5, 1, np.inf],
    "P": [-np.inf, 4, 7, 10, 15, np.inf],
    "C": [-np.inf, 1, 2, 3, 5, np.inf],        # % organic C, after unit conversion
}
SCORE_LABELS = [1, 2, 3, 4, 5]

# Scores for the nine pH intervals (Table 3, Sulaeman et al., 2021):
# acidic side (<= 5 to > 5.8 - 6) -> 1 to 4; optimum (> 6 - 7) -> 5;
# alkaline side (> 7 - 7.4 to > 8.2) -> 4 to 1.
PH_LABELS = [1, 2, 3, 4, 5, 4, 3, 2, 1]

# Texture classification (Table 3), from Texture_code (step 1)
TEXTURE_CLASS = {
    "S": 1, "Si": 1,
    "LS": 2,
    "SL": 3, "L": 3, "SiL": 3,
    "SiC": 4, "CL": 4, "SCL": 4,
    "SiCL": 5, "SC": 5, "C": 5,
}

# Organic carbon unit conversion: g/kg -> %
CONVERT_C_GKG_TO_PERCENT = True

SCORE_COLUMNS = ["pH_score", "CEC_score", "K_score", "P_score", "C_score", "texture_score"]


# ============================================================
# FUNCTIONS
# ============================================================

def load_prepared_data(path: Path) -> pd.DataFrame:
    """Load the data prepared by 01_data_preparation.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found -> {path}. Did you run 01_data_preparation.py first?"
        )
    return pd.read_csv(path)


def convert_carbon_units(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """Convert C from g/kg to % if enabled=True."""
    if enabled:
        df["C"] = df["C"] / 10
    print("=== C statistics (after optional unit conversion) ===")
    print(df["C"].describe())
    return df


def score_variable(series: pd.Series, bins: list, labels: list) -> pd.Series:
    """Cut a continuous variable into classes 1-5 per Table 3 thresholds.

    ordered=False is required because pH labels repeat (1..5..1).
    Bins are right-closed: (a, b].
    """
    return pd.cut(series, bins=bins, labels=labels, ordered=False)


def score_texture(df: pd.DataFrame, texture_class: dict) -> pd.Series:
    """Derive the texture score (1-5) from Texture_code (produced in step 1)."""
    if "Texture_code" not in df.columns:
        raise KeyError(
            "Column 'Texture_code' is missing: check that it was produced "
            "by 01_data_preparation.py."
        )
    scores = df["Texture_code"].map(texture_class)

    unmatched = df.loc[scores.isna() & df["Texture_code"].notna(), "Texture_code"].unique()
    if len(unmatched) > 0:
        print(f"WARNING: unrecognized texture codes in TEXTURE_CLASS: {list(unmatched)}. "
              f"Check TEXTURE_CODE in 01_data_preparation.py.")
    return scores


def apply_all_scores(df: pd.DataFrame, bins_config: dict, labels: list, texture_class: dict) -> pd.DataFrame:
    """Apply Table 3 scoring to all MSFI indicators."""
    for var, bins in bins_config.items():
        var_labels = PH_LABELS if var == "pH" else labels
        if len(var_labels) != len(bins) - 1:
            raise ValueError(f"{var}: {len(bins) - 1} bins but {len(var_labels)} labels.")
        df[f"{var}_score"] = score_variable(df[var], bins, var_labels)
    df["texture_score"] = score_texture(df, texture_class)
    return df


def standardize_scores(df: pd.DataFrame, score_cols: list, n_classes: int) -> pd.DataFrame:
    """
    Standardize each score: Sij . p, with p = 1/n_classes (equation 7).
    Applied only once here — do not divide the SFI by n_classes again
    later (see 05_sfi_calculation.py).
    """
    p = 1 / n_classes
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") * p
    return df


def check_score_quality(df: pd.DataFrame, score_cols: list) -> None:
    """Check for missing values after scoring (consistent with steps 1-3)."""
    n_na = df[score_cols].isna().sum()
    print("\n=== Missing values per score (after standardization) ===")
    print(n_na)
    fully_empty = n_na[n_na == len(df)].index.tolist()
    if fully_empty:
        print(f"\nWARNING: fully missing score(s) -> {fully_empty}.")


def save_scored_data(df: pd.DataFrame, path: Path) -> None:
    """Save the scored DataFrame, reloaded explicitly by 05_sfi_calculation.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved scored data -> {path}")


def run_scoring(
    data_file: Path = PREPARED_DATA_FILE,
    output_file: Path = SCORED_DATA_FILE,
    bins_config: dict = SCORE_BINS,
    labels: list = SCORE_LABELS,
    texture_class: dict = TEXTURE_CLASS,
    n_classes: int = N_CLASSES,
    convert_carbon: bool = CONVERT_C_GKG_TO_PERCENT,
) -> pd.DataFrame:
    """Full pipeline for step 4."""
    data = load_prepared_data(data_file)

    required = ["pH", "CEC", "K", "P", "C", "Texture_code"]
    missing = [v for v in required if v not in data.columns]
    if missing:
        raise KeyError(
            f"Required columns missing from `data`: {missing}. "
            f"Check 01_data_preparation.py."
        )

    data = convert_carbon_units(data, convert_carbon)
    data = apply_all_scores(data, bins_config, labels, texture_class)

    print("\n=== Texture class distribution (before standardization) ===")
    print(data["texture_score"].value_counts(dropna=False))

    data = standardize_scores(data, SCORE_COLUMNS, n_classes)
    check_score_quality(data, SCORE_COLUMNS)

    print("\n=== Standardized scores (Sij . p) ===")
    print(data[SCORE_COLUMNS].describe())

    save_scored_data(data, output_file)
    return data


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    data = run_scoring()
