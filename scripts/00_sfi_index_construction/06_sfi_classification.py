# ============================================================
# STEP 6 — FINAL SFI CLASSIFICATION (Table 5, Bagherzadeh et al., 2018)
# ============================================================
"""
Purpose
-------
Classify each observation's SFI into 5 levels (Table 5) and produce the
final pipeline outputs (full dataset + a summary table ready for
statistical analyses / manuscript figures).

Input  : outputs/data_sfi.csv, produced by 05_sfi_calculation.py
Output : - outputs/data_sfi_classified.csv (full dataset + SFI_class)
         - outputs/sfi_final_summary.csv   (Site, LandUse, Depth_class,
           SFI, SFI_method, SFI_class — ready for Kruskal-Wallis, Dunn,
           figures, etc.)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs")
SFI_DATA_FILE = OUTPUT_DIR / "data_sfi.csv"

CLASSIFIED_FULL_FILE = OUTPUT_DIR / "data_sfi_classified.csv"
FINAL_SUMMARY_FILE = OUTPUT_DIR / "sfi_final_summary.csv"

# Table 5, Bagherzadeh et al., 2018
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
CLASS_BREAKS = [-np.inf, 0.25, 0.50, 0.75, 0.90, np.inf]

SUMMARY_COLUMNS = ["Site", "LandUse", "Depth_class", "SFI", "SFI_method", "SFI_class"]


# ============================================================
# FUNCTIONS
# ============================================================

def load_sfi_data(path: Path) -> pd.DataFrame:
    """Load the dataset with the SFI already calculated, from 05_sfi_calculation.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found -> {path}. Did you run 05_sfi_calculation.py?"
        )
    return pd.read_csv(path)


def classify_sfi(df: pd.DataFrame, breaks: list, labels: list) -> pd.DataFrame:
    """
    Classify the SFI into 5 levels (Table 5). No re-division by 5: the
    SFI is already on the expected scale (see module docstring). pd.cut
    returns NaN for missing SFI values ("insufficient" observations from
    step 5), which is the intended behaviour.
    """
    df["SFI_class"] = pd.cut(df["SFI"], bins=breaks, labels=labels)
    return df


def check_classification_consistency(df: pd.DataFrame) -> None:
    """Check that SFI_class is missing exactly where SFI_method == 'insufficient'."""
    na_class = df["SFI_class"].isna()
    na_sfi = df["SFI"].isna()
    if not (na_class == na_sfi).all():
        mismatched = (na_class != na_sfi).sum()
        print(f"WARNING: SFI_class / SFI mismatch on {mismatched} observation(s) — investigate.")
    else:
        print("OK: SFI_class is consistent with missing SFI values.")


def report_results(df: pd.DataFrame) -> None:
    """Print the class distribution and the observed SFI range."""
    print("=== SFI_class distribution (Table 5, Bagherzadeh et al., 2018) ===")
    print(df["SFI_class"].value_counts(dropna=False))

    print("\n=== Observed SFI min/max (expected within [0.2, 1.0], see docstring) ===")
    print(df["SFI"].agg(["min", "max"]))


def save_outputs(df: pd.DataFrame, summary_cols: list, full_path: Path, summary_path: Path) -> None:
    """Save the full classified dataset and a summary, in outputs/, with no duplicate content."""
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)
    df[summary_cols].to_csv(summary_path, index=False)
    print(f"\nSaved full classified dataset -> {full_path}")
    print(f"Saved final summary -> {summary_path}")


def run_classification(
    sfi_data_file: Path = SFI_DATA_FILE,
    breaks: list = CLASS_BREAKS,
    labels: list = CLASS_LABELS,
    summary_cols: list = SUMMARY_COLUMNS,
    full_path: Path = CLASSIFIED_FULL_FILE,
    summary_path: Path = FINAL_SUMMARY_FILE,
) -> pd.DataFrame:
    """Full pipeline for step 6."""
    data = load_sfi_data(sfi_data_file)

    if "SFI" not in data.columns:
        raise KeyError("Column 'SFI' is missing from `data`. Check 05_sfi_calculation.py.")

    data = classify_sfi(data, breaks, labels)
    check_classification_consistency(data)
    report_results(data)

    missing_summary_cols = [c for c in summary_cols if c not in data.columns]
    if missing_summary_cols:
        raise KeyError(f"Expected final summary columns missing: {missing_summary_cols}.")

    save_outputs(data, summary_cols, full_path, summary_path)
    return data


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    data = run_classification()
