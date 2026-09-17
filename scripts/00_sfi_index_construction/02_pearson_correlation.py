# ============================================================
# STEP 2 — PEARSON CORRELATION MATRIX + P-VALUES
# (Python equivalent of Hmisc::rcorr + corrplot)
# ============================================================
"""
Purpose
-------
Compute the Pearson correlation matrix (r and p-value) between soil
properties, export it as a figure and as tables, and flag strongly
correlated pairs (|r| > 0.8) that motivate the selection of the Minimum
Soil Fertility Indicators (MSFI).

Input  : outputs/data_prepared.csv, produced by 01_data_preparation.py
Output : - outputs/correlation_heatmap.png
         - outputs/pearson_correlation_matrix.csv (full r matrix)
         - outputs/pearson_pvalue_matrix.csv      (full p matrix)
         - outputs/pearson_significant_pairs.csv
         - outputs/pearson_strong_pairs.csv (|r| > 0.8, MSFI threshold)
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

# ============================================================
# CONFIGURATION
# ============================================================

# Variables included in the correlation analysis, using the same names
# produced by 01_data_preparation.py (see NUMERIC_COLUMNS)
CORR_VARIABLES = ["N", "P", "C", "CEC", "pH", "K", "clay", "silt", "sand"]

ALPHA = 0.05

# |r| threshold used for MSFI collinearity screening (variables above
# this threshold are candidates for exclusion — see note above for the
# clay/sand case, which additionally rests on expert judgement)
MSFI_CORRELATION_THRESHOLD = 0.8

OUTPUT_DIR = Path("outputs")
PREPARED_DATA_FILE = OUTPUT_DIR / "data_prepared.csv"


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


def compute_pearson_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Equivalent of Hmisc::rcorr: computes the Pearson correlation matrix
    (r) and the associated p-value matrix for every pair of columns in
    `df`. Each pair is computed once (the matrix is symmetric) and then
    mirrored on both sides.
    """
    cols = df.columns
    r_matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    p_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    for col_a, col_b in combinations(cols, 2):
        valid = df[[col_a, col_b]].dropna()
        r, p = pearsonr(valid[col_a], valid[col_b])
        r_matrix.loc[col_a, col_b] = r_matrix.loc[col_b, col_a] = r
        p_matrix.loc[col_a, col_b] = p_matrix.loc[col_b, col_a] = p

    return r_matrix, p_matrix


def plot_correlation_heatmap(cor_matrix: pd.DataFrame, output_path: Path) -> None:
    """Generate and save the correlation heatmap (equivalent of corrplot)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cor_matrix, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
        square=True, cbar_kws={"shrink": 0.8},
    )
    plt.title("Pearson correlation matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def extract_pair_results(cor_matrix: pd.DataFrame, p_matrix: pd.DataFrame) -> pd.DataFrame:
    """Build a long table (one row per variable pair, no Var1/Var2 <-> Var2/Var1 duplicate)."""
    cols = cor_matrix.columns
    rows = [
        {
            "Var1": col_a, "Var2": col_b,
            "r": cor_matrix.loc[col_a, col_b],
            "p": p_matrix.loc[col_a, col_b],
        }
        for col_a, col_b in combinations(cols, 2)
    ]
    return pd.DataFrame(rows).sort_values("r", key=abs, ascending=False).reset_index(drop=True)


def select_significant_pairs(pairs: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Filter statistically significant pairs (p < alpha)."""
    return pairs[pairs["p"] < alpha].reset_index(drop=True)


def select_msfi_candidates(pairs: pd.DataFrame, r_threshold: float) -> pd.DataFrame:
    """
    Filter pairs whose |r| exceeds the collinearity threshold retained
    for MSFI selection. These are the pairs where only one of the two
    variables should be kept in the analysis.
    """
    return pairs[pairs["r"].abs() > r_threshold].reset_index(drop=True)


def run_pearson_analysis(
    data_file: Path = PREPARED_DATA_FILE,
    variables: list = CORR_VARIABLES,
    alpha: float = ALPHA,
    msfi_threshold: float = MSFI_CORRELATION_THRESHOLD,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Full pipeline for step 2. Loads the data, analyses, exports, and returns the results."""
    data = load_prepared_data(data_file)

    missing = [v for v in variables if v not in data.columns]
    if missing:
        raise KeyError(
            f"Columns missing from `data`: {missing}. "
            f"Check that they were produced by 01_data_preparation.py."
        )

    sub = data[variables]
    cor_matrix, p_matrix = compute_pearson_matrices(sub)

    output_dir.mkdir(parents=True, exist_ok=True)
    cor_matrix.to_csv(output_dir / "pearson_correlation_matrix.csv")
    p_matrix.to_csv(output_dir / "pearson_pvalue_matrix.csv")
    plot_correlation_heatmap(cor_matrix, output_dir / "correlation_heatmap.png")

    all_pairs = extract_pair_results(cor_matrix, p_matrix)
    sig_pairs = select_significant_pairs(all_pairs, alpha)
    msfi_pairs = select_msfi_candidates(sig_pairs, msfi_threshold)

    sig_pairs.to_csv(output_dir / "pearson_significant_pairs.csv", index=False)
    msfi_pairs.to_csv(output_dir / "pearson_strong_pairs.csv", index=False)

    print(f"=== Significant correlations (p < {alpha}): {len(sig_pairs)} pairs ===")
    print(sig_pairs)
    print(f"\n=== Pairs with |r| > {msfi_threshold} — statistical basis for MSFI collinearity screening ===")
    print(msfi_pairs)

    return {
        "cor_matrix": cor_matrix,
        "p_matrix": p_matrix,
        "all_pairs": all_pairs,
        "sig_pairs": sig_pairs,
        "msfi_pairs": msfi_pairs,
    }


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    # Explicitly loads outputs/data_prepared.csv (produced by
    # 01_data_preparation.py) — no dependency on an in-memory session.
    results = run_pearson_analysis()
