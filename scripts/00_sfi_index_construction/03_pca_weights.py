# ============================================================
# STEP 3 — PRINCIPAL COMPONENT ANALYSIS (PCA)
# + Wj WEIGHT CALCULATION PER EQUATION (1)
#   Wj = |xjk| . Wk
#   xjk : loading of variable j on ITS corresponding component k
#         (the retained component where |loading| is maximal)
#   Wk  : eigenvalue of that component k
# (Python equivalent of FactoMineR::PCA(..., scale.unit=TRUE))
# ============================================================
"""
Purpose
-------
Run PCA on the 6 retained MSFI indicators (pH, CEC, clay, K, P, C —
selected after the collinearity screening in step 2), then compute the
weight Wj of each indicator per equation (1) of the methodology.

Input  : outputs/data_prepared.csv, produced by 01_data_preparation.py
Output : - outputs/pca_eigenvalues.csv
         - outputs/pca_variable_coordinates.csv
         - outputs/pca_variable_contributions.csv
         - outputs/pca_weights_Wj.csv (used in step 5 for the SFI)
         - outputs/pca_correlation_circle.png (variable correlation circle, PC1 x PC2)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs")
PREPARED_DATA_FILE = OUTPUT_DIR / "data_prepared.csv"

# The 6 MSFI indicators retained in the methodology (after the
# collinearity screening in step 2): pH, CEC, clay (continuous proxy for
# texture, retained over sand/silt), K, P, C content.
PCA_VARIABLES = ["pH", "CEC", "clay", "K", "P", "C"]

# Kaiser criterion: components retained if eigenvalue > this threshold
KAISER_THRESHOLD = 1.0

# Normalize Wj weights to sum = 1 (see docstring note above)
NORMALIZE_WEIGHTS = True

# Mapping to the score column names used in steps 4/5 ("clay" ->
# "texture_score" because the final score for this indicator is derived
# from the Table 4 texture class, not directly from % clay)
VAR_TO_SCORE_NAME = {
    "pH": "pH_score",
    "CEC": "CEC_score",
    "clay": "texture_score",
    "K": "K_score",
    "P": "P_score",
    "C": "C_score",
}


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


def select_pca_data(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Select the MSFI variables and drop incomplete observations (PCA needs complete cases)."""
    missing_cols = [v for v in variables if v not in df.columns]
    if missing_cols:
        raise KeyError(f"MSFI columns missing from `data`: {missing_cols}")

    pca_data = df[variables].dropna()
    n_dropped = len(df) - len(pca_data)
    pct_dropped = 100 * n_dropped / len(df) if len(df) else 0
    print(f"Complete observations for PCA: {len(pca_data)}/{len(df)} "
          f"({n_dropped} excluded, {pct_dropped:.1f}%)")
    if pct_dropped > 10:
        print("WARNING: more than 10% of observations are excluded due to missing values.")
    return pca_data


def standardize(pca_data: pd.DataFrame) -> np.ndarray:
    """Standardization (equivalent of scale.unit=TRUE in FactoMineR::PCA)."""
    return StandardScaler().fit_transform(pca_data)


def run_pca(X_scaled: np.ndarray, variables: list) -> tuple[PCA, pd.DataFrame]:
    """Fit the PCA and build the eigenvalue table (res.pca$eig)."""
    pca = PCA()
    pca.fit(X_scaled)

    eigenvalues = pca.explained_variance_
    variance_pct = pca.explained_variance_ratio_ * 100
    cum_variance_pct = np.cumsum(variance_pct)

    eig_table = pd.DataFrame(
        {
            "eigenvalue": eigenvalues,
            "percentage_of_variance": variance_pct,
            "cumulative_percentage": cum_variance_pct,
        },
        index=[f"Dim.{i + 1}" for i in range(len(eigenvalues))],
    )
    return pca, eig_table


def compute_variable_coordinates(pca: PCA, eig_table: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Variable coordinates on the components (res.pca$var$coord): loading x sqrt(eigenvalue)."""
    loadings = pca.components_.T
    var_coord = loadings * np.sqrt(eig_table["eigenvalue"].values)
    return pd.DataFrame(var_coord, index=variables, columns=eig_table.index)


def compute_contributions(var_coord_df: pd.DataFrame) -> pd.DataFrame:
    """Variable contributions to each component (res.pca$var$contrib, in %)."""
    contrib = var_coord_df ** 2
    return contrib.div(contrib.sum(axis=0), axis=1) * 100


def select_retained_components(eig_table: pd.DataFrame, threshold: float) -> list:
    """Kaiser criterion: components with eigenvalue > threshold."""
    retained = list(eig_table.index[eig_table["eigenvalue"] > threshold])
    if not retained:
        raise ValueError(f"No component with eigenvalue > {threshold}: check the input data.")
    print(f"Retained components (eigenvalue > {threshold}): {retained}")
    return retained


def compute_wj_weights(
    var_coord_df: pd.DataFrame,
    eig_table: pd.DataFrame,
    variables: list,
    retained_dims: list,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute the weight Wj of each indicator per equation (1):
    Wj = |xjk| . Wk, where xjk is the loading of variable j on its
    "corresponding" component k (the retained component where |loading|
    is maximal), and Wk the eigenvalue of that component.
    """
    loadings_retained = var_coord_df[retained_dims]

    assigned_component = {}
    raw_weights = {}
    for var in variables:
        abs_loadings = loadings_retained.loc[var].abs()
        best_dim = abs_loadings.idxmax()
        xjk = loadings_retained.loc[var, best_dim]
        Wk = eig_table.loc[best_dim, "eigenvalue"]
        raw_weights[var] = abs(xjk) * Wk
        assigned_component[var] = best_dim

    raw_weights = pd.Series(raw_weights, name="Wj_raw")
    weights = raw_weights / raw_weights.sum() if normalize else raw_weights.copy()
    weights.name = "Wj_normalized" if normalize else "Wj_raw"

    weights_table = pd.DataFrame({
        "matching_component": pd.Series(assigned_component),
        "abs_loading": loadings_retained.abs().max(axis=1),
        "eigenvalue_Wk": pd.Series(
            {v: eig_table.loc[assigned_component[v], "eigenvalue"] for v in variables}
        ),
        "Wj_raw": raw_weights,
        "Wj_normalized": raw_weights / raw_weights.sum(),
    })
    return weights_table


def plot_correlation_circle(var_coord_df: pd.DataFrame, eig_table: pd.DataFrame, output_path: Path) -> None:
    """Plot the PCA variable correlation circle (PC1 x PC2), one arrow per property."""
    pc1_pct = eig_table.loc["Dim.1", "percentage_of_variance"]
    pc2_pct = eig_table.loc["Dim.2", "percentage_of_variance"]

    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, fill=False, color="grey", linestyle="--")
    ax.add_patch(circle)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)

    for var in var_coord_df.index:
        x, y = var_coord_df.loc[var, "Dim.1"], var_coord_df.loc[var, "Dim.2"]
        ax.annotate(
            "", xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
        )
        ax.text(x * 1.1, y * 1.1, var, ha="center", va="center", fontsize=10)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}%)")
    ax.set_title("PCA correlation circle — MSFI indicators")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pca_analysis(
    data_file: Path = PREPARED_DATA_FILE,
    variables: list = PCA_VARIABLES,
    kaiser_threshold: float = KAISER_THRESHOLD,
    normalize_weights: bool = NORMALIZE_WEIGHTS,
    var_to_score: dict = VAR_TO_SCORE_NAME,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Full pipeline for step 3. Loads, computes, exports, and returns the results."""
    data = load_prepared_data(data_file)
    pca_data = select_pca_data(data, variables)

    X_scaled = standardize(pca_data)
    pca, eig_table = run_pca(X_scaled, variables)
    var_coord_df = compute_variable_coordinates(pca, eig_table, variables)
    contrib_df = compute_contributions(var_coord_df)
    retained_dims = select_retained_components(eig_table, kaiser_threshold)
    weights_table = compute_wj_weights(
        var_coord_df, eig_table, variables, retained_dims, normalize=normalize_weights
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    eig_table.to_csv(output_dir / "pca_eigenvalues.csv")
    var_coord_df.to_csv(output_dir / "pca_variable_coordinates.csv")
    contrib_df.to_csv(output_dir / "pca_variable_contributions.csv")
    weights_table.to_csv(output_dir / "pca_weights_Wj.csv")
    plot_correlation_circle(var_coord_df, eig_table, output_dir / "pca_correlation_circle.png")

    print("\n=== Eigenvalues (res.pca$eig) ===")
    print(eig_table)
    print("\n=== Variable coordinates (res.pca$var$coord) ===")
    print(var_coord_df)
    print("\n=== Variable contributions (res.pca$var$contrib, %) ===")
    print(contrib_df)
    print("\n=== Wj weights (equation 1) ===")
    print(weights_table)

    weights_scores = weights_table["Wj_normalized"].rename(index=var_to_score)
    print("\n=== Wj weights (indexed on score column names, used in step 5) ===")
    print(weights_scores)

    return {
        "pca": pca,
        "eig_table": eig_table,
        "var_coord": var_coord_df,
        "contrib": contrib_df,
        "retained_dims": retained_dims,
        "weights_table": weights_table,
        "weights_scores": weights_scores,
    }


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    results = run_pca_analysis()
