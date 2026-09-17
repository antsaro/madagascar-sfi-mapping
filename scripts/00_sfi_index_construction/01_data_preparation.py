# ============================================================
# STEP 1 — DATA IMPORT AND VARIABLE PREPARATION
# ============================================================
"""
Purpose
-------
Import the raw soil fertility dataset and prepare the categorical
factors, numeric variables, depth-in-cm, and texture code needed for
the Soil Fertility Index (SFI) pipeline (Pearson correlation, PCA,
indicator scoring, SFI calculation, classification).

Input  : raw CSV, ';'-separated, comma decimal
Output : outputs/data_prepared.csv, reused as-is by every subsequent
         step (02_pearson_correlation.py, 03_pca_weights.py, etc.)
"""

import sys
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIGURATION — the only block to edit when switching datasets
# ============================================================

INPUT_FILE = Path("DEFINITIVEMENT_SJS_AMEN.csv")
PREPARED_DATA_FILE = Path("outputs/data_prepared.csv")

# Unique observation identifier
ID_COLUMN = "ID_LABO"

# Raw column -> final column name, converted to numeric (decimal = point)
NUMERIC_COLUMNS = {
    "PH": "pH",
    "K": "K",
    "P": "P",
    "CEC": "CEC",
    "ARGILE": "clay",
    "LIMON": "silt",
    "SABLE": "sand",
    "N": "N",
    "C": "C",
}

# Depth is coded numerically (1-5); same depth classes across all sites
DEPTH_ORDER = [1, 2, 3, 4, 5]
DEPTH_TO_CM = {1: 10, 2: 20, 3: 30, 4: 60, 5: 90}

# Texture -> abbreviated code (USDA-style), used later for the
# Table 4 (Sulaeman et al., 2021) texture-based scoring in step 4.
# NOTE: "SILT" and "SILT LOAM" are required for the VL and M classes
# of the texture scoring — do not remove them.
TEXTURE_CODE = {
    "CLAY": "C", "CLAY LOAM": "CL", "LOAM": "L", "SAND": "S",
    "SANDY CLAY": "SC", "SANDY CLAY LOAM": "SCL", "SANDY LOAM": "SL",
    "LOAMY SAND": "LS", "SILTY CLAY": "SiC", "SILTY CLAY LOAM": "SiCL",
    "SILT": "Si", "SILT LOAM": "SiL",
}


# ============================================================
# FUNCTIONS
# ============================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV with automatic field-separator detection."""
    if not path.exists():
        sys.exit(f"Error: file not found -> {path}")

    df = pd.read_csv(path, sep=None, engine="python", decimal=".")

    if df.shape[1] == 1:
        sys.exit(
            "Error: only one column detected — the field separator was "
            "likely not identified correctly. Check the file or set "
            "sep=... explicitly."
        )
    return df


def convert_numeric_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Convert comma-decimal text columns to float and rename them."""
    for raw_col, final_col in mapping.items():
        df[final_col] = pd.to_numeric(
            df[raw_col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return df


def add_categorical_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Define categorical factors: Site, LandUse, Texture."""
    df["Site"] = df["Site"].astype("category")
    df["LandUse"] = df["Landuse"].astype("category")
    df["Texture"] = df["TEXTURE"].astype("category")
    return df


def add_depth_variables(df: pd.DataFrame, order: list, cm_mapping: dict) -> pd.DataFrame:
    """Add depth as an ordered factor (Depth_class) and its value in cm (Depth_cm)."""
    df["Depth_class"] = pd.Categorical(df["Depth"], categories=order, ordered=True)
    df["Depth_cm"] = df["Depth"].map(cm_mapping)
    return df


def add_texture_code(df: pd.DataFrame, code_mapping: dict) -> pd.DataFrame:
    """Add a 'Texture_code' column (short abbreviation, e.g. SiCL, CL, S)."""
    df["Texture_code"] = df["TEXTURE"].str.upper().map(code_mapping)
    unmatched = df.loc[df["Texture_code"].isna() & df["TEXTURE"].notna(), "TEXTURE"].unique()
    if len(unmatched) > 0:
        print(f"WARNING: unrecognized texture classes in TEXTURE_CODE: {list(unmatched)}")
    return df


def check_id_uniqueness(df: pd.DataFrame, id_col: str) -> None:
    """Check that the identifier column has no missing values or duplicates."""
    if id_col not in df.columns:
        print(f"WARNING: identifier column '{id_col}' is missing from the file.")
        return
    n_missing = df[id_col].isna().sum()
    n_duplicates = df[id_col].duplicated().sum()
    print(f"=== Identifier check ('{id_col}') ===")
    print(f"Observations: {len(df)} | Missing: {n_missing} | Duplicates: {n_duplicates}")
    if n_missing or n_duplicates:
        print("WARNING: check the identifiers before proceeding with the analysis.")


def check_data_quality(df: pd.DataFrame, columns: list) -> None:
    """Report missing values per column and flag any fully-empty column."""
    n_na = df[columns].isna().sum()
    print("=== Missing values per column (after conversion) ===")
    print(n_na)
    fully_empty = n_na[n_na == len(df)].index.tolist()
    if fully_empty:
        print(f"\nWARNING: fully empty column(s) -> {fully_empty}. "
              f"Check the corresponding raw column name.")


def save_prepared_data(df: pd.DataFrame, path: Path) -> None:
    """Save the prepared DataFrame, to be reloaded explicitly by later steps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved prepared data -> {path}")


def prepare_data(input_file: Path = INPUT_FILE, output_file: Path = PREPARED_DATA_FILE) -> pd.DataFrame:
    """Full pipeline: load + prepare variables."""
    df = load_raw_data(input_file)
    df = convert_numeric_columns(df, NUMERIC_COLUMNS)
    df = add_categorical_factors(df)
    df = add_depth_variables(df, DEPTH_ORDER, DEPTH_TO_CM)
    df = add_texture_code(df, TEXTURE_CODE)

    check_id_uniqueness(df, ID_COLUMN)
    check_data_quality(df, list(NUMERIC_COLUMNS.values()))

    print("\n=== Prepared dataset preview ===")
    print(df.info())
    print(df.head())

    save_prepared_data(df, output_file)
    return df


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    data = prepare_data()
