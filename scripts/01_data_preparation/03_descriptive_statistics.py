"""
Descriptive statistics (min/max/mean/median/std/CV/missing count) for
every numeric covariate, exported to an Excel workbook for the
manuscript's supplementary material.
"""

# -----------------------------------------------------------------------
# STATS DESCRIPTIVES DES VARIABLES NUMERIQUES -> EXCEL
# -----------------------------------------------------------------------
NUMERIC_COVARIATES = [c for c in COVARIATE_NAMES if c not in CATEGORICAL_COVARIATES]

desc_stats = extracted_df[NUMERIC_COVARIATES].describe().T
desc_stats['cv'] = desc_stats['std'] / desc_stats['mean'] * 100
desc_stats['missing'] = extracted_df[NUMERIC_COVARIATES].isna().sum()
desc_stats = desc_stats.rename_axis('variable').reset_index()

STATS_XLSX = os.path.join(RESULTS_DIR, 'numeric_covariates_stats.xlsx')
desc_stats.to_excel(STATS_XLSX, index=False, sheet_name='descriptive_stats')

print(f"Saved descriptive stats -> {STATS_XLSX}")
print(desc_stats)
