# Changelog

## 2026-11-03: manuscript title and authors
- README and `CITATION.cff` updated to the new manuscript title, *Spatial prediction of soil fertility across tropical
  forest-grassland mosaics*, with the full author list, numbered affiliations (including Université catholique de Louvain),
  equal-contribution note and corresponding author.

## 2026-11-03: publication figures
- New `scripts/08_publication_figures/01_nature_style_figures.py`: Nature-style figures (Arial, exact journal widths,
  panel letters, 600 dpi PNG + vector PDF) in original and colour-blind palettes, with four combined main figures,
  a supplementary diagnostics figure, draft figure legends and an index of files.

## 2026-11-02: SFI assessment update and mixed-model analysis

### Changed: SFI construction (`scripts/00_sfi_index_construction/`, `notebooks/sfi_index_construction.ipynb`)
- **Collinearity threshold** for MSFI screening lowered from |r| > 0.8 to |r| > 0.75
  (`02_pearson_correlation.py`). The flagged pairs are unchanged (N–C, clay–sand, silt–sand).
- **pH scoring is now two-sided** (`04_indicator_scoring.py`): nine intervals
  (≤5, 5–5.4, 5.4–5.8, 5.8–6, 6–7, 7–7.4, 7.4–7.8, 7.8–8.2, >8.2) scored
  1-2-3-4-5-4-3-2-1, so alkaline soils are penalised as well as acidic ones.
  Previously pH > 6 always scored 5. Includes a check that each indicator has
  one label per bin.
- Table and equation numbers aligned with the revised manuscript: scoring thresholds
  = Table 3, SFI classes = Table 4, weights = equation (6), SFI = equation (7).
- Input file renamed to `data/SFI_CALCULATION.csv`. Outputs go to `outputs/sfi_index/`.
  Paths are relative to the repository root.

### Added: linear mixed-effects analysis (`scripts/07_sfi_mixed_models/`)
- `01_sfi_lmm_analysis.py`: random-effect and fixed-effect structure comparison,
  likelihood-ratio tests of interactions, estimated marginal means, Holm-adjusted
  contrasts and compact letters, Nakagawa R² / ICC, RMSE/RPD/RPIQ, residual diagnostics,
  Excel report (`outputs/lmm/tables/SFI_results.xlsx`) and figure annotation tables.
- `02_sfi_lmm_figures.py`: boxplot and depth-profile figures (PNG 300 dpi + PDF).
- Notebooks `notebooks/sfi_lmm_analysis.ipynb` and `notebooks/sfi_lmm_figures.ipynb`.
- Input `data/DATA_LM.csv` (1,523 samples, 307 plots).

### Added: data, outputs, housekeeping
- `data/` with both input tables; `outputs/` with regenerated results of stages 00 and 07.
- `.gitignore` (it was listed in the README but missing).
- `patsy` and minimum versions in `requirements.txt` / `environment.yml`.
- README: updated structure, data description (1,523 samples / 307 plots, which replaces "309 profiles"),
  SFI method, new Section 2.8 (mixed models), run commands, outputs table, references.
