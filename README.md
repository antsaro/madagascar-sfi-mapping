# Madagascar Soil Fertility Index (SFI) Mapping — Random Forest Regression-Kriging

Code and analysis pipeline for the manuscript:

> **Comparative assessment of soil fertility across different land cover types in three bioclimatic regions of Madagascar**
> Sarobidy Rafidimanantsoa <sup>1,2</sup>, Nandrianina Ramifehiarivo <sup>1,2</sup>, Lovafitia Ratovoarimanana <sup>1</sup>, Andry Andriamananjara <sup>1</sup>, Steven Bouillon <sup>3</sup>, Adam Devenish <sup>4,5</sup>, Herintsitohaina Razakamanarivo <sup>1</sup>
> <sup>1</sup> Laboratoire des Radioisotopes, University of Antananarivo, Madagascar
> <sup>2</sup> École Supérieure des Sciences Agronomiques, University of Antananarivo, Madagascar
> <sup>3</sup> Department of Earth & Environmental Sciences, KU Leuven, Belgium
> <sup>4</sup> Royal Botanic Gardens Edinburgh, Edinburgh, United Kingdom
> <sup>5</sup> Royal Botanic Gardens Kew, London, United Kingdom
>
> *Submitted to Geoderma*

---

## Repository description

This repository holds the full analysis pipeline used to compute, validate, and spatially predict a multi-depth **Soil Fertility Index (SFI)** across three bioclimatic regions of Madagascar (dry northwest, sub-humid central highlands, sub-humid southeast), and to compare SFI across land-cover types (forest, shrubland, grassland, reforestation). The pipeline starts from raw physical/chemical soil-profile measurements, **constructs the SFI itself** (Minimum Soil Fertility Indicator selection, PCA-based weighting, indicator scoring and classification), then combines **Random Forest regression** on remote-sensing/terrain/climate covariates with **kriging of the spatial residuals** (regression-kriging), producing continuous raster maps of predicted soil fertility with associated uncertainty, plus figures used in the manuscript.

## Highlights

- SFI (from physical texture and chemical properties — pH, CEC, exchangeable K, available P, total C, total N) decreases significantly with soil depth.
- SFI is significantly higher in the central highlands (0.52 ± 0.11) than in the dry northwest (0.41 ± 0.09).
- SFI declines along land-cover degradation gradients: forest (0.51 ± 0.10) > shrubland (0.48 ± 0.11) ≈ grassland (0.47 ± 0.13) ≈ reforested soils (0.47 ± 0.10).
- SFI response on reforestation plots is region-specific.

---

## Repository structure

```
.
├── README.md                          <- this file (methodology + math)
├── requirements.txt                   <- CPU-only Python dependencies
├── requirements-gpu.txt               <- optional GPU (RAPIDS/cuML) dependencies
├── environment.yml                    <- conda environment (CPU + optional GPU extras)
├── LICENSE
├── CITATION.cff
├── .gitignore
├── notebooks/
│   ├── sfi_index_construction.ipynb   <- original end-to-end SFI-construction notebook
│   └── tanety_full_pipeline.ipynb     <- original end-to-end Kaggle notebook (RF/RK mapping)
└── scripts/
    ├── 00_sfi_index_construction/
    │   ├── 01_data_preparation.py
    │   ├── 02_pearson_correlation.py
    │   ├── 03_pca_weights.py
    │   ├── 04_indicator_scoring.py
    │   ├── 05_sfi_calculation.py
    │   └── 06_sfi_classification.py
    ├── 01_data_preparation/
    │   ├── 01_inspect_input_geopackage.py
    │   ├── 02_extract_covariates_at_points.py
    │   └── 03_descriptive_statistics.py
    ├── 02_feature_selection/
    │   └── 01_vif_multicollinearity_selection.py
    ├── 03_random_forest_modeling/
    │   ├── 01_optuna_tuning_and_nested_cv.py
    │   ├── 02_nested_cv_execution_and_summary.py
    │   ├── 03_observed_vs_predicted_plots.py
    │   ├── 04_final_model_training_and_shap.py
    │   └── 05_shap_figures.py
    ├── 04_rf_mapping/
    │   ├── 01_bootstrap_mean_std_prediction_gpu.py
    │   └── 02_regional_maps_mean_std.py
    ├── 05_regression_kriging_modeling/
    │   ├── 01_rk_optuna_tuning_and_nested_cv.py
    │   ├── 02_rk_results_summary_and_plots.py
    │   └── 03_rk_shap_analysis.py
    └── 06_rk_mapping/
        ├── 01_point_estimate_prediction_gpu_kriging.py
        ├── 02_point_estimate_prediction_cpu_kriging.py
        └── 03_regional_maps_point_estimate.py
```

Each numbered folder is a stage of the pipeline, meant to be run in order (`00_...` → `06_...`); scripts within a folder are also numbered in execution order. Stage `00_sfi_index_construction/` builds the SFI response variable itself from raw soil-profile measurements; stages `01_...` through `06_...` then extract covariates at those profile locations, model, and spatially predict SFI. The two original notebooks (`notebooks/sfi_index_construction.ipynb` and `notebooks/tanety_full_pipeline.ipynb`) each contain their part of the pipeline end-to-end and are kept for full reproducibility/provenance; the `scripts/` folder splits the same code into standalone, documented modules.

---

## 1. Data

- **Response variable**: `ISF_a`, `ISF_b`, `ISF_c`, `ISF_d`, `ISF_e` — the Soil Fertility Index at five depth intervals (0–10, 10–20, 20–30, 30–60, 60–90 cm respectively), derived from measured physical (texture) and chemical (pH, CEC, exchangeable K, available P, total C, total N) topsoil/subsoil properties (309 profiles). The index is built by `scripts/00_sfi_index_construction/` (see §2.0) from the raw per-depth soil-profile measurements; that stage outputs one `SFI` value per profile × depth (long format), which is pivoted to the wide, per-depth `ISF_a`…`ISF_e` columns expected by `scripts/01_data_preparation/` onward.
- **Predictors (covariates)**: automatically discovered from a directory of aligned 30 m rasters — elevation, slope, TPI, TWI, mean annual precipitation (MAP) and temperature (MAT), net primary productivity (NPP), canopy height, tree cover, Sentinel-2 reflectance bands (B2–B12) and derived spectral indices (NDVI, SAVI, GRVI, NBR, MNDWI, CIre, BSI, clay index, NIRI), plus two categorical layers (ESA WorldCover land-cover class, soil type).
- **Coordinates**: UTM 38S (`lon`, `lat` columns; metres, *not* geographic degrees) — used directly for Euclidean-distance kriging.
- **Area of interest (AOI)**: a dissolved polygon layer with a `Region` attribute (3 bioclimatic regions) used to crop and compare prediction maps.

---

## 2. Methodology

### 2.0 Soil Fertility Index (SFI) construction

Before any covariate is touched, the SFI response itself is built from the raw soil-profile measurements (`scripts/00_sfi_index_construction/`), following Sulaeman et al. (2021) / Bagherzadeh et al. (2018):

1. **Data preparation** (`01_data_preparation.py`) — parses the raw lab CSV, converts comma-decimal numeric fields, and derives the depth class/`Depth_cm`, and a texture code (USDA-style) from the measured texture class.
2. **Collinearity screening / MSFI selection** (`02_pearson_correlation.py`) — computes the Pearson correlation matrix and p-values between the candidate soil properties (N, P, C, CEC, pH, K, clay, silt, sand); pairs with |r| > 0.8 flag redundant variables. Combined with expert judgement, this narrows the set down to six **Minimum Soil Fertility Indicators (MSFI)**: pH, CEC, clay (retained over sand/silt as the texture proxy), K, P, C.
3. **PCA-based weighting** (`03_pca_weights.py`) — a PCA (`scale.unit = TRUE` equivalent) is run on the standardized MSFI indicators; components with eigenvalue > 1 are retained (Kaiser criterion). Each indicator *j* is assigned the retained component *k* on which its loading is largest, and its weight follows equation (1):
   ```
   Wj = |xjk| . Wk
   ```
   where `xjk` is the loading of indicator *j* on its assigned component *k*, and `Wk` the eigenvalue of that component. Weights are normalized to sum to 1.
4. **Indicator scoring** (`04_indicator_scoring.py`) — each indicator is cut into 5 fertility classes (Table 4, Sulaeman et al., 2021: thresholds for pH, CEC, exchangeable K, available P, total C, plus a texture-class score derived from the USDA texture code), scored 1 (very low) to 5 (very high), then standardized as `Sij . p` with `p = 1/n` (`n = 5` classes).
5. **SFI calculation** (`05_sfi_calculation.py`) — the index is aggregated per equation (2):
   ```
   SFIi = Σ_j Wj . Sij . p
   ```
   using only observations with all 6 MSFI indicators available (`p` is already embedded in the standardized scores from step 4 and is not re-applied here).
6. **Classification** (`06_sfi_classification.py`) — each computed SFI is assigned one of 5 fertility levels (Table 5, Bagherzadeh et al., 2018): Very low (≤ 0.25), Low (0.25–0.50), Moderate (0.50–0.75), High (0.75–0.90), Very high (> 0.90).

The output (`outputs/sfi_summary.csv`, one row per profile × depth) is pivoted to the wide `ISF_a`…`ISF_e` per-depth columns before being consumed by `scripts/01_data_preparation/` for covariate extraction.

### 2.1 Covariate extraction

Raster covariate values are sampled at each soil-profile location (point sampling, nearest pixel), reprojecting points to each raster's native CRS before sampling. Extracted values feed directly into feature selection and model training.

### 2.2 Multicollinearity screening (VIF)

Before model fitting, numeric predictors are screened for multicollinearity using the **Variance Inflation Factor**:

```
VIF_j = 1 / (1 - R_j^2)
```

where `R_j^2` is the coefficient of determination of predictor *j* regressed on all remaining predictors. A stepwise elimination removes, at each iteration, the non-priority predictor with the highest VIF (threshold VIF ≥ 10), while a small set of ecologically important predictors (`MAP_30m`, `Elevation_30m`) is always retained regardless of VIF. Elimination stops once no non-priority predictor exceeds the threshold.

### 2.3 Random Forest regression (trend model)

For each depth response, a **Random Forest regressor** is trained:

```
f(x) = (1 / T) * Σ_{t=1}^{T} f_t(x)
```

where each of the `T` regression trees `f_t` is grown on a bootstrap resample of the training data with a random subset of predictors considered at each split (`max_features`). Hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`) are tuned with **Optuna** (Tree-structured Parzen Estimator sampler), minimizing inner-fold RMSE.

**Nested cross-validation** is used throughout to obtain unbiased performance estimates:
- *Outer loop* (`K = 5` folds): held-out folds used only for final evaluation.
- *Inner loop* (`K = 3`–`4` folds, within each outer training fold): used exclusively for Optuna hyperparameter search — the outer test fold never participates in tuning or imputation/encoding fitting (leakage-safe: medians, modes, and one-hot encoders are fit on the training fold only).

**Evaluation metrics** (computed on pooled out-of-fold predictions):

```
R²   = 1 − Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²

RMSE = sqrt( (1/n) Σ (y_i − ŷ_i)² )

MAE  = (1/n) Σ |y_i − ŷ_i|

RPIQ = IQR(y) / RMSE          (IQR = 75th − 25th percentile of observed y)

CCC  = 2·s_xy / (s_x² + s_y² + (x̄ − ȳ)²)     (Lin's concordance correlation coefficient)
```

where `x = y_obs`, `y = y_pred`, `s_xy` is their covariance, and `s_x², s_y²` their variances.

### 2.4 Model interpretability (SHAP)

Feature importance and directionality are assessed with **SHAP (SHapley Additive exPlanations)** values, computed via `TreeExplainer`. For a prediction `f(x)`, the contribution of feature *i* is the Shapley value:

```
φ_i = Σ_{S ⊆ F\{i}}  [ |S|! (|F| − |S| − 1)! / |F|! ]  ·  [ f(S ∪ {i}) − f(S) ]
```

summed over all subsets `S` of the other features `F\{i}`, so that predictions decompose additively: `f(x) = φ_0 + Σ_i φ_i`.

### 2.5 Regression-Kriging (RK) of spatial residuals

Random Forest captures non-linear covariate relationships but ignores spatial autocorrelation in the residuals. **Regression-Kriging** corrects this by adding a kriged interpolation of the RF residuals to the RF trend:

```
Ẑ(s₀) = m̂(s₀) + ê(s₀)
```

where `m̂(s₀)` is the RF trend prediction at location `s₀`, and `ê(s₀)` is the **Ordinary Kriging** prediction of the trend's residuals `e(s) = y(s) − m̂(s)`.

**Semivariogram.** Spatial dependence of the residuals is described by the empirical semivariogram:

```
γ(h) = (1 / 2N(h)) · Σ_{i=1}^{N(h)} [ e(s_i) − e(s_i + h) ]²
```

where `h` is the separation distance and `N(h)` the number of residual pairs at that lag. A parametric model is fit to the empirical semivariogram; five candidate families are searched (selected per depth by inner-CV kriging RMSE, together with the number of lag bins and whether closer lags are up-weighted in the least-squares fit):

```
Linear:       γ(h) = slope·h + nugget

Power:        γ(h) = scale·h^exponent + nugget

Gaussian:     γ(h) = psill·(1 − exp(−h² / (4r²/7))) + nugget

Exponential:  γ(h) = psill·(1 − exp(−h / (r/3))) + nugget

Spherical:    γ(h) = psill·(3h/2r − h³/2r³) + nugget,   h ≤ r
              γ(h) = psill + nugget,                      h > r
```

with `r` the range, `psill` the partial sill, and `nugget` the nugget effect.

**Ordinary Kriging (local, moving-window).** For a query location `s₀`, its `K` nearest sampled residuals are used. The kriging weights `λ = (λ₁, …, λ_K)` and Lagrange multiplier `μ` solve the linear system enforcing unbiasedness (`Σλ_i = 1`):

```
⎡ γ(s₁,s₁) … γ(s₁,s_K)  1 ⎤ ⎡ λ₁ ⎤   ⎡ γ(s₁,s₀) ⎤
⎢     ⋮      ⋱     ⋮    ⋮ ⎥ ⎢ ⋮  ⎥ = ⎢    ⋮     ⎥
⎢ γ(s_K,s₁) … γ(s_K,s_K) 1 ⎥ ⎢λ_K ⎥   ⎢ γ(s_K,s₀)⎥
⎣     1      …     1     0 ⎦ ⎣ μ  ⎦   ⎣    1     ⎦
```

and the kriged residual estimate is:

```
ê(s₀) = Σ_{i=1}^{K} λ_i · e(s_i)
```

Two numerically-equivalent implementations of this local OK step are provided and cross-validated against each other:
- **GPU** (`06_rk_mapping/01_...gpu_kriging.py`): variogram fit on CPU via PyKrige (cheap, done once), then per-pixel KNN search + batched `(K+1)×(K+1)` linear solves on GPU (`cuML` KNN + `CuPy` batched `linalg.solve`) for scalable prediction over large rasters.
- **CPU** (`06_rk_mapping/02_...cpu_kriging.py`): identical mathematical formulation, executed entirely through PyKrige's native `OrdinaryKriging.execute(backend='C')`, used as a reference/fallback and for cross-checking the GPU implementation (`validate_gpu_kriging` / `validate_pykrige_cpu` both report max |CPU − GPU| discrepancy on a synthetic case).

### 2.6 GPU-accelerated raster prediction

The final RF trend is compiled to Forest Inference Library (**FIL**) format (`cuML`) for fast batched GPU inference across every valid pixel of the covariate raster stack, tiled to bound memory use. Two prediction modes are implemented:
- **Bootstrap mean ± std** (`04_rf_mapping/`, RF-only trend): `N_BOOTSTRAP` models are refit on bootstrap resamples of the training data; per-pixel mean and standard deviation across the ensemble give a point estimate and an uncertainty band, `mean(s₀) = (1/B) Σ_b f_b(s₀)`, `std(s₀) = sqrt( (1/B) Σ_b (f_b(s₀) − mean(s₀))² )`.
- **Point estimate** (`06_rk_mapping/`, RF + kriged residual): a single deterministic map per depth, `Ẑ(s₀) = m̂(s₀) + ê(s₀)`, using the already-tuned final model and variogram (no resampling).

### 2.7 Regional comparison maps

Prediction rasters are cropped to each bioclimatic region using the AOI polygon layer, then classified into **quantile (equal-count) bins** — rather than equal-width bins — for visual contrast on skewed environmental data, and rendered as multi-panel (depth × region) figures using a consistent "fertility" colour ramp (red/orange = low → yellow = moderate → green = high).

---

## 3. Reproducing the pipeline

```bash
# 1) Environment
conda env create -f environment.yml
conda activate sfi-madagascar
# (GPU steps additionally require a CUDA-enabled machine with RAPIDS cuML/cuPy —
#  see requirements-gpu.txt; CPU-only kriging/mapping scripts do not need this.)

# 2) Run stages in order (adjust the hard-coded /kaggle/... paths at the top of
#    each script to your own data/output directories first)
python scripts/00_sfi_index_construction/01_data_preparation.py
python scripts/00_sfi_index_construction/02_pearson_correlation.py
python scripts/00_sfi_index_construction/03_pca_weights.py
python scripts/00_sfi_index_construction/04_indicator_scoring.py
python scripts/00_sfi_index_construction/05_sfi_calculation.py
python scripts/00_sfi_index_construction/06_sfi_classification.py
# -> pivot outputs/sfi_final_summary.csv (long: one row per profile x depth)
#    to wide ISF_a...ISF_e columns before continuing
python scripts/01_data_preparation/02_extract_covariates_at_points.py
python scripts/02_feature_selection/01_vif_multicollinearity_selection.py
python scripts/03_random_forest_modeling/02_nested_cv_execution_and_summary.py
python scripts/03_random_forest_modeling/04_final_model_training_and_shap.py
python scripts/04_rf_mapping/01_bootstrap_mean_std_prediction_gpu.py     # GPU
python scripts/04_rf_mapping/02_regional_maps_mean_std.py
python scripts/05_regression_kriging_modeling/01_rk_optuna_tuning_and_nested_cv.py
python scripts/06_rk_mapping/01_point_estimate_prediction_gpu_kriging.py # GPU
python scripts/06_rk_mapping/02_point_estimate_prediction_cpu_kriging.py # CPU fallback
python scripts/06_rk_mapping/03_regional_maps_point_estimate.py
```

Scripts were originally authored and run in a Kaggle GPU notebook environment (`/kaggle/working/...`, `/kaggle/input/...` paths); these paths are the single block of `CONFIGURATION` constants at the top of every script and are the only things that need to change to run elsewhere.

---

## 4. Outputs

| Stage | Output |
|---|---|
| SFI — correlation / MSFI screening | `outputs/correlation_heatmap.png`, `outputs/pearson_*_matrix.csv`, `outputs/pearson_strong_pairs.csv` |
| SFI — PCA weighting | `outputs/pca_eigenvalues.csv`, `outputs/pca_weights_Wj.csv`, `outputs/pca_correlation_circle.png` |
| SFI — scoring / calculation / classification | `outputs/data_scored.csv`, `outputs/data_sfi.csv`, `outputs/sfi_summary.csv`, `outputs/data_sfi_classified.csv`, `outputs/sfi_final_summary.csv` |
| Feature selection | `VIF/vif_final.csv`, VIF barplot figure |
| RF nested CV | per-response metrics table (R², RMSE, MAE, RPIQ, CCC), observed-vs-predicted scatter figure |
| RF final model | `models/final_model_{response}.joblib` (fitted RF + medians/modes/encoder bundle) |
| SHAP | per-response SHAP summary CSVs, SHAP bar/beeswarm/dependence figures |
| RF bootstrap maps | `Maps/ISF/{response}/{response}_mean.tif`, `{response}_std.tif` |
| RK nested CV | RF-only vs RF+Kriging metrics comparison, tuned variogram parameters |
| RK final model | `models/final_rk_model_{response}.joblib` (fitted RF + tuned variogram + preprocessing bundle) |
| RK point-estimate maps | `Maps/ISF_RK_point/{response}/{response}.tif` (GPU), `Maps/ISF_RK_point_cpuKrige/{response}/{response}.tif` (CPU) |
| Regional figures | `ISF_by_region_5x3_mean/std.png/pdf`, `ISF_by_region_pointEstimate_GPU/CPU.png/pdf` |

---

## 5. Citation

If you use this code, please cite the associated manuscript (citation to be updated upon publication — see `CITATION.cff`):

> Rafidimanantsoa, S., Ramifehiarivo, N., Ratovoarimanana, L., Andriamananjara, A., Bouillon, S., Devenish, A., Razakamanarivo, H. *Comparative assessment of soil fertility across different land cover types in three bioclimatic regions of Madagascar*. Submitted to *Geoderma*.

## 6. License

Code released under the MIT License (see `LICENSE`). Soil and covariate data are not redistributed in this repository; contact the corresponding author for data access.

## 7. Acknowledgments

Laboratoire des Radioisotopes (University of Antananarivo), École Supérieure des Sciences Agronomiques, KU Leuven, Royal Botanic Gardens Edinburgh and Kew.
