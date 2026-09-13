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

This repository holds the full analysis pipeline used to build, validate, and spatially predict a multi-depth **Soil Fertility Index (SFI)** across three bioclimatic regions of Madagascar (dry northwest, sub-humid central highlands, sub-humid southeast), and to compare SFI across land-cover types (forest, shrubland, grassland, reforestation). The pipeline combines **Random Forest regression** on remote-sensing/terrain/climate covariates with **kriging of the spatial residuals** (regression-kriging), producing continuous raster maps of predicted soil fertility with associated uncertainty, plus figures used in the manuscript.

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
│   └── tanety_full_pipeline.ipynb     <- original end-to-end Kaggle notebook
└── scripts/
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

Each numbered folder is a stage of the pipeline, meant to be run in order (`01_...` → `06_...`); scripts within a folder are also numbered in execution order. The original Kaggle notebook (`notebooks/tanety_full_pipeline.ipynb`) contains all stages end-to-end and is kept for full reproducibility/provenance; the `scripts/` folder splits the same code into standalone, documented modules.

---

## 1. Data

- **Response variable**: `ISF_a`, `ISF_b`, `ISF_c`, `ISF_d`, `ISF_e` — the Soil Fertility Index at five depth intervals (0–10, 10–20, 20–30, 30–60, 60–90 cm respectively), derived from measured physical (texture) and chemical (pH, CEC, exchangeable K, available P, total C, total N) topsoil/subsoil properties (309 profiles). The index construction itself (weighting/aggregation of the individual soil properties into a single fertility score) is documented in the manuscript; this repository starts from the already-computed depth-wise index values.
- **Predictors (covariates)**: automatically discovered from a directory of aligned 30 m rasters — elevation, slope, TPI, TWI, mean annual precipitation (MAP) and temperature (MAT), net primary productivity (NPP), canopy height, tree cover, Sentinel-2 reflectance bands (B2–B12) and derived spectral indices (NDVI, SAVI, GRVI, NBR, MNDWI, CIre, BSI, clay index, NIRI), plus two categorical layers (ESA WorldCover land-cover class, soil type).
- **Coordinates**: UTM 38S (`lon`, `lat` columns; metres, *not* geographic degrees) — used directly for Euclidean-distance kriging.
- **Area of interest (AOI)**: a dissolved polygon layer with a `Region` attribute (3 bioclimatic regions) used to crop and compare prediction maps.

---

## 2. Methodology

### 2.1 Covariate extraction

Raster covariate values are sampled at each soil-profile location (point sampling, nearest pixel), reprojecting points to each raster's native CRS before sampling. Extracted values feed directly into feature selection and model training.

### 2.2 Multicollinearity screening (VIF)

Before model fitting, numeric predictors are screened for multicollinearity using the **Variance Inflation Factor**:

$$VIF_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination of predictor *j* regressed on all remaining predictors. A stepwise elimination removes, at each iteration, the non-priority predictor with the highest VIF (threshold VIF ≥ 10), while a small set of ecologically important predictors (`MAP_30m`, `Elevation_30m`) is always retained regardless of VIF. Elimination stops once no non-priority predictor exceeds the threshold.

### 2.3 Random Forest regression (trend model)

For each depth response, a **Random Forest regressor** is trained:

$$f(x) = \frac{1}{T} \sum_{t=1}^{T} f_t(x)$$

where each of the $T$ regression trees $f_t$ is grown on a bootstrap resample of the training data with a random subset of predictors considered at each split (`max_features`). Hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`) are tuned with **Optuna** (Tree-structured Parzen Estimator sampler), minimizing inner-fold RMSE.

**Nested cross-validation** is used throughout to obtain unbiased performance estimates:
- *Outer loop* (`K = 5` folds): held-out folds used only for final evaluation.
- *Inner loop* (`K = 3`–`4` folds, within each outer training fold): used exclusively for Optuna hyperparameter search — the outer test fold never participates in tuning or imputation/encoding fitting (leakage-safe: medians, modes, and one-hot encoders are fit on the training fold only).

**Evaluation metrics** (computed on pooled out-of-fold predictions):

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

$$RMSE = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}$$

$$MAE = \frac{1}{n}\sum_i \left|y_i - \hat{y}_i\right|$$

$$RPIQ = \frac{IQR(y)}{RMSE} \qquad \big(IQR = \text{75th} - \text{25th percentile of observed } y\big)$$

$$CCC = \frac{2\,s_{xy}}{s_x^2 + s_y^2 + (\bar{x} - \bar{y})^2} \qquad \text{(Lin's concordance correlation coefficient)}$$

where $x = y_{obs}$, $y = y_{pred}$, $s_{xy}$ is their covariance, and $s_x^2, s_y^2$ their variances.

### 2.4 Model interpretability (SHAP)

Feature importance and directionality are assessed with **SHAP (SHapley Additive exPlanations)** values, computed via `TreeExplainer`. For a prediction $f(x)$, the contribution of feature $i$ is the Shapley value:

$$\phi_i = \sum_{S \,\subseteq\, F \setminus \{i\}} \frac{|S|!\,\big(|F| - |S| - 1\big)!}{|F|!} \;\Big[ f\big(S \cup \{i\}\big) - f(S) \Big]$$

summed over all subsets $S$ of the other features $F \setminus \{i\}$, so that predictions decompose additively: $f(x) = \phi_0 + \sum_i \phi_i$.

### 2.5 Regression-Kriging (RK) of spatial residuals

Random Forest captures non-linear covariate relationships but ignores spatial autocorrelation in the residuals. **Regression-Kriging** corrects this by adding a kriged interpolation of the RF residuals to the RF trend:

$$\hat{Z}(s_0) = \hat{m}(s_0) + \hat{e}(s_0)$$

where $\hat{m}(s_0)$ is the RF trend prediction at location $s_0$, and $\hat{e}(s_0)$ is the **Ordinary Kriging** prediction of the trend's residuals $e(s) = y(s) - \hat{m}(s)$.

**Semivariogram.** Spatial dependence of the residuals is described by the empirical semivariogram:

$$\gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} \big[e(s_i) - e(s_i + h)\big]^2$$

where $h$ is the separation distance and $N(h)$ the number of residual pairs at that lag. A parametric model is fit to the empirical semivariogram; five candidate families are searched (selected per depth by inner-CV kriging RMSE, together with the number of lag bins and whether closer lags are up-weighted in the least-squares fit):

**Linear:**
$$\gamma(h) = \text{slope} \cdot h + \text{nugget}$$

**Power:**
$$\gamma(h) = \text{scale} \cdot h^{\text{exponent}} + \text{nugget}$$

**Gaussian:**
$$\gamma(h) = \text{psill} \left(1 - \exp\!\left(-\frac{h^2}{4r^2/7}\right)\right) + \text{nugget}$$

**Exponential:**
$$\gamma(h) = \text{psill} \left(1 - \exp\!\left(-\frac{h}{r/3}\right)\right) + \text{nugget}$$

**Spherical:**
$$\gamma(h) = \begin{cases} \text{psill}\left(\dfrac{3h}{2r} - \dfrac{h^3}{2r^3}\right) + \text{nugget}, & h \le r \\ \text{psill} + \text{nugget}, & h > r \end{cases}$$

with $r$ the range, $\text{psill}$ the partial sill, and $\text{nugget}$ the nugget effect.

**Ordinary Kriging (local, moving-window).** For a query location $s_0$, its $K$ nearest sampled residuals are used. The kriging weights $\lambda = (\lambda_1, \ldots, \lambda_K)$ and Lagrange multiplier $\mu$ solve the linear system enforcing unbiasedness ($\sum_i \lambda_i = 1$):

$$\begin{bmatrix} \gamma(s_1,s_1) & \cdots & \gamma(s_1,s_K) & 1 \\ \vdots & \ddots & \vdots & \vdots \\ \gamma(s_K,s_1) & \cdots & \gamma(s_K,s_K) & 1 \\ 1 & \cdots & 1 & 0 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \vdots \\ \lambda_K \\ \mu \end{bmatrix} = \begin{bmatrix} \gamma(s_1,s_0) \\ \vdots \\ \gamma(s_K,s_0) \\ 1 \end{bmatrix}$$

and the kriged residual estimate is:

$$\hat{e}(s_0) = \sum_{i=1}^{K} \lambda_i \cdot e(s_i)$$

Two numerically-equivalent implementations of this local OK step are provided and cross-validated against each other:
- **GPU** (`06_rk_mapping/01_...gpu_kriging.py`): variogram fit on CPU via PyKrige (cheap, done once), then per-pixel KNN search + batched `(K+1)×(K+1)` linear solves on GPU (`cuML` KNN + `CuPy` batched `linalg.solve`) for scalable prediction over large rasters.
- **CPU** (`06_rk_mapping/02_...cpu_kriging.py`): identical mathematical formulation, executed entirely through PyKrige's native `OrdinaryKriging.execute(backend='C')`, used as a reference/fallback and for cross-checking the GPU implementation (`validate_gpu_kriging` / `validate_pykrige_cpu` both report max |CPU − GPU| discrepancy on a synthetic case).

### 2.6 GPU-accelerated raster prediction

The final RF trend is compiled to Forest Inference Library (**FIL**) format (`cuML`) for fast batched GPU inference across every valid pixel of the covariate raster stack, tiled to bound memory use. Two prediction modes are implemented:
- **Bootstrap mean ± std** (`04_rf_mapping/`, RF-only trend): `N_BOOTSTRAP` models are refit on bootstrap resamples of the training data; per-pixel mean and standard deviation across the ensemble give a point estimate and an uncertainty band, $\text{mean}(s_0) = \frac{1}{B}\sum_b f_b(s_0)$, $\text{std}(s_0) = \sqrt{\frac{1}{B}\sum_b \big(f_b(s_0) - \text{mean}(s_0)\big)^2}$.
- **Point estimate** (`06_rk_mapping/`, RF + kriged residual): a single deterministic map per depth, $\hat{Z}(s_0) = \hat{m}(s_0) + \hat{e}(s_0)$, using the already-tuned final model and variogram (no resampling).

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
