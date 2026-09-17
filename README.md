# Madagascar Soil Fertility Index (SFI) Mapping: Random Forest Regression-Kriging

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

This repository contains the full analysis pipeline we used to build, validate, and spatially predict a Soil Fertility Index (SFI) at five soil depth intervals across three bioclimatic regions of Madagascar (dry northwest, sub-humid central highlands, sub-humid southeast), and to compare fertility across land cover types (forest, shrubland, grassland, reforestation). We start from raw physical and chemical soil-profile measurements and construct the SFI itself (indicator selection, PCA-based weighting, scoring, and classification). We then model the index with Random Forest regression on remote sensing, terrain, and climate covariates, and add kriging of the spatial residuals (regression-kriging) to produce continuous raster maps of predicted soil fertility with associated uncertainty.

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
│   ├── sfi_index_construction.ipynb   <- original end-to-end SFI construction notebook
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

Each numbered folder is a stage of the pipeline, meant to be run in order (`00_...` through `06_...`), and the scripts within a folder are numbered in execution order too. Stage `00_sfi_index_construction/` builds the SFI response itself from raw soil-profile measurements. Stages `01_...` through `06_...` then extract covariates at those profile locations, train the models, and produce the spatial predictions. The two original notebooks (`notebooks/sfi_index_construction.ipynb` and `notebooks/tanety_full_pipeline.ipynb`) each hold their part of the pipeline end to end and are kept for provenance; the `scripts/` folder splits the same code into standalone, documented modules.

---

## 1. Data

Our response variable is the Soil Fertility Index computed at five depth intervals (0-10, 10-20, 20-30, 30-60, and 60-90 cm), derived from measured physical (texture) and chemical (pH, CEC, exchangeable K, available P, total C, total N) topsoil and subsoil properties across 309 profiles. The index is built by `scripts/00_sfi_index_construction/` (see Section 2.0) directly from the raw per-depth soil-profile measurements. That stage outputs one SFI value per profile and depth in long format, which is then pivoted to a wide table with one column per depth interval, the format expected by `scripts/01_data_preparation/` onward.

Predictors are covariates automatically discovered from a directory of aligned 30 m rasters: elevation, slope, TPI, TWI, mean annual precipitation and temperature, net primary productivity, canopy height, tree cover, Sentinel-2 reflectance bands (B2-B12), derived spectral indices (NDVI, SAVI, GRVI, NBR, MNDWI, CIre, BSI, clay index, NIRI), and two categorical layers (ESA WorldCover land cover class, soil type).

Coordinates are given in UTM 38S (`lon`, `lat` columns, in metres rather than geographic degrees), used directly for Euclidean-distance kriging. The area of interest is a dissolved polygon layer with a `Region` attribute covering the three bioclimatic regions, used to crop and compare the prediction maps.

---

## 2. Methodology

### 2.0 Soil Fertility Index (SFI) construction

Before any covariate is touched, we build the SFI response itself from the raw soil-profile measurements (`scripts/00_sfi_index_construction/`), following Sulaeman et al. (2021) and Bagherzadeh et al. (2018).

1. **Data preparation** (`01_data_preparation.py`) parses the raw lab CSV, converts comma-decimal numeric fields, and derives the depth class, the depth in centimetres, and a texture code (USDA-style) from the measured texture class.
2. **Collinearity screening and MSFI selection** (`02_pearson_correlation.py`) computes the Pearson correlation matrix and associated p-values between candidate soil properties (N, P, C, CEC, pH, K, clay, silt, sand). Pairs with an absolute correlation above 0.8 flag redundant variables. Combined with expert judgement, this narrows the set down to six Minimum Soil Fertility Indicators (MSFI): pH, CEC, clay (retained over sand and silt as the texture proxy), K, P, and C.
3. **PCA-based weighting** (`03_pca_weights.py`) runs a PCA (equivalent to `scale.unit = TRUE`) on the standardized MSFI indicators, retaining components with an eigenvalue above 1 (Kaiser criterion). Each indicator $j$ is assigned to the retained component $k$ on which its loading is largest, and its weight follows equation (1):

$$W_j = |x_{jk}| \cdot W_k$$

   where $x_{jk}$ is the loading of indicator $j$ on its assigned component $k$, and $W_k$ is the eigenvalue of that component. Weights are then normalized to sum to 1.

4. **Indicator scoring** (`04_indicator_scoring.py`) cuts each indicator into 5 fertility classes (Table 4 in Sulaeman et al., 2021: thresholds for pH, CEC, exchangeable K, available P, total C, plus a texture-class score derived from the USDA texture code), scored from 1 (very low) to 5 (very high), then standardized as $S_{ij} \cdot p$ with $p = 1/n$ and $n = 5$ classes.
5. **SFI calculation** (`05_sfi_calculation.py`) aggregates the index per equation (2):

$$SFI_i = \sum_j W_j \cdot S_{ij} \cdot p$$

   using only observations for which all 6 MSFI indicators are available. Note that $p$ is already embedded in the standardized scores from step 4 and is not reapplied here.

6. **Classification** (`06_sfi_classification.py`) assigns each computed SFI to one of 5 fertility levels (Table 5 in Bagherzadeh et al., 2018): very low (0.25 or below), low (0.25 to 0.50), moderate (0.50 to 0.75), high (0.75 to 0.90), and very high (above 0.90).

The output (`outputs/sfi_summary.csv`, one row per profile and depth) is pivoted to a wide table, one column per depth interval, before being consumed by `scripts/01_data_preparation/` for covariate extraction.

### 2.1 Covariate extraction

Raster covariate values are sampled at each soil-profile location (point sampling, nearest pixel), reprojecting points to each raster's native CRS before sampling. Extracted values feed directly into feature selection and model training.

### 2.2 Multicollinearity screening (VIF)

Before model fitting, numeric predictors are screened for multicollinearity using the Variance Inflation Factor:

$$VIF_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination of predictor $j$ regressed on all remaining predictors. A stepwise elimination removes, at each iteration, the non-priority predictor with the highest VIF (threshold VIF of 10 or above), while a small set of ecologically important predictors (mean annual precipitation, elevation) is always retained regardless of VIF. Elimination stops once no non-priority predictor exceeds the threshold.

### 2.3 Random Forest regression (trend model)

For each depth, a Random Forest regressor is trained:

$$f(x) = \frac{1}{T} \sum_{t=1}^{T} f_t(x)$$

where each of the $T$ regression trees $f_t$ is grown on a bootstrap resample of the training data, with a random subset of predictors considered at each split. Hyperparameters (number of trees, max depth, minimum samples per split and per leaf, max features) are tuned with Optuna (Tree-structured Parzen Estimator sampler), minimizing inner-fold RMSE.

Nested cross-validation is used throughout to obtain unbiased performance estimates. The outer loop (5 folds) holds out folds used only for final evaluation. The inner loop (3 to 4 folds within each outer training fold) is used exclusively for the Optuna hyperparameter search, so the outer test fold never participates in tuning or in fitting imputation and encoding (medians, modes, and one-hot encoders are fit on the training fold only).

Evaluation metrics are computed on pooled out-of-fold predictions, with $x = y_{obs}$ and $y = y_{pred}$:

$$R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}$$

$$RMSE = \sqrt{\frac{1}{n} \sum (y_i - \hat y_i)^2}$$

$$MAE = \frac{1}{n} \sum |y_i - \hat y_i|$$

$$RPIQ = \frac{IQR(y)}{RMSE}$$

$$CCC = \frac{2 \, s_{xy}}{s_x^2 + s_y^2 + (\bar x - \bar y)^2}$$

where $IQR(y)$ is the interquartile range of the observed values (75th minus 25th percentile), $s_{xy}$ is the covariance of $x$ and $y$, $s_x^2$ and $s_y^2$ their variances, and CCC is Lin's concordance correlation coefficient.

### 2.4 Model interpretability (SHAP)

Feature importance and directionality are assessed with SHAP (SHapley Additive exPlanations) values, computed via `TreeExplainer`. For a prediction $f(x)$, the contribution of feature $i$ is the Shapley value:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! \, (|F| - |S| - 1)!}{|F|!} \Big[ f(S \cup \{i\}) - f(S) \Big]$$

summed over all subsets $S$ of the other features $F \setminus \{i\}$, so predictions decompose additively:

$$f(x) = \phi_0 + \sum_i \phi_i$$

### 2.5 Regression-kriging of spatial residuals

Random Forest captures non-linear covariate relationships but ignores spatial autocorrelation in the residuals. Regression-kriging corrects this by adding a kriged interpolation of the RF residuals to the RF trend:

$$\hat Z(s_0) = \hat m(s_0) + \hat e(s_0)$$

where $\hat m(s_0)$ is the RF trend prediction at location $s_0$, and $\hat e(s_0)$ is the Ordinary Kriging prediction of the trend's residuals $e(s) = y(s) - \hat m(s)$.

**Semivariogram.** Spatial dependence of the residuals is described by the empirical semivariogram:

$$\gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} \big[ e(s_i) - e(s_i + h) \big]^2$$

where $h$ is the separation distance and $N(h)$ the number of residual pairs at that lag. A parametric model is fit to the empirical semivariogram, with five candidate families searched and selected per depth by inner-CV kriging RMSE, along with the number of lag bins and whether closer lags are up-weighted in the least-squares fit:

$$\text{Linear: } \gamma(h) = \text{slope} \cdot h + \text{nugget}$$

$$\text{Power: } \gamma(h) = \text{scale} \cdot h^{\text{exponent}} + \text{nugget}$$

$$\text{Gaussian: } \gamma(h) = \text{psill} \left(1 - \exp\left(-\frac{h^2}{4r^2/7}\right)\right) + \text{nugget}$$

$$\text{Exponential: } \gamma(h) = \text{psill} \left(1 - \exp\left(-\frac{h}{r/3}\right)\right) + \text{nugget}$$

$$\text{Spherical: } \gamma(h) = \text{psill} \left(\frac{3h}{2r} - \frac{h^3}{2r^3}\right) + \text{nugget}, \quad h \le r$$

$$\gamma(h) = \text{psill} + \text{nugget}, \quad h > r$$

with $r$ the range, $\text{psill}$ the partial sill, and $\text{nugget}$ the nugget effect.

**Ordinary Kriging (local, moving window).** For a query location $s_0$, its $K$ nearest sampled residuals are used. The kriging weights $\lambda_1, \dots, \lambda_K$ and Lagrange multiplier $\mu$ solve the system that enforces unbiasedness, for every $i = 1, \dots, K$:

$$\sum_{l=1}^{K} \lambda_l \, \gamma(s_l, s_i) + \mu = \gamma(s_i, s_0)$$

together with the unbiasedness constraint:

$$\sum_{l=1}^{K} \lambda_l = 1$$

and the kriged residual estimate is:

$$\hat e(s_0) = \sum_{i=1}^{K} \lambda_i \cdot e(s_i)$$

We provide two numerically equivalent implementations of this local Ordinary Kriging step and cross-validate them against each other. The main implementation (`06_rk_mapping/02_...cpu_kriging.py`) is executed entirely through PyKrige's native `OrdinaryKriging.execute(backend='C')` on CPU, and is what we use throughout the analysis. An optional GPU version (`06_rk_mapping/01_...gpu_kriging.py`) is also provided for scalable prediction over large rasters: it fits the variogram on CPU via PyKrige (cheap, done once), then runs a per-pixel KNN search and batched linear solves on GPU (cuML for the KNN step, CuPy for the batched `linalg.solve`). The two are cross-checked against each other (`validate_gpu_kriging` and `validate_pykrige_cpu` both report the maximum discrepancy between CPU and GPU on a synthetic case).

### 2.6 Raster prediction

Raster prediction is run on CPU. The main deliverable is the point estimate (`06_rk_mapping/`, RF plus kriged residual), a single deterministic map per depth using the already-tuned final model and variogram, without resampling:

$$\hat Z(s_0) = \hat m(s_0) + \hat e(s_0)$$

For users who only want the RF trend on its own, without the kriging step, an optional bootstrap mean and standard deviation mode is also provided (`04_rf_mapping/`, RF-only trend): $B$ models are refit on bootstrap resamples of the training data, and the per-pixel mean and standard deviation across the ensemble give a point estimate and an uncertainty band:

$$mean(s_0) = \frac{1}{B} \sum_b f_b(s_0)$$

$$std(s_0) = \sqrt{\frac{1}{B} \sum_b \big(f_b(s_0) - mean(s_0)\big)^2}$$

For both modes, an optional GPU path is available for large raster stacks: the final RF trend can be compiled to Forest Inference Library (FIL) format via cuML for fast batched GPU inference across every valid pixel, tiled to bound memory use.

### 2.7 Regional comparison maps

Prediction rasters are cropped to each bioclimatic region using the AOI polygon layer, then classified into quantile (equal-count) bins rather than equal-width bins, for better visual contrast on skewed environmental data. They are rendered as multi-panel (depth by region) figures using a consistent fertility colour ramp, from red or orange for low fertility, through yellow for moderate, to green for high.

---

## 3. Reproducing the pipeline

```bash
# 1) Environment
conda env create -f environment.yml
conda activate sfi-madagascar
# (GPU steps are optional and additionally require a CUDA-enabled machine with
#  RAPIDS cuML/cuPy, see requirements-gpu.txt; the main pipeline runs entirely
#  on CPU, including PyKrige-based kriging.)

# 2) Run stages in order (adjust the hard-coded /kaggle/... paths at the top of
#    each script to your own data/output directories first)
python scripts/00_sfi_index_construction/01_data_preparation.py
python scripts/00_sfi_index_construction/02_pearson_correlation.py
python scripts/00_sfi_index_construction/03_pca_weights.py
python scripts/00_sfi_index_construction/04_indicator_scoring.py
python scripts/00_sfi_index_construction/05_sfi_calculation.py
python scripts/00_sfi_index_construction/06_sfi_classification.py
# then pivot outputs/sfi_final_summary.csv (long: one row per profile x depth)
# to a wide table, one column per depth interval, before continuing
python scripts/01_data_preparation/02_extract_covariates_at_points.py
python scripts/02_feature_selection/01_vif_multicollinearity_selection.py
python scripts/03_random_forest_modeling/02_nested_cv_execution_and_summary.py
python scripts/03_random_forest_modeling/04_final_model_training_and_shap.py
python scripts/05_regression_kriging_modeling/01_rk_optuna_tuning_and_nested_cv.py
python scripts/06_rk_mapping/02_point_estimate_prediction_cpu_kriging.py # main path (CPU, PyKrige)
python scripts/06_rk_mapping/03_regional_maps_point_estimate.py
python scripts/06_rk_mapping/01_point_estimate_prediction_gpu_kriging.py # optional GPU
# optional: RF trend only, without kriging, for mean/std uncertainty maps
python scripts/04_rf_mapping/01_bootstrap_mean_std_prediction_gpu.py     # optional, RF-only, GPU
python scripts/04_rf_mapping/02_regional_maps_mean_std.py
```

Scripts were originally authored and run in a Kaggle notebook environment (`/kaggle/working/...`, `/kaggle/input/...` paths), with GPU acceleration used only for the optional steps noted above. These paths sit in a single `CONFIGURATION` block at the top of every script and are the only things that need to change to run elsewhere.

---

## 4. Outputs

| Stage | Output |
|---|---|
| SFI: correlation and MSFI screening | `outputs/correlation_heatmap.png`, `outputs/pearson_*_matrix.csv`, `outputs/pearson_strong_pairs.csv` |
| SFI: PCA weighting | `outputs/pca_eigenvalues.csv`, `outputs/pca_weights_Wj.csv`, `outputs/pca_correlation_circle.png` |
| SFI: scoring, calculation, classification | `outputs/data_scored.csv`, `outputs/data_sfi.csv`, `outputs/sfi_summary.csv`, `outputs/data_sfi_classified.csv`, `outputs/sfi_final_summary.csv` |
| Feature selection | `VIF/vif_final.csv`, VIF barplot figure |
| RF nested CV | per-depth metrics table (R², RMSE, MAE, RPIQ, CCC), observed-vs-predicted scatter figure |
| RF final model | fitted RF plus medians, modes, and encoder bundle, one per depth |
| SHAP | per-depth SHAP summary CSVs, SHAP bar, beeswarm, and dependence figures |
| RK nested CV | RF-only vs RF+Kriging metrics comparison, tuned variogram parameters |
| RK final model | fitted RF plus tuned variogram and preprocessing bundle, one per depth |
| RK point-estimate maps | point-estimate rasters per depth (main deliverable, CPU/PyKrige; optional GPU variant) |
| RF bootstrap maps (optional) | mean and standard deviation rasters, one pair per depth, for RF-only use without kriging |
| Regional figures | multi-panel depth by region maps for the point-estimate and, where produced, the mean/std predictions |

---

## 5. Citation

If you use this code, please cite the associated manuscript (citation to be updated upon publication, see `CITATION.cff`):

> Rafidimanantsoa, S., Ramifehiarivo, N., Ratovoarimanana, L., Andriamananjara, A., Bouillon, S., Devenish, A., Razakamanarivo, H. *Comparative assessment of soil fertility across different land cover types in three bioclimatic regions of Madagascar*. Submitted to *Geoderma*.

## 6. License

Code released under the MIT License (see `LICENSE`). Soil and covariate data are not redistributed in this repository; contact the corresponding author for data access.

## 7. Acknowledgments

Laboratoire des Radioisotopes (University of Antananarivo), École Supérieure des Sciences Agronomiques, KU Leuven, Royal Botanic Gardens Edinburgh and Kew.
