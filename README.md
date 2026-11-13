# Spatial prediction of soil fertility across tropical forest-grassland mosaics

**Soil Fertility Index (SFI) construction, Random Forest regression-kriging mapping and mixed-effects analysis in Madagascar**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Manuscript: submitted](https://img.shields.io/badge/manuscript-submitted%20to%20Geoderma%20Regional-orange.svg)](#6-citation)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-mixed%20models-4B8BBE.svg)](https://www.statsmodels.org/)
[![PyKrige](https://img.shields.io/badge/PyKrige-regression--kriging-2E8B57.svg)](https://geostat-framework.readthedocs.io/projects/pykrige/)
[![Study area: Madagascar](https://img.shields.io/badge/study%20area-Madagascar-009A44.svg)](#1-data)

Code and analysis pipeline for the manuscript:

> **Spatial prediction of soil fertility across tropical forest-grassland mosaics**
>
> Lovanirina Sarobidy Rafidimanantsoa<sup>1,2,5</sup>, Nandrianina Ramifehiarivo<sup>1,2,\*,+</sup>, Lovafitia Ratovoarimanana<sup>1</sup>, Andry Andriamananjara<sup>1</sup>, Adam Devenish<sup>3,4</sup>, Antsa Sarobidy Randrianantenaina<sup>1,2,6</sup>, Herintsitohaina Razakamanarivo<sup>1</sup>
>
> <sup>1</sup> Laboratoire des Radioisotopes (LRI), Université d'Antananarivo, Route d'Andraisoro, BP 3383, Antananarivo 101, Madagascar<br>
> <sup>2</sup> École Supérieure des Sciences Agronomiques (ESSA), Université d'Antananarivo, Route d'Ambohitsaina, BP 175, Antananarivo 101, Madagascar<br>
> <sup>3</sup> Royal Botanic Gardens Edinburgh, 20A Inverleith Row, Edinburgh EH3 5LR, United Kingdom<br>
> <sup>4</sup> Royal Botanic Gardens, Kew, Richmond, Surrey TW9 3AE, United Kingdom<br>
> <sup>5</sup> Université catholique de Louvain, 1 place de l'Université, 1348 Louvain-la-Neuve, Belgium<br>
> <sup>6</sup> Mention Biodiversité, Écologie et Évolution, Faculté des Sciences et Ingénierie, Sorbonne Université, 4 Place Jussieu, 75005 Paris, France
>
> <sup>+</sup> The first two authors contributed equally. <sup>\*</sup> Corresponding author: [ranandrianina@gmail.com](mailto:ranandrianina@gmail.com)
>
> *Submitted to Geoderma Regional*

---

## Repository description

This repository contains the full analysis pipeline we used to build, validate, and spatially predict a Soil Fertility Index (SFI) at five soil depth intervals across three bioclimatic regions of Madagascar (dry northwest, sub-humid central highlands, sub-humid southeast), and to compare fertility across land cover types (forest, shrubland, grassland, reforestation). We start from raw physical and chemical soil-profile measurements and construct the SFI itself (indicator selection, PCA-based weighting, scoring, and classification). We then model the index with Random Forest regression on remote sensing, terrain, and climate covariates, and add kriging of the spatial residuals (regression-kriging) to produce continuous raster maps of predicted soil fertility with associated uncertainty. Finally, we test how the SFI differs among landscapes, land cover types and soil depths with linear mixed-effects models that treat the plot as the sampling unit, so that the five depth increments of one profile are not counted as independent observations.

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
├── data/
│   ├── SFI_CALCULATION.csv            <- raw soil-profile measurements (input of stage 00)
│   └── DATA_LM.csv                    <- per-sample SFI + design factors (input of stage 07)
├── notebooks/
│   ├── sfi_index_construction.ipynb   <- end-to-end SFI construction notebook
│   ├── sfi_lmm_analysis.ipynb         <- mixed-model analysis notebook (same code as stage 07, script 01)
│   ├── sfi_lmm_figures.ipynb          <- mixed-model figures notebook (same code as stage 07, script 02)
│   └── tanety_full_pipeline.ipynb     <- original end-to-end Kaggle notebook (RF/RK mapping)
├── outputs/
│   ├── sfi_index/                     <- results of stage 00 (tables + PCA / correlation figures)
│   ├── lmm/                           <- results of stage 07 (tables/, figures/, logs/)
│   └── figures_publication/           <- stage 08: journal-ready figures (original + colour-blind palettes)
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
    ├── 06_rk_mapping/
    │   ├── 01_point_estimate_prediction_gpu_kriging.py
    │   ├── 02_point_estimate_prediction_cpu_kriging.py
    │   └── 03_regional_maps_point_estimate.py
    ├── 07_sfi_mixed_models/
    │   ├── 01_sfi_lmm_analysis.py
    │   └── 02_sfi_lmm_figures.py
    └── 08_publication_figures/
        └── 01_nature_style_figures.py
```

Each numbered folder is a stage of the pipeline, and the scripts within a folder are numbered in execution order. Stage `00_sfi_index_construction/` builds the SFI response itself from raw soil-profile measurements. Stages `01_...` through `06_...` then extract covariates at those profile locations, train the models, and produce the spatial predictions. Stage `07_sfi_mixed_models/` is the statistical comparison of the SFI among landscapes, land covers and depths; it only needs the SFI table (`data/DATA_LM.csv`) and can be run independently of the mapping stages 01 to 06. The notebooks hold the same code end to end and are kept for provenance; the `scripts/` folder splits it into standalone, documented modules. Stage 00 and stage 07 scripts use paths relative to the repository root, so run them from there; the notebooks use paths relative to `notebooks/`.

---

## 1. Data

Our response variable is the Soil Fertility Index computed at five depth intervals (0-10, 10-20, 20-30, 30-60, and 60-90 cm), derived from measured physical (texture) and chemical (pH, CEC, exchangeable K, available P, total C, total N) topsoil and subsoil properties: 1,523 soil samples from 307 plots (profiles) in three landscapes (NW: 534, CH: 555, SE: 434 samples). Eight plots have fewer than five sampled depths. The index is built by `scripts/00_sfi_index_construction/` (see Section 2.0) directly from the raw per-depth soil-profile measurements in `data/SFI_CALCULATION.csv` (semicolon-separated; `ID_LABO, Site, Depth, Landuse, PH, P, K, N, C, CEC, SABLE, ARGILE, LIMON, TEXTURE`). That stage outputs one SFI value per profile and depth in long format, which is then pivoted to a wide table with one column per depth interval, the format expected by `scripts/01_data_preparation/` onward.

Predictors are covariates automatically discovered from a directory of aligned 30 m rasters: elevation, slope, TPI, TWI, mean annual precipitation and temperature, net primary productivity, canopy height, tree cover, Sentinel-2 reflectance bands (B2-B12), derived spectral indices (NDVI, SAVI, GRVI, NBR, MNDWI, CIre, BSI, clay index, NDMI), and two categorical layers (ESA WorldCover land cover class, soil type).

`data/DATA_LM.csv` is the input of the mixed-model analysis (Section 2.8). It holds one row per soil sample with the same soil properties plus `Samples_ID` (plot identifier followed by the depth code, e.g. `Iv_4_Comp12_3` = plot `Iv_4_Comp12`, depth 3), `SFI` and `SFI_class`. Site codes are AF = North-West (NW), AT = Central Highlands (CH), IV = South-East (SE); depth codes 1 to 5 = 0-10, 10-20, 20-30, 30-60, 60-90 cm; land covers are Forest, Shrubland, Grassland and Reforestation.

Coordinates are given in UTM 38S (`lon`, `lat` columns, in metres rather than geographic degrees), used directly for Euclidean-distance kriging. The area of interest is a dissolved polygon layer with a `Region` attribute covering the three bioclimatic regions, used to crop and compare the prediction maps.

---

## 2. Methodology

### 2.0 Soil Fertility Index (SFI) construction

Before any covariate is touched, we build the SFI response itself from the raw soil-profile measurements (`scripts/00_sfi_index_construction/`), following Sulaeman et al. (2021) and Bagherzadeh et al. (2018).

1. **Data preparation** (`01_data_preparation.py`) parses the raw lab CSV, converts comma-decimal numeric fields, and derives the depth class, the depth in centimetres, and a texture code (USDA-style) from the measured texture class.
2. **Collinearity screening and MSFI selection** (`02_pearson_correlation.py`) computes the Pearson correlation matrix and associated p-values between candidate soil properties (N, P, C, CEC, pH, K, clay, silt, sand). Pairs with an absolute correlation above 0.75 flag redundant variables (in our data: N and C, r = 0.95; clay and sand, r = -0.94; silt and sand, r = -0.85). Combined with expert judgement, this narrows the set down to six Minimum Soil Fertility Indicators (MSFI): pH, CEC, clay (retained over sand and silt as the texture proxy), K, P, and C.
3. **PCA-based weighting** (`03_pca_weights.py`) runs a PCA (equivalent to `scale.unit = TRUE`) on the standardized MSFI indicators, retaining components with an eigenvalue above 1 (Kaiser criterion; two components, 59.8% of the variance in our data). Each indicator $j$ is assigned to the retained component $k$ on which its loading is largest, and its weight follows equation (6) of the manuscript:

$$W_j = |x_{jk}| \cdot W_k$$

   where $x_{jk}$ is the loading of indicator $j$ on its assigned component $k$, and $W_k$ is the eigenvalue of that component. Weights are then normalized to sum to 1.

4. **Indicator scoring** (`04_indicator_scoring.py`) cuts each indicator into fertility classes following Table 3 of the manuscript (after Sulaeman et al., 2021: thresholds for pH, CEC, exchangeable K, available P, total C, plus a texture-class score derived from the USDA texture code), scored from 1 (very low) to 5 (very high), then standardized as $S_{ij} \cdot p$ with $p = 1/n$ and $n = 5$ classes. CEC, K, P and C are scored on monotonic increasing scales. Soil pH is scored on a two-sided (non-monotonic) scale over nine intervals: the optimum range (above 6 to 7) scores 5, and scores decrease towards both acidic and alkaline conditions:

| pH | ≤ 5 | 5–5.4 | 5.4–5.8 | 5.8–6 | **6–7** | 7–7.4 | 7.4–7.8 | 7.8–8.2 | > 8.2 |
|---|---|---|---|---|---|---|---|---|---|
| Score | 1 | 2 | 3 | 4 | **5** | 4 | 3 | 2 | 1 |

   Intervals are right-closed, $(a, b]$. Small gaps between some classes in the published table (e.g. Ex-K 0.3–0.4 and 0.5–0.6) are absorbed into the adjacent higher class.
5. **SFI calculation** (`05_sfi_calculation.py`) aggregates the index per equation (7) of the manuscript:

$$SFI_i = \sum_j W_j \cdot S_{ij} \cdot p$$

   using only observations for which all 6 MSFI indicators are available. Note that $p$ is already embedded in the standardized scores from step 4 and is not reapplied here.

6. **Classification** (`06_sfi_classification.py`) assigns each computed SFI to one of 5 fertility levels (Table 4 of the manuscript, after Bagherzadeh et al., 2018): very low (0.25 or below), low (0.25 to 0.50), moderate (0.50 to 0.75), high (0.75 to 0.90), and very high (above 0.90).

The output (`outputs/sfi_index/sfi_final_summary.csv`, one row per sample) is pivoted to a wide table, one column per depth interval, before being consumed by `scripts/01_data_preparation/` for covariate extraction.

### 2.1 Covariate extraction

Raster covariate values are sampled at each soil-profile location (point sampling, nearest pixel), reprojecting points to each raster's native CRS before sampling. Extracted values feed directly into feature selection and model training.

### 2.2 Multicollinearity screening (VIF)

Before model fitting, numeric predictors are screened for multicollinearity using the Variance Inflation Factor:

$$VIF_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination of predictor $j$ regressed on all remaining predictors. A stepwise elimination removes, at each iteration, the predictor with the highest VIF (threshold VIF of 10 or above).

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

### 2.8 Linear mixed-effects analysis of landscape, land cover and depth effects

`scripts/07_sfi_mixed_models/` tests how the SFI varies among the three landscapes (NW, CH, SE), the four land cover types and the five depths, using `statsmodels` linear mixed models fitted to all 1,523 samples.

**Non-independence of depths.** The plot is the grouping unit. Each plot gets a random intercept and a random linear slope on the centred depth rank $d_c \in \{-2, \dots, 2\}$. The intercept and slope variances are estimated separately for each landscape (intercept and slope uncorrelated within a landscape); the residual variance is common. For the sample at depth $t$ in plot $p$ of landscape $s$:

$$SFI_{pt} = \mathbf{x}_{pt}^\top \boldsymbol\beta + u_{0p} + u_{1p} \, d_{c,t} + \varepsilon_{pt}, \quad u_{0p} \sim N(0, \sigma^2_{0,s}), \; u_{1p} \sim N(0, \sigma^2_{1,s}), \; \varepsilon_{pt} \sim N(0, \sigma^2)$$

This structure was selected by AIC among four random-effect structures fitted by maximum likelihood with the same fixed effects: (A) random intercept, homogeneous variance; (B) correlated random intercept and depth slope, homogeneous variance; (C) random intercept, landscape-specific variance; (D) random intercept and depth slope, landscape-specific variances (retained).

**Fixed effects.** The retained fixed-effect structure, specified a priori, is

$$SFI \sim \text{landscape} \times \text{land cover} + \text{landscape} \times \text{depth}$$

Six pre-specified candidate structures (M0: main effects only; M1: + landscape × land cover; M2: + landscape × depth; M3: both, retained; M4: M3 + land cover × depth; M5: full three-way interaction) are compared by AIC and BIC; there is no stepwise selection. Interactions are tested with likelihood-ratio tests on ML fits; land cover × depth and the three-way interaction are supplementary. The final model is refitted by REML.

**Contrasts and inference.** Estimated marginal means average model predictions with equal weights over the factors not shown, with 95% confidence intervals. Pairwise contrasts use t-based Wald tests with degrees of freedom equal to the number of plots minus the 12 between-plot fixed-effect parameters (intercept, 2 landscape, 3 land cover, 6 landscape × land cover). This is a conservative approximation, since statsmodels has no Kenward-Roger or Satterthwaite correction. p-values are Holm-adjusted within each family of comparisons, and compact letter displays are built from the Holm-adjusted p-values. Global tests of a factor within a panel are joint Wald F-tests. Comparisons that involve both land cover and depth use M4 (profiles by land cover) or M5 (land cover vs grassland at each depth within each landscape, Holm-adjusted across depths).

**Fit and assumptions.** Marginal and conditional $R^2$ follow Nakagawa & Schielzeth (2013) and Nakagawa et al. (2017):

$$R^2_m = \frac{\sigma^2_f}{\sigma^2_f + \sigma^2_r + \sigma^2}, \qquad R^2_c = \frac{\sigma^2_f + \sigma^2_r}{\sigma^2_f + \sigma^2_r + \sigma^2}$$

where $\sigma^2_f$ is the variance of the fixed-effect predictions, $\sigma^2_r$ the average random-effect variance $\sigma^2_{0,s} + d_c^2 \sigma^2_{1,s}$ over the samples, and $\sigma^2$ the residual variance. The intraclass correlation is taken from a null model with a random plot intercept. RMSE, RPD (SD/RMSE) and RPIQ (IQR/RMSE) are reported for the in-sample fitted values, with fixed effects only and with fixed plus random effects. Residual diagnostics (QQ-plots of standardised residuals and of plot random intercepts, residuals vs fitted and by group, Levene tests and residual SD by group) are saved as `fig_S_residual_diagnostics.png`.

**Figures.** `02_sfi_lmm_figures.py` reads letters, p-values, estimated means and CIs from the tables written by `01_sfi_lmm_analysis.py`; nothing is recomputed, so every figure matches the reported statistics. Comparisons without depth are boxplots (median and IQR, whiskers at 1.5 × IQR, + = mean, points = samples); every comparison involving depth is a depth profile of estimated means ± 95% CI.

### 2.9 Publication figures

`scripts/08_publication_figures/01_nature_style_figures.py` redraws every figure of stages 00 and 07 for publication. It reads the
tables those stages write and re-estimates nothing, apart from the residuals of the final mixed model, which it caches on first use.

* **Format:** compact figures (46–170 mm wide, within Nature's 183 mm page width) with large type for their size: Arial at
  8 pt (ticks) and 9 pt (axis labels), and bold lowercase panel letters (12 pt) at the top left of every panel. Boxplots use narrow
  boxes with a white mean circle, sample sizes above each box, letters above the whiskers and light horizontal gridlines.
  Axes are 0.6 pt, and there are no titles or captions inside the figures.
  Output is 600 dpi PNG plus vector PDF with embedded TrueType fonts.
* **Palettes:** `original` (the colours of the analysis figures) and `colorblind` (Okabe–Ito for land cover, Paul Tol *bright* for
  landscapes, viridis for depth, PuOr for correlations).
* **Sets:** `individual/` has one file per figure; `combined/` has four multi-panel main figures (SFI construction; landscape and
  land cover; depth; land cover view) plus the model diagnostics as a supplementary figure.
* `figure_legends.md` has draft figure legends that include the statistics, and `figure_index.md` lists every file with its size.

Arial is proprietary and is not distributed here. Point `--font-dir` at a folder containing `ARIAL.TTF`, `ARIALBD.TTF`, `ARIALI.TTF`
and `ARIALBI.TTF`; without it the script falls back to Liberation Sans, which has the same metrics. Use `--no-single-letters` to drop
the panel letter from single-panel figures.

---

## 3. Reproducing the pipeline

```bash
# 1) Environment
conda env create -f environment.yml
conda activate sfi-madagascar
# (GPU steps are optional and additionally require a CUDA-enabled machine with
#  RAPIDS cuML/cuPy, see requirements-gpu.txt; the main pipeline runs entirely
#  on CPU, including PyKrige-based kriging.)

# 2) Run stages in order from the repository root.
#    Stages 00 and 07 run as-is on the files in data/.
#    Stages 01-06 need the covariate rasters and AOI (not distributed): adjust the
#    hard-coded /kaggle/... paths at the top of each of those scripts first.
python scripts/00_sfi_index_construction/01_data_preparation.py
python scripts/00_sfi_index_construction/02_pearson_correlation.py
python scripts/00_sfi_index_construction/03_pca_weights.py
python scripts/00_sfi_index_construction/04_indicator_scoring.py
python scripts/00_sfi_index_construction/05_sfi_calculation.py
python scripts/00_sfi_index_construction/06_sfi_classification.py
# then pivot outputs/sfi_index/sfi_final_summary.csv (long: one row per profile x depth)
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

# 3) Mixed-model analysis of landscape, land cover and depth effects (independent of 01-06)
python scripts/07_sfi_mixed_models/01_sfi_lmm_analysis.py   # stats + Excel report + figure tables (~4-5 min)
python scripts/07_sfi_mixed_models/02_sfi_lmm_figures.py    # figures (PNG 300 dpi + PDF)
# both accept --data <csv> and --out <dir> (defaults: data/DATA_LM.csv, outputs/lmm)

# 4) Journal-ready figures (Nature style, 600 dpi PNG + vector PDF, two palettes)
python scripts/08_publication_figures/01_nature_style_figures.py --font-dir /path/to/arial/fonts
```

Stages 01 to 06 were originally authored and run in a Kaggle notebook environment (`/kaggle/working/...`, `/kaggle/input/...` paths), with GPU acceleration used only for the optional steps noted above. These paths sit in a single `CONFIGURATION` block at the top of each script and are the only things that need to change to run elsewhere. Exact package versions used for stage 07 are written to `outputs/lmm/logs/environment.txt`.

---

## 4. Outputs

| Stage | Output |
|---|---|
| SFI: data preparation | `outputs/sfi_index/data_prepared.csv` |
| SFI: correlation and MSFI screening | `outputs/sfi_index/correlation_heatmap.png`, `outputs/sfi_index/pearson_*_matrix.csv`, `outputs/sfi_index/pearson_significant_pairs.csv`, `outputs/sfi_index/pearson_strong_pairs.csv` |
| SFI: PCA weighting | `outputs/sfi_index/pca_eigenvalues.csv`, `pca_variable_coordinates.csv`, `pca_variable_contributions.csv`, `pca_weights_Wj.csv`, `pca_correlation_circle.png` (the notebook additionally writes `pca_sfi_correlation_circle.png` and `pca_summary_table.csv/.png`) |
| SFI: scoring, calculation, classification | `outputs/sfi_index/data_scored.csv`, `data_sfi.csv`, `sfi_summary.csv`, `data_sfi_classified.csv`, `sfi_final_summary.csv` |
| Feature selection | `VIF/vif_final.csv`, VIF barplot figure |
| RF nested CV | per-depth metrics table (R², RMSE, MAE, RPIQ, CCC), observed-vs-predicted scatter figure |
| RF final model | fitted RF plus medians, modes, and encoder bundle, one per depth |
| SHAP | per-depth SHAP summary CSVs, SHAP bar, beeswarm, and dependence figures |
| RK nested CV | RF-only vs RF+Kriging metrics comparison, tuned variogram parameters |
| RK final model | fitted RF plus tuned variogram and preprocessing bundle, one per depth |
| RK point-estimate maps | point-estimate rasters per depth (main deliverable, CPU/PyKrige; optional GPU variant) |
| RF bootstrap maps (optional) | mean and standard deviation rasters, one pair per depth, for RF-only use without kriging |
| Regional figures | multi-panel depth by region maps for the point-estimate and, where produced, the mean/std predictions |
| Mixed models: report | `outputs/lmm/tables/SFI_results.xlsx`, one sheet per analysis: `1_Design`, `2_Model_selection`, `3_Interactions`, `4_Land_cover`, `5_Depth`, `6_Landscapes`, `7_Suppl_LC_by_depth`, `8_Model_fit`, `9_Letters` (raw and Holm-adjusted p-values) |
| Mixed models: figure tables | `outputs/lmm/tables/fig_all_annotations.csv`, `fig_depth_annotations.csv`, `fig_depth_site_annotations.csv`, `fig_lc_depth_site_profiles.csv` |
| Mixed models: figures | `outputs/lmm/figures/`: `fig_box_01`–`04` (boxplots by landscape, land cover, land cover within landscape, landscape within land cover), `fig_prof_01`–`04` (depth profiles: pooled, by landscape, by land cover, land cover within landscape; vertical and horizontal layouts), `fig_S_residual_diagnostics` |
| Publication figures | `outputs/figures_publication/{original,colorblind}/{individual,combined}/*.png|pdf`, `figure_legends.md`, `figure_index.md` |
| Mixed models: logs | `outputs/lmm/logs/analysis_log.txt`, `outputs/lmm/logs/environment.txt` |

---

## 5. References

References cited in the mixed-model code (the SFI method references, Sulaeman et al. 2021 and Bagherzadeh et al. 2018, are given in full in the manuscript).

- Gelman, A., Stern, H. (2006). The difference between "significant" and "not significant" is not itself statistically significant. *The American Statistician*, 60, 328–331.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6, 65–70.
- Nakagawa, S., Schielzeth, H. (2013). A general and simple method for obtaining R² from generalized linear mixed-effects models. *Methods in Ecology and Evolution*, 4, 133–142.
- Nakagawa, S., Johnson, P. C. D., Schielzeth, H. (2017). The coefficient of determination R² and intra-class correlation coefficient from generalized linear mixed-effects models revisited and expanded. *Journal of the Royal Society Interface*, 14, 20170213.
- Seabold, S., Perktold, J. (2010). Statsmodels: econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*.

## 6. Citation

If you use this code, please cite the associated manuscript (citation to be updated upon publication, see `CITATION.cff`):

> Rafidimanantsoa, L. S., Ramifehiarivo, N., Ratovoarimanana, L., Andriamananjara, A., Devenish, A., Randrianantenaina, A. S., Razakamanarivo, H. *Spatial prediction of soil fertility across tropical forest-grassland mosaics*. Submitted to *Geoderma Regional*.

## 7. License

Code released under the MIT License (see `LICENSE`). The soil-profile tables needed to rebuild the SFI and rerun the mixed-model analysis are provided in `data/`. Covariate rasters and sampling coordinates are not redistributed in this repository; contact the corresponding author for access.

## 8. Acknowledgments

Laboratoire des Radioisotopes (LRI) and École Supérieure des Sciences Agronomiques (ESSA), Université d'Antananarivo; KU Leuven; Royal Botanic Gardens Edinburgh; Royal Botanic Gardens, Kew; Université catholique de Louvain; Sorbonne Université.
