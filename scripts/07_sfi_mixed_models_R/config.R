# =====================================================================================================
# config.R : shared settings (names, orders, colours, thresholds). EDIT HERE ONLY.
# Everything is in English: landscapes NW / CH / SE, land covers, depths in cm. Only the SFI is analysed.
# =====================================================================================================

# ---- Files ----------------------------------------------------------------------------------------
DATA_FILE      <- "DATA_LM_GPS.xlsx"    # workbook containing the sheet below
DATA_SHEET     <- "PPT_TOUT"            # 1 row = 1 sample (plot x depth)
OUT_DIR        <- "results"

# ---- Thresholds -----------------------------------------------------------------------------------
ALPHA <- 0.05

# Residual variance by depth: flagged in the diagnostics if (max SD / min SD across depths) > DEPTH_VAR_THRESHOLD
DEPTH_VAR_THRESHOLD <- 2

# Random structure: NULL = lowest AIC (REML); or "A","B","C","D","E" to impose it on all variables
#   A random intercept, common variance             B intercept + slope correlated, common variance
#   C random intercept, variance per site           D intercept + slope UNCORRELATED, variances per site
#   E intercept + slope CORRELATED, variances/covariance per site (9 variance-covariance parameters)
FORCED_STRUCTURE <- NULL

# Holm family used for the stars "land cover differs from grassland at a given depth" (fig_prof_04).
# The stars come from the all-pairs contrasts of each site x depth cell (model M5, 90 contrasts in total):
#   "p_holm_6_cell" (6 contrasts of a site x depth cell) | "p_holm_30_site" (30 contrasts of a site) | "p_holm_90" (all)
STAR_FAMILY <- "p_holm_30_site"

# ---- Factors: the FIRST level is the reference of contrasts -------------------------------------------
SITES        <- c("NW", "CH", "SE")      # NW = north-west (Ankarafantsika), CH = central highlands, SE = south-east
# Codes found in the data file -> site codes used here (AF -> NW, AT -> CH, IV -> SE)
SITE_CODE    <- c(AF = "NW", AT = "CH", IV = "SE", NW = "NW", CH = "CH", SE = "SE")
LANDCOVERS   <- c("Grassland", "Forest", "Shrubland", "Reforestation")
LC_ORDER     <- c("Forest", "Shrubland", "Grassland", "Reforestation")           # order in the figures
LC_REF       <- LANDCOVERS[1]
DEPTHS       <- c("0-10", "10-20", "20-30", "50-60", "80-90")
DEPTH_MID_CM <- c(5, 15, 25, 55, 85)     # MID-POINT of each layer (0-10, 10-20, 20-30, 50-60, 80-90 cm)
names(DEPTH_MID_CM) <- DEPTHS
# Depth coding for the RANDOM SLOPE (fixed effects keep depth as a factor):
#   "cm"   = layer mid-point (5, 15, 25, 55, 85), centred and scaled     <- main choice
#   "log"  = log of the layer mid-point, centred and scaled
#   "rank" = rank 1..5, centred and scaled (makes layers equidistant)
# 01_analysis.R compares the three codings by AIC (sheet 8_Fit_diagnostics); the main choice stays this one.
SLOPE_CODING <- "cm"

# Codes of the data file -> labels
LC_CODE    <- c("Forest" = "Forest", "Shrubland" = "Shrubland", "Grassland" = "Grassland", "Reforestation" = "Reforestation")
DEPTH_CODE <- setNames(DEPTHS, 1:5)

# ---- Labels and colours of the figures ------------------------------------------------------------------
SITE_LABEL <- c(NW = "North-West (NW)", CH = "Central Highlands (CH)", SE = "South-East (SE)")   # panel titles
SITE_ALIGN_RIGHT <- "CH"                 # site whose letters are drawn on the right / above of the points (fig_prof_02)
SITE_ALIGN_LEFT  <- "NW"                 # site whose letters are drawn on the left / below
COL_SITE   <- c(NW = "#A0522D", CH = "#2E8B3E", SE = "#1F3FD0")
COL_SITE_D <- c(NW = "#6B3410", CH = "#1B5E26", SE = "#14288F")
COL_LC     <- c("Forest" = "#2E8B57", "Shrubland" = "#98FB98", "Grassland" = "#DAA520", "Reforestation" = "#A0522D")
COL_LC_D   <- c("Forest" = "#1E5E3B", "Shrubland" = "#3F9F5F", "Grassland" = "#A9780F", "Reforestation" = "#6B3410")
COL_DEPTH  <- setNames(c("#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"), DEPTHS)
COL_DEPTH_D <- setNames(c("#5B7FA6", "#4A7BA8", "#3A6FA3", "#1F5A94", "#08306B"), DEPTHS)
SIZE_DEPTH <- setNames(c(1.6, 2.2, 2.8, 3.4, 4.0), DEPTHS)   # point size (mm) increasing with depth
LC_DODGE   <- 0.20

# ---- Variable analysed (SFI only) -----------------------------------------------------------------------------
# col: column name in the data sheet | scale: multiplicative factor | axis: y/x axis label of the figures
VARIABLES <- list(
  SFI = list(col = "SFI", name = "Soil fertility index (SFI)", axis = "Soil Fertility Index", unit = "", scale = 1)
)

# ---- Figures: drawn DIRECTLY at final print size (1 inch in the code = 1 printed inch) -----------------------
# Usual widths (Elsevier): 1 column 90 mm = 3.54 in; 1.5 columns 140 mm = 5.51 in; 2 columns 190 mm = 7.48 in.
# TO BE ADAPTED to the target journal. With this rule the sizes below are the REAL printed sizes.
WIDTH_1COL   <- 3.54
WIDTH_15COL  <- 5.51
WIDTH_2COL   <- 7.48
PT_TEXT      <- 7.5     # axes (ticks), legends, n= labels            (required minimum: 7 pt)
PT_TITLE     <- 8.5     # axis titles, strip text
PT_LETTER    <- 9       # group letters (bold)
PT_CAPTION   <- 6       # caption printed under the figure (discreet; use 7 to meet the Elsevier minimum)
COL_CAPTION  <- "#7f7f7f"
PT_STAR      <- 13      # asterisks
LW_BAR       <- 0.5     # thickness (ggplot unit, ~1.07 pt) of CI bars and whiskers
LW_BOX       <- 0.5     # box outline
LW_CURVE     <- 0.9     # lines joining the means (~1.9 pt)
CAP_CM       <- 2.5     # width of the CI end caps, axis in cm (~2.8 % of the 0-90 cm axis)
CAP_CLASS    <- 0.14    # same, axis in equidistant depth classes (~2.7 % of the axis)

