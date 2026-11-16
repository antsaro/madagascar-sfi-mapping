# =====================================================================================================
# 01_analysis.R : linear mixed models (LMM) for the soil fertility index (SFI)
#
# Models: random structure chosen by AIC (REML) among A-E; fixed effects M0-M5 (M3 = main model).
# Overall effects: Satterthwaite F-tests (lmerTest / emmeans); interactions: likelihood-ratio tests (chi2, ML).
# Estimated marginal means (equal weights, 95% CI); pairwise comparisons with Holm correction within each family.
#
# ORDER OF THE TESTS (same order as the Methods and as the sheets of the Excel workbook)
#   LAND COVER  3.1 all landscapes and depths: estimated means, joint F (3 df), 6 pairwise comparisons      (M3)
#               3.2 consistency across landscapes: landscape x land cover interaction (LRT, M3 vs M2)
#               3.3 within each landscape: joint F, 6 comparisons                                            (M3)
#               3.4 at each depth: joint F, 6 comparisons                                                    (M4)
#               3.5 landscapes compared within each land cover (3 pairs)                                     (M3)
#   DEPTH       4.1 all landscapes and land covers: joint F (4 df), 10 pairwise comparisons                  (M3)
#               4.2 landscape x depth interaction (LRT, M3 vs M1; confirmed by Satterthwaite F of M3)
#               4.3 within each landscape: joint F, 10 comparisons                                           (M3)
#               4.4 within each land cover: joint F, 10 comparisons                                          (M4)
#               4.5 landscapes compared at each depth (3 pairs)                                              (M3)
#   COMPLEMENTARY  5. land cover x depth (LRT, M4 vs M3) and triple interaction (LRT, M5 vs M4)
#   SUPPLEMENTARY  6. landscapes compared overall (used by fig_box_01) | 7. M5 contrasts (stars of fig_prof_04)
#   FIT            8. R2, ICC, variance components, coefficients, residual diagnostics, convergence, depth coding
#                  9. singularity check
# Outputs: results/SFI/SFI_LMM_results.xlsx, results/SFI/tables/*.csv, results/SFI/figures/
#
# Usage (from the scripts folder):   Rscript 01_analysis.R
# Packages: readxl dplyr lme4 lmerTest emmeans openxlsx
# =====================================================================================================
suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(lme4); library(lmerTest); library(emmeans); library(openxlsx)
})
source("config.R")
emm_options(lmer.df = "satterthwaite", lmerTest.limit = 20000, pbkrtest.limit = 20000)
set.seed(20260611)

LEVELS <- list(site = SITES, lc = LANDCOVERS, depth = DEPTHS)
FX  <- c(M0 = "y ~ site + lc + depth",
         M1 = "y ~ site*lc + depth",
         M2 = "y ~ site + lc + site*depth",
         M3 = "y ~ site*lc + site*depth",                 # main model
         M4 = "y ~ site*lc + site*depth + lc:depth",
         M5 = "y ~ site*lc*depth")
STRUCT_LABEL <- c(A = "A. Random intercept, common variance",
                  B = "B. Intercept + depth slope (correlated), common variance",
                  C = "C. Random intercept, variance per landscape",
                  D = "D. Intercept + slope uncorrelated, variances per landscape",
                  E = "E. Intercept + slope correlated, variances/covariance per landscape")

# ---- Small statistical helpers ----------------------------------------------------------------------
skewness <- function(x) { x <- x[!is.na(x)]; m <- mean(x); mean((x - m)^3) / mean((x - m)^2)^1.5 }
excess_kurtosis <- function(x) { x <- x[!is.na(x)]; m <- mean(x); mean((x - m)^4) / mean((x - m)^2)^2 - 3 }
round_df <- function(df, nd = 6) { for (j in names(df)) if (is.numeric(df[[j]])) df[[j]] <- signif(df[[j]], nd); df }

# ---- 1. Data ------------------------------------------------------------------------------------------
# Depth coding for the RANDOM SLOPE only (fixed effects keep depth as a factor): layer mid-point (DEPTH_MID_CM),
# its log, or the rank, always centred and scaled over the 5 levels.
add_slope <- function(d, type = SLOPE_CODING) {
  v5 <- switch(type, cm = DEPTH_MID_CM, log = log(DEPTH_MID_CM), rank = seq_along(DEPTH_MID_CM),
               stop("SLOPE_CODING must be cm, log or rank"))
  v5 <- setNames(as.numeric(v5), DEPTHS)
  d$dc <- unname((v5[as.character(d$depth)] - mean(v5)) / sd(v5))
  for (s in SITES) d[[paste0("s", s)]] <- d[[paste0("i", s)]] * d$dc
  d
}

load_data <- function() {
  d <- as.data.frame(read_excel(DATA_FILE, sheet = DATA_SHEET))
  d$plot_id <- sub("_[0-9]+$", "", d$Samples_ID)
  d$site  <- factor(unname(SITE_CODE[toupper(d$Site)]), levels = SITES)
  d$lc    <- factor(unname(LC_CODE[as.character(d$Landuse)]), levels = LANDCOVERS)
  d$depth <- factor(unname(DEPTH_CODE[as.character(d$Depth)]), levels = DEPTHS)
  stopifnot(!anyNA(d$site), !anyNA(d$lc), !anyNA(d$depth), anyDuplicated(d$Samples_ID) == 0)
  chk <- d %>% group_by(plot_id) %>% summarise(ns = n_distinct(site), nl = n_distinct(lc), .groups = "drop")
  stopifnot(all(chk$ns == 1), all(chk$nl == 1))
  for (s in SITES) d[[paste0("i", s)]] <- as.numeric(d$site == s)   # landscape indicators: variance components specific to each landscape
  d <- add_slope(d)                                       # dc (centred-scaled depth) and slopes by landscape sNW, sCH, sSE
  d$plot_id <- factor(d$plot_id)
  list(d = d)
}

# ---- 2. Models ------------------------------------------------------------------------------------------
random_formula <- function(struct) {
  i <- paste0("i", SITES); s <- paste0("s", SITES)
  switch(struct,
         A = "(1 | plot_id)",
         B = "(1 + dc | plot_id)",
         C = paste0("(0 + ", i, " | plot_id)", collapse = " + "),
         D = paste0("(0 + ", c(i, s), " | plot_id)", collapse = " + "),
         E = paste0("(0 + ", i, " + ", s, " | plot_id)", collapse = " + "))
}

fit_lmm <- function(d, struct, reml = TRUE, fixed = FX[["M3"]]) {
  f <- as.formula(paste(fixed, "+", random_formula(struct)))
  warn <- character()
  m <- withCallingHandlers(
    lmer(f, data = d, REML = reml,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                               check.nobs.vs.nRE = "ignore")),   # landscape indicators: many structurally null random effects
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(msg) invokeRestart("muffleMessage"))
  list(m = m, warn = warn, singular = isSingular(m))
}

ll   <- function(m) as.numeric(logLik(m))
npar <- function(m) attr(logLik(m), "df")

# random variance attributed to each observation: z' Sigma z (for Nakagawa R2)
random_var_obs <- function(m, d) {
  vc <- VarCorr(m); tot <- rep(0, nrow(d))
  for (nm in names(vc)) {
    M <- as.matrix(vc[[nm]]); v <- rownames(M)
    Z <- matrix(sapply(v, function(x) if (x == "(Intercept)") rep(1, nrow(d)) else d[[x]]), nrow = nrow(d))
    tot <- tot + rowSums((Z %*% M) * Z)
  }
  tot
}

r2_nakagawa <- function(m, d) {
  eta <- as.vector(getME(m, "X") %*% fixef(m))
  vf <- mean((eta - mean(eta))^2); vr <- mean(random_var_obs(m, d)); ve <- sigma(m)^2
  c(var_fixed = vf, var_random = vr, var_resid = ve, R2_marginal = vf / (vf + vr + ve), R2_conditional = (vf + vr) / (vf + vr + ve))
}

# parameters estimated ON THE BOUNDARY: correlation = +/-1 or variance ~ 0 (cause of a singular fit)
boundary_terms <- function(m) {
  vc <- as.data.frame(VarCorr(m)); vc <- vc[vc$grp != "Residual", ]
  site_of <- function(v) ifelse(grepl("^[is](NW|CH|SE)$", v), sub("^[is]", "", v), "all")
  isvar <- is.na(vc$var2)
  b <- (isvar & vc$sdcor < 1e-4) | (!isvar & abs(vc$sdcor) > 0.99)
  if (!any(b)) return(data.frame(landscape = character(), term = character(), type = character()))
  data.frame(landscape = site_of(vc$var1[b]), term = ifelse(isvar[b], vc$var1[b], paste(vc$var1[b], "x", vc$var2[b])),
             type = ifelse(isvar[b], "variance ~ 0", "correlation = +/-1"), stringsAsFactors = FALSE)
}

# ---- 3. Estimated marginal means, contrasts, F tests ----------------------------------------------------
emm_f <- function(m, form) emmeans(m, form)

df_emm <- function(e) {
  s <- as.data.frame(summary(e))
  names(s)[names(s) == "emmean"] <- "emm"
  names(s)[names(s) == "lower.CL"] <- "ci_low"; names(s)[names(s) == "upper.CL"] <- "ci_high"
  s
}

# pairwise contrasts (combn order of levels), unadjusted CIs, raw p and Holm p within each group
pairs_tab <- function(e, levels_, by = NULL) {
  ct <- pairs(e)
  s1 <- as.data.frame(summary(ct, infer = c(TRUE, TRUE), adjust = "none"))
  s2 <- as.data.frame(summary(ct, adjust = "holm"))
  cmb <- combn(levels_, 2); ng <- nrow(s1) / ncol(cmb)
  grp <- if (is.null(by)) rep("All", nrow(s1)) else as.character(s1[[by]])
  data.frame(group = grp, a = rep(cmb[1, ], ng), b = rep(cmb[2, ], ng),
             contrast = paste(rep(cmb[1, ], ng), "-", rep(cmb[2, ], ng)),
             estimate = s1$estimate, SE = s1$SE, df = s1$df, ci_low = s1$lower.CL, ci_high = s1$upper.CL,
             p = s1$p.value, p_holm = s2$p.value, stringsAsFactors = FALSE)
}

# joint F test (Satterthwaite df) of consecutive contrasts, within each "by" group
joint_F <- function(m, e) {
  ct <- contrast(e, "consec"); L <- ct@linfct; by <- ct@misc$by.vars
  grp <- if (is.null(by)) rep("All", nrow(L)) else as.character(ct@grid[[by[1]]])
  do.call(rbind, lapply(unique(grp), function(k) {
    r <- contestMD(m, L[grp == k, , drop = FALSE])
    data.frame(group = k, F = r[["F value"]], df1 = r[["NumDF"]], df2 = r[["DenDF"]], p = r[["Pr(>F)"]])
  }))
}

# compact letters (clique algorithm, ported from the compact_letters function of the Python script)
compact_letters <- function(levels_, means, P, alpha = ALPHA) {
  ord <- levels_[order(-means[levels_])]
  differ <- function(a, b) isTRUE(P[a, b] < alpha)
  cliques <- list()
  for (r in length(ord):1) {
    for (cc in combn(ord, r, simplify = FALSE)) {
      ok <- TRUE
      if (length(cc) > 1) for (pr in combn(cc, 2, simplify = FALSE)) if (differ(pr[1], pr[2])) { ok <- FALSE; break }
      already <- length(cliques) > 0 && any(vapply(cliques, function(o) all(cc %in% o), logical(1)))
      if (ok && !already) cliques[[length(cliques) + 1]] <- cc
    }
  }
  links <- function(cs) {
    out <- character()
    for (cc in cs) {
      out <- c(out, paste0(cc, "|", cc))
      if (length(cc) > 1) for (pr in combn(cc, 2, simplify = FALSE)) out <- c(out, paste(sort(pr), collapse = "|"))
    }
    unique(out)
  }
  keep <- rep(TRUE, length(cliques)); ids <- seq_along(cliques)
  for (i in ids[order(lengths(cliques))]) {
    rest <- cliques[keep & ids != i]
    if (all(links(list(cliques[[i]])) %in% links(rest))) keep[i] <- FALSE
  }
  cliques <- cliques[keep]
  keys <- vapply(cliques, function(cc) paste(sprintf("%03d", match(cc, ord)), collapse = ""), "")
  cliques <- cliques[order(keys)]
  out <- setNames(rep("", length(levels_)), levels_)
  for (i in seq_along(cliques)) for (lv in cliques[[i]]) out[lv] <- paste0(out[lv], letters[i])
  out
}

p_matrix <- function(levels_, pp) {
  P <- matrix(1, length(levels_), length(levels_), dimnames = list(levels_, levels_))
  for (r in seq_len(nrow(pp))) { P[pp$a[r], pp$b[r]] <- pp$p_holm[r]; P[pp$b[r], pp$a[r]] <- pp$p_holm[r] }
  P
}

# panels of the figures / tables: (id, FACET, X variable, model). CAUTION: the id is only a label read by 02_figures.R.
#   landcover_by_depth = depth compared WITHIN each land cover (facet lc, x depth, M4)
#   depth_by_landcover = land covers compared AT each depth (facet depth, x lc, M4)
SPECS <- list(
  list("global_landcover", NULL, "lc", "M3"), list("site_by_landcover", "site", "lc", "M3"), list("depth_by_landcover", "depth", "lc", "M4"),
  list("landcover_by_site", "lc", "site", "M3"),
  list("global_depth", NULL, "depth", "M3"), list("depth_by_site", "site", "depth", "M3"), list("landcover_by_depth", "lc", "depth", "M4"),
  list("site_by_depth", "depth", "site", "M3"),
  list("global_site", NULL, "site", "M3"))

annotate_specs <- function(mods, d) {
  do.call(rbind, lapply(SPECS, function(sp) {
    id <- sp[[1]]; facet <- sp[[2]]; x <- sp[[3]]; m <- mods[[sp[[4]]]]
    form <- if (is.null(facet)) reformulate(x) else as.formula(paste("~", x, "|", facet))
    e <- emm_f(m, form); est <- df_emm(e); pw <- pairs_tab(e, LEVELS[[x]], by = facet); jf <- joint_F(m, e)
    panels <- if (is.null(facet)) "All" else LEVELS[[facet]]
    do.call(rbind, lapply(panels, function(pn) {
      es <- if (is.null(facet)) est else est[as.character(est[[facet]]) == pn, ]
      pp <- pw[pw$group == pn, ]; jj <- jf[jf$group == pn, ]
      mu <- setNames(es$emm, as.character(es[[x]]))
      let <- compact_letters(LEVELS[[x]], mu, p_matrix(LEVELS[[x]], pp))
      do.call(rbind, lapply(LEVELS[[x]], function(lv) {
        msk <- d[[x]] == lv
        if (!is.null(facet)) msk <- msk & d[[facet]] == pn
        r <- es[as.character(es[[x]]) == lv, ]
        data.frame(figure = id, model = sp[[4]], facet = if (is.null(facet)) "" else facet, x = x, panel = pn,
                   level_x = lv, letter = unname(let[lv]), n_samples = sum(msk), n_plots = n_distinct(d$plot_id[msk]),
                   emm = r$emm, ci_low = r$ci_low, ci_high = r$ci_high,
                   F = jj$F, df1 = jj$df1, df2 = jj$df2, p_panel = jj$p, stringsAsFactors = FALSE)
      }))
    }))
  }))
}

# Families of pairwise comparisons: (label, formula, x variable, by variable, model, key). Holm WITHIN each group of a family.
FAMILIES <- list(
  list("Land cover (all landscapes, all depths)",     ~lc,           "lc",    NULL,    "M3", "lc_all"),
  list("Land cover within landscape",                 ~lc | site,    "lc",    "site",  "M3", "lc_in_site"),
  list("Land cover at depth",                         ~lc | depth,   "lc",    "depth", "M4", "lc_at_depth"),
  list("Landscape within land cover",                 ~site | lc,    "site",  "lc",    "M3", "site_in_lc"),
  list("Depth (all landscapes, all land covers)",     ~depth,        "depth", NULL,    "M3", "depth_all"),
  list("Depth within landscape",                      ~depth | site, "depth", "site",  "M3", "depth_in_site"),
  list("Depth within land cover",                     ~depth | lc,   "depth", "lc",    "M4", "depth_in_lc"),
  list("Landscape at depth",                          ~site | depth, "site",  "depth", "M3", "site_at_depth"),
  list("Landscape (all land covers, all depths)",     ~site,         "site",  NULL,    "M3", "site_all"))

all_pairs <- function(mods) {
  do.call(rbind, lapply(FAMILIES, function(fm) {
    p <- pairs_tab(emm_f(mods[[fm[[5]]]], fm[[2]]), LEVELS[[fm[[3]]]], by = fm[[4]])
    p$family <- if (is.null(fm[[4]])) fm[[1]] else paste(fm[[1]], p$group)
    p$key <- fm[[6]]; p$model <- fm[[5]]
    p
  }))
}

# all pairwise land-cover contrasts in each landscape x depth cell (model with the triple interaction M5): 6 x 15 = 90 contrasts
triple_pairs <- function(m5) {
  e <- emm_f(m5, ~ lc | site * depth)
  s <- as.data.frame(summary(pairs(e), infer = c(TRUE, TRUE), adjust = "none"))
  lab <- gsub("[()]", "", as.character(s$contrast))
  parts <- strsplit(lab, " [-/] ")
  t <- data.frame(site = as.character(s$site), depth = as.character(s$depth),
                  a = vapply(parts, `[`, "", 1), b = vapply(parts, `[`, "", 2), contrast = lab,
                  estimate = s$estimate, SE = s$SE, df = s$df, ci_low = s$lower.CL, ci_high = s$upper.CL, p = s$p.value,
                  stringsAsFactors = FALSE)
  t <- t %>% group_by(site, depth) %>% mutate(p_holm_6_cell = p.adjust(p, "holm")) %>% ungroup() %>%
    group_by(site) %>% mutate(p_holm_30_site = p.adjust(p, "holm")) %>% ungroup() %>% mutate(p_holm_90 = p.adjust(p, "holm"))
  list(contrasts = as.data.frame(t), emm = df_emm(emm_f(m5, ~ site * lc * depth)))
}

# ---- 4. Diagnostics -------------------------------------------------------------------------------------
diagnostics <- function(m, d, png_file) {
  r <- resid(m); rs <- r / sigma(m); f <- fitted(m)
  png(png_file, width = 4500, height = 2400, res = 300); par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
  qqnorm(rs, pch = 16, cex = 0.3, main = "Q-Q plot of standardised residuals", xlab = "Theoretical quantiles", ylab = "Observed quantiles"); qqline(rs, col = "red")
  plot(f, rs, pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.35), xlab = "Fitted values", ylab = "Standardised residuals", main = "Residuals vs fitted values"); abline(h = 0, col = "red")
  boxplot(rs ~ d$depth, main = "Residuals by depth", xlab = "", ylab = ""); boxplot(rs ~ d$site, main = "Residuals by landscape", xlab = "", ylab = "")
  boxplot(rs ~ d$lc, main = "Residuals by land cover", xlab = "", ylab = "")
  re <- ranef(m)$plot_id; site_of <- sapply(split(as.character(d$site), as.character(d$plot_id)), function(z) z[1])
  col <- vapply(rownames(re), function(p) if ("(Intercept)" %in% names(re)) "(Intercept)" else paste0("i", unname(site_of[p])), "")
  ri <- re[cbind(seq_len(nrow(re)), match(col, names(re)))]; ri <- ave(ri, unname(site_of[rownames(re)]), FUN = function(z) z / sd(z))
  qqnorm(ri, pch = 16, cex = 0.4, main = "Q-Q plot of random intercepts (plots)", xlab = "Theoretical quantiles", ylab = "Observed quantiles"); qqline(ri, col = "red")
  dev.off()
  sd_by <- function(v) tapply(r, v, sd)
  sdp <- sd_by(d$depth)
  rows <- data.frame(indicator = c("Residual skewness", "Residual excess kurtosis", "Max/min residual SD ratio across depths",
                                   paste0("Residual SD, landscape ", names(sd_by(d$site))), paste0("Residual SD, depth ", names(sdp)),
                                   paste0("Residual SD, land cover ", names(sd_by(d$lc)))),
                     value = c(skewness(r), excess_kurtosis(r), max(sdp) / min(sdp), sd_by(d$site), sdp, sd_by(d$lc)))
  list(tab = rows, depth_ratio = max(sdp) / min(sdp), skew = skewness(r), kurt = excess_kurtosis(r))
}

# joint Wald/F tests (emmeans::joint_tests): interactions (used by the singularity check)
WALD_NAMES <- c("p_landscape_x_landcover", "p_landscape_x_depth")
wald_na <- function() as.data.frame(as.list(setNames(rep(NA_real_, length(WALD_NAMES)), WALD_NAMES)))
wald_tests <- function(m) {
  tryCatch({
    jt <- as.data.frame(emmeans::joint_tests(m))
    g <- function(term) { k <- jt[["model term"]] == term; if (any(k)) jt$p.value[k][1] else NA_real_ }
    as.data.frame(as.list(setNames(c(g("site:lc"), g("site:depth")), WALD_NAMES)))
  }, error = function(e) wald_na())
}

# Sensitivity of the depth coding in the random slope (retained structure kept; no effect for structures A and C)
sensitivity_slope_coding <- function(d, retained) {
  lab <- c(cm = "Layer mid-point (cm), centred-scaled", log = "log(layer mid-point), centred-scaled", rank = "Rank 1 to 5, centred-scaled")
  out <- do.call(rbind, lapply(names(lab), function(tp) tryCatch({
    f <- fit_lmm(add_slope(d, tp), retained, TRUE)
    data.frame(slope_coding = lab[[tp]], is_main_coding = (tp == SLOPE_CODING), structure = retained,
               AIC_REML = AIC(f$m), singular = f$singular, message = "", stringsAsFactors = FALSE)
  }, error = function(e) data.frame(slope_coding = lab[[tp]], is_main_coding = (tp == SLOPE_CODING), structure = retained,
                                    AIC_REML = NA_real_, singular = NA, message = conditionMessage(e), stringsAsFactors = FALSE))))
  out$delta_AIC <- out$AIC_REML - min(out$AIC_REML, na.rm = TRUE)
  out
}

# Singularity check: when the retained fit (M3) is singular, refit M3 with structure D (slope uncorrelated, never on the
# boundary here) and compare the interaction tests (same joint tests for both fits).
sensitivity_structure_D <- function(d, retained, M3) {
  row <- function(label, fit) data.frame(structure = label, singular = fit$singular, AIC_REML = AIC(fit$m), wald_tests(fit$m),
                                         note = "", stringsAsFactors = FALSE)
  if (retained == "D" || !isTRUE(M3$singular)) {
    out <- data.frame(structure = STRUCT_LABEL[[retained]], singular = M3$singular, AIC_REML = NA_real_, wald_na(),
                      note = "Not run: the retained fit is not singular (or already structure D).", stringsAsFactors = FALSE)
    out$delta_AIC <- NA_real_; return(out)
  }
  fitD <- tryCatch(fit_lmm(d, "D", TRUE), error = function(e) NULL)
  if (is.null(fitD)) {
    out <- data.frame(structure = STRUCT_LABEL[[retained]], singular = M3$singular, AIC_REML = NA_real_, wald_na(),
                      note = "Refit with structure D failed.", stringsAsFactors = FALSE)
    out$delta_AIC <- NA_real_; return(out)
  }
  out <- rbind(row(STRUCT_LABEL[[retained]], M3), row(STRUCT_LABEL[["D"]], fitD))
  out$delta_AIC <- out$AIC_REML - min(out$AIC_REML); out
}

# ---- 5. Excel export ---------------------------------------------------------------------------------------
block <- function(title, tab, note = "") list(title = title, tab = round_df(as.data.frame(tab)), note = note)
section <- function(sheet, title, blocks) list(sheet = sheet, title = title, blocks = blocks)

write_xlsx_report <- function(sections, path) {
  wb <- createWorkbook()
  st_t <- createStyle(textDecoration = "bold", fontSize = 14); st_b <- createStyle(textDecoration = "bold")
  st_h <- createStyle(textDecoration = "bold", fgFill = "#D9E2F3", halign = "center", border = "topbottom", wrapText = TRUE)
  st_n <- createStyle(textDecoration = "italic", fontSize = 9)
  for (sec in sections) {
    ws <- substr(sec$sheet, 1, 31); addWorksheet(wb, ws)
    writeData(wb, ws, sec$title, startRow = 1, startCol = 1); addStyle(wb, ws, st_t, rows = 1, cols = 1)
    r <- 3
    for (b in sec$blocks) {
      writeData(wb, ws, b$title, startRow = r, startCol = 1); addStyle(wb, ws, st_b, rows = r, cols = 1); r <- r + 1
      writeData(wb, ws, b$tab, startRow = r, startCol = 1, headerStyle = st_h); r <- r + nrow(b$tab) + 1
      if (nzchar(b$note)) { writeData(wb, ws, b$note, startRow = r, startCol = 1); addStyle(wb, ws, st_n, rows = r, cols = 1); r <- r + 1 }
      r <- r + 1
    }
    setColWidths(wb, ws, cols = 1:14, widths = "auto")
  }
  saveWorkbook(wb, path, overwrite = TRUE)
}

# ---- 6. Analysis of one variable (SFI) ---------------------------------------------------------------------------
analyse_variable <- function(key, base) {
  sp <- VARIABLES[[key]]; d <- base$d
  out_dir <- file.path(OUT_DIR, key)
  dir.create(file.path(out_dir, "tables"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(out_dir, "figures"), recursive = TRUE, showWarnings = FALSE)
  d$raw <- d[[sp$col]] * sp$scale; stopifnot(!anyNA(d$raw))
  d$y <- d$raw                                       # SFI analysed on its original scale (no transformation)
  n_obs <- nrow(d); n_plots <- n_distinct(d$plot_id)

  # missing values relative to the complete design (plots x 5 depths)
  plots <- d %>% distinct(plot_id, site, lc)
  design <- merge(plots, data.frame(depth = factor(DEPTHS, levels = DEPTHS)))
  miss <- anti_join(design, d[, c("plot_id", "depth")], by = c("plot_id", "depth"))
  miss_tab <- as.data.frame(table(Landscape = miss$site, Depth = miss$depth))
  names(miss_tab) <- c("Landscape", "Depth", "Missing measurements")

  # random structures (REML, same fixed effects M3)
  fits <- lapply(setNames(names(STRUCT_LABEL), names(STRUCT_LABEL)), function(s) fit_lmm(d, s, TRUE))
  struct <- do.call(rbind, lapply(names(fits), function(s) data.frame(
    Structure = STRUCT_LABEL[[s]], Code = s, Parameters = npar(fits[[s]]$m), logLik_REML = ll(fits[[s]]$m),
    AIC = AIC(fits[[s]]$m), BIC = BIC(fits[[s]]$m), Singular = fits[[s]]$singular,
    Warnings = paste(unique(fits[[s]]$warn), collapse = " | "))))
  struct$delta_AIC <- struct$AIC - min(struct$AIC)
  retained <- if (!is.null(FORCED_STRUCTURE)) FORCED_STRUCTURE else struct$Code[which.min(struct$AIC)]

  # candidate fixed-effect models (ML) and likelihood-ratio tests
  ml <- lapply(FX, function(f) fit_lmm(d, retained, FALSE, f)$m)
  cand <- do.call(rbind, lapply(names(ml), function(k) data.frame(Model = k, Fixed_effects = FX[[k]], Parameters = npar(ml[[k]]),
                                                                  logLik_ML = ll(ml[[k]]), AIC = AIC(ml[[k]]), BIC = BIC(ml[[k]]))))
  cand$delta_AIC <- cand$AIC - min(cand$AIC)
  lrt_row <- function(big, small, name) { chi <- 2 * (ll(ml[[big]]) - ll(ml[[small]])); ddf <- npar(ml[[big]]) - npar(ml[[small]])
  data.frame(Term = name, Models = paste(big, "vs", small), Chi2 = chi, df = ddf, p = pchisq(chi, ddf, lower.tail = FALSE)) }
  lrt <- rbind(lrt_row("M3", "M2", "Landscape x land cover"), lrt_row("M3", "M1", "Landscape x depth"),
               lrt_row("M4", "M3", "Land cover x depth (complementary)"),
               lrt_row("M5", "M4", "Landscape x land cover x depth (complementary)"))

  # final models (REML)
  M3 <- fit_lmm(d, retained, TRUE, FX[["M3"]]); M4 <- fit_lmm(d, retained, TRUE, FX[["M4"]]); M5 <- fit_lmm(d, retained, TRUE, FX[["M5"]])
  m <- M3$m; mods <- list(M3 = m, M4 = M4$m)
  conv <- data.frame(Model = c("M3 (retained)", "M4 (+ land cover x depth)", "M5 (triple interaction)"),
                     Singular = c(M3$singular, M4$singular, M5$singular),
                     Warnings = c(paste(unique(M3$warn), collapse = " | "), paste(unique(M4$warn), collapse = " | "),
                                  paste(unique(M5$warn), collapse = " | ")), stringsAsFactors = FALSE)
  # Satterthwaite F of the interaction terms of M3 (REML): confirmation of the likelihood-ratio tests
  aov_sat <- as.data.frame(anova(m)); aov_sat <- cbind(Term = rownames(aov_sat), aov_sat)
  names(aov_sat) <- c("Term", "SumSq", "MeanSq", "df1", "df2", "F", "p")
  sat_row <- function(term, label) { z <- aov_sat[aov_sat$Term == term, c("F", "df1", "df2", "p")]; cbind(Term = label, z) }
  r2 <- r2_nakagawa(m, d)
  m0 <- lmer(y ~ 1 + (1 | plot_id), data = d); v0 <- as.data.frame(VarCorr(m0))$vcov
  icc <- v0[1] / sum(v0)
  bnd <- boundary_terms(m)
  vc <- as.data.frame(VarCorr(m))
  vc_tab <- data.frame(Group = vc$grp, Term1 = vc$var1, Term2 = vc$var2, Variance = vc$vcov, SD_or_correlation = vc$sdcor)
  vc_tab$At_boundary <- ifelse(vc$grp == "Residual", FALSE,
                               ifelse(is.na(vc$var2), vc$sdcor < 1e-4, abs(vc$sdcor) > 0.99))
  coefs <- as.data.frame(coef(summary(m))); coefs <- cbind(Term = rownames(coefs), coefs)
  names(coefs) <- c("Term", "Estimate", "SE", "df", "t", "p")

  # annotations, contrasts, tests
  annot <- annotate_specs(mods, d)
  pw <- all_pairs(mods)
  tri <- triple_pairs(M5$m)
  dgn <- diagnostics(m, d, file.path(out_dir, "figures", "fig_S_residual_diagnostics.png"))
  sens_slope <- sensitivity_slope_coding(d, retained)
  sens_D <- sensitivity_structure_D(d, retained, M3)

  # ---- CSV tables for the figures ---------------------------------------------------------------------------
  write.csv(annot, file.path(out_dir, "tables", "fig_annotations.csv"), row.names = FALSE)
  prof <- tri$emm; prof$depth_cm <- unname(DEPTH_MID_CM[as.character(prof$depth)])
  star <- tri$contrasts[tri$contrasts$a == LC_REF, c("site", "b", "depth", STAR_FAMILY)]   # land cover vs reference (grassland)
  names(star) <- c("site", "lc", "depth", "p_star")
  prof <- merge(prof, star, by = c("site", "lc", "depth"), all.x = TRUE)
  prof$star <- !is.na(prof$p_star) & prof$p_star < ALPHA
  write.csv(prof, file.path(out_dir, "tables", "fig_profiles_lc_site.csv"), row.names = FALSE)
  write.csv(cbind(variable = key, sens_slope), file.path(out_dir, "tables", "sensitivity_slope_coding.csv"), row.names = FALSE)
  write.csv(cbind(variable = key, sens_D), file.path(out_dir, "tables", "sensitivity_structure_D.csv"), row.names = FALSE)
  write.csv(data.frame(plot_id = d$plot_id, site = d$site, lc = d$lc, depth = d$depth, value = d$raw),
            file.path(out_dir, "tables", "data.csv"), row.names = FALSE)

  # ---- Excel workbook (sheets in the order of the Methods) -------------------------------------------------------------
  eff <- d %>% group_by(site, lc) %>% summarise(plots = n_distinct(plot_id), samples = n(), .groups = "drop")
  names(eff) <- c("Landscape", "Land cover", "Plots (n)", "Samples (n)")
  pwt <- data.frame(Family = pw$family, Key = pw$key, Model = pw$model, Contrast = pw$contrast, Estimate = pw$estimate, SE = pw$SE, df = pw$df,
                    CI_low = pw$ci_low, CI_high = pw$ci_high, p = pw$p, p_Holm = pw$p_holm)
  sel <- function(keys) { z <- pwt[pwt$Key %in% keys, ]; z$Key <- NULL; z }
  wald <- function(fig) { a <- annot[annot$figure == fig, ]; a <- a[!duplicated(a$panel), ]
  data.frame(Panel = a$panel, F = a$F, df1 = a$df1, df2 = a$df2, p = a$p_panel) }
  emm_tab <- function(fig) { a <- annot[annot$figure == fig, ]
  data.frame(Panel = a$panel, Level = a$level_x, Estimated_mean = a$emm, CI_low = a$ci_low, CI_high = a$ci_high,
             Samples = a$n_samples, Plots = a$n_plots, Letter = a$letter) }
  tri_t <- tri$contrasts[, c("site", "depth", "contrast", "estimate", "SE", "df", "ci_low", "ci_high", "p",
                             "p_holm_6_cell", "p_holm_30_site", "p_holm_90")]
  names(tri_t) <- c("Landscape", "Depth", "Contrast", "Estimate", "SE", "df", "CI_low", "CI_high", "p",
                    "p_Holm_6_per_cell", "p_Holm_30_per_landscape", "p_Holm_90_total")
  lrt_sat <- function(i, term, label) rbind(
    data.frame(Test = "Likelihood-ratio test (ML)", Models = lrt$Models[i], Statistic = lrt$Chi2[i], df1 = lrt$df[i], df2 = NA_real_, p = lrt$p[i]),
    setNames(data.frame(Test = "Satterthwaite F test of M3 (REML), confirmation", Models = "M3", sat_row(term, label)[, c("F", "df1", "df2", "p")]),
             c("Test", "Models", "Statistic", "df1", "df2", "p")))
  info <- data.frame(Item = c("Variable", "Column in data file", "Analysed scale", "Retained random structure", "Singular fit",
                              "Samples (n)", "Plots (n)", "Missing measurements (out of 5 x plots)",
                              "Max/min residual SD ratio across depths", "Residual variance differs between depths (ratio > threshold)", "Model"),
                     Value = c(sp$name, sp$col, "original scale (no transformation)", STRUCT_LABEL[[retained]], M3$singular, n_obs, n_plots,
                               nrow(miss), round(dgn$depth_ratio, 3), dgn$depth_ratio > DEPTH_VAR_THRESHOLD,
                               "y ~ landscape*land cover + landscape*depth + random plot effect; Satterthwaite df; Holm per family"))
  sections <- list(
    section("0_Variable", paste("Linear mixed model results:", sp$name), list(block("Information", info))),
    section("1_Design", "1. Sampling design and missing data",
            list(block("Plots and samples by landscape and land cover", eff),
                 block("Missing measurements by landscape and depth (relative to the complete design plots x 5 depths)", miss_tab,
                       "Assumption: missing at random conditionally on landscape and depth (MAR)."))),
    section("2_Model_choice", "2. Model choice",
            list(block("Random structures (REML, fixed effects M3)", struct, "Lowest AIC = best structure; Singular = a parameter estimated on the boundary (variance ~ 0 or correlation = +/-1)."),
                 block("Candidate fixed-effect structures (ML)", cand, "Predefined candidates, not selected stepwise; M3 retained as the main model."))),
    section("3_Land_cover", "3. Land cover (overall effects: Satterthwaite F; interactions: likelihood-ratio tests)",
            list(block("3.1a Estimated marginal means of land cover, all landscapes and depths (model M3; equal weights; 95% CI)", emm_tab("global_landcover")),
                 block("3.1b Joint Satterthwaite F test of land cover (3 numerator df; model M3)", wald("global_landcover")),
                 block("3.1c Six pairwise comparisons of land cover, all landscapes and depths (model M3)", sel("lc_all"),
                       "Estimate: difference. Holm-adjusted p. To be read descriptively if the landscape x land cover interaction is significant."),
                 block("3.2 Consistency across landscapes: landscape x land cover interaction (M3 vs M2)",
                       lrt_sat(1, "site:lc", "Landscape x land cover")),
                 block("3.3a Joint F test of land cover within each landscape (model M3)", wald("site_by_landcover")),
                 block("3.3b Six pairwise comparisons within each landscape (model M3)", sel("lc_in_site"), "Holm-adjusted p within each landscape."),
                 block("3.3c Estimated means and letters, by landscape (model M3)", emm_tab("site_by_landcover")),
                 block("3.4a Joint F test of land cover at each depth (model M4)", wald("depth_by_landcover")),
                 block("3.4b Six pairwise comparisons at each depth (model M4)", sel("lc_at_depth"), "Holm-adjusted p within each depth. Landscapes pooled with equal weights."),
                 block("3.4c Estimated means and letters, by depth (model M4)", emm_tab("depth_by_landcover")),
                 block("3.5a Landscapes compared within each land cover: joint F test (model M3)", wald("landcover_by_site")),
                 block("3.5b Landscapes compared within each land cover: three pairs (model M3)", sel("site_in_lc"), "Holm-adjusted p within each land cover."),
                 block("3.5c Estimated means and letters, by land cover (model M3)", emm_tab("landcover_by_site")))),
    section("4_Depth", "4. Depth (overall effect: Satterthwaite F; interactions: likelihood-ratio tests)",
            list(block("4.1a Estimated marginal means of depth, all landscapes and land covers (model M3; equal weights; 95% CI)", emm_tab("global_depth")),
                 block("4.1b Joint Satterthwaite F test of depth (4 numerator df; model M3)", wald("global_depth")),
                 block("4.1c Ten pairwise comparisons among layers (model M3)", sel("depth_all"), "Holm-adjusted p. To be read descriptively if the interactions are significant."),
                 block("4.2 Landscape x depth interaction (M3 vs M1)", lrt_sat(2, "site:depth", "Landscape x depth")),
                 block("4.3a Joint F test of depth within each landscape (model M3)", wald("depth_by_site")),
                 block("4.3b Ten pairwise comparisons within each landscape (model M3)", sel("depth_in_site"), "Holm-adjusted p within each landscape."),
                 block("4.3c Estimated means and letters, by landscape (model M3)", emm_tab("depth_by_site")),
                 block("4.4a Joint F test of depth within each land cover (model M4)", wald("landcover_by_depth")),
                 block("4.4b Ten pairwise comparisons within each land cover (model M4)", sel("depth_in_lc"), "Holm-adjusted p within each land cover. Landscapes pooled with equal weights."),
                 block("4.4c Estimated means and letters, by land cover (model M4)", emm_tab("landcover_by_depth")),
                 block("4.5a Landscapes compared at each depth: joint F test (model M3)", wald("site_by_depth")),
                 block("4.5b Landscapes compared at each depth: three pairs (model M3)", sel("site_at_depth"), "Holm-adjusted p within each depth."),
                 block("4.5c Estimated means and letters, by depth (model M3)", emm_tab("site_by_depth")))),
    section("5_Complementary_LRT", "5. Complementary likelihood-ratio tests (ML)",
            list(block("Land cover x depth (M4 vs M3) and triple interaction (M5 vs M4)", lrt[3:4, ]))),
    section("6_Landscape_overall", "6. Supplementary: landscapes compared overall (used by fig_box_01)",
            list(block("Estimated marginal means of landscape (model M3; equal weights; 95% CI)", emm_tab("global_site")),
                 block("Joint Satterthwaite F test of landscape (2 numerator df)", wald("global_site")),
                 block("Three pairwise comparisons between landscapes (model M3)", sel("site_all")))),
    section("7_LC_by_landscape_depth", "7. Supplementary: land cover contrasts in each landscape x depth cell (model M5, triple interaction)",
            list(block("All pairwise contrasts (6 per cell, 90 in total)", tri_t,
                       paste0("Three Holm families provided: 6 per landscape x depth cell, 30 per landscape, 90 in total. Triple-interaction LRT p = ", signif(lrt$p[4], 3),
                              ". Stars in fig_prof_04 use the family set in config.R (STAR_FAMILY) on the contrasts versus ", LC_REF, ".")))),
    section("8_Fit_diagnostics", "8. Fit and diagnostics",
            list(block("Nakagawa R2 and ICC", data.frame(Indicator = c("Marginal R2", "Conditional R2", "ICC (null model)"),
                                                         Value = c(r2[["R2_marginal"]], r2[["R2_conditional"]], icc)),
                       "Computed on the fitting data, without external validation."),
                 block("Variance components", vc_tab, "At_boundary = TRUE: variance ~ 0 or correlation = +/-1 (singular fit)."),
                 block(paste0("Fixed-effect coefficients (reference: ", SITES[1], ", ", LC_REF, ", ", DEPTHS[1], " cm)"), coefs),
                 block("Residual diagnostics", dgn$tab, "Judge on the graphs (fig_S_residual_diagnostics.png); numerical indicators are guidelines."),
                 block("Convergence and singularity of the final models (REML)", conv,
                       "Any mention of non-convergence must be examined before interpreting M4 and M5."),
                 block("Depth coding in the random slope (retained structure, REML)", sens_slope,
                       "Lower AIC = better fit; no effect if the retained structure has no slope (A, C)."))),
    section("9_Singularity_check", "9. Singularity check",
            list(block("Parameters estimated on the boundary (retained structure, model M3)", if (nrow(bnd)) bnd else data.frame(note = "None: the fit is not singular."),
                       "Typically a correlation between the random intercept and slope of one landscape estimated at +1 or -1."),
                 block("Retained structure vs structure D (intercept and slope uncorrelated): interaction tests", sens_D,
                       "Same joint tests in both fits; if the p-values agree, the singularity does not affect the conclusions. Not run when the retained fit is not singular."))))
  write_xlsx_report(sections, file.path(out_dir, paste0(key, "_LMM_results.xlsx")))

  # ---- summary row -------------------------------------------------------------------------------------------------
  panel_p <- function(fig) { z <- annot[annot$figure == fig & !duplicated(paste(annot$figure, annot$panel)), ]; setNames(z$p_panel, z$panel) }
  g1 <- panel_p("global_landcover"); g2 <- panel_p("global_depth"); po <- panel_p("site_by_landcover"); pp <- panel_p("depth_by_site")
  res <- data.frame(variable = key, name = sp$name, structure = retained, singular = M3$singular,
                    boundary = paste(unique(bnd$landscape), collapse = "+"),
                    p_F_landcover = g1[["All"]], p_LRT_landscape_x_landcover = lrt$p[1],
                    p_F_depth = g2[["All"]], p_LRT_landscape_x_depth = lrt$p[2],
                    p_LRT_lc_x_depth = lrt$p[3], p_LRT_triple = lrt$p[4])
  for (s in SITES) res[[paste0("p_landcover_", s)]] <- po[[s]]
  for (s in SITES) res[[paste0("p_depth_", s)]] <- pp[[s]]
  res <- cbind(res, data.frame(R2_marginal = r2[["R2_marginal"]], R2_conditional = r2[["R2_conditional"]], ICC = icc,
                               residual_skewness = dgn$skew, excess_kurtosis = dgn$kurt, sd_ratio_depths = dgn$depth_ratio))
  res$missing <- nrow(miss)
  write.csv(res, file.path(out_dir, "tables", "summary.csv"), row.names = FALSE)
  res
}

# ---- 7. Main program -------------------------------------------------------------------------------------------------
if (sys.nframe() == 0) {
  base <- load_data()
  dir.create(OUT_DIR, showWarnings = FALSE)
  for (key in names(VARIABLES)) {
    cat("\n====", key, "====\n"); t0 <- Sys.time()
    r <- tryCatch(analyse_variable(key, base), error = function(e) { cat("[FAILED]", key, ":", conditionMessage(e), "\n"); NULL })
    if (!is.null(r)) cat("[OK]", key, round(as.numeric(difftime(Sys.time(), t0, units = "secs"))), "s\n")
  }
  writeLines(capture.output(sessionInfo()), file.path(OUT_DIR, "R_and_package_versions.txt"))
}
