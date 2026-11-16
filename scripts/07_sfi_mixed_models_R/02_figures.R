# =====================================================================================================
# 02_figures.R : publication figures, in English, with ggplot2.
# Reads results/<VAR>/tables/*.csv written by 01_analysis.R: letters, p-values, estimated means and CIs come from
# the model and are never recomputed here (only the raw data of the box plots are read).
#
#   fig_box_01_global_site ... fig_box_04_landcover_by_site      box plots
#   fig_prof_01_global_depth                                     global depth profile
#   fig_prof_02_by_landscape_vertical / horizontal               profile by site
#   fig_prof_03_by_landcover_*  (4 variants)                     profile by land cover
#   fig_prof_04_landcover_within_landscape_*                     land cover within each site (* = differs from grassland)
#
# RULES OF THIS VERSION
#   - Each figure is drawn AT ITS FINAL PRINT SIZE (WIDTH_* of config.R): the text sizes (PT_*) and line widths (LW_*)
#     of config.R are therefore the REAL sizes (text >= 7 pt).
#   - No title; a SHORT grey note (essentials only) is printed UNDER the plot and also written to
#     figures/figure_legends.txt. Legends are placed on the RIGHT of the plot.
#   - Drawing order of the profiles: line -> POINTS -> ERROR BARS (bars are never hidden by a point).
#   - End caps of constant width: CAP_CM (axis in cm) or CAP_CLASS (axis in classes); boxes: 0.3.
#   - Depth in true cm = layer MID-POINT (5, 15, 25, 55, 85 cm), see DEPTH_MID_CM in config.R.
# Usage:  Rscript 02_figures.R            (all variables)   |   Rscript 02_figures.R SFI pH
# =====================================================================================================
suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })
source("config.R")

LEVELS_FIG <- list(site = SITES, lc = LC_ORDER, depth = DEPTHS)           # display order
PAL   <- list(site = COL_SITE, lc = COL_LC, depth = COL_DEPTH)
PAL_D <- list(site = COL_SITE_D, lc = COL_LC_D, depth = COL_DEPTH_D)
LEGEND_TITLE <- c(site = "Landscape", lc = "Land cover", depth = "Depth (cm)")
FACTOR_NAME  <- c(site = "landscape", lc = "land cover", depth = "depth")
PLURAL       <- c(site = "landscapes", lc = "land covers", depth = "depths")
SITE_MID     <- setdiff(SITES, c(SITE_ALIGN_RIGHT, SITE_ALIGN_LEFT))     # site(s) whose letters are centred

mm <- function(pt) pt / .pt        # ggplot text size (mm) for a wanted size in points

theme_fig <- theme_minimal(base_size = PT_TEXT) +
  theme(text = element_text(size = PT_TEXT),
        axis.text = element_text(size = PT_TEXT, colour = "black"),
        axis.title = element_text(size = PT_TITLE, face = "bold"),
        panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = "#e5e5e5", linewidth = 0.3),
        axis.line = element_line(colour = "black", linewidth = 0.4), axis.ticks = element_line(colour = "black", linewidth = 0.4),
        strip.background = element_rect(fill = "#f2f2f2", colour = NA), strip.text = element_text(size = PT_TITLE, face = "bold"),
        legend.text = element_text(size = PT_TEXT), legend.title = element_text(size = PT_TITLE, face = "bold"),
        legend.position = "right", legend.box = "vertical", legend.box.just = "left", legend.key.size = unit(3.5, "mm"),
        legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(3, "mm"), legend.spacing.y = unit(3, "mm"),
        plot.margin = margin(4, 6, 4, 4), plot.background = element_rect(fill = "white", colour = NA))

fmt_p <- function(p, nd = 2) ifelse(p < 0.001, formatC(p, format = "e", digits = nd), formatC(p, format = "f", digits = 3))
wrap <- function(s, w) paste(strwrap(s, width = w), collapse = "\n")

NOTE_PTS <- "Points: estimated marginal means \u00b1 95% CI."
NOTE_SAME <- "the same letter indicates no significant difference."

LEG <- new.env()    # captions proposed for the manuscript, written to figure_legends.txt
save_fig <- function(p, name, folder, w, h, leg = NULL) {
  if (!is.null(leg)) {      # caption specific to each figure, printed under the plot (the height increases accordingly)
    lines <- unlist(lapply(leg, strwrap, width = floor(w * 17 * PT_TEXT / PT_CAPTION)))   # one paragraph per element of leg
    p <- p + labs(caption = paste(lines, collapse = "\n")) +
      theme(plot.caption = element_text(size = PT_CAPTION, colour = COL_CAPTION, hjust = 0.5, lineheight = 1.05, margin = margin(t = 6)), plot.caption.position = "plot")
    h <- h + length(lines) * PT_CAPTION * 1.25 / 72 + 0.12
  }
  ggsave(file.path(folder, paste0(name, ".png")), p, width = w, height = h, dpi = 600, bg = "white")
  ggsave(file.path(folder, paste0(name, ".pdf")), p, width = w, height = h, device = if (capabilities("cairo")) cairo_pdf else "pdf")
  if (!is.null(leg)) assign(name, paste(leg, collapse = " "), envir = LEG)
  cat("   ", name, sprintf("(%.2f x %.2f in)", w, h), "\n")
}

# ---- A. Box plots -----------------------------------------------------------------------------------------
box_limits <- function(d) {
  v <- d$value; wh_lo <- c(); wh_hi <- c()
  for (g in list("site", "lc", c("site", "lc"))) {
    for (x in split(v, interaction(d[g], drop = TRUE))) {
      q <- quantile(x, c(0.25, 0.75)); i <- q[2] - q[1]
      wh_hi <- c(wh_hi, max(x[x <= q[2] + 1.5 * i])); wh_lo <- c(wh_lo, min(x[x >= q[1] - 1.5 * i]))
    }
  }
  hi <- min(max(v), max(quantile(v, 0.99), max(wh_hi))); lo <- max(min(v), min(quantile(v, 0.01), min(wh_lo))); rng <- hi - lo
  tk <- pretty(c(lo, hi), n = 5); tk <- tk[tk >= lo - 1e-9 & tk <= hi + 1e-9]
  # below the data area: n labels then p; above: letters (Holm)
  list(lo = lo, hi = hi, ylim = c(lo - 0.17 * rng, hi + 0.16 * rng), n = lo - 0.035 * rng, p = lo - 0.11 * rng,
       letter = hi + 0.03 * rng, ticks = tk)
}

box_plot <- function(d, ann, fid, x, facet, name, axis_lab, lim, folder) {
  a <- ann[ann$figure == fid, ]; lev <- LEVELS_FIG[[x]]
  d$xv <- factor(d[[x]], levels = lev); d$col <- unname(PAL[[x]][as.character(d$xv)])
  panels <- if (is.null(facet)) "All" else LEVELS_FIG[[facet]]
  d$panel <- if (is.null(facet)) factor("All", levels = panels) else factor(d[[facet]], levels = panels)
  a$xv <- factor(a$level_x, levels = lev); a$panel <- factor(a$panel, levels = panels)
  a$colf <- unname(PAL_D[[x]][as.character(a$xv)])
  pdat <- a[!duplicated(a$panel), ]; pdat$lab <- paste0("p = ", fmt_p(pdat$p_panel)); pdat$xx <- (length(lev) + 1) / 2
  lab_pan <- if (!is.null(facet) && facet == "site") SITE_LABEL else setNames(panels, panels)
  pooled <- PLURAL[setdiff(c("site", "lc", "depth"), c(x, facet))]
  outside <- d$value < lim$lo | d$value > lim$hi          # extreme values outside the axis area (not drawn as points)
  dj <- d[!outside, ]
  leg <- paste0("Boxes: median and interquartile range; whiskers: 1.5 x IQR; + = mean; points = individual samples (", paste(pooled, collapse = " and "),
                " pooled); n = number of samples",
                if (any(outside)) paste0("; ", sum(outside), " extreme value(s) beyond the axis not drawn as points") else "",
                ". Letters: Holm-adjusted pairwise comparisons of ", FACTOR_NAME[[x]], " levels from the linear mixed model",
                if (!is.null(facet)) paste0(", within each ", FACTOR_NAME[[facet]]) else "", "; ", NOTE_SAME,
                " p: Satterthwaite F-test of ", FACTOR_NAME[[x]], if (!is.null(facet)) paste0(" within each ", FACTOR_NAME[[facet]]) else "", ".")
  p <- ggplot(d, aes(xv, value)) +
    stat_boxplot(geom = "errorbar", coef = 1.5, width = 0.3, colour = "#404040", linewidth = LW_BAR) +   # end caps of the whiskers
    geom_boxplot(aes(fill = xv), width = 0.65, alpha = 0.55, outlier.shape = NA, coef = 1.5, colour = "#404040", linewidth = LW_BOX) +
    geom_jitter(data = dj, aes(colour = col), position = position_jitter(width = 0.17, height = 0, seed = 1), size = 0.7, alpha = 0.5) +
    stat_summary(fun = mean, geom = "point", shape = 3, size = 2.2, stroke = 0.9, colour = "black") +
    geom_text(data = a, aes(x = xv, label = letter, colour = colf), y = lim$letter, vjust = 0, fontface = "bold", size = mm(PT_LETTER), inherit.aes = FALSE) +
    geom_text(data = a, aes(x = xv, label = paste0("n=", n_samples)), y = lim$n, vjust = 1,
              fontface = "italic", size = mm(PT_TEXT), colour = "#595959", inherit.aes = FALSE) +
    geom_text(data = pdat, aes(x = xx, label = lab), y = lim$p, vjust = 1, fontface = "bold.italic", size = mm(PT_TEXT + 0.5), inherit.aes = FALSE) +
    facet_wrap(~panel, nrow = 1, labeller = labeller(panel = lab_pan)) +
    scale_fill_manual(values = PAL[[x]], labels = if (x == "depth") paste(lev, "cm") else lev,
                      name = LEGEND_TITLE[[x]]) + scale_colour_identity() +
    scale_y_continuous(breaks = lim$ticks) + coord_cartesian(ylim = lim$ylim) +
    labs(x = NULL, y = axis_lab) +
    theme_fig + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), panel.grid.major.x = element_blank())
  width <- if (is.null(facet)) WIDTH_15COL else WIDTH_2COL
  save_fig(p, name, folder, width, 4.4, leg)
}

# ---- B1. Global profile and profiles by site (depth in true cm, layer mid-point) -----------------------------------
depth_guide <- guides(size = guide_legend(title = "Depth (cm)", override.aes = list(fill = "#999999", colour = "#333333", shape = 21)))
AXIS_CM <- "Depth (layer mid-point, cm)"

fig_prof_01 <- function(ann, n_obs, axis_lab, folder) {
  g <- ann[ann$figure == "global_depth", ]; g$cm <- unname(DEPTH_MID_CM[g$level_x]); g$dep <- factor(g$level_x, levels = DEPTHS)
  sp <- max(g$ci_high) - min(g$ci_low)
  leg <- c(paste0("n=", n_obs, " | Depth effect (Satterthwaite F-test): p=", fmt_p(g$p_panel[1], 1)),
           paste(NOTE_PTS, "Depths sharing a letter do not differ (Holm)."))
  p <- ggplot(g, aes(emm, cm)) + geom_path(colour = "#404040", linewidth = LW_CURVE) +
    geom_point(aes(size = dep), shape = 21, fill = "#999999", colour = "#333333", stroke = 0.5) +
    geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = CAP_CM, colour = "#1a1a1a", linewidth = LW_BAR) +
    geom_text(aes(x = ci_high + 0.02 * sp, label = letter), hjust = 0, fontface = "bold", colour = "#404040", size = mm(PT_LETTER)) +
    scale_y_reverse(breaks = unname(DEPTH_MID_CM), labels = DEPTHS, limits = c(92, -2)) +
    scale_x_continuous(limits = c(min(g$ci_low) - 0.12 * sp, max(g$ci_high) + 0.18 * sp)) +
    scale_size_manual(values = SIZE_DEPTH) + depth_guide +
    labs(x = axis_lab, y = AXIS_CM) + theme_fig
  save_fig(p, "fig_prof_01_global_depth", folder, WIDTH_15COL, 4.8, leg)
}

fig_prof_02 <- function(ann, d, axis_lab, folder) {
  a1 <- ann[ann$figure == "depth_by_site", ]; a2 <- ann[ann$figure == "site_by_depth", ]
  g <- merge(data.frame(site = a1$panel, depth = a1$level_x, emm = a1$emm, ci_low = a1$ci_low, ci_high = a1$ci_high,
                        minus = a1$letter, n = a1$n_samples, p = a1$p_panel),
             data.frame(site = a2$level_x, depth = a2$panel, maj = toupper(a2$letter)), by = c("site", "depth"))
  g$label <- paste0(g$maj, g$minus); g$cm <- unname(DEPTH_MID_CM[g$depth]); g$dep <- factor(g$depth, levels = DEPTHS)
  g$site <- factor(g$site, levels = SITES); g <- g[order(g$site, g$cm), ]; sp <- max(g$ci_high) - min(g$ci_low); lo <- min(g$ci_low); hi <- max(g$ci_high)
  ps <- g[!duplicated(g$site), ]; ps$n <- vapply(as.character(ps$site), function(s) sum(d$site == s), 1L)
  by_site <- function(f) paste(vapply(SITES, function(s) paste0(s, ": ", f(ps[ps$site == s, ])), ""), collapse = " | ")
  leg <- c(by_site(function(r) paste0("n=", r$n)),
           paste0("Depth effect within landscape (Satterthwaite F-test): ", by_site(function(r) paste0("p=", fmt_p(r$p, 1)))),
           paste(NOTE_PTS, "Letters: capital = landscapes compared within a depth; lowercase = depths compared within a landscape (Holm);", NOTE_SAME))
  # lines, bars and letters in dark tone; points filled in light tone (the bars remain visible inside the point)
  guid <- guides(fill = guide_legend(order = 1, title = "Landscape", override.aes = list(shape = 21, size = 3.5, colour = "white")), colour = "none",
                 size = guide_legend(order = 2, title = "Depth (cm)", override.aes = list(fill = "#999999", colour = "#333333", shape = 21)))
  scl <- list(scale_colour_manual(values = COL_SITE_D), scale_fill_manual(values = COL_SITE, name = "Landscape"),
              scale_size_manual(values = SIZE_DEPTH), guid)
  # --- vertical (depth on the y axis) ---
  gv <- g; gv$tx <- ifelse(gv$site == SITE_ALIGN_RIGHT, gv$ci_high + 0.02 * sp, ifelse(gv$site == SITE_ALIGN_LEFT, gv$ci_low - 0.02 * sp, gv$emm))
  gv$ty <- ifelse(gv$site %in% SITE_MID, gv$cm - 6, gv$cm); gv$hj <- ifelse(gv$site == SITE_ALIGN_RIGHT, 0, ifelse(gv$site == SITE_ALIGN_LEFT, 1, 0.5))
  pv <- ggplot(gv, aes(emm, cm, colour = site, group = site)) + geom_path(linewidth = LW_CURVE) +
    geom_point(aes(size = dep, fill = site), shape = 21, colour = "white", stroke = 0.5) +
    geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = CAP_CM, linewidth = LW_BAR) +
    geom_text(aes(x = tx, y = ty, label = label, hjust = hj), fontface = "bold", size = mm(PT_LETTER), show.legend = FALSE) +
    scale_y_reverse(breaks = unname(DEPTH_MID_CM), labels = DEPTHS, limits = c(93, -7)) + scale_x_continuous(limits = c(lo - 0.10 * sp, hi + 0.16 * sp)) +
    scl + labs(x = axis_lab, y = AXIS_CM) + theme_fig
  save_fig(pv, "fig_prof_02_by_landscape_vertical", folder, WIDTH_15COL, 5.6, leg)
  # --- horizontal (depth on the x axis) ---
  gh <- g; gh$ty <- ifelse(gh$site == SITE_ALIGN_RIGHT, gh$ci_high + 0.025 * sp, ifelse(gh$site == SITE_ALIGN_LEFT, gh$ci_low - 0.025 * sp, gh$emm + 0.02 * sp))
  gh$tx <- ifelse(gh$site %in% SITE_MID, gh$cm + 6, gh$cm); gh$vj <- ifelse(gh$site == SITE_ALIGN_RIGHT, 0, ifelse(gh$site == SITE_ALIGN_LEFT, 1, 0.5))
  ph <- ggplot(gh, aes(cm, emm, colour = site, group = site)) + geom_path(linewidth = LW_CURVE) +
    geom_point(aes(size = dep, fill = site), shape = 21, colour = "white", stroke = 0.5) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = CAP_CM, linewidth = LW_BAR) +
    geom_text(aes(x = tx, y = ty, label = label, vjust = vj), fontface = "bold", size = mm(PT_LETTER), show.legend = FALSE) +
    scale_x_continuous(breaks = unname(DEPTH_MID_CM), labels = DEPTHS, limits = c(-4, 96)) + coord_cartesian(ylim = c(lo - 0.10 * sp, hi + 0.16 * sp)) +
    scl + labs(x = AXIS_CM, y = axis_lab) + theme_fig
  save_fig(ph, "fig_prof_02_by_landscape_horizontal", folder, WIDTH_2COL, 4.2, leg)
}

# ---- B2. Equidistant depth classes, land covers offset (fig_prof_03 and fig_prof_04) ---------------------------------
auto_limits <- function(df, vert, extra = 0) {
  lo <- min(df$ci_low); hi <- max(df$ci_high); sp <- hi - lo
  c(lo - 0.03 * sp, hi + (if (vert) 0.15 else 0.12) * sp + extra)
}

equidistant <- function(df, layout, dodge, lim, axis_lab, labels = TRUE, stars = FALSE, lc_legend = TRUE) {
  vert <- layout == "vertical"
  df$di <- match(as.character(df$depth), DEPTHS); df$dep <- factor(df$depth, levels = DEPTHS)
  df <- df[order(df$site, df$lc, df$di), ]
  off <- setNames((seq_along(LC_ORDER) - (length(LC_ORDER) + 1) / 2) * LC_DODGE, LC_ORDER)
  df$pos <- df$di + if (dodge) unname(off[as.character(df$lc)]) else 0
  df$dk <- unname(COL_LC_D[as.character(df$lc)]); df$lcf <- factor(df$lc, levels = LC_ORDER)
  pad <- 0.012 * (lim[2] - lim[1])
  # drawing order: line -> point -> error bar (dark bars, always visible above the points)
  if (vert) {
    p <- ggplot(df, aes(x = emm, y = pos)) + geom_path(aes(colour = dk, group = interaction(lc, site)), linewidth = LW_CURVE) +
      geom_point(aes(size = dep, fill = lcf), shape = 21, colour = "#333333", stroke = 0.4) +
      geom_errorbar(aes(xmin = ci_low, xmax = ci_high, colour = dk), orientation = "y", width = CAP_CLASS, linewidth = LW_BAR) +
      scale_y_reverse(breaks = 1:5, labels = DEPTHS, limits = c(5.6, 0.4)) + coord_cartesian(xlim = lim) +
      labs(x = axis_lab, y = "Depth (cm)")
  } else {
    p <- ggplot(df, aes(x = pos, y = emm)) + geom_path(aes(colour = dk, group = interaction(lc, site)), linewidth = LW_CURVE) +
      geom_point(aes(size = dep, fill = lcf), shape = 21, colour = "#333333", stroke = 0.4) +
      geom_errorbar(aes(ymin = ci_low, ymax = ci_high, colour = dk), width = CAP_CLASS, linewidth = LW_BAR) +
      scale_x_continuous(breaks = 1:5, labels = DEPTHS, limits = c(0.4, 5.6)) + coord_cartesian(ylim = lim) +
      labs(x = "Depth (cm)", y = axis_lab)
  }
  p <- p + scale_colour_identity() + scale_fill_manual(values = COL_LC, name = "Land cover", drop = FALSE) +
    scale_size_manual(values = SIZE_DEPTH) + theme_fig
  if (labels) {
    p <- p + (if (vert) geom_text(aes(x = ci_high + pad, label = label, colour = dk), hjust = 0, fontface = "bold", size = mm(PT_LETTER - 0.5))
              else geom_text(aes(y = ci_high + pad, label = label, colour = dk), vjust = 0, fontface = "bold", size = mm(PT_LETTER - 0.5)))
  }
  if (stars) {
    st <- df[df$star, ]
    if (nrow(st) > 0) p <- p + (if (vert) geom_text(data = st, aes(x = ci_high + pad, label = "*", colour = dk), hjust = 0, fontface = "bold", size = mm(PT_STAR))
                                else geom_text(data = st, aes(y = ci_high + pad, label = "*", colour = dk), vjust = 0, fontface = "bold", size = mm(PT_STAR)))
  }
  g_lc <- if (lc_legend) guide_legend(order = 1, override.aes = list(size = 3.2, shape = 21)) else "none"
  p + guides(fill = g_lc, size = guide_legend(order = 2, title = "Depth (cm)", override.aes = list(fill = "#999999", colour = "#333333", shape = 21)))
}

fig_prof_03 <- function(ann, d, axis_lab, folder) {
  low <- ann[ann$figure == "landcover_by_depth", ]; up <- ann[ann$figure == "depth_by_landcover", ]
  g3 <- merge(data.frame(lc = low$panel, depth = low$level_x, emm = low$emm, ci_low = low$ci_low, ci_high = low$ci_high, minus = low$letter, p = low$p_panel),
              data.frame(lc = up$level_x, depth = up$panel, maj = toupper(up$letter)), by = c("lc", "depth"))
  g3$label <- paste0(g3$maj, g3$minus); g3$site <- "all"
  n3 <- table(d$lc); p3 <- g3[!duplicated(g3$lc), ]; p3 <- setNames(p3$p, p3$lc)
  leg_fn <- function(facet) c(paste(vapply(LC_ORDER, function(g) paste0(g, ": n=", n3[[g]]), ""), collapse = " | "),
                              paste0("Depth effect within land cover (Satterthwaite F-test): ",
                                     paste(vapply(LC_ORDER, function(g) paste0(g, " p=", fmt_p(p3[[g]], 1)), ""), collapse = " | ")),
                              paste(NOTE_PTS, "Letters: capital = land covers compared within a depth; lowercase = depths compared within a land cover (Holm);", NOTE_SAME,
                                    "Depth classes equally spaced (not to scale)."))
  specs <- list(c("vertical", "dodge", WIDTH_15COL, 6.0), c("vertical", "facet", WIDTH_2COL, 4.8),
                c("horizontal", "dodge", WIDTH_2COL, 4.2), c("horizontal", "facet", WIDTH_2COL, 5.5))
  for (s in specs) {
    layout <- s[1]; mode <- s[2]; w <- as.numeric(s[3]); h <- as.numeric(s[4]); vert <- layout == "vertical"
    lim <- auto_limits(g3, vert)
    if (mode == "dodge") {
      p <- equidistant(g3, layout, TRUE, lim, axis_lab)
    } else {
      p <- equidistant(g3, layout, FALSE, lim, axis_lab, lc_legend = FALSE) +
        facet_wrap(~factor(lc, levels = LC_ORDER), nrow = if (vert) 1 else 2)
    }
    save_fig(p, paste0("fig_prof_03_by_landcover_", layout, "_", mode), folder, w, h, leg_fn(mode == "facet"))
  }
}

fig_prof_04 <- function(lp, axis_lab, folder) {
  lp$site <- factor(lp$site, levels = SITES); lp$lc <- as.character(lp$lc); lp$depth <- as.character(lp$depth)
  lp$star <- as.logical(lp$star); lp$label <- ""
  leg <- c(paste(NOTE_PTS, "Depth classes equally spaced (not to scale)."),
           paste0("* = land cover differs from ", tolower(LC_REF), " at this depth within the landscape (Holm, p < 0.05)."))
  lab_site <- labeller(site = SITE_LABEL)
  extra <- 0.12 * (max(lp$ci_high) - min(lp$ci_low))
  pv <- equidistant(lp, "vertical", TRUE, auto_limits(lp, TRUE, extra), axis_lab, labels = FALSE, stars = TRUE) +
    facet_wrap(~site, nrow = 1, labeller = lab_site)
  save_fig(pv, "fig_prof_04_landcover_within_landscape_vertical", folder, WIDTH_2COL, 5.0, leg)
  ph <- equidistant(lp, "horizontal", TRUE, auto_limits(lp, FALSE, extra), axis_lab, labels = FALSE, stars = TRUE) +
    facet_wrap(~site, ncol = 1, labeller = lab_site)
  save_fig(ph, "fig_prof_04_landcover_within_landscape_horizontal", folder, WIDTH_15COL, 7.0, leg)
}

# ---- Main program ---------------------------------------------------------------------------------------------------
figures_variable <- function(key) {
  sp <- VARIABLES[[key]]; dos <- file.path(OUT_DIR, key); tab <- file.path(dos, "tables"); fig <- file.path(dos, "figures")
  dir.create(fig, showWarnings = FALSE, recursive = TRUE)
  rm(list = ls(LEG), envir = LEG)
  read_tab <- function(f) read.csv(file.path(tab, f), stringsAsFactors = FALSE, encoding = "UTF-8", check.names = FALSE)
  ann <- read_tab("fig_annotations.csv"); d <- read_tab("data.csv"); lp <- read_tab("fig_profiles_lc_site.csv")
  d$site <- as.character(d$site); d$lc <- as.character(d$lc); d$depth <- as.character(d$depth)
  lim <- box_limits(d)
  cat("Box plots\n")
  specs <- list(c("global_site", "site", NA), c("global_landcover", "lc", NA), c("site_by_landcover", "lc", "site"), c("landcover_by_site", "site", "lc"))
  for (i in seq_along(specs)) {
    s <- specs[[i]]; box_plot(d, ann, s[1], s[2], if (is.na(s[3])) NULL else s[3], sprintf("fig_box_%02d_%s", i, s[1]), sp$axis, lim, fig)
  }
  cat("Depth profiles\n")
  fig_prof_01(ann, nrow(d), sp$axis, fig); fig_prof_02(ann, d, sp$axis, fig)
  fig_prof_03(ann, d, sp$axis, fig); fig_prof_04(lp, sp$axis, fig)
  # captions to paste into the manuscript (the images contain neither title nor caption)
  names_ <- sort(ls(LEG))
  txt <- unlist(lapply(names_, function(n) c(paste0("[", n, "]"), strwrap(get(n, envir = LEG), width = 110), "")))
  con <- file(file.path(fig, "figure_legends.txt"), open = "w", encoding = "UTF-8")
  writeLines(c(paste("Proposed captions for the manuscript:", sp$name), "", txt), con); close(con)
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  keys <- if (length(args)) args else names(VARIABLES)
  for (key in keys) {
    if (!file.exists(file.path(OUT_DIR, key, "tables", "fig_annotations.csv"))) { cat("[SKIPPED]", key, ": run 01_analysis.R first\n"); next }
    cat("\n====", key, "====\n")
    tryCatch(figures_variable(key), error = function(e) cat("[FAILED]", key, ":", conditionMessage(e), "\n"))
  }
}
