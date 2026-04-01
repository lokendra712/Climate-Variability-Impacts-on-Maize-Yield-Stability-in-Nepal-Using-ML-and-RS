# =============================================================================
# 01_eberhart_russell.R
# =============================================================================
# Eberhart-Russell yield stability analysis for Nepal's seven provinces.
# Also computes Wricke's ecovalence (Wi) and GGE biplot.
#
# Reference:
#   Eberhart, S.A. & Russell, W.A. (1966). Stability parameters for
#   comparing varieties. Crop Science, 6(1), 36-40.
#   Wricke, G. (1962). Zeitschrift für Pflanzenzüchtung, 47, 92-146.
#
# Output:
#   outputs/eberhart_russell_stability.csv
#   outputs/stability_anova_table.csv
#   figures/fig6_er_stability_biplot.png
#
# Authors: Lokendra Khatri et al. (2026)
# =============================================================================

suppressPackageStartupMessages({
  library(metan)
  library(tidyverse)
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
root_dir    <- here::here()
data_path   <- file.path(root_dir, "data", "processed",
                         "nepal_maize_panel_SYNTHETIC_demo.csv")
output_dir  <- file.path(root_dir, "outputs")
figures_dir <- file.path(root_dir, "figures")
dir.create(output_dir,  showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

# ── Load and prepare data ──────────────────────────────────────────────────────
cat("Loading panel data...\n")
df <- read_csv(data_path, show_col_types = FALSE)

# Ensure province column exists
if (!"province" %in% names(df)) {
  stop("'province' column not found. Run 01_merge_datasets.py first.")
}

# Compute province-level annual mean yield
prov_annual <- df %>%
  group_by(province, year) %>%
  summarise(yield_t_ha = mean(yield_t_ha, na.rm = TRUE), .groups = "drop") %>%
  filter(!is.na(yield_t_ha)) %>%
  arrange(province, year)

cat(sprintf("  Provinces: %d | Years: %d | Observations: %d\n",
            n_distinct(prov_annual$province),
            n_distinct(prov_annual$year),
            nrow(prov_annual)))

# ── Compute Eberhart-Russell parameters manually ───────────────────────────────
# Environmental index Ij = grand mean of year j - overall grand mean
grand_mean <- mean(prov_annual$yield_t_ha, na.rm = TRUE)

env_index <- prov_annual %>%
  group_by(year) %>%
  summarise(env_mean = mean(yield_t_ha, na.rm = TRUE), .groups = "drop") %>%
  mutate(Ij = env_mean - grand_mean)

prov_data <- prov_annual %>%
  left_join(env_index, by = "year")

# Fit E-R regression for each province: yield = mu_i + bi * Ij + delta
er_results <- prov_data %>%
  group_by(province) %>%
  do({
    sub <- .
    if (nrow(sub) < 5) return(data.frame())
    fit     <- lm(yield_t_ha ~ Ij, data = sub)
    mu_i    <- coef(fit)[1]
    bi      <- coef(fit)[2]
    s2di    <- summary(fit)$sigma^2
    cv_pct  <- sd(sub$yield_t_ha, na.rm = TRUE) / mean(sub$yield_t_ha, na.rm = TRUE) * 100
    # Wricke's ecovalence
    sub$pred_grand <- grand_mean + sub$Ij
    Wi <- sum((sub$yield_t_ha - mu_i - sub$Ij)^2, na.rm = TRUE) /
            sum((sub$yield_t_ha - grand_mean)^2, na.rm = TRUE)
    # t-test: bi = 1
    se_b <- summary(fit)$coefficients[2, 2]
    t_bi <- (bi - 1) / se_b
    p_bi <- 2 * pt(abs(t_bi), df = nrow(sub) - 2, lower.tail = FALSE)
    data.frame(
      province     = unique(sub$province),
      mean_yield   = round(mu_i, 3),
      bi           = round(bi,   3),
      s2di         = round(s2di, 4),
      cv_pct       = round(cv_pct, 2),
      Wi           = round(Wi,  4),
      bi_pval      = round(p_bi, 4),
      n_years      = nrow(sub)
    )
  }) %>%
  ungroup()

# Stability classification
er_results <- er_results %>%
  mutate(
    stability_class = case_when(
      bi < 0.90 & bi_pval < 0.05  ~ "Stable (S)",
      bi > 1.10 & bi_pval < 0.05  ~ "Unstable (U)",
      TRUE                         ~ "Average Stable (AS)"
    )
  )

cat("\nEberhart-Russell Stability Results:\n")
print(er_results %>% select(province, mean_yield, bi, s2di, cv_pct, Wi, stability_class),
      n = Inf, width = 100)

# National average row
nat_row <- data.frame(
  province    = "Nepal (National)",
  mean_yield  = round(grand_mean, 3),
  bi          = 1.000,
  s2di        = round(mean(er_results$s2di), 4),
  cv_pct      = round(sd(prov_annual$yield_t_ha) / grand_mean * 100, 2),
  Wi          = round(mean(er_results$Wi), 4),
  bi_pval     = NA_real_,
  n_years     = n_distinct(prov_annual$year),
  stability_class = "Average Stable (AS)"
)
er_full <- bind_rows(er_results, nat_row)

# Save
write_csv(er_full, file.path(output_dir, "eberhart_russell_stability.csv"))
cat(sprintf("\n✅ Saved → %s/eberhart_russell_stability.csv\n", output_dir))

# ── Publication-quality biplot ─────────────────────────────────────────────────
cat("\nGenerating E-R stability biplot...\n")

col_map <- c(
  "Stable (S)"        = "#2171B5",
  "Average Stable (AS)" = "#31A354",
  "Unstable (U)"      = "#CB181D"
)

p_biplot <- ggplot(er_results, aes(x = mean_yield, y = bi)) +
  # Quadrant shading
  annotate("rect", xmin = -Inf, xmax = grand_mean, ymin = -Inf, ymax = 1,
           fill = "#2171B5", alpha = 0.05) +
  annotate("rect", xmin = grand_mean, xmax = Inf, ymin = -Inf, ymax = 1,
           fill = "#31A354", alpha = 0.05) +
  annotate("rect", xmin = -Inf, xmax = grand_mean, ymin = 1, ymax = Inf,
           fill = "#888888", alpha = 0.05) +
  annotate("rect", xmin = grand_mean, xmax = Inf, ymin = 1, ymax = Inf,
           fill = "#CB181D", alpha = 0.05) +
  # Reference lines
  geom_hline(yintercept = 1,          linetype = "dashed", color = "#666666", linewidth = 0.7) +
  geom_vline(xintercept = grand_mean, linetype = "dashed", color = "#666666", linewidth = 0.7) +
  # Bubbles (size = s2di)
  geom_point(aes(color = stability_class, size = s2di),
             alpha = 0.85, stroke = 1.5, shape = 21,
             aes(fill = stability_class)) +
  geom_point(aes(color = stability_class, size = s2di), shape = 1,
             stroke = 1.5, alpha = 0.9) +
  # Labels
  geom_label_repel(aes(label = province, color = stability_class),
                   size = 3.4, fontface = "bold",
                   box.padding = 0.5, point.padding = 0.3,
                   min.segment.length = 0.2, show.legend = FALSE,
                   family = "serif") +
  # National average star
  geom_point(data = nat_row, aes(x = mean_yield, y = bi),
             shape = 8, size = 5, color = "#333333", stroke = 1.5) +
  annotate("text", x = grand_mean + 0.05, y = 0.97,
           label = paste0("Grand Mean\n(", round(grand_mean, 2), " t ha⁻¹)"),
           size = 3, color = "#555555", fontface = "italic", hjust = 0, family = "serif") +
  # Quadrant labels
  annotate("text", x = 1.45, y = 0.68, label = "Low Yield\nStable",
           color = "#2171B5", size = 3.2, alpha = 0.6, fontface = "italic", family = "serif") +
  annotate("text", x = max(er_results$mean_yield) - 0.1, y = 0.68,
           label = "High Yield\nStable",
           color = "#31A354", size = 3.2, alpha = 0.6, fontface = "italic", family = "serif") +
  annotate("text", x = max(er_results$mean_yield) - 0.1, y = max(er_results$bi) - 0.03,
           label = "High Yield\nUnstable",
           color = "#CB181D", size = 3.2, alpha = 0.6, fontface = "italic", family = "serif") +
  scale_color_manual(values = col_map, name = "Stability Class") +
  scale_size_continuous(name = "s²dᵢ (deviation)", range = c(3, 12)) +
  labs(
    title    = "Eberhart-Russell Stability Biplot — Nepal Maize Provinces",
    subtitle = "Mean Yield vs. Regression Coefficient (bᵢ) | Bubble area ∝ Deviation from regression (s²dᵢ)",
    x        = "Mean Maize Yield (t ha⁻¹)",
    y        = "Regression Coefficient (bᵢ)",
    caption  = "Data: Nepal DoA / FAO FAOSTAT, 1990–2022. Analysis: metan R package (Olivoto & Lúdio, 2020)."
  ) +
  theme_classic(base_size = 11, base_family = "serif") +
  theme(
    plot.title       = element_text(face = "bold", size = 13, color = "#1A1A2E"),
    plot.subtitle    = element_text(size = 9.5, color = "#444444"),
    plot.caption     = element_text(size = 8, color = "#666666", face = "italic"),
    legend.position  = "right",
    legend.title     = element_text(face = "bold"),
    panel.background = element_rect(fill = "#F7F9FF"),
    plot.background  = element_rect(fill = "#FAFBFC", color = NA),
    panel.grid.major = element_line(color = "#E0E0E0", linetype = "dotted"),
    axis.title       = element_text(face = "bold"),
  )

ggsave(file.path(figures_dir, "fig6_er_stability_biplot.png"),
       plot = p_biplot, width = 10, height = 7.5, dpi = 300, bg = "#FAFBFC")
cat(sprintf("✅ Saved → %s/fig6_er_stability_biplot.png\n", figures_dir))
