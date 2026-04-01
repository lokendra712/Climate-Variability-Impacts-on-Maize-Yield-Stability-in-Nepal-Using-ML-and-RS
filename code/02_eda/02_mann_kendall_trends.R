# =============================================================================
# 02_mann_kendall_trends.R
# =============================================================================
# Non-parametric Mann-Kendall trend analysis with Sen's slope estimation
# for all climate variables and maize yield.
#
# Methods:
#   - Pre-whitening to remove serial autocorrelation (Zhang et al. 2001)
#   - Mann-Kendall test (Mann 1945; Kendall 1975)
#   - Sen's slope estimator (Sen 1968)
#
# Output:
#   outputs/mk_trend_results.csv
#   outputs/mk_trend_results.html  (formatted table)
#
# Authors: Lokendra Paudel et al. (2025)
# =============================================================================

library(trend)       # Mann-Kendall + Sen's slope
library(tidyverse)
library(knitr)
library(kableExtra)

# ── Paths ─────────────────────────────────────────────────────────────────────
root_dir    <- here::here()
data_path   <- file.path(root_dir, "data", "processed",
                         "nepal_maize_panel_SYNTHETIC_demo.csv")
output_dir  <- file.path(root_dir, "outputs")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading panel data...\n")
df <- read_csv(data_path, show_col_types = FALSE)

# Compute national annual means for trend analysis
national_ts <- df %>%
  group_by(year) %>%
  summarise(
    yield_t_ha   = mean(yield_t_ha,   na.rm = TRUE),
    rain_annual  = mean(rain_annual,  na.rm = TRUE),
    tmax_mean    = mean(tmax_mean,    na.rm = TRUE),
    tmin_mean    = mean(tmin_mean,    na.rm = TRUE),
    spi3_mean    = mean(spi3_mean,    na.rm = TRUE),
    sm_mean      = mean(sm_mean,      na.rm = TRUE),
    ndvi_gs_mean = mean(ndvi_gs_mean, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(year)

cat(sprintf("  National time series: %d years (%d–%d)\n",
            nrow(national_ts),
            min(national_ts$year),
            max(national_ts$year)))

# ── Pre-whitening function ─────────────────────────────────────────────────────
prewhiten <- function(x) {
  # Remove lag-1 autocorrelation before MK test (Zhang et al. 2001)
  x_clean <- na.omit(x)
  if (length(x_clean) < 4) return(x)
  ar1 <- cor(x_clean[-length(x_clean)], x_clean[-1], use = "complete.obs")
  if (abs(ar1) > 0.05) {
    x_pw <- x_clean[-1] - ar1 * x_clean[-length(x_clean)]
    return(x_pw)
  }
  return(x_clean)
}

# ── Run MK + Sen's slope for each variable ─────────────────────────────────────
variables <- list(
  list(name = "Annual Rainfall (mm yr⁻¹)",         col = "rain_annual",  unit = "mm yr⁻¹"),
  list(name = "Max Temperature (°C decade⁻¹)",      col = "tmax_mean",    unit = "°C decade⁻¹"),
  list(name = "Min Temperature (°C decade⁻¹)",      col = "tmin_mean",    unit = "°C decade⁻¹"),
  list(name = "Maize Yield (t ha⁻¹ yr⁻¹)",         col = "yield_t_ha",   unit = "t ha⁻¹ yr⁻¹"),
  list(name = "SPI-3 Index",                         col = "spi3_mean",    unit = "per yr"),
  list(name = "Soil Moisture (kg m⁻² yr⁻¹)",       col = "sm_mean",      unit = "kg m⁻² yr⁻¹"),
  list(name = "NDVI (growing season, 2001–2022)",    col = "ndvi_gs_mean", unit = "per yr")
)

results <- lapply(variables, function(v) {
  x_raw <- national_ts[[v$col]]
  if (all(is.na(x_raw))) return(NULL)

  # Pre-whiten
  x_pw <- prewhiten(x_raw)

  # Mann-Kendall test
  mk   <- mk.test(x_pw)
  # Sen's slope (on original series)
  ss   <- sens.slope(na.omit(x_raw))

  sig <- ifelse(mk$p.value < 0.001, "***",
         ifelse(mk$p.value < 0.01,  "**",
         ifelse(mk$p.value < 0.05,  "*", "ns")))

  direction <- ifelse(ss$estimates > 0, "Increasing", "Decreasing")

  data.frame(
    Variable      = v$name,
    Sens_Slope    = round(ss$estimates, 4),
    MK_Z_stat     = round(mk$statistic, 3),
    p_value       = ifelse(mk$p.value < 0.001, "< 0.001", round(mk$p.value, 3)),
    Direction     = direction,
    Unit          = v$unit,
    Significance  = sig,
    stringsAsFactors = FALSE
  )
})

results_df <- bind_rows(Filter(Negate(is.null), results))
cat("\nMann-Kendall Trend Analysis Results:\n")
print(results_df, row.names = FALSE)

# ── Save outputs ───────────────────────────────────────────────────────────────
write_csv(results_df, file.path(output_dir, "mk_trend_results.csv"))
cat(sprintf("\n✅ Saved → %s/mk_trend_results.csv\n", output_dir))

# HTML table
results_df %>%
  kbl(caption = "Table 6. Mann-Kendall Trend Statistics and Sen's Slope (1990–2022)",
      align = c("l","r","r","r","l","l","c")) %>%
  kable_styling(bootstrap_options = c("striped","hover","condensed"),
                full_width = FALSE) %>%
  column_spec(7, bold = TRUE,
              color = ifelse(results_df$Significance %in% c("***","**","*"),
                             "darkgreen", "gray")) %>%
  save_kable(file.path(output_dir, "mk_trend_results.html"))

cat("✅ Saved HTML table → outputs/mk_trend_results.html\n")
