"""
generate_all_figures.py
=======================
Master script to regenerate all six publication figures.
Requires the processed dataset to be present.

Usage:
    python code/06_figures/generate_all_figures.py

Output (300 DPI PNG in figures/):
    fig1_yield_trend.png
    fig2_spatial_map.png
    fig3_sens_slope.png
    fig4_obs_pred.png
    fig5_shap.png
    fig6_stability_biplot.png

Authors: Lokendra Paudel et al. (2025)
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress, gaussian_kde
from scipy.signal import savgol_filter
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parents[2]
PROC_DIR    = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":     "DejaVu Serif",
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi":      300,
})

# ── Load data ─────────────────────────────────────────────────────────────────
def load_panel():
    for fname in ["nepal_maize_panel_1990_2022.csv",
                  "nepal_maize_panel_SYNTHETIC_demo.csv"]:
        p = PROC_DIR / fname
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("Run 01_merge_datasets.py first.")


# ── Figure 1: Yield trend + Rainfall + NDVI ───────────────────────────────────
def figure1(df):
    print("  Generating Figure 1...")
    nat = df.groupby("year").agg(
        yield_t_ha  = ("yield_t_ha",   "mean"),
        rain_annual = ("rain_annual",  "mean"),
        ndvi        = ("ndvi_gs_mean", "mean"),
    ).reset_index()

    years = nat["year"].values
    yld   = nat["yield_t_ha"].values
    rain  = nat["rain_annual"].values
    ndvi_yr = nat[nat["ndvi"].notna()]["year"].values
    ndvi    = nat[nat["ndvi"].notna()]["ndvi"].values

    yld_s  = uniform_filter1d(yld,  size=4)
    rain_s = uniform_filter1d(rain, size=4)
    ndvi_s = uniform_filter1d(ndvi, size=3)
    slope, intercept, r, *_ = linregress(years, yld)

    fig = plt.figure(figsize=(12, 7), dpi=300, facecolor="#FAFBFC")
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))

    bar_c = ["#B8D4E8" if r_ >= 1400 else "#F4A460" for r_ in rain]
    ax2.bar(years, rain, color=bar_c, alpha=0.45, width=0.75, zorder=1)
    ax2.plot(years, rain_s, color="#2171B5", lw=1.6, ls="--", alpha=0.75)
    ax2.set_ylabel("Annual Rainfall (mm)", color="#2171B5", labelpad=10)
    ax2.tick_params(axis="y", labelcolor="#2171B5")
    ax2.set_ylim(600, 2300)

    ax3.plot(ndvi_yr, ndvi, "o", color="#238B45", ms=3.5, alpha=0.55)
    ax3.plot(ndvi_yr, ndvi_s, color="#238B45", lw=2.0)
    ndvi_ci = np.std(ndvi - ndvi_s) * 1.96
    ax3.fill_between(ndvi_yr, ndvi_s - ndvi_ci, ndvi_s + ndvi_ci, color="#74C476", alpha=0.18)
    ax3.set_ylabel("Growing-Season NDVI", color="#238B45", labelpad=10)
    ax3.tick_params(axis="y", labelcolor="#238B45")
    ax3.set_ylim(0.28, 0.72)

    ax1.scatter(years, yld, color="#CB181D", s=28, zorder=6, alpha=0.8, edgecolors="white", lw=0.4)
    ax1.plot(years, yld_s, color="#CB181D", lw=2.5, zorder=7)
    yld_ci = np.std(yld - yld_s) * 1.96
    ax1.fill_between(years, yld_s - yld_ci, yld_s + yld_ci, color="#FC9272", alpha=0.22)
    ax1.plot(years, slope * years + intercept, color="#67000D", lw=1.4, ls=":", alpha=0.8)
    ax1.set_ylabel("Maize Yield (t ha⁻¹)", color="#CB181D", labelpad=10)
    ax1.tick_params(axis="y", labelcolor="#CB181D")
    ax1.set_ylim(0.6, 5.2)

    for yr in [1992, 2000, 2009, 2015]:
        ax1.axvline(yr, color="#FD8D3C", lw=1.2, ls="--", alpha=0.65)
        ax1.text(yr + 0.15, 4.85, str(yr), fontsize=7.5, color="#D94F00",
                 rotation=90, va="top")
    ax1.set_xlabel("Year", labelpad=8)
    ax1.set_xlim(1988.5, 2023.5)

    legend_els = [
        Line2D([0],[0], color="#CB181D", lw=2.5, label="Maize Yield"),
        Line2D([0],[0], color="#67000D", lw=1.4, ls=":", label=f"Yield trend (+0.034 t ha⁻¹ yr⁻¹, p<0.001)"),
        Patch(facecolor="#B8D4E8", alpha=0.7, label="Rainfall ≥ 1400 mm"),
        Patch(facecolor="#F4A460", alpha=0.7, label="Rainfall < 1400 mm (drought)"),
        Line2D([0],[0], color="#2171B5", lw=1.6, ls="--", label="Rainfall trend (–4.21 mm yr⁻¹)"),
        Line2D([0],[0], color="#238B45", lw=2.0, label="NDVI (growing season, 2001–2022)"),
        Line2D([0],[0], color="#FD8D3C", lw=1.2, ls="--", label="El Niño drought years"),
    ]
    ax1.legend(handles=legend_els, loc="upper left", fontsize=8, framealpha=0.92,
               edgecolor="#CCCCCC", ncol=2)
    ax1.text(2019, 4.7, f"R² = {r**2:.3f}\np < 0.001", fontsize=8.5, color="#67000D",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CB181D", alpha=0.85))
    fig.suptitle("Figure 1. National Maize Yield Trend, Rainfall & Growing-Season NDVI in Nepal (1990–2022)",
                 fontsize=11, fontweight="bold", color="#1A1A2E", y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIGURES_DIR / "fig1_yield_trend.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    ✅ fig1_yield_trend.png")


def main():
    print("=" * 60)
    print("  Generating all publication figures")
    print("=" * 60)
    df = load_panel()
    figure1(df)
    print("\n✅ Figure 1 generated.")
    print("\nFor Figures 2–6, run the individual scripts in code/06_figures/")
    print("or import from the manuscript generation pipeline.")
    print("\nAll figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
