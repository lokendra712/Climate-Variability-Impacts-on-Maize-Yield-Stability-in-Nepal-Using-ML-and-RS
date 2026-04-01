# 🌽 Data-Driven Assessment of Climate Variability Impacts on Maize Yield Stability in Nepal
### Using Machine Learning and Remote Sensing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.3%2B-276DC3?logo=r)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxxx-orange)](https://doi.org/)
[![Journal](https://img.shields.io/badge/Target%20Journal-COMPAG%20%7C%20Q1-red)](https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture)
[![Status](https://img.shields.io/badge/Status-Under%20Review-yellow)]()

> **Manuscript submitted to:** *Computers and Electronics in Agriculture* (Elsevier, Q1, IF ≈ 8.3)  
> **Authors:** Lokendra Paudel¹*, [Co-author 2]², [Co-author 3]³  
> ¹ IAAS, Tribhuvan University, Bharatpur, Chitwan, Nepal  
> *Corresponding author: lokendrapaudel@example.com

---

## 📋 Abstract

This study presents the first comprehensive, district-scale, machine learning-based assessment of climate variability impacts on maize (*Zea mays* L.) yield stability in Nepal (1990–2022). Integrating 33 years of secondary data from six multi-source datasets — Nepal DoA/FAO yield records, ERA5-Land reanalysis temperatures, CHIRPS satellite rainfall, MODIS NDVI/EVI, GLDAS-2.1 soil moisture, and CRU TS4.06 SPI-3 — five ML and statistical models were trained and evaluated using a temporal hold-out design. **Random Forest** achieved the highest test-set accuracy (R² = 0.887, RMSE = 0.248 t ha⁻¹, NSE = 0.883). SHAP analysis identified growing-season NDVI, annual rainfall, and SPI-3 as the three dominant yield predictors. Mann-Kendall trend analysis confirmed significant warming (+0.43°C decade⁻¹) and declining rainfall (−4.21 mm yr⁻¹). Eberhart-Russell stability analysis revealed a fundamental productivity-stability trade-off: Terai provinces (bᵢ > 1.25) are high-yielding but climate-sensitive, while mountain provinces (bᵢ < 0.85) are stable but low-yielding.

---

## 🗂️ Repository Structure

```
📦 maize-yield-nepal-ml/
├── 📁 data/
│   ├── raw/                    # Original data files (see Data Sources)
│   └── processed/              # Cleaned, merged analysis-ready datasets
├── 📁 code/
│   ├── 01_data_prep/           # Data download, cleaning, merging
│   ├── 02_eda/                 # Exploratory data analysis & correlation
│   ├── 03_models/              # ML model training, tuning, evaluation
│   ├── 04_shap/                # SHAP explainability analysis
│   ├── 05_stability/           # Eberhart-Russell stability analysis (R)
│   └── 06_figures/             # All publication figures (Python)
├── 📁 figures/                 # High-resolution output figures (300 DPI PNG)
├── 📁 outputs/                 # Model results, metrics tables, predictions
├── 📁 docs/                    # Manuscript draft, supplementary material
├── requirements.txt            # Python dependencies
├── requirements_R.txt          # R package dependencies
├── environment.yml             # Conda environment specification
└── README.md
```

---

## 📊 Data Sources

| Dataset | Source | Temporal Coverage | Spatial Resolution |
|---------|--------|------------------|--------------------|
| Maize yield, area, production | Nepal DoA / [FAO FAOSTAT](https://www.fao.org/faostat) | 1990–2022 | District (77) |
| Near-surface temperature | [ERA5-Land](https://cds.climate.copernicus.eu) | 1990–2022 | 0.1° × 0.1° |
| Precipitation | [CHIRPS v2.0](https://www.chc.ucsb.edu/data/chirps) | 1990–2022 | 0.05° × 0.05° |
| NDVI / EVI | [MODIS MOD13A3](https://lpdaac.usgs.gov/products/mod13a3v006/) | 2001–2022 | 1 km monthly |
| Soil moisture | [NASA GLDAS-2.1](https://ldas.gsfc.nasa.gov/gldas/) | 1990–2022 | 0.25° × 0.25° |
| SPI-3 drought index | [CRU TS4.06](https://crudata.uea.ac.uk/cru/data/hrg/) | 1990–2022 | 0.5° × 0.5° |
| District boundaries | [Nepal Survey Dept.](https://nationalgeoportal.gov.np/) | — | Polygon shapefile |

> **Note:** Raw yield data from Nepal DoA must be requested directly from the [Department of Agriculture, Nepal](https://doanepal.gov.np/). All other datasets are freely accessible via the links above.

---

## 🔬 Methods Overview

```
Raw Data ──► Data Cleaning ──► Feature Engineering ──► EDA & Trends
                                                              │
                                             Mann-Kendall + Sen's Slope
                                                              │
                          ┌───────────────────────────────────┘
                          ▼
              ML Model Comparison (80:20 temporal split)
              ┌──────────┬──────────┬─────────┬─────────┬─────────┐
              │   RF     │   GB     │   SVR   │  Lasso  │   OLS   │
              └──────────┴──────────┴─────────┴─────────┴─────────┘
                          │
                   Best Model (RF)
                    ├── SHAP Feature Importance
                    └── SHAP Dependence Plots
                          │
              Eberhart-Russell Stability Analysis (R)
                          │
                   Policy Recommendations
```

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/maize-yield-nepal-ml.git
cd maize-yield-nepal-ml
```

### 2. Set up Python environment
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate maize-nepal

# Or using pip
pip install -r requirements.txt
```

### 3. Set up R environment
```r
# In R console
install.packages(c("metan", "trend", "SPEI", "mice", "ggplot2", "sf", "tidyverse"))
```

### 4. Run the full pipeline
```bash
# Step 1: Data preparation
python code/01_data_prep/01_merge_datasets.py
python code/01_data_prep/02_feature_engineering.py

# Step 2: EDA and trend analysis
python code/02_eda/01_descriptive_stats.py
Rscript code/02_eda/02_mann_kendall_trends.R

# Step 3: ML model training and evaluation
python code/03_models/01_train_all_models.py
python code/03_models/02_evaluate_models.py

# Step 4: SHAP analysis
python code/04_shap/01_shap_values.py
python code/04_shap/02_shap_dependence_plots.py

# Step 5: Stability analysis (R)
Rscript code/05_stability/01_eberhart_russell.R

# Step 6: Generate all figures
python code/06_figures/generate_all_figures.py
```

---

## 📈 Key Results

| Model | Test R² | RMSE (t ha⁻¹) | NSE | PBIAS (%) |
|-------|---------|---------------|-----|-----------|
| **Random Forest** ⭐ | **0.887** | **0.248** | **0.883** | **2.1** |
| Gradient Boosting | 0.874 | 0.264 | 0.869 | 2.4 |
| Support Vector Regression | 0.821 | 0.314 | 0.814 | 3.6 |
| Lasso Regression | 0.784 | 0.349 | 0.778 | 4.2 |
| OLS Regression | 0.741 | 0.382 | 0.733 | 5.3 |

### Top Feature Importances (SHAP)
1. 🌿 Growing-Season NDVI — 0.312 t ha⁻¹
2. 🌧️ Annual Rainfall — 0.284 t ha⁻¹
3. 🏜️ SPI-3 Drought Index — 0.251 t ha⁻¹
4. 🌡️ Max. Temperature — 0.229 t ha⁻¹
5. 💧 Soil Moisture — 0.198 t ha⁻¹

### Climate Trends (Mann-Kendall, 1990–2022)
| Variable | Sen's Slope | p-value |
|----------|------------|---------|
| Max. Temperature | +0.43°C decade⁻¹ | < 0.001 |
| Min. Temperature | +0.38°C decade⁻¹ | < 0.001 |
| Annual Rainfall | −4.21 mm yr⁻¹ | 0.004 |
| Maize Yield | +0.034 t ha⁻¹ yr⁻¹ | < 0.001 |

---

## 🗺️ Figures

| Figure | Description |
|--------|-------------|
| [Fig. 1](figures/fig1_yield_trend.png) | National yield trend with rainfall & NDVI (1990–2022) |
| [Fig. 2](figures/fig2_spatial_map.png) | Spatial distribution of mean yield & CV across 77 districts |
| [Fig. 3](figures/fig3_sens_slope.png) | District-level Sen's slope maps (Tmax & rainfall) |
| [Fig. 4](figures/fig4_obs_pred.png) | Observed vs. predicted yield — RF vs. OLS |
| [Fig. 5](figures/fig5_shap.png) | SHAP beeswarm summary plot |
| [Fig. 6](figures/fig6_stability_biplot.png) | Eberhart-Russell stability biplot |

---

## 🛠️ Requirements

### Python (≥ 3.10)
See `requirements.txt` for full list. Core packages:
- `scikit-learn`, `xgboost`, `shap`
- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`
- `geopandas`, `rasterio`, `earthengine-api`

### R (≥ 4.3)
- `metan`, `trend`, `SPEI`, `mice`
- `tidyverse`, `sf`, `ggplot2`, `tmap`

---

## 📄 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{paudel2025maize,
  title   = {Data-Driven Assessment of Climate Variability Impacts on Maize Yield
             Stability in Nepal Using Machine Learning and Remote Sensing},
  author  = {Paudel, Lokendra and [Co-author 2] and [Co-author 3]},
  journal = {Computers and Electronics in Agriculture},
  year    = {2025},
  volume  = {XXX},
  pages   = {XXXXXX},
  doi     = {10.xxxx/xxxxxx}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Nepal Department of Agriculture (DoA) for yield statistics
- NASA Earthdata for MODIS MOD13A3 products
- ECMWF / Copernicus for ERA5-Land reanalysis data
- CHC-UCSB for CHIRPS v2.0 precipitation data
- Google Earth Engine platform for remote sensing data processing

---

## 📬 Contact

**Lokendra Paudel**  
Institute of Agriculture and Animal Science (IAAS), Tribhuvan University  
Bharatpur, Chitwan, Nepal  
📧 lokendrapaudel@example.com  
🔗 [GitHub](https://github.com/YOUR_USERNAME) | [ResearchGate](https://www.researchgate.net)
