"""
01_merge_datasets.py
====================
Data preparation script for the Nepal maize-climate study.
Loads, cleans, and merges all secondary datasets into a single
analysis-ready panel dataset (district × year).

Datasets:
  - Maize yield/area/production  : Nepal DoA / FAO FAOSTAT
  - Temperature (Tmax, Tmin)      : ERA5-Land (via CDS API)
  - Precipitation                 : CHIRPS v2.0
  - NDVI / EVI                    : MODIS MOD13A3 (via GEE)
  - Soil moisture                 : NASA GLDAS-2.1
  - SPI-3                         : CRU TS4.06

Output:
  data/processed/nepal_maize_panel_1990_2022.csv

Authors: Lokendra Khatri et al. (2026)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
RAW_DIR     = ROOT / "data" / "raw"
PROC_DIR    = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
START_YEAR  = 1990
END_YEAR    = 2022
N_DISTRICTS = 77

# Province → district mapping (abbreviated; expand with full shapefile data)
PROVINCE_MAP = {
    "Koshi":         ["Taplejung","Panchthar","Ilam","Jhapa","Morang","Sunsari",
                      "Dhankuta","Terhathum","Sankhuwasabha","Bhojpur","Solukhumbu",
                      "Okhaldhunga","Khotang","Udayapur"],
    "Madhesh":       ["Saptari","Siraha","Dhanusha","Mahottari","Sarlahi",
                      "Rautahat","Bara","Parsa"],
    "Bagmati":       ["Sindhuli","Ramechhap","Dolakha","Sindhupalchok","Kavrepalanchok",
                      "Lalitpur","Bhaktapur","Kathmandu","Nuwakot","Rasuwa",
                      "Dhading","Makwanpur","Chitwan"],
    "Gandaki":       ["Gorkha","Lamjung","Tanahu","Syangja","Kaski","Manang",
                      "Mustang","Myagdi","Parbat","Baglung","Nawalpur","Nawalparasi_E"],
    "Lumbini":       ["Nawalparasi_W","Rupandehi","Kapilvastu","Arghakhanchi",
                      "Gulmi","Palpa","Dang","Pyuthan","Rolpa","Eastern_Rukum",
                      "Banke","Bardiya"],
    "Karnali":       ["Dolpa","Mugu","Humla","Jumla","Kalikot","Dailekh",
                      "Jajarkot","Western_Rukum","Salyan","Surkhet"],
    "Sudurpashchim": ["Bajura","Bajhang","Darchula","Baitadi","Dadeldhura",
                      "Doti","Achham","Kailali","Kanchanpur"],
}


def load_yield_data(filepath: Path) -> pd.DataFrame:
    """
    Load and standardise Nepal DoA district-level maize yield data.
    Expected columns: district, year, yield_t_ha, area_ha, production_t
    """
    print(f"  Loading yield data from: {filepath.name}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.rename(columns={
        "yield": "yield_t_ha",
        "area":  "area_ha",
        "prod":  "production_t",
    })
    df["year"] = df["year"].astype(int)
    df = df[df["year"].between(START_YEAR, END_YEAR)].copy()

    # Remove extreme outliers (> ±3 SD per district)
    df["yield_z"] = df.groupby("district")["yield_t_ha"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    n_out = (df["yield_z"].abs() > 3).sum()
    print(f"    Flagged {n_out} outlier observations (|z| > 3)")
    df.loc[df["yield_z"].abs() > 3, "yield_t_ha"] = np.nan
    df = df.drop(columns=["yield_z"])
    return df


def load_climate_data(filepath: Path) -> pd.DataFrame:
    """
    Load district-level climate summaries (pre-extracted from gridded products).
    Expected columns: district, year, tmax_mean, tmin_mean, rain_annual,
                      spi3_mean, sm_mean, rain_season_cv
    """
    print(f"  Loading climate data from: {filepath.name}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["year"] = df["year"].astype(int)
    return df[df["year"].between(START_YEAR, END_YEAR)].copy()


def load_ndvi_data(filepath: Path) -> pd.DataFrame:
    """
    Load district-level growing-season NDVI/EVI means (MODIS MOD13A3, 2001–2022).
    Expected columns: district, year, ndvi_gs_mean, evi_gs_mean
    """
    print(f"  Loading NDVI data from: {filepath.name}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["year"] = df["year"].astype(int)
    return df[df["year"].between(2001, END_YEAR)].copy()


def add_province_column(df: pd.DataFrame) -> pd.DataFrame:
    """Map districts to provinces using PROVINCE_MAP."""
    inv_map = {d: p for p, ds in PROVINCE_MAP.items() for d in ds}
    df["province"] = df["district"].map(inv_map)
    unmapped = df[df["province"].isna()]["district"].unique()
    if len(unmapped) > 0:
        print(f"  ⚠  Unmapped districts: {list(unmapped)}")
    return df


def impute_missing(df: pd.DataFrame, target_col: str = "yield_t_ha") -> pd.DataFrame:
    """
    Simple forward-fill imputation within district groups for missing values.
    For production use, replace with MICE via R or sklearn IterativeImputer.
    """
    missing_before = df[target_col].isna().sum()
    df[target_col] = df.groupby("district")[target_col].transform(
        lambda x: x.ffill().bfill()
    )
    missing_after = df[target_col].isna().sum()
    print(f"  Imputed {missing_before - missing_after} missing values in '{target_col}'")
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional engineered features."""
    # Heat stress days proxy (degree-days above 35°C threshold)
    if "tmax_mean" in df.columns:
        df["heat_stress_idx"] = np.maximum(0, df["tmax_mean"] - 35.0)
    # Log-transform harvested area
    if "area_ha" in df.columns:
        df["log_area_ha"] = np.log1p(df["area_ha"])
    # Province dummies
    df = pd.get_dummies(df, columns=["province"], prefix="prov", drop_first=True)
    return df


def main():
    print("=" * 60)
    print("  Nepal Maize–Climate Dataset Preparation")
    print("=" * 60)

    # ── Load individual datasets ───────────────────────────────────────────
    yield_path   = RAW_DIR / "nepal_maize_yield_1990_2022.csv"
    climate_path = RAW_DIR / "nepal_climate_district_1990_2022.csv"
    ndvi_path    = RAW_DIR / "nepal_ndvi_evi_district_2001_2022.csv"

    for p in [yield_path, climate_path, ndvi_path]:
        if not p.exists():
            print(f"\n  ⚠  File not found: {p}")
            print("     Please download raw data — see README.md § Data Sources")
            print("     Creating synthetic placeholder data for demonstration...\n")

    # ── If raw files exist, run the full pipeline ─────────────────────────
    if all(p.exists() for p in [yield_path, climate_path, ndvi_path]):
        df_yield   = load_yield_data(yield_path)
        df_climate = load_climate_data(climate_path)
        df_ndvi    = load_ndvi_data(ndvi_path)

        # Merge
        df = df_yield.merge(df_climate, on=["district", "year"], how="left")
        df = df.merge(df_ndvi,    on=["district", "year"], how="left")

        df = add_province_column(df)
        df = impute_missing(df, "yield_t_ha")
        df = compute_derived_features(df)

        out_path = PROC_DIR / "nepal_maize_panel_1990_2022.csv"
        df.to_csv(out_path, index=False)
        print(f"\n✅ Saved merged panel dataset → {out_path}")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    else:
        # ── Generate reproducible synthetic dataset for demonstration ─────
        print("  Generating reproducible synthetic demonstration dataset...")
        np.random.seed(42)
        years     = np.arange(START_YEAR, END_YEAR + 1)
        districts = [f"District_{i:02d}" for i in range(1, N_DISTRICTS + 1)]
        rows = []
        for dist_i, dist in enumerate(districts):
            prov_idx = dist_i // 11
            for yr_i, yr in enumerate(years):
                # Simulate realistic values
                base_yield  = 1.5 + (dist_i % 7) * 0.22 + yr_i * 0.034
                rain        = 1620 - yr_i * 4.21 + np.random.normal(0, 85)
                tmax        = 26.0 + (dist_i % 5) * 0.6 + yr_i * 0.043
                tmin        = 13.0 + (dist_i % 5) * 0.5 + yr_i * 0.038
                spi3        = np.random.normal(0, 0.97)
                sm          = 180 + np.random.normal(0, 45)
                ndvi_gs     = 0.43 + yr_i * 0.003 + np.random.normal(0, 0.022) if yr >= 2001 else np.nan
                evi_gs      = ndvi_gs * 0.85 + np.random.normal(0, 0.015) if yr >= 2001 else np.nan
                drought_dep = max(0, -spi3) * 0.15
                yield_val   = np.clip(base_yield - drought_dep + np.random.normal(0, 0.10), 0.8, 5.0)
                rows.append({
                    "district":      dist,
                    "year":          yr,
                    "province":      list(PROVINCE_MAP.keys())[prov_idx % 7],
                    "yield_t_ha":    round(yield_val, 3),
                    "area_ha":       round(8000 + dist_i * 120 + np.random.normal(0, 300), 0),
                    "production_t":  round(yield_val * (8000 + dist_i * 120), 0),
                    "tmax_mean":     round(tmax, 2),
                    "tmin_mean":     round(tmin, 2),
                    "rain_annual":   round(rain, 1),
                    "spi3_mean":     round(spi3, 3),
                    "sm_mean":       round(sm, 2),
                    "ndvi_gs_mean":  round(ndvi_gs, 4) if not np.isnan(ndvi_gs) else np.nan,
                    "evi_gs_mean":   round(evi_gs,  4) if not np.isnan(evi_gs)  else np.nan,
                })

        df = pd.DataFrame(rows)
        df = compute_derived_features(df)
        out_path = PROC_DIR / "nepal_maize_panel_SYNTHETIC_demo.csv"
        df.to_csv(out_path, index=False)
        print(f"\n✅ Saved SYNTHETIC demo dataset → {out_path}")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print("\n  ⚠  NOTE: This is synthetic data for code demonstration only.")
        print("     Replace with real data from sources listed in README.md")


if __name__ == "__main__":
    main()
