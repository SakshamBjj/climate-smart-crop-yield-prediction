# Data Pipeline & Feature Engineering

*Deep-dive companion to [README.md](../README.md). Covers source processing, spatial/temporal harmonization, feature engineering, and data quality decisions.*

---

## The Core Problem

Three data sources. Three incompatible formats. Harmonizing them without leaking future information or introducing spatial bias was the majority of the engineering work.

| Source | Format | Resolution | Challenge |
|--------|--------|------------|-----------|
| ICRISAT TCI | Tabular CSV | Annual, district boundaries | Irregular polygons, missing years |
| NASA POWER API | Gridded lat-lon | Daily, 0.5° × 0.5° (~55km) | Grid cells don't match district boundaries |
| ISRO VEDAS | Raster imagery | 16-day composites, 500m | 15% cloud contamination during monsoon |

**System contribution:** The novel component is not individual techniques (zonal statistics, GDD calculation) but the integrated pipeline that automates alignment across these sources while maintaining temporal validity and spatial precision.

---

## Data Sources

### 1. ICRISAT TCI (Target Variable)

*Purpose: establish the yield target and filter to a clean, bias-free district sample.*

**Coverage:** 300 districts, 20 states, 2008–2017 (filtered from 571 districts, 1966–2020)  
**Key fields:** District_Code, Year, Crop, Area, Production, **Yield (kg/ha)** ← target

**Quality issues:**
- District boundary changes across census years — handled via 1991/2001 baseline apportionment
- ~8% missing values (sparse coverage for some districts/years)
- Yield outliers >10,000 kg/ha flagged as likely reporting errors

**Preprocessing steps:**
```
1. Filter: 2008–2017, crops = {Rice, Wheat, Maize}
2. Drop districts with >30% missing years
3. Impute remaining missing yields via spatial IDW (3 nearest districts)
4. Final: 300 districts × 10 years × 3 crops = ~7,200 complete records
```

---

### 2. NASA POWER API (Climate)

*Purpose: derive physiologically meaningful climate aggregates — not raw temperatures, but crop-development metrics.*

**Coverage:** Global gridded at 0.5° × 0.5°  
**Resolution:** Daily → aggregated to growing season  
**Key parameters:** T2M (temperature), PRECTOTCORR (precipitation), RH2M (humidity), PS (pressure)

**Derived features:**
- **GDD (Growing Degree Days):** `Σ max(0, (T_max + T_min)/2 − T_base)` over growing season. Crop-specific T_base: Wheat = 0°C, Rice/Maize = 10°C
- **CDD18_3:** Cooling degree days — heat stress index during flowering
- **Temp_Range:** T2M_MAX − T2M — diurnal variation as water stress indicator

**Preprocessing steps:**
```
1. Query: For each district centroid (lat, lon), fetch daily data 2008–2017
2. Temporal aggregation: Daily → growing season aggregate
   - T2M: Mean during growing season window
   - PRECTOT: Cumulative sum during growing season
   - GDD: Cumulative sum during growing season
3. Spatial alignment: Grid → district boundaries (see Spatial Harmonization below)
```

---

### 3. ISRO VEDAS (Satellite Vegetation Indices)

*Purpose: recover crop health signal from satellite data despite monsoon cloud contamination at the worst possible time.*

**Coverage:** India, 500m × 500m  
**Resolution:** 16-day composites → aggregated to annual  
**Key indices:**
- **NDVI:** `(NIR − Red) / (NIR + Red)` — vegetation health (−1 to +1, higher = healthier)
- **VCI (%):** `100 × (NDVI − NDVI_min) / (NDVI_max − NDVI_min)` — relative to historical range

**Preprocessing steps:**
```
1. Download 230 16-day composites per district (23/year × 10 years)
2. Cloud masking: Remove pixels with cloud cover > 30%
3. Spatial aggregation: 500m pixels → district-level mean (zonal statistics)
4. Temporal aggregation: 16-day → growing season mean (NDVI) + min at grain-filling (VCI)
5. Gap filling: Cloud cover > 50% → Maximum Value Composite (see below)
```

**The cloud contamination problem:** 15% of monsoon-season composites are unusable. Monsoon (June–September) is the most important period for Kharif crop NDVI signal — the data is missing exactly when it matters most.

**Solution — Maximum Value Composite (MVC):**
- For each 16-day period, take the highest NDVI across that and adjacent periods
- Rationale: Clouds suppress NDVI to near-zero; vegetation maxima approximate the cloud-free value
- Validation: −45 kg/ha RMSE vs simple mean imputation

---

## Spatial Harmonization

**Why this matters:** Three data sources use three different spatial formats. Naive alignment (district centroid → nearest grid cell) loses signal for large, climatically diverse districts. Area-weighted aggregation captures within-district climate variation — validated at 12% RMSE improvement. Getting this wrong propagates into every feature.

**Problem:** NASA POWER uses a regular 0.5° lat-lon grid. ICRISAT uses irregular administrative district polygons. MODIS pixels are 500m squares. None align.

**Implementation — Zonal Statistics:**
```python
# Pseudocode (implementation proprietary)
for each district:
    boundary = load_shapefile(district_id)

    # NASA climate: area-weighted average of overlapping grid cells
    cells = find_intersecting_cells(boundary, nasa_grid)
    weights = area_overlap_fraction(boundary, cells)
    climate_value = weighted_average(cells, weights)

    # MODIS satellite: mean of all pixels falling within boundary
    pixels = extract_pixels_within(boundary, modis_raster_500m)
    ndvi_value = mean(pixels, ignore_nodata=True)
```

**Boundary size rule:**
- Districts < 2,500 km² → use nearest NASA grid cell centroid
- Districts ≥ 2,500 km² → area-weighted average of 4 nearest cells

**Validation:** Area-weighted aggregation improved RMSE by 12% vs centroid-only approach.

**Tools:** GeoPandas (shapefiles), Rasterio (raster extraction), GDAL (spatial operations)

---

## Temporal Alignment

**Why this matters:** Matching daily climate data to annual yield is not a simple aggregation — the window must match each crop's biology. Using a calendar year instead of a crop year misattributes pre-season climate to the wrong harvest. Using the same window for all crops ignores that wheat grows in winter while rice grows in monsoon. These errors compound into systematic regional bias.

**Problem:** Daily climate + 16-day satellite composites + annual yield must all be matched to the same crop year.

**Implementation — Crop-Specific Growing Season Windows:**
```
Kharif crops (Rice, Maize): June Year_N → October Year_N
Rabi crops (Wheat):         November Year_N → March Year_(N+1)

For each crop year:
  - GDD: Cumulative sum within crop's window
  - PRECTOT: Cumulative sum within crop's window
  - NDVI: Mean during peak growth phase within window
  - VCI: Minimum during grain-filling phase within window

Match all aggregates to yield reported for harvest year.
```

**Example (wheat 2015):**
```
Yield: reported at harvest March 2016
Climate aggregates: November 2015 → March 2016
Satellite indices: November 2015 → March 2016 composites
```

**Known limitation:** Annual aggregates lose sub-seasonal timing. Late monsoon onset (July vs June) can reduce Kharif yields by 30%, but that signal is averaged out. Monthly sequences would capture it — documented in [EVALUATION.md](EVALUATION.md) as the highest-priority improvement.

---

## Feature Engineering

### Final Feature Vector (14 features)

**Climate (11):**

| Feature | Type | Notes |
|---------|------|-------|
| GDD | Derived | Crop-specific base temp — most important feature (32%) |
| CDD18_3 | Derived | Cooling degree days — heat stress at flowering |
| PRECTOTCORR | Raw | Cumulative precipitation during growing season |
| T2M | Raw | Mean temperature during growing season |
| T2M_MAX | Raw | Maximum temperature |
| Temp_Range | Derived | T2M_MAX − T2M (diurnal variation = water stress) |
| RH2M | Raw | Mean relative humidity |
| PS | Raw | Surface pressure |
| QV2M | Raw | Specific humidity |
| T2MDEW | Raw | Dew point temperature |

**Vegetation (1):**
- NDVI mean during growing season — second most important (18%)

**Geospatial (2):**
- Latitude — agro-climatic zone proxy (north/south)
- Longitude — east-west monsoon gradient proxy

---

### GDD Calculation Detail

```
For each day in growing season:
    T_avg = (T_max + T_min) / 2
    GDD_day = max(0, T_avg − T_base)

Annual_GDD = Σ GDD_day
```

| Crop | T_base | Rationale |
|------|--------|-----------|
| Wheat | 0°C | Winter crop; grows through near-freezing temps |
| Rice | 10°C | Tropical crop; growth ceases below 10°C |
| Maize | 10°C | Warm-season crop; same threshold as rice |

**Validation against agronomic benchmarks:**
- Punjab wheat expected: 2,200–2,400 °C-days
- Observed mean: 2,315 °C-days ✓

Generic T_base = 10°C for all crops would have reduced wheat prediction accuracy by 28 kg/ha RMSE.

---

## Data Quality Control

### Outlier Handling

**Method:** IQR flagging (flag, not remove — outliers are informative for failure analysis)

```
Outlier threshold: < Q1 − 1.5×IQR  OR  > Q3 + 1.5×IQR
```

**Flagged cases:**
- Yield > 8,000 kg/ha for wheat → likely reporting error
- Negative NDVI → water body misclassified as cropland
- GDD > 5,000 °C-days → tropical districts with year-round cropping

### Missing Value Summary

| Source | Missing % | Strategy |
|--------|-----------|----------|
| ICRISAT Yield | 8% | Spatial IDW from 3 nearest districts |
| NASA Climate | 2% | Linear temporal interpolation |
| MODIS NDVI | **15%** | **Maximum Value Composite** |

23 districts dropped entirely: >30% missing years, insufficient temporal coverage for reliable training.

---

## Final Unified Dataset

```
District_Code | Year | Crop | Latitude | Longitude | T2M | PRECTOT | GDD | NDVI | VCI | ... | Yield
```

| Dimension | Value |
|-----------|-------|
| Total samples | 7,200 |
| Features | 14 |
| Target | Yield (kg/ha) |
| Training set | 5,760 (2008–2015) |
| Test set | 1,440 (2016–2017) |

---

## Documented Limitations

- **Temporal resolution:** Annual aggregates lose sub-seasonal timing (late monsoon onset, flowering-stage heat stress)
- **Spatial independence:** Districts treated independently — spatial autocorrelation ignored
- **Cloud contamination:** MVC reduces gap rate from 15% → 8%, not to zero
- **No field-level validation:** Ground truth available only at district level

*See [EVALUATION.md](EVALUATION.md) for quantified impact of each limitation and proposed fixes.*

---

## Patent vs Standard Practice

**Patent-protected (system integration):**
- Automated three-source harmonization pipeline (ICRISAT + NASA + ISRO)
- Crop-specific temporal alignment (Kharif vs Rabi windowing)
- Multi-resolution spatial aggregation (500m pixels → irregular district polygons)
- Scalable architecture (300 → 5,000+ districts without redesign)

**Standard practice (not patent claims):**
- Zonal statistics (GDAL standard)
- GDD calculation (established agronomic method)
- Maximum Value Composite (common in remote sensing)
- RF and XGBoost algorithms (open-source)

*The patent covers the integrated system architecture, not individual techniques.*
