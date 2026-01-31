# Data Pipeline & Feature Engineering

## Overview

The core technical challenge: **harmonizing three heterogeneous data sources** with different temporal resolutions, spatial formats, and collection methodologies.

---

## Data Sources

### 1. ICRISAT TCI Database (Target Variable)
- **What:** District-level crop yield statistics for India
- **Coverage:** 300 districts, 20 states, 2008-2017 (filtered from 571 districts, 1966-2020)
- **Resolution:** Annual, administrative boundaries
- **Key fields:** District_Code, Year, Crop, Area, Production, **Yield (kg/ha)** ← target variable

**Quality Issues:**
- District boundary changes (handled via census baseline apportionment)
- Missing years for some districts (~8% missing values)
- Outliers from reporting errors (yields > 10,000 kg/ha flagged)

**Preprocessing:**
```
1. Filter: 2008-2017, major crops (Rice, Wheat, Maize)
2. Remove districts with >30% missing years
3. Impute missing yields using spatial neighbors (inverse distance weighting)
4. Final: 300 districts × 10 years × 3 crops = ~7,200 complete records
```

---

### 2. NASA POWER API (Climate Data)
- **What:** Meteorological parameters from NASA's POWER project
- **Coverage:** Global, gridded at 0.5° × 0.5° (~55 km at equator)
- **Resolution:** **Daily** (aggregated to annual)
- **Key parameters:** T2M (temperature), PRECTOTCORR (precipitation), RH2M (humidity), PS (pressure)

**Derived Features:**
- **GDD (Growing Degree Days):** `sum(max(0, (T_max + T_min)/2 - T_base))` for growing season
  - Crop-specific T_base: Wheat = 0°C, Rice/Maize = 10°C
- **CDD18_3:** Cooling Degree Days (heat stress during flowering)
- **Temp_Range:** T2M_MAX - T2M (diurnal variation = water stress indicator)

**Preprocessing:**
```
1. API query: For each district centroid (lat, lon), fetch daily data (2008-2017)
2. Temporal aggregation: Daily → Annual
   - T2M: Mean during growing season
   - PRECTOT: Sum during growing season
   - GDD: Cumulative sum during growing season
3. Spatial alignment: Grid centroids → District boundaries
   - Small districts (<2,500 km²): Use nearest grid cell
   - Large districts (>2,500 km²): Weighted average of 4 nearest cells
```

---

### 3. ISRO VEDAS (Satellite Vegetation Indices)
- **What:** Vegetation health indices from MODIS satellite
- **Coverage:** India, 500m × 500m spatial resolution
- **Resolution:** **16-day composites** (aggregated to annual)
- **Key indices:**
  - **NDVI:** (NIR - Red) / (NIR + Red), range -1 to +1 (higher = healthier vegetation)
  - **VCI (%):** `100 × (NDVI - NDVI_min) / (NDVI_max - NDVI_min)` (relative to historical range)

**Preprocessing:**
```
1. Download 16-day NDVI composites (23 per year × 10 years = 230 images per district)
2. Cloud masking: Remove pixels with cloud cover > 30%
3. Spatial aggregation: 500m pixels → District-level mean (zonal statistics)
4. Temporal aggregation: 16-day → Annual
   - NDVI: Mean of growing season composites
   - VCI: Minimum during critical growth stages (flowering, grain filling)
5. Gap filling: If cloud cover > 50% → Maximum Value Composite (MVC)
```

**Challenge:** NDVI has high cloud contamination during monsoon (June-September)

**Solution: Maximum Value Composite (MVC)**
- For each 16-day period, take **highest NDVI** (clouds have low NDVI)
- If still insufficient → borrow from adjacent 16-day periods
- Validation: Reduced RMSE by **45 kg/ha** vs. simple mean imputation

---

## Data Harmonization Strategy

### Temporal Alignment

**Problem:** Daily climate + 16-day satellite + annual yield

**Solution: Crop-Specific Growing Season Windowing**
```
1. Define "crop year":
   - Kharif crops (Rice, Maize): June Year_N to October Year_N
   - Rabi crops (Wheat): November Year_N to March Year_(N+1)

2. Align climate data:
   - Sum GDD during crop-specific growing season
   - Sum precipitation during growing season
   - Mean temperature during growing season

3. Align satellite data:
   - Mean NDVI during peak growth stage (heading/flowering)
   - Min VCI during stress-sensitive stage (grain filling)

4. Match to annual yield:
   - Yield reported in Year_N → Use climate/satellite from Year_N growing season
```

**Example:**
```
Wheat yield 2015 (reported for harvest in March 2016)
    ← Climate: Nov 2015 - Mar 2016 aggregates
    ← Satellite: Nov 2015 - Mar 2016 NDVI/VCI
```

**Why this matters:** Late monsoon onset (July vs June) can reduce yields by 30%, but annual aggregates lose this signal. This is a **documented limitation** (see EVALUATION.md).

---

### Spatial Harmonization

**Problem:** District boundaries (irregular polygons) vs NASA grid (regular lat-lon) vs MODIS pixels (500m grid)

**Solution: Zonal Statistics**
```python
# Pseudocode (actual implementation proprietary)
for each district:
    district_boundary = load_shapefile(district_id)
    
    # NASA climate: Find overlapping grid cells
    overlapping_cells = find_grids_intersecting(district_boundary, nasa_grid)
    weights = calculate_area_overlap(district_boundary, overlapping_cells)
    climate_value = weighted_average(overlapping_cells, weights)
    
    # MODIS satellite: Extract pixels within boundary
    pixels_in_district = extract_pixels(district_boundary, modis_raster)
    ndvi_mean = mean(pixels_in_district, na.rm=TRUE)
    
    # Store aligned record
    aligned_data[district_id] = {
        'climate': climate_value,
        'ndvi': ndvi_mean,
        'yield': icrisat_yield[district_id]
    }
```

**Tools:** GeoPandas (shapefile), Rasterio (raster extraction), GDAL (spatial operations)

---

## Feature Engineering

### Final Feature Vector (14 features)

**Climate (11 features):**
- T2M (mean temperature during growing season)
- PRECTOTCORR (total precipitation during growing season)
- RH2M (mean relative humidity)
- PS (surface pressure)
- QV2M (specific humidity)
- T2MDEW (dew point temperature)
- T2M_MAX (maximum temperature)
- **GDD (Growing Degree Days)** – Most important (32% feature importance)
- CDD18_3 (Cooling Degree Days, heat stress index)
- **Temp_Range (T2M_MAX - T2M)** – Diurnal variation indicates water stress

**Vegetation (1 feature):**
- **NDVI** (mean during growing season) – Second most important (18% feature importance)

**Geospatial (2 features):**
- Latitude (proxy for agro-climatic zone)
- Longitude (east-west monsoon gradient)

### Derived Feature: Growing Degree Days (GDD)

**Formula:**
```
For each day in growing season:
    T_avg = (T_max + T_min) / 2
    GDD_day = max(0, T_avg - T_base)
    
Annual_GDD = sum(GDD_day for day in growing_season)
```

**Crop-Specific Base Temperatures:**
- Rice: T_base = 10°C
- Wheat: T_base = 0°C
- Maize: T_base = 10°C

**Validation:** Compared computed GDD against agronomic benchmarks
- Punjab wheat: Expected 2,200-2,400 °C-days
- Observed mean: 2,315 °C-days ✓

**Why this matters:** GDD accounts for **32% of predictive power** (feature importance analysis). It's a biologically meaningful metric, not just a temperature average.

---

## Data Quality Control

### Outlier Detection

**Method:** Interquartile Range (IQR) flagging

```
For each feature:
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    Outliers: Values < Q1 - 1.5×IQR  OR  > Q3 + 1.5×IQR
```

**Outliers Flagged (not removed):**
- Extreme yields (>8,000 kg/ha for wheat → likely reporting error)
- Negative NDVI (water bodies misclassified as cropland)
- GDD > 5,000°C-days (tropical districts with year-round cropping)

### Missing Value Handling

| Source | Missing % | Strategy |
|--------|-----------|----------|
| ICRISAT Yield | 8% | Spatial interpolation (IDW from 3 nearest districts) |
| NASA Climate | 2% | Temporal interpolation (linear between adjacent days) |
| MODIS NDVI | **15%** | **Maximum Value Composite (MVC)** to reduce cloud impact |

**Districts Dropped:** 23 districts with >30% missing years (insufficient temporal coverage)

---

## Final Unified Dataset

**Schema:**
```
District_Code | Year | Crop | Latitude | Longitude | T2M | PRECTOT | GDD | NDVI | VCI | ... | Yield
```

**Dimensions:**
- **Samples:** 7,200 (300 districts × 10 years × 3 crops, after cleaning)
- **Features:** 14 (11 climate + 1 vegetation + 2 geospatial)
- **Target:** Yield (kg/ha)

**Train/Test Split:**
- **Train:** 2008-2015 (8 years) = 5,760 samples
- **Test:** 2016-2017 (2 years) = 1,440 samples
- **Rationale:** Temporal split simulates real forecasting (predict future from past)

---

## Key Engineering Decisions

### Decision 1: Why District-Level (Not State or Field)?
- **State-level:** Too coarse, loses local climate variation
- **Field-level:** Ground truth unavailable at scale in India
- **District-level:** Matches administrative units for policy implementation (sweet spot)

### Decision 2: Why Temporal Split (Not Random Split)?
- **Simulates real forecasting:** Predict 2016-2017 from 2008-2015
- **Exposes generalization weakness:** Model struggles with 2016 drought (unseen pattern)
- **Prevents data leakage:** No future information in training set

### Decision 3: Why MVC for NDVI (Not Simple Mean)?
- **Problem:** 15% of 16-day composites unusable during monsoon
- **Alternative 1:** Simple mean imputation → loses signal during critical growth stages
- **Alternative 2:** Delete missing → loses 15% of data
- **Our solution:** Maximum Value Composite → clouds have low NDVI, max preserves vegetation signal
- **Validation:** Reduced RMSE by 45 kg/ha

### Decision 4: Why Crop-Specific GDD Base Temperatures?
- **Generic T_base = 10°C:** Works for most cereals but suboptimal
- **Wheat:** Grows in winter (Nov-Mar), T_base = 0°C more appropriate
- **Rice/Maize:** Summer crops, T_base = 10°C standard
- **Impact:** Improved wheat predictions by 28 kg/ha RMSE

---

## What's Novel (Patent-Protected)

**System design novelty:**
1. **Automated pipeline** for ICRISAT + NASA + ISRO harmonization
2. **Crop-specific temporal alignment** (Kharif vs Rabi growing seasons)
3. **Multi-resolution spatial aggregation** (500m pixels → irregular districts)
4. **Scalable to 5,000+ districts** (current implementation: 300, pipeline generalizes)

**Standard practice (not patented):**
- Zonal statistics (GDAL standard)
- Temporal aggregation (common in agricultural remote sensing)
- GDD calculation (established agronomic method)

**Patent covers:** System integration and automation, not individual techniques.

---

## Documented Limitations

1. **Temporal resolution:** Annual aggregates lose critical timing (late monsoon onset not captured)
2. **Spatial independence:** Model treats districts independently (ignores spatial autocorrelation)
3. **Cloud contamination:** MVC helps but doesn't eliminate all gaps (15% → 8% remaining)
4. **No field-level validation:** Ground truth only available at district level

**Proposed improvements:** See EVALUATION.md for detailed recommendations (monthly sequences, spatial features, district adjacency).

---

**Related:** [EVALUATION.md](EVALUATION.md) for model performance and error analysis