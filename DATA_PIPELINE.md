# Data Integration Pipeline

## Overview

The core technical challenge of this project is **harmonizing three heterogeneous data sources** with different temporal resolutions, spatial formats, and collection methodologies.

---

## Data Sources

### 1. ICRISAT TCI Database (Target Variable)
**What:** District-level crop yield statistics for India  
**Coverage:** 571 districts, 20 states, 1966-2020 (we used 2008-2017)  
**Temporal Resolution:** Annual  
**Spatial Resolution:** Administrative boundaries (districts)  

**Key Fields:**
- `District_Code` (unique identifier)
- `Year` (2008-2017 for this study)
- `Crop` (Rice, Wheat, Maize)
- `Area` (1000 hectares)
- `Production` (1000 tons)
- `Yield` (kg/ha) ← **TARGET VARIABLE**

**Data Quality Issues:**
- District boundary changes (handled via 1991/2001 census baseline apportionment)
- Missing years for some districts (~8% missing values)
- Outliers from reporting errors (yields > 10,000 kg/ha flagged)

**Preprocessing:**
```
1. Filter years: 2008-2017
2. Filter crops: Rice, Wheat, Maize (major staples)
3. Remove districts with >30% missing years
4. Impute missing yields using spatial neighbors (inverse distance weighting)
5. Final dataset: 300 districts × 10 years × 3 crops = ~9,000 records
   (After removing missing values: ~7,200 complete records)
```

---

### 2. NASA POWER API (Climate Data)
**What:** Meteorological parameters from NASA's POWER project  
**Coverage:** Global, gridded  
**Temporal Resolution:** **Daily**  
**Spatial Resolution:** 0.5° × 0.5° latitude-longitude grid (~55 km at equator)

**Key Parameters:**
- `T2M`: Temperature at 2 meters (°C)
- `PRECTOTCORR`: Precipitation (mm/day)
- `RH2M`: Relative humidity (%)
- `PS`: Surface pressure (kPa)
- `QV2M`: Specific humidity (g/kg)
- `T2MDEW`: Dew point temperature (°C)
- `T2M_MAX`: Maximum temperature (°C)

**Derived Features:**
- `GDD`: Growing Degree Days = Σ max(0, (T_max + T_min)/2 - T_base)
  - T_base = 10°C (standard for cereals)
- `CDD18_3`: Cooling Degree Days (heat stress index)
- `Temp_Range`: T2M_MAX - T2M (diurnal temperature variation)

**Preprocessing:**
```
1. API query: For each district centroid (lat, lon), fetch daily data (2008-2017)
2. Temporal aggregation: Daily → Annual
   - T2M: Mean annual temperature
   - PRECTOT: Sum annual precipitation
   - RH2M: Mean annual humidity
   - GDD: Cumulative annual sum
3. Spatial alignment: Grid centroids → District boundaries
   - If district > grid cell → area-weighted average of overlapping cells
   - If district < grid cell → use nearest grid centroid
4. Quality control: Flag unrealistic values (T2M < -10°C or > 50°C)
```

**Challenge:** NASA grid cells don't align with district boundaries.

**Solution:**
- For small districts (area < 2,500 km²): Use nearest grid cell
- For large districts (area > 2,500 km²): Weighted average of 4 nearest cells
- Weight = 1 / (distance from district centroid)²

---

### 3. ISRO VEDAS (Satellite Vegetation Indices)
**What:** Vegetation health indices from MODIS satellite imagery  
**Coverage:** India  
**Temporal Resolution:** **16-day composites**  
**Spatial Resolution:** 500m × 500m

**Key Indices:**
- `NDVI`: Normalized Difference Vegetation Index
  - Formula: (NIR - Red) / (NIR + Red)
  - Range: -1 to +1 (higher = healthier vegetation)
- `VCI (%)`: Vegetation Condition Index
  - Formula: 100 × (NDVI - NDVI_min) / (NDVI_max - NDVI_min)
  - Range: 0-100% (relative to historical range)

**Preprocessing:**
```
1. Acquisition: Download 16-day NDVI composites from VEDAS portal
   - 23 composites per year × 10 years = 230 images per district
2. Cloud masking: Remove pixels with cloud cover > 30%
3. Spatial aggregation: 500m pixels → District-level mean
   - Zonal statistics: Mean NDVI across all pixels within district boundary
4. Temporal aggregation: 16-day → Annual
   - NDVI: Mean of growing season composites (Kharif: June-Oct, Rabi: Nov-Mar)
   - VCI: Minimum VCI during critical growth stages (flowering, grain filling)
5. Gap filling: If cloud cover ruins >50% of composites → linear interpolation
```

**Challenge:** NDVI has high cloud contamination during monsoon season (June-September).

**Solution:**
- Use **maximum value composite** (MVC) method: For each 16-day period, take highest NDVI (clouds have low NDVI)
- If still insufficient → borrow from adjacent 16-day periods

---

## Data Harmonization Strategy

### Temporal Alignment

**Problem:** Daily climate + 16-day satellite + annual yield

**Solution:**
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
   - Yield reported in Year_N → Use climate/satellite from crop year_N growing season
```

**Example:**
```
Wheat yield 2015 (reported for harvest in March 2016)
    ← Climate: Nov 2015 - Mar 2016 aggregates
    ← Satellite: Nov 2015 - Mar 2016 NDVI/VCI
```

### Spatial Harmonization

**Problem:** District boundaries (irregular polygons) vs. NASA grid (regular lat-lon) vs. MODIS pixels (500m grid)

**Solution: Zonal Statistics**
```
# Pseudocode
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

**Tools Used:**
- Shapefile manipulation: GeoPandas (Python)
- Raster extraction: Rasterio, RasterStats
- Spatial operations: GDAL

---

## Feature Engineering

### Derived Climate Features

1. **Growing Degree Days (GDD)**
```
   GDD = Σ max(0, (T_max + T_min)/2 - T_base)
```
   - **Why:** Biologically relevant measure of heat accumulation
   - **Crop-specific T_base:** 
     - Rice: 10°C
     - Wheat: 0°C
     - Maize: 10°C

2. **Temperature Range**
```
   Temp_Range = T_max - T_min
```
   - **Why:** Large diurnal swings indicate water stress

3. **Cooling Degree Days (CDD18_3)**
   - **Why:** Measures heat stress (temperatures > 18°C during flowering reduce yield)

### Normalized Vegetation Indices

1. **VCI Normalization**
```
   VCI = 100 × (NDVI_current - NDVI_min_historic) / (NDVI_max_historic - NDVI_min_historic)
```
   - **Why:** Accounts for regional baseline differences (semi-arid vs. humid tropics)

---

## Data Quality Control

### Outlier Detection

**Method:** Interquartile Range (IQR) flagging

For each feature:
- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1
- Outliers: Values < Q1 - 1.5×IQR  OR  > Q3 + 1.5×IQR

**Outliers flagged (not removed):**
- Extreme yields (>8,000 kg/ha for wheat → likely reporting error)
- Negative NDVI (water bodies misclassified as cropland)
- GDD > 5,000°C-days (tropical districts with year-round cropping)

### Missing Value Handling

| Source | Missing % | Strategy |
|--------|-----------|----------|
| ICRISAT Yield | 8% | Spatial interpolation (IDW from 3 nearest districts) |
| NASA Climate | 2% | Temporal interpolation (linear between adjacent days) |
| MODIS NDVI | 15% | Maximum value composite (MVC) to reduce cloud impact |

**Districts dropped:** 23 districts with >30% missing years (insufficient temporal coverage)

---

## Final Unified Dataset

**Schema:**
| District_Code | Year | Crop  | Latitude | Longitude | T2M | PRECTOT | ... | NDVI | VCI | Yield |
|---------------|------|-------|----------|-----------|-----|---------|-----|------|-----|-------|
| AP001         | 2008 | Rice  | 16.5     | 80.6      | 28.3| 1,234   | ... | 0.72 | 65  | 3,420 |


**Dimensions:**
- **Samples:** 7,200 (300 districts × 10 years × 3 crops, after cleaning)
- **Features:** 14 (11 climate + 1 vegetation + 2 geospatial)
- **Target:** 1 (Yield in kg/ha)

**Train/Test Split:**
- **Train:** 2008-2015 (8 years) = 5,760 samples
- **Test:** 2016-2017 (2 years) = 1,440 samples
- **Rationale:** Temporal split simulates real forecasting (predict future from past)

---

## Patent-Relevant Innovation

**What's novel:**
1. **Automated pipeline** for ICRISAT + NASA + ISRO harmonization
2. **Crop-specific temporal alignment** (Kharif vs. Rabi growing seasons)
3. **Multi-resolution spatial aggregation** (500m pixels → irregular districts)
4. **Scalable to 5,000+ districts** (current implementation handles 300, but pipeline generalizes)

**What's standard practice:**
- Zonal statistics (GDAL standard)
- Temporal aggregation (common in agricultural remote sensing)
- GDD calculation (established agronomic method)

**Patent claims:** System design and automation, not individual techniques.

---

**See Also:**
- MODELING_AND_EXPERIMENTS.md (how this data was used in training)
- ERROR_ANALYSIS.md (data quality impact on model performance)

---