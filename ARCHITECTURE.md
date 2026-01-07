# System Architecture

## High-Level Pipeline

1. **Data Ingestion**
   - ICRISAT district-level crop yield data (2008–2017)
   - NDVI/VCI vegetation indices from satellite sources
   - Climate variables via NASA POWER API

2. **Preprocessing & Harmonization**
   - Temporal alignment across datasets
   - Spatial aggregation at district level
   - Missing value handling and normalization

3. **Feature Engineering**
   - Climatic indices (GDD, temperature ranges)
   - Vegetation health metrics
   - Geographical context (latitude, longitude)

4. **Modeling Layer**
   - Traditional ML models for tabular data
   - Deep learning architectures for heterogeneous feature fusion

5. **Evaluation & Interpretation**
   - Temporal validation across growing seasons
   - Region-wise error analysis
   - Feature importance and sensitivity analysis

---

## Design Considerations

- **Heterogeneous data sources** required careful alignment
- **Temporal validation** was prioritized to simulate real forecasting
- **Interpretability** was treated as a first-class requirement
