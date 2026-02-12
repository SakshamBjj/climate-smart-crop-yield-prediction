# Climate-Smart Crop Yield Prediction System

[![Patent Published](https://img.shields.io/badge/Patent-202541116475_A-blue)](https://ipindiaonline.gov.in/patentsearch/)
[![Research](https://img.shields.io/badge/Research-VIT%20Vellore-orange)]()
[![Status](https://img.shields.io/badge/Status-Under%20Examination-yellow)]()

**District-level crop yield forecasting across 300 Indian districts using multi-source data integration (satellite imagery, climate APIs, agricultural records)**

---

## Problem & Motivation

Agricultural yield forecasting in India faces a data integration problem:
- **Climate data:** Daily granularity, API-based (NASA POWER)
- **Satellite imagery:** 16-day composites, 500m grids (ISRO VEDAS)
- **Yield records:** Annual aggregates, irregular district boundaries (ICRISAT)

No single source provides complete predictive signal. Traditional approaches either:
1. Use only climate (miss vegetation health)
2. Use only satellite (miss temperature extremes)
3. Aggregate to state-level (lose local variation)

This system harmonizes all three sources at district resolution to capture climate-crop interactions while remaining policy-relevant (districts = administrative units for agricultural intervention).

---

## What This System Does

**Input:** District ID + crop type + year  
**Output:** Predicted yield (kg/ha) with uncertainty estimates

**Coverage:**
- 300 districts across 20 Indian states
- 2008-2017 (training: 2008-2015, testing: 2016-2017)
- Primary crops: wheat, rice, maize

**Performance:**
- **R² = 0.78** (explains 78% of yield variance)
- **RMSE = 575 kg/ha** (±12% error on average yields)
- Competitive with USDA county-level models despite data constraints

---

## Engineering Decisions

### 1. Why Random Forest over Deep Learning?

Tested 4 architectures: Random Forest, XGBoost, DeepFusionNN (custom CNN-LSTM), CNN-LSTM baseline.

**Result:** Tree ensembles outperformed deep learning by 22% (RMSE).

| Model | RMSE (kg/ha) | R² | Training Time | Parameters |
|-------|--------------|-----|---------------|------------|
| **Random Forest** | **578** | **0.78** | 12 min | ~1K trees |
| **XGBoost** | **572** | **0.78** | 18 min | ~500 trees |
| DeepFusionNN | 690 | 0.66 | 2h 45m | 170K params |
| CNN-LSTM | 750 | 0.59 | 1h 15m | 220K params |

**Why tree methods won:**
- Data regime: 7,200 samples insufficient for 170K-parameter networks
- Sample efficiency: 0.0006 samples/param (RF) vs 0.034 (DL)
- Biological interpretability: Feature importance matches agronomic knowledge

**Decision:** Chose simpler model with better generalization. Deep learning requires 5,000+ districts (~50K samples) based on learning curve extrapolation.

---

### 2. Handling Data Quality Issues

**Problem:** 15% of satellite NDVI values missing due to cloud contamination (monsoon season).

**Options considered:**
1. Drop districts with missing data → Lose 45 districts
2. Mean imputation → Introduces temporal bias
3. Forward-fill → Violates causality
4. Maximum Value Composite (MVC) → Standard in remote sensing

**Decision:** Implemented MVC (take highest NDVI across adjacent 16-day periods).
- **Rationale:** Clouds suppress NDVI, maximum approximates cloud-free vegetation
- **Validation:** Reduced RMSE by 45 kg/ha vs simple mean imputation

---

### 3. Temporal Validation Strategy

**Decision:** Train on 2008-2015, test on 2016-2017 (no random split).

**Why:**
- Simulates real forecasting scenario (predict future from past)
- Exposes model to unseen climate patterns (2015-2016 drought years)
- Prevents data leakage (future information in training)

**Trade-off:** Lower test accuracy but honest performance estimate.

---

### 4. District-Level Granularity

**Why not state-level?**
- Too coarse: A state like Maharashtra spans 7 agro-climatic zones
- Loses actionable signal for regional policy

**Why not field-level?**
- Ground truth unavailable at scale in India
- Satellite resolution (500m) doesn't support field boundaries

**Why district:**
- Matches administrative units (tractable policy implementation)
- Balance between signal preservation and data availability
- 300 districts = sufficient statistical power

---

## Technical Implementation

### Data Pipeline

**Sources:**
1. **ICRISAT TCI Database** — District yield records (institutional access required)
2. **NASA POWER API** — Daily climate (T2M, precipitation, humidity, radiation)
3. **ISRO VEDAS** — 16-day NDVI/VCI composites (500m resolution)

**Alignment challenges solved:**
- **Temporal:** Daily climate → crop-specific growing season windows (Kharif vs Rabi) → annual aggregates
- **Spatial:** 500m grid cells → zonal statistics (mean, std, percentiles) → irregular district polygons
- **Missing data:** Cloud contamination handling via MVC, outlier detection

### Feature Engineering (14 features)

**Climate-derived (11):**
- Growing Degree Days (GDD) — Heat accumulation using crop-specific base temperatures
- Cooling Degree Days (CDD18_3) — Heat stress metric
- Precipitation totals, extremes, variability
- Temperature range, mean, extremes
- Relative humidity (disease pressure proxy)

**Vegetation-derived (1):**
- NDVI mean during critical growth phases (vegetation health proxy)

**Geospatial (2):**
- Latitude, longitude (agro-climatic zone proxy)

**Why these features:**
- Biologically interpretable (not data-mined)
- Align with crop physiology literature
- Enable model debugging (feature importance = domain validation)

### Model Training

**Framework:** Scikit-learn (Random Forest), XGBoost
**Hyperparameters:** 5-fold CV, grid search for tree depth/count
**Validation:** Temporal split (2008-2015 train, 2016-2017 test)
**Metrics:** RMSE (primary), R², MAE, crop-specific breakdowns

---

## Results & Failure Modes

### Overall Performance

<p align="center">
  <img src="assets/model_comparison_rmse.png" width="45%" />
  <img src="assets/model_comparison_r2.png" width="45%" />
</p>

Random Forest achieved **R² = 0.78, RMSE = 575 kg/ha** on held-out 2016-2017 test set.

---

### Feature Importance

<p align="center">
  <img src="assets/feature_importance_rf.png" width="65%" />
</p>

**Top 3 features (64% of predictive power):**
1. **GDD (32%)** — Growing Degree Days (heat accumulation)
2. **NDVI (18%)** — Vegetation health from satellite
3. **PRECTOT (14%)** — Total precipitation (critical for rainfed crops)

Model aligns with agronomic knowledge (not spurious pattern-matching).

---

### Where It Fails (Error Analysis)

#### Spatial Patterns
- **Semi-arid regions:** +30% RMSE (high climate variability, rainfed agriculture)
- **Indo-Gangetic Plain:** -22% RMSE (irrigation infrastructure stabilizes yields)

#### Temporal Sensitivity
- **Normal years:** RMSE ~600 kg/ha
- **Drought years (2015, 2016):** RMSE ~800 kg/ha (+25-30% error)
- **Root cause:** Only 2 drought years in 8-year training set (poor generalization)

#### Crop-Specific Performance
| Crop  | R² | RMSE | Why Different |
|-------|-----|------|---------------|
| Wheat | 0.82 | 520 kg/ha | Rabi season (predictable), 80% irrigated |
| Rice  | 0.76 | 605 kg/ha | Monsoon-dependent, NDVI cloud issues |
| Maize | 0.71 | 680 kg/ha | Heterogeneous systems, data scarcity |

---

### Learning Curve Analysis

<p align="center">
  <img src="assets/learning_curves_deepfusion.png" width="70%" />
</p>

**Key finding:** Validation loss plateaus at ~3,000 samples.

**Implication:** Data scarcity, not model architecture, is the bottleneck.

**Estimated crossover point:** 5,000+ districts (~50K samples) needed for deep learning to outperform Random Forest based on power-law extrapolation.

---

## Limitations & Next Steps

### Current Limitations

**1. Temporal resolution:** Annual aggregates lose critical timing information
- Late monsoon onset not captured
- Flowering-stage heat stress averaged out

**2. Spatial independence:** Model treats districts as independent
- Ignores spatial autocorrelation (neighboring districts' yields correlate)

**3. Drought undersampling:** Only 2 extreme drought years in training
- Poor generalization to climate extremes
- Model underestimates variance in tail events

**4. Single-model approach:** One RF for all crops
- Wheat/rice pooled despite different physiology
- Maize suffers from data scarcity (only 15% of records)

---

### Proposed Improvements (Ordered by Expected Impact)

**High impact:**
1. **Monthly climate sequences** → LSTM on 12-month windows (estimated +5-8% R² improvement)
2. **Crop-specific models** → Separate RF for wheat/rice/maize (estimated +3-5% R²)
3. **Drought augmentation** → Synthetic data generation or oversampling extreme years

**Medium impact:**
4. **Spatial features** → District adjacency matrix (neighboring yields as predictors)
5. **Sub-seasonal satellite** → Weekly NDVI instead of 16-day (cloud mitigation)

**Low impact (data unavailable):**
6. Field-level validation
7. Real-time API deployment

---

## Repository Structure

```
climate-smart-crop-yield-prediction/
├── assets/                   # Evaluation visualizations
│   ├── model_comparison_rmse.png
│   ├── model_comparison_r2.png
│   ├── learning_curves_deepfusion.png
│   └── feature_importance_rf.png
├── docs/
│   ├── EVALUATION.md        # Detailed error analysis
│   └── PIPELINE_OVERVIEW.md # Data integration methodology
└── README.md
```

**Note on code availability:**
This project is part of a patent-published system (Application No. 202541116475 A) currently under examination. Full implementation code and proprietary dataset pipelines are not publicly available to protect IP and comply with institutional policies.

**What's public:**
- Complete methodology and system architecture
- Evaluation results with failure mode analysis
- Reproducible visualizations
- Engineering decision rationale

**What's protected:**
- Dataset ETL pipelines (ICRISAT institutional access required)
- Model training code
- DeepFusionNN architecture internals (patent-protected)

**For recruiters:** Private code walkthrough available upon request.

---

## What This Demonstrates

**Data engineering:**
- Multi-source integration across incompatible formats
- Temporal and spatial alignment at scale
- Real-world data quality handling (missing data, outliers)

**Model selection:**
- Empirical comparison over default choices
- Data regime analysis driving architecture decisions
- Sample efficiency vs model complexity tradeoffs

**Evaluation rigor:**
- Honest failure mode documentation
- Learning curve analysis quantifying data requirements
- Spatial/temporal/crop-specific error decomposition

**Domain understanding:**
- Biologically interpretable features
- Agriculture-specific data quality issues
- Policy-relevant granularity choices

---

## Citation

```bibtex
@mastersthesis{bajaj2025crop,
  title={Deep Learning Approach for Crop Yield Prediction using Intelligent Climate Change Prediction},
  author={Bajaj, Saksham and Srivastava, Rishabhraj and Kumar, Harshit Vijay},
  year={2025},
  school={Vellore Institute of Technology},
  note={Patent Application No. 202541116475 A}
}
```

---

**Author:** Saksham Bajaj  
**Contact:** [LinkedIn](https://www.linkedin.com/in/saksham-bjj/) | [GitHub](https://github.com/SakshamBjj)  
**Last Updated:** Feb 2026
