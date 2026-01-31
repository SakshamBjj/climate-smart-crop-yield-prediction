# Climate-Smart Crop Yield Prediction System

[![Patent Published](https://img.shields.io/badge/Patent-202541116475_A-blue)](https://ipindiaonline.gov.in/patentsearch/)
[![Research](https://img.shields.io/badge/Research-VIT%20Vellore-orange)]()
[![Status](https://img.shields.io/badge/Status-Under%20Examination-yellow)]()

> **Applied ML Engineering Project**: District-level crop yield prediction across 300 Indian districts using multi-source data integration (satellite imagery, climate APIs, agricultural records)

---

## 🔒 **IP / Patent Notice**

This project is part of a **patent-published system** (Application No: 202541116475 A) and is currently under grant process. To protect IP and comply with institutional policies, the full implementation and proprietary training dataset pipelines are not public.

This repository shares **methodology, evaluation, key engineering decisions, and reproducible visualizations** to demonstrate ML execution and outcomes. Recruiters can request a **private walkthrough / code review**.

**What's Public:**
- ✅ System architecture and data integration methodology
- ✅ Feature engineering approach and model comparison
- ✅ Complete evaluation results with failure mode analysis
- ✅ Reproducible visualization scripts

**What's Protected:**
- 🔒 Full implementation code (dataset ETL, model training pipelines)
- 🔒 Proprietary dataset extraction scripts (ICRISAT institutional access required)
- 🔒 DeepFusionNN architecture internals (patent-protected)

---

## 🎯 **Problem Statement**

Traditional crop yield forecasting fails to capture:
- Non-linear climate-crop interactions
- Spatial heterogeneity across districts
- Temporal dependencies in seasonal data
- Multi-source data integration complexity (satellite 16-day, climate daily, yield annual)

**This project:** District-level prediction system harmonizing ICRISAT agricultural records (2008-2017), NASA POWER climate APIs, and ISRO VEDAS satellite imagery across 300 districts, 20 states.

---

## 📊 **Results Summary**

### Model Performance (Test Set: 2016-2017)

| Model | RMSE (kg/ha) | R² | Training Time | Inference Cost |
|-------|--------------|-----|---------------|----------------|
| **Random Forest** | **578.84** | **0.7796** | 12 min | $0.15/1k preds |
| **XGBoost** | **572.07** | **0.7784** | 18 min | $0.12/1k preds |
| DeepFusionNN | 690.20 | 0.6550 | 2h 45m | $0.42/1k preds |
| CNN-LSTM | 749.77 | 0.5907 | 1h 15m | $0.38/1k preds |

**Key Finding:** Tree-based ML achieved **78% variance explained** (R² = 0.78), outperforming deep learning by 22% on RMSE.

### Visual Evidence

<p align="center">
  <img src="assets/model_comparison_rmse.png" width="45%" />
  <img src="assets/model_comparison_r2.png" width="45%" />
</p>

<p align="center">
  <img src="assets/learning_curves_deepfusion.png" width="70%" />
</p>

**Critical Insight:** Learning curves show validation loss plateaus at ~3,000 samples. Analysis indicates **5,000+ districts needed** (50K samples) for deep learning to outperform Random Forest. This quantifies the data requirement gap.

<p align="center">
  <img src="assets/feature_importance_rf.png" width="65%" />
</p>

**Top 3 Features (64% of predictive power):**
- **GDD (32%)** – Growing Degree Days (heat accumulation)
- **NDVI (18%)** – Vegetation health from satellite
- **PRECTOT (14%)** – Precipitation (rainfed crops)

Model is **biologically interpretable**, not just pattern-matching.

---

## 🏗️ **What Makes This Non-Trivial**

### 1. Multi-Source Data Harmonization
- **Temporal alignment:** Daily climate → 16-day satellite → annual yield
- **Spatial alignment:** 500m satellite grids → irregular district boundaries
- **Missing data:** 15% NDVI cloud contamination handled via Maximum Value Composite (MVC)

### 2. District-Level Resolution
- Not too broad (state/national averages lose signal)
- Not too granular (field-level data unavailable)
- **Policy-relevant:** Aligns with Indian agricultural administrative units

### 3. Honest Model Comparison
- **Most studies hide when DL fails.** We documented it with root cause analysis.
- **Learning curve analysis:** Estimated 5,000+ districts needed for DL superiority
- **Crossover hypothesis:** Validated via power-law extrapolation

### 4. Error Analysis Rigor
**Spatial patterns:**
- Semi-arid regions: +30% RMSE (climate variability)
- Indo-Gangetic Plain: -22% RMSE (irrigation stability)

**Temporal patterns:**
- Normal years: RMSE ~600 kg/ha
- Drought years (2015, 2016): RMSE ~800 kg/ha (+25-30% error)
- Root cause: Only 2 drought years in 8-year training period

**Crop-specific:**
- Wheat: R² = 0.82 (Rabi season = predictable, 80% irrigated)
- Rice: R² = 0.76 (monsoon-dependent, NDVI cloud issues)
- Maize: R² = 0.71 (heterogeneous systems, data scarcity)

*See [EVALUATION.md](EVALUATION.md) for detailed failure mode analysis*

---

## 🛠️ **Technical Overview**

### Data Sources
1. **ICRISAT TCI Database** – District-level yield records (300 districts, 2008-2017)
2. **NASA POWER API** – Daily climate parameters (T2M, precipitation, humidity, GDD)
3. **ISRO VEDAS** – 16-day satellite NDVI/VCI (500m resolution)

### Feature Engineering (14 features)
- **Climate (11):** GDD, CDD18_3, precipitation sum, temperature ranges, humidity
- **Vegetation (1):** NDVI mean during growing season
- **Geospatial (2):** Latitude, longitude (agro-climatic zone proxy)

### Modeling Approach
- **Temporal validation:** Train 2008-2015, test 2016-2017 (simulates real forecasting)
- **Hyperparameter tuning:** 5-fold CV for RF/XGBoost, Bayesian optimization for DL
- **Evaluation metrics:** RMSE (primary), R², MAE, crop-specific breakdowns

*See [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for data integration methodology*

---

## 🎤 **My Contribution**

As the primary technical lead (handled ~90% of implementation):

**Data Engineering:**
- Designed and implemented end-to-end data integration pipeline across 3 heterogeneous sources
- Built temporal alignment logic (crop-specific growing season windowing: Kharif vs Rabi)
- Implemented spatial harmonization (zonal statistics: 500m pixels → district boundaries)
- Handled missing NDVI data (15% cloud contamination) via Maximum Value Composite method

**Feature Engineering:**
- Engineered domain-specific features: GDD calculation (crop-specific base temperatures), VCI normalization
- Created derived climate metrics: temperature range, cooling degree days (CDD18_3)

**Modeling & Evaluation:**
- Executed comparative experiments: Random Forest, XGBoost, DeepFusionNN, CNN-LSTM
- Performed systematic error analysis across spatial/temporal/crop dimensions
- Conducted learning curve analysis to quantify data requirements for DL viability

**Key Deliverables:**
- Achieved R² = 0.78, RMSE = 575 kg/ha on district-level prediction (competitive with USDA county models)
- Identified crossover point: 5,000+ districts needed for DL to outperform tree methods
- Documented failure modes: drought sensitivity (+25% error), spatial heterogeneity patterns

---

## 🔍 **Key Technical Decisions**

**Why Random Forest beat DeepFusionNN:**
- Data size bottleneck: 7,200 samples insufficient for 170K-parameter DL model
- Tree methods are more sample-efficient for tabular data (0.0006 vs 0.034 samples/parameter)
- Learning curve plateau at ~3,000 samples confirms data, not architecture, is limiting factor

**Why temporal split (not random split):**
- Simulates real-world forecasting scenario (predict 2016-2017 from 2008-2015)
- Exposes model vulnerability to unseen climate patterns (drought generalization)
- Prevents data leakage from future information

**Why district-level (not state or field-level):**
- State-level: Too coarse, loses local climate variation
- Field-level: Ground truth unavailable at scale in India
- District-level: Matches administrative units for policy implementation

**Handling NDVI cloud contamination:**
- 15% of 16-day composites unusable during monsoon season
- Solution: Maximum Value Composite (MVC) – take highest NDVI across adjacent periods
- Validation: Reduced RMSE by 45 kg/ha vs simple mean imputation

---

## 📈 **Documented Limitations & Next Steps**

**Current Limitations:**
1. **Temporal resolution:** Annual aggregates lose critical timing information (late monsoon onset not captured)
2. **Spatial independence:** Model treats districts independently (ignores spatial autocorrelation)
3. **Drought undersampling:** Only 2 drought years in 8-year training → poor generalization
4. **Crop heterogeneity:** Single model for all crops (maize suffers from data scarcity)

**Proposed Improvements:**
- **Monthly climate sequences:** LSTM on 12-month windows (estimated +5-8% R² improvement)
- **Spatial features:** District adjacency matrix (neighboring districts' yields as features)
- **Drought augmentation:** Synthetic data generation or oversampling extreme years
- **Crop-specific models:** Separate RF models for wheat/rice/maize

*See [EVALUATION.md](EVALUATION.md) for detailed recommendations*

---

## 📚 **Citation**

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
**Last Updated:** January 2026