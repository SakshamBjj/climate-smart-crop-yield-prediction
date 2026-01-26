# Climate-Smart Crop Yield Prediction System

[![Patent Published](https://img.shields.io/badge/Patent-Published-blue)](https://ipindiaonline.gov.in/patentsearch/)
[![Research](https://img.shields.io/badge/Research-VIT%20Vellore-orange)]()
[![Status](https://img.shields.io/badge/Status-Under%20Examination-yellow)]()

## Intellectual Property Notice

This work is the subject of a **published patent application**:

- **Title:** Deep Fusion Neural Network System for Crop Yield Prediction
- **Application No.:** 202541116475 A
- **Applicants:** Vellore Institute of Technology
- **Inventors:** Dr. Jayakumar K, Saksham Bajaj, Rishabhraj Srivastava, Harshit Vijay Kumar
- **Publication Date:** December 12, 2025
- **Jurisdiction:** India

---

## Overview

An **end-to-end machine learning system** for district-level crop yield prediction across India, integrating:

- **Historical Agricultural Data:** ICRISAT district-level yield records (2008-2017)
- **Satellite Vegetation Indices:** NDVI, VCI from ISRO VEDAS
- **Climate Variables:** NASA POWER API (temperature, precipitation, humidity, GDD)

**Target Use Case:** Climate-aware agricultural planning at policy-relevant administrative resolution (district-level).

---

## Problem Statement

Traditional crop yield forecasting methods struggle with:

1. **Non-linear climate-crop interactions** → Simple regression fails
2. **Spatial heterogeneity** → National models miss local patterns
3. **Temporal dependencies** → Annual aggregates lose seasonal signals
4. **Data integration complexity** → Combining satellite (16-day), climate (daily), and yield (annual) data

**This project addresses:** How to build a district-level prediction system that harmonizes multi-source, multi-resolution agricultural data.

---

## Performance Summary

### Quantitative Results

| Model | RMSE (kg/ha) | R² | MAE | Training Cost | Inference Cost |
|-------|--------------|-----|-----|---------------|----------------|
| **Random Forest** | 578.84 | **0.7796** | 489.5 | Low (12 min) | $0.15/1k preds |
| **XGBoost** | 572.07 | **0.7784** | 503.5 | Low (18 min) | $0.12/1k preds |
| CNN-LSTM | 749.77 | 0.5907 | 338.6 | Medium (1h 15m) | $0.38/1k preds |
| DeepFusionNN | 690.20 | 0.6550 | 366.7 | High (2h 45m) | $0.42/1k preds |

**Dataset:** 300 districts, 20 states, 10 years (2008-2017)  
**Validation Strategy:** Temporal split (train: 2008-2015, test: 2016-2017)

### Key Findings

✅ **Tree-based ML models achieved 78% variance explained** (R² = 0.78)  
✅ **RMSE of ~575 kg/ha** for district-level predictions  
⚠️ **Deep learning underperformed** due to limited dataset size (~3,000 samples)  
📊 **GDD and NDVI** were dominant predictive features (RF feature importance: 32% and 18%)

---

## What Makes This Non-Trivial

### 1. Multi-Source Data Harmonization
- **Temporal alignment:** Daily climate → 16-day satellite → annual yield
- **Spatial alignment:** 500m satellite grids → irregular district boundaries
- **Missing data handling:** NDVI cloud cover, incomplete climate records

### 2. District-Level Resolution
- Not too broad (state/national averages lose local signal)
- Not too granular (field-level requires unavailable data)
- **Policy-relevant:** Aligns with Indian agricultural administration units

### 3. Honest Model Comparison
- **Most studies hide when DL fails.** We documented it.
- **Root cause analysis:** Data size, not architecture quality
- **Crossover hypothesis:** Estimated 5,000+ districts needed for DL superiority

### 4. Patent-Grade System Design
- Modular architecture (data acquisition → preprocessing → fusion → prediction)
- Generalizable to other crops/regions
- Scalable to nationwide deployment

---

## Repository Structure
```
├── README.md                          ← You are here
├── ARCHITECTURE.md                    ← System design and DeepFusionNN specs
├── DATA_PIPELINE.md                   ← Multi-source integration methodology
├── MODELING_AND_EXPERIMENTS.md        ← Model descriptions and training strategy
├── RESULTS_AND_EVALUATION.md          ← Performance metrics and analysis
├── ERROR_ANALYSIS.md                  ← Failure modes and spatial/temporal breakdown
├── LIMITATIONS_AND_FUTURE_WORK.md     ← Honest limitations and research directions
├── PATENT_CONTEXT.md                  ← IP scope and code availability policy
└── results/                           ← Performance visualizations
    ├── model_comparison_rmse.png
    ├── model_comparison_r2.png
    ├── model_comparison_mae.png
    ├── learning_curves_deepfusion.png
    └── feature_importance_rf.png
```

---

## Code Availability

**Status:** Implementation code is available to **verified academic researchers** and **institutional partners**.

**Why controlled access?**
1. **Patent examination in progress** (Application No. 202541116475 A)
2. **Dataset licensing restrictions** (ICRISAT TCI database requires institutional agreement)
3. **Institutional IP compliance** (VIT technology transfer policies)

**What's publicly available:**
- ✅ System architecture and design specifications
- ✅ Model architecture definitions (layer-by-layer specs)
- ✅ Feature engineering methodology
- ✅ Complete evaluation results with visualizations
- ✅ Error analysis and failure mode documentation

**Post-patent grant roadmap:**
- 📦 Reference implementation (Apache 2.0 + Patent Grant license)
- 📊 Preprocessed sample datasets (100 districts, non-commercial use)
- 🤖 Trained model weights (research license)

**For collaboration inquiries:**  
📧 Academic: saksham.bajaj2021@vitstudent.ac.in  
🏛️ Commercial licensing: patents@vit.ac.in

---

## Citation

If you reference this work, please cite:
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

## Acknowledgments

- **Supervisor:** Dr. Jayakumar K (Associate Professor Sr., VIT SCOPE)
- **Data Sources:** ICRISAT TCI Database, NASA POWER API, ISRO VEDAS Dashboard
- **Institution:** Vellore Institute of Technology (VIT Vellore)
- **Patent Applicant:** Vellore Institute of Technology

---

## FAQ

**Q: Why is R² = 0.78 good for agricultural prediction?**  
A: District-level yield has high natural variance (weather, soil, practices). 78% explained variance is competitive with published agricultural forecasting systems. For context, USDA models achieve R² = 0.65-0.82 for county-level corn yield.

**Q: Why did deep learning underperform?**  
A: Insufficient data. DL needs 10-100× more samples. Our analysis (see ERROR_ANALYSIS.md) shows learning curves plateauing at ~2,000 samples. Estimated 5,000+ districts needed for DL to outperform RF/XGBoost.

**Q: Can I reproduce the results?**  
A: Data pipeline and model specs are fully documented. Researchers can request preprocessed data samples and validation protocols. Full reproduction requires ICRISAT institutional access.

**Q: What's the patent actually covering?**  
A: System architecture for multi-source agricultural data integration, not individual ML components (attention mechanisms are prior art). See PATENT_CONTEXT.md for details.

**Q: When will code be released?**  
A: Post-patent examination (estimated 18-24 months). Early access available to academic collaborators under NDA.

---

**Last Updated:** January 2026