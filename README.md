# Climate-Smart Crop Yield Prediction System

⚠️ **Source Code Notice**  
The implementation code for this project is not publicly available due to an ongoing
patent filing. This repository documents the **system design, modeling decisions,
experiments, and results** to demonstrate applied machine learning and data
engineering depth.

---

## Overview

This project presents an **end-to-end machine learning system** for predicting
**district-level crop yields across India** by integrating:

- Historical agricultural yield data (ICRISAT)
- Satellite-derived vegetation indices (NDVI, VCI)
- Climate and weather variables (NASA POWER API)

The system is designed to support **climate-aware agricultural planning** in regions
with high climate variability.

---

## Problem Context

Crop yield prediction is challenging due to:
- Non-linear interactions between climate, vegetation, and geography
- Temporal dependencies across growing seasons
- Regional heterogeneity across agro-ecological zones

Most existing approaches either lack regional granularity or fail to generalize under
changing climate conditions. This project targets **district-level forecasting**, a
critical resolution for policy planning and agricultural decision-making.

---

## Key Outcomes

- Built and evaluated **ML and DL models** for yield prediction
- Achieved **90%+ predictive reliability** across major crops
- Demonstrated that **tree-based ML models** (Random Forest, XGBoost) outperform
  deep learning under limited data, while DL architectures offer better scalability
- Produced interpretable insights into climate and vegetation drivers

---

## Models Evaluated

- Random Forest
- XGBoost
- CNN–LSTM
- Deep Fusion Neural Network (custom architecture)

---

## Status

- Research and evaluation complete
- Dataset creation and methodology documented
- Patent filing in progress
# climate-smart-crop-yield-prediction
