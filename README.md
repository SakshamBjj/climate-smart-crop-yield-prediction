# Climate-Smart Crop Yield Prediction System

## Intellectual Property Notice

This work is the subject of a **published patent application**:

**Title:** Deep Fusion Neural Network System for Crop Yield Prediction  
**Applicants:** Vellore Institute of Technology  
**Inventors:** Dr. Jayakumar K, Saksham Bajaj, Rishabhraj Srivastava, Harshit Vijay Kumar  
**Publication Date:** December 12, 2025  
**Jurisdiction:** India

To protect intellectual property during examination, the source code is not publicly released.
This repository documents the system design, modeling rationale, and experimental outcomes.

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
- Patent application published in the Indian Patent Office Journal (Dec 2025)
