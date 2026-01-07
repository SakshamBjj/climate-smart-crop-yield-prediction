# Modeling and Experiments

## Baseline Models

Tree-based models were used as strong baselines:

### Random Forest
- Robust to non-linear relationships
- Provided feature importance for interpretability
- Low training and inference cost

### XGBoost
- Efficient gradient boosting on tabular data
- Strong performance with missing values
- Best overall RMSE among tested models

---

## Deep Learning Models

### CNN–LSTM
- Combined convolutional layers for feature extraction
- LSTM layers captured temporal dependencies
- Improved modeling of sequential climate patterns

### Deep Fusion Neural Network
A custom architecture designed to:
- Process climate, vegetation, and geographic features separately
- Fuse representations using attention mechanisms
- Improve scalability as dataset size increases

---

## Experimentation Strategy

- 80/20 temporal train–test split (2008–2015 vs 2016–2017)
- 5-fold stratified cross-validation
- Evaluation using RMSE, R², MAE, MSE, MedAE, and bias metrics

---

## Key Learnings

- Tree-based ML models outperformed DL on limited data
- DL models showed higher computational cost but better scalability
- Vegetation indices and GDD were dominant predictors
