# Results and Evaluation

## Performance Summary

| Model         | RMSE (kg/ha) | R²   | Training Cost |
|---------------|-------------|------|---------------|
| Random Forest | ~579        | 0.78 | Low           |
| XGBoost       | ~572        | 0.78 | Low           |
| CNN–LSTM      | ~750        | 0.59 | Medium        |
| DeepFusionNN  | ~690        | 0.65 | High          |

---

## Interpretation

- ML models were more **cost-effective and accurate** for current dataset size
- DL models are better suited for **larger, higher-resolution datasets**
- Errors increased during drought years, highlighting climate sensitivity

---

## Practical Implications

- Enables early yield estimation
- Supports climate-aware agricultural planning
- Provides interpretable insights for policymakers
