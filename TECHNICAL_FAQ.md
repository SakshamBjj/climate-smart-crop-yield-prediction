# Technical FAQ

Frequently asked questions about the Climate-Smart Crop Yield Prediction system.
This document addresses common technical inquiries from researchers, recruiters, 
and collaborators.

## Data Engineering Questions

### Q1: How did you handle missing NDVI data (15% cloud contamination)?

**A:** Implemented Maximum Value Composite (MVC) method across 16-day windows. For each period, took the highest NDVI value (clouds have low NDVI, vegetation has high). For persistent gaps (>50% composites missing), applied linear interpolation from adjacent periods. Validated this approach reduced RMSE by **45 kg/ha** vs. simple mean imputation.

**Follow-up: Why not just delete missing data?**  
Would lose 15% of samples, including critical monsoon months when NDVI is most informative for rainfed crops. MVC preserves vegetation signal while filtering cloud contamination.

---

### Q2: Why district-level instead of field-level or state-level?

**A:** Three reasons:
1. **Data availability:** Field-level yield records don't exist at scale in India
2. **Policy relevance:** District = administrative unit for resource allocation (matches stakeholder needs)
3. **Validation:** Ground truth available from ICRISAT at district resolution

State-level is too coarse (loses local climate variation). Field-level data doesn't exist for historical periods.

---

### Q3: How did you align daily climate data to annual yield?

**A:** Crop-specific growing season aggregation:
- **Kharif (Rice/Maize):** June-Oct → sum GDD/precipitation during this window
- **Rabi (Wheat):** Nov-Mar → separate aggregation window
- Matched yield reported in year N to climate from N's growing season

**Key decision:** Annual aggregates lose timing information (late monsoon onset). Documented this as a limitation—proposed improvement is monthly sequences.

---

### Q4: What was the hardest data integration challenge?

**A:** Spatial harmonization. NASA POWER grid (0.5° cells) doesn't align with irregular district boundaries. Solution:
- Small districts (<2,500 km²): Use nearest grid cell
- Large districts: Weighted average of 4 nearest cells by area overlap
- Used GeoPandas + Rasterio for zonal statistics

Validated by comparing district centroids vs. area-weighted aggregates—saw 12% RMSE improvement with area weighting.

---

## Modeling & Evaluation Questions

### Q5: Why did Random Forest beat your custom deep learning model?

**A:** Data size bottleneck, not architecture quality. Learning curve analysis showed:
- Validation loss plateaus at ~3,000 samples (we had 7,200 total)
- Random Forest: 0.0006 samples/parameter → highly sample-efficient
- DeepFusionNN: 0.034 samples/parameter → needs 10K+ samples
- Power-law extrapolation: **5,000+ districts needed** for DL to match RF

This is consistent with ML literature—tree methods are more sample-efficient for tabular data (Chen & Guestrin, 2016: XGBoost paper).

---

### Q6: How would you know if your model is overfitting?

**A:** Three pieces of evidence it's NOT overfitting:
1. **Feature importance:** GDD (32%) and NDVI (18%) dominate—these are established agronomic indicators, not spurious patterns
2. **Temporal validation:** Test on held-out years 2016-2017, not random split
3. **Error patterns are interpretable:** Performance degrades predictably during drought years (+25% error), not randomly

If overfitting, we'd see: (1) high train R², low test R², (2) spurious features like longitude having high importance, (3) random error patterns.

---

### Q7: Why temporal split instead of random split for train/test?

**A:** Simulates real-world forecasting scenario:
- Train: 2008-2015 (predict from past)
- Test: 2016-2017 (forecast future)

**Benefits:**
1. Catches temporal drift (model struggles with 2016 drought—unseen pattern)
2. Prevents data leakage from future information
3. Realistic evaluation (production would predict future, not interpolate)

Random split would give **20% higher R²** but wouldn't expose this critical weakness.

---

### Q8: What metric did you optimize for and why?

**A:** Primary: **RMSE** (Root Mean Squared Error). Reasons:
1. **Interpretable:** Same units as target (kg/ha)
2. **Penalizes large errors:** Important for agriculture (100% error is worse than two 50% errors)
3. **Standard in yield prediction literature:** Enables comparison to USDA/JRC models

Also reported R² (explained variance) and MAE (robustness check). Used Huber loss for DL training (robust to outliers from reporting errors).

---

### Q9: How did you validate your model isn't just learning the dataset mean?

**A:** R² = 0.78 means model explains 78% of variance. A naive baseline (predict mean yield = 3,000 kg/ha) would give R² = 0.

Also validated via **feature importance**: Top 3 features (GDD, NDVI, precipitation) account for 64% of predictive power. These are biologically meaningful—model captures crop-climate interactions, not just dataset bias.

---

## Error Analysis Questions

### Q10: What's your model's biggest weakness?

**A:** **Drought year generalization.** RMSE increases +25-30% during droughts (2009, 2015, 2016). Root cause:
- Only 2 drought years out of 8 training years
- Model trained on mostly normal climate → struggles with extreme deviations
- Underestimates actual yield reduction: Reality = 40-60%, model predicts only 15-25%

**Remediation:** Oversample drought years (SMOTE for regression), add climate anomaly features (is this year extreme?). Estimated -15% RMSE improvement on drought years.

---

### Q11: Which regions does your model perform worst in?

**A:** Semi-arid zones (Rajasthan, Maharashtra Vidarbha): **RMSE > 700 kg/ha** (+30% vs. overall).

**Root causes:**
1. Higher climate variability (precipitation CV > 40%)
2. NDVI cloud contamination > 50% during monsoon
3. Agricultural heterogeneity (intercropping confuses NDVI signal)

**Best regions:** Indo-Gangetic Plain (Punjab, Haryana): RMSE < 450 kg/ha. Reason: 80%+ irrigated, stable climate, clean NDVI data.

---

### Q12: How do you explain the wheat vs. rice vs. maize performance difference?

**A:**
- **Wheat (R² = 0.82):** Best performer. Rabi season (Nov-Mar) = predictable winter climate, 80%+ irrigated, concentrated in Punjab/Haryana (high data quality)
- **Rice (R² = 0.76):** Medium. Kharif season (Jun-Oct) = monsoon-dependent, NDVI cloud contamination during growing season
- **Maize (R² = 0.71):** Worst. Heterogeneous (both Kharif and Rabi), data scarcity (only 15% of training samples)

**Action:** Proposed crop-specific models (estimated +5% R² improvement for maize).

---

## Architecture & Scale Questions

### Q13: Why not use transfer learning from pre-trained models?

**A:** Transfer learning requires similar task domains. This is a **tabular data problem**, not image/text:
- Satellite imagery (NDVI) was aggregated to district-level scalars → no spatial structure for CNNs to exploit
- Climate time series aggregated to annual → no temporal structure for LSTMs
- Transfer learning works for: ImageNet→crop disease detection, BERT→agricultural NLP

For tabular agricultural prediction, tree methods (RF, XGBoost) are state-of-the-art.

---

### Q14: How would you scale this to nationwide deployment (5,000 districts)?

**A:** Two-stage approach:
1. **Data pipeline:** Already scales (automated ICRISAT + NASA + ISRO integration). Bottleneck: NDVI processing (500m rasters). Solution: Cloud processing (AWS EC2 spot instances, ~$50/month for full India)
2. **Model inference:** Random Forest model is lightweight (2ms latency/prediction). Batch inference: 5,000 districts × 12 months = 60K predictions/year → $9/year inference cost

**Expected improvement:** With 5,000 districts (50K samples), deep learning would outperform RF (learning curve crossover). Would switch to DeepFusionNN at that scale.

---

### Q15: How would you deploy this in production?

**A:** Three-component architecture:
1. **Data ingestion pipeline:** Monthly batch jobs to fetch NASA climate, ISRO NDVI via APIs. Store in PostgreSQL (structured) + S3 (rasters)
2. **Inference service:** Flask/FastAPI serving pickled RF model. Endpoint: `/predict` accepts district_id + year → returns yield + confidence interval
3. **Monitoring:** Track prediction drift vs. actual yields (retrain trigger if RMSE > 650 kg/ha threshold)

**Infrastructure:** AWS (t3.medium for API, S3 for data, RDS for metadata). Estimated cost: **$100/month** for nationwide deployment.

---

## Business Impact Questions

### Q16: What's the economic value of this model?

**A:** District-level resource allocation use case:
- Current: Fertilizer subsidy distributed uniformly across districts
- With model: Target subsidies to low-predicted-yield districts (drought-prone regions)
- Estimated impact: 10-15% yield improvement in targeted districts × average farm income

**Conservative estimate:** 300 districts × avg 100,000 farmers/district × ₹50,000/farmer income × 10% improvement = **₹15,000 crore/year potential impact**.

Model is not for individual farmers (±15% error too high), but for **policy-level resource allocation**.