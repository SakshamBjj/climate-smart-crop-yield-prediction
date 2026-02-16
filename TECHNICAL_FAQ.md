# Technical FAQ

*Common questions from engineers, researchers, and recruiters. For full methodology detail, see [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) and [EVALUATION.md](EVALUATION.md).*

## System Classification

### Q0: Isn't this just feature engineering and model comparison?

No — and the distinction is worth stating clearly before anything else.

A standard tabular ML workflow:
```
dataset → feature engineering → model selection → evaluation
```

This system:
```
heterogeneous sources (3 formats, 3 resolutions, 3 institutions)
    → spatial alignment (500m rasters → irregular district polygons)
    → crop-specific temporal alignment (Kharif vs Rabi growing windows)
    → physiological feature construction (GDD, MVC-corrected NDVI, heat stress)
    → regime check (samples/parameter ratio → model selection)
    → model layer (tree ensembles now; DeepFusionNN at national scale)
```

The contribution is the automated harmonization and adaptive modeling architecture. Any individual step — zonal statistics, GDD calculation, MVC — is standard practice. The system that connects them, aligns them without leakage, and selects the appropriate model based on data regime is not.

**On regime-based selection:** The system evaluates sample/parameter ratios and learning curve convergence to determine which model family is appropriate for the current data scale. At 7,200 samples (0.034 samples/parameter for DeepFusionNN vs 0.0006 for RF), tree ensembles are optimal. At projected 50K+ samples, the architecture transitions to DeepFusionNN. This adaptive selection is the framework's core contribution.

To put it concretely: removing any one connection in the chain above changes the results. Replacing crop-specific temporal windows with calendar-year aggregation loses the monsoon onset signal — late monsoon onset (July vs June) causes 30–48% yield errors in affected districts, a signal that annual aggregates erase entirely (see case studies in [EVALUATION.md](EVALUATION.md)). Using centroid-only spatial aggregation produced 12% higher RMSE. Switching to a random split instead of temporal split would have hidden the drought-year failure entirely. A feature engineering + model comparison workflow has no equivalent to these structural decisions.

---

## Data Engineering

### Q1: How did you handle 15% missing NDVI from cloud contamination?

Maximum Value Composite (MVC): for each 16-day window, take the highest NDVI value across that period and adjacent periods. Clouds suppress NDVI to near-zero; vegetation maxima approximate the cloud-free signal. **Result: −45 kg/ha RMSE vs simple mean imputation.**

Deleting gaps was rejected — monsoon months are the most informative period for Kharif crops; losing that signal was worse than the approximation.

---

### Q2: Why district-level instead of state or field?

- **State-level:** Too coarse — a single state like Maharashtra spans 7 agro-climatic zones
- **Field-level:** Ground truth doesn't exist at historical scale in India; ICRISAT only records at district resolution
- **District-level:** Matches agricultural administrative units (resource allocation, subsidy targeting); ground truth available from ICRISAT

---

### Q3: How did you align daily climate data to annual yield?

Crop-specific growing season windows:
- **Kharif (Rice/Maize):** June–October of crop year
- **Rabi (Wheat):** November–March (spans two calendar years)

Climate variables are aggregated within the relevant window and matched to the yield reported at harvest. Example: wheat yield 2015 uses climate from November 2015–March 2016.

**Known limitation:** Annual aggregates lose sub-seasonal timing. Late monsoon onset (July vs June) reduces yields by up to 30% but disappears in annual GDD sums. Monthly sequences would capture it — highest-priority improvement in [EVALUATION.md](EVALUATION.md).

---

### Q4: What was the hardest data integration challenge?

Spatial harmonization. NASA POWER uses a regular 0.5° lat-lon grid; district boundaries are irregular administrative polygons. Solution: area-weighted averaging — small districts use the nearest grid cell; large districts (>2,500 km²) use the weighted average of 4 nearest cells by area overlap.

Centroid-only approach vs area-weighted: **12% RMSE improvement** with area weighting.

---

## Modeling & Evaluation

### Q5: Why did Random Forest beat your custom deep learning architecture?

The framework selected tree ensembles for this dataset — that is the correct outcome, not a failure of the architecture.

DeepFusionNN is the interaction modeling layer of the system, designed to learn coupled relationships between climate accumulation, vegetation state, and regional geography. It is the appropriate component at national scale. At 7,200 samples it is not — learning curve analysis shows why:

- Validation loss plateaus at ~3,000 samples
- DeepFusionNN: 170K parameters → 0.034 samples/parameter
- Random Forest: ~0.0006 samples/parameter → sample-efficient for tabular data at this scale

Power-law extrapolation puts the crossover at **5,000+ districts (~50K samples)**. The experiment validated the selection logic. The system chose correctly.

---

### Q5a: So why build DeepFusionNN if RF worked?

The goal was a scalable national forecasting system, not a single-dataset solution. RF is optimal at ~7K samples; the architecture transitions to DeepFusionNN at ~50K. Building and validating both components — and demonstrating the regime check that selects between them — is the contribution.

Functionally, the two models handle features differently. Tree models treat GDD, NDVI, and precipitation as independent conditional splits. DeepFusionNN models them as coupled relationships across growing phases — it learns crop response behavior rather than correlation rules. That distinction matters at scale, where interactions between climate variables become the signal.

---

### Q6: How do you know the model isn't overfitting?

Three indicators:

1. **Feature importance is biologically interpretable** — GDD (32%), NDVI (18%), PRECTOT (14%) are established agronomic indicators; no spurious high-importance features
2. **Temporal test set** — evaluated on held-out years 2016–2017, not a random split
3. **Error patterns are interpretable** — performance degrades predictably during drought years (+25% error) and in semi-arid regions (+30%), not randomly

An overfit model would show: high train R² / low test R², spurious features (longitude at 30% importance), random error distribution.

---

### Q7: Why temporal split instead of random split?

Temporal split simulates the real deployment scenario: predict 2016–2017 from 2008–2015. Benefits:
- Catches temporal drift — model's drought-year weakness would be invisible in a random split
- Prevents data leakage — no future climate patterns in training
- Honest benchmark — random split would give ~20% higher R² but that number is fictitious for a forecasting system

---

### Q8: What metric did you optimize for?

Primary: **RMSE** (same units as target, penalizes large errors more than MAE, standard in yield prediction literature). Also reported R² and MAE for completeness. Used Huber loss for DL training — more robust to yield outliers from agricultural reporting errors.

---

### Q9: How do you know the model isn't just learning the dataset mean?

R² = 0.78 means 78% of variance is explained. A mean-prediction baseline gives R² = 0. Additionally, feature importance validates that the model captures crop-climate interactions — if it were regressing to the mean, GDD and NDVI wouldn't dominate.

---

## Error Analysis

### Q10: What's the model's biggest weakness?

**Drought year generalization.** RMSE is 25–30% higher in drought years (2009, 2015, 2016). Root cause: only 2 drought years in 8 training years. The model systematically underestimates drought impact — actual yield reduction is 40–60%; model predicts 15–25%.

Fix: oversample drought years + add climate anomaly features. Estimated improvement: −15% RMSE on drought years. Full detail in [EVALUATION.md](EVALUATION.md).

---

### Q11: Which regions perform worst?

**Worst:** Semi-arid zones (Rajasthan, Maharashtra Vidarbha, Northern Karnataka) — RMSE >700 kg/ha (+30%). High precipitation variability, heavy monsoon cloud contamination, intercropping confuses NDVI signal.

**Best:** Indo-Gangetic Plain (Punjab, Haryana) — RMSE <450 kg/ha (−22%). 80%+ irrigated, stable climate, clean Rabi-season NDVI.

---

### Q12: Why does wheat outperform rice and maize?

- **Wheat (R²=0.82):** Rabi season → predictable winter climate; 80%+ irrigated; concentrated in high-data-quality Punjab/Haryana
- **Rice (R²=0.76):** Kharif season → monsoon-dependent; cloud contamination during growing season
- **Maize (R²=0.71):** Grown in both Kharif and Rabi; high regional heterogeneity; only 15% of training samples

Proposed fix: crop-specific models. Estimated R² gain for maize: 0.71 → 0.76.

---

## Architecture & Scale

### Q13: Why not use transfer learning?

Transfer learning requires similar domains. This is a **tabular aggregation problem** — satellite imagery was reduced to district-level scalar NDVI values (no spatial structure for CNNs), climate was aggregated to annual totals (no temporal structure for sequence models). For tabular data at <10K samples, tree methods are state-of-the-art. Transfer learning would apply if using raw satellite imagery per pixel, or sub-seasonal time series.

---

### Q14: How would you scale to 5,000+ districts nationally?

**Data pipeline:** Already architected to scale — automated ICRISAT + NASA + ISRO integration. Main bottleneck is NDVI raster processing; addressable with cloud batch jobs (estimated ~$50/month on AWS spot instances for full India).

**Model inference:** RF is lightweight (2ms latency). 5,000 districts × 12 months = 60K predictions/year → ~$9/year inference.

**Model choice at scale:** At 50K samples, DL overtakes RF per learning curve extrapolation. Architecture would switch to DeepFusionNN at that point.

---

### Q15: How would you deploy this in production?

Three components:
1. **Ingestion:** Monthly batch jobs fetching NASA climate and ISRO NDVI via APIs; stored in PostgreSQL (structured) + S3 (rasters)
2. **Inference:** Flask/FastAPI serving pickled RF model — endpoint accepts district_id + year, returns yield + confidence interval
3. **Monitoring:** Track prediction drift vs. actuals; retrain trigger at RMSE > 650 kg/ha threshold

**Estimated infrastructure cost:** ~$100/month on AWS (t3.medium for API, S3 for data, RDS for metadata) for nationwide deployment.

---

## Reproducibility

### Q16: Can I reproduce the results?

Methodology is fully documented. Partial reproduction is possible without institutional access; full reproduction requires ICRISAT institutional agreement.

| Component | Available |
|-----------|-----------|
| Preprocessing methodology | ✅ PIPELINE_OVERVIEW.md |
| Model evaluation methodology | ✅ EVALUATION.md |
| NASA POWER climate data | ✅ Public API |
| ISRO VEDAS NDVI | ✅ Registration at vedas.sac.gov.in |
| ICRISAT yield data | Institutional agreement required |
| Model training code | Protected (patent) — walkthrough on request |

---

### Q17: Can I use this methodology in my own project?

Yes — methodology is not patented. The patent covers the specific automated system integration. You can freely use: the documented data integration approach, feature engineering techniques (GDD, NDVI aggregation), model comparison framework, and evaluation methodology. Cite accordingly.
