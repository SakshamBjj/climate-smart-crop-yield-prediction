# System Architecture

## High-Level System Design
```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION LAYER                    │
├─────────────────────────────────────────────────────────────────┤
│  ICRISAT TCI          NASA POWER API       ISRO VEDAS           │
│  (Annual, District)   (Daily, Grid)        (16-day, 500m)       │
│  • Yield records      • Temperature        • NDVI               │
│  • Area, Production   • Precipitation      • VCI                │
│  • 2008-2017          • Humidity, GDD      • Vegetation health  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING & HARMONIZATION                  │
├─────────────────────────────────────────────────────────────────┤
│  Temporal Alignment         Spatial Harmonization               │
│  • Daily → Annual           • Grid → District boundaries        │
│  • 16-day → Annual          • Zonal statistics aggregation      │
│                                                                  │
│  Feature Engineering        Quality Control                     │
│  • GDD calculation          • Missing value detection           │
│  • VCI normalization        • Outlier flagging                  │
│  • Temperature ranges       • Cross-source validation           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        UNIFIED FEATURE VECTOR                    │
├─────────────────────────────────────────────────────────────────┤
│  Climate (11): T2M, PRECTOT, RH2M, GDD, CDD18_3, PS, QV2M,     │
│                T2MDEW, T2M_MAX, Temp_Range, NDVI                │
│  Vegetation (1): VCI (%)                                        │
│  Geospatial (2): Latitude, Longitude                            │
│                                                                  │
│  Target: Crop Yield (kg/ha) per district per year              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                          MODELING LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐        ┌──────────────────┐               │
│  │  Traditional ML │        │  Deep Learning   │               │
│  ├─────────────────┤        ├──────────────────┤               │
│  │ • Random Forest │        │ • DeepFusionNN   │               │
│  │ • XGBoost       │        │ • CNN-LSTM       │               │
│  │                 │        │                  │               │
│  │ ✓ Best RMSE     │        │ ✗ Underperformed │               │
│  │ ✓ Low cost      │        │ ✗ Data-hungry    │               │
│  └─────────────────┘        └──────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PREDICTION & EVALUATION                     │
├─────────────────────────────────────────────────────────────────┤
│  District-Level Yield Forecasts                                 │
│  • Point predictions (kg/ha)                                    │
│  • Uncertainty quantification (future work)                     │
│  • Crop-specific outputs (Rice, Wheat, Maize)                  │
│                                                                  │
│  Validation & Interpretation                                    │
│  • Temporal validation (2008-2015 train, 2016-2017 test)       │
│  • Spatial error analysis (which districts fail?)               │
│  • Feature importance (which variables drive yield?)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## DeepFusionNN Architecture Specification

### Conceptual Design

The DeepFusionNN implements a **multi-branch architecture** where different data modalities are processed separately before fusion:
```
Input (14 features)
    │
    ├──────────────────────┬──────────────────────┐
    │                      │                      │
Climate Features (11)  Vegetation (1)    Geospatial (2)
    │                      │                      │
Climate Branch (128)   [Merged w/ Climate]   Geo Branch (64)
    │                                             │
    └───────────────── Attention Fusion ─────────┘
                           │
                      Fused Repr (192)
                           │
                    Prediction Head (96 → 1)
                           │
                    Yield Prediction (kg/ha)
```

### Layer-by-Layer Specification

#### Branch 1: Climate + Vegetation Encoder
```
Input: [T2M, PRECTOT, RH2M, NDVI, VCI, GDD, CDD18_3, PS, QV2M, T2MDEW, T2M_MAX, Temp_Range]
    ↓
Linear(12 → 128)
    ↓
LayerNorm(128)
    ↓
ReLU activation
    ↓
Dropout(p=0.3)
    ↓
Output: climate_features [batch_size, 128]
```

**Rationale:**
- Climate features are **dense** (12 dimensions with high correlation)
- Larger representation (128) captures complex interactions (e.g., GDD × precipitation)
- Dropout prevents overfitting on limited data

#### Branch 2: Geospatial Encoder
```
Input: [Latitude, Longitude]
    ↓
Linear(2 → 64)
    ↓
LayerNorm(64)
    ↓
ReLU activation
    ↓
Dropout(p=0.2)
    ↓
Output: geo_features [batch_size, 64]
```

**Rationale:**
- Geographic features are **sparse** (only 2 dimensions)
- Smaller representation (64) prevents overfitting
- Encodes spatial patterns (latitude ≈ climate zones, longitude ≈ monsoon gradients)

#### Fusion Layer: Multi-Head Attention
```
Query/Key/Value: concat(climate_features, geo_features) → [batch_size, 192]
    ↓
MultiHeadAttention(embed_dim=192, num_heads=4)
    ↓
Output: attended_features [batch_size, 192]
```

**Rationale:**
- **Why attention?** Learns which climate features matter for specific locations
  - Example: Coastal districts → humidity weights up
  - Example: Inland districts → temperature range weights up
- **Why 4 heads?** Balances expressiveness vs. overfitting risk
- **Why self-attention vs. cross-attention?** Patent application uses self-attention on concatenated features (simpler, works for tabular data)

#### Prediction Head
```
attended_features [batch_size, 192]
    ↓
Linear(192 → 96)
    ↓
ReLU activation
    ↓
Dropout(p=0.3)
    ↓
Linear(96 → 1)
    ↓
Output: yield_prediction [batch_size, 1]
```

**Rationale:**
- Two-layer head provides non-linearity without excessive parameters
- Final output is unbounded (yields can vary widely across crops/districts)

### Training Configuration

- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-5)
- **Loss Function:** Huber Loss (robust to outliers in yield data)
- **Batch Size:** 64
- **Epochs:** 100 (with early stopping on validation RMSE)
- **Regularization:**
  - Dropout: 0.3 (climate branch), 0.2 (geo branch)
  - Gradient clipping: max_norm=1.0
- **Hardware:** Apple M1 (MPS backend) / NVIDIA GPU (CUDA)

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Climate branch | 12×128 + 128 = 1,664 |
| Geo branch | 2×64 + 64 = 192 |
| Attention | 192×192×4 + biases ≈ 150K |
| Prediction head | 192×96 + 96×1 ≈ 18.5K |
| **Total** | **~170K parameters** |

**Context:** Random Forest with 500 trees has ~10M effective parameters, yet outperformed DeepFusionNN. This confirms the hypothesis: **model capacity isn't the bottleneck, data size is.**

---

## CNN-LSTM Architecture Specification

### Conceptual Design

Treats input features as a **time-series sequence** (even though data is annual, features represent temporal accumulation).
```
Input Features (14) → Sequence Length 1
    ↓
1D Convolution (extract local patterns)
    ↓
LSTM (model temporal dependencies)
    ↓
Prediction Head
    ↓
Yield (kg/ha)
```

### Layer Specification
```
Input: [batch_size, 1, 14] (unsqueezed for Conv1D)
    ↓
Conv1D(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    ↓
BatchNorm1D(32)
    ↓
ReLU
    ↓
MaxPool1D(kernel_size=2) → [batch_size, 32, 7]
    ↓
Conv1D(32 → 64, kernel_size=3, padding=1)
    ↓
BatchNorm1D(64)
    ↓
ReLU
    ↓
MaxPool1D(kernel_size=2) → [batch_size, 64, 3]
    ↓
Permute to [batch_size, 3, 64] (sequence_length, features)
    ↓
Bidirectional LSTM(input_size=64, hidden_size=128, num_layers=2)
    ↓
Take final hidden state: [batch_size, 256] (128×2 for bidirectional)
    ↓
Linear(256 → 128) → ReLU → Dropout(0.3)
    ↓
Linear(128 → 1)
    ↓
Output: yield_prediction
```

**Why this underperformed:**
- Designed for **sequential data** (e.g., monthly climate time series)
- Our data is **annual aggregates** → no temporal structure to exploit
- LSTM overhead without temporal benefit

---

## Traditional ML Baselines

### Random Forest
- **Estimators:** 500 trees
- **Max Depth:** 12
- **Criterion:** MSE
- **Features:** All 14 (no feature selection)
- **Why it won:** Handles non-linear interactions, robust to outliers, interpretable feature importance

### XGBoost
- **Estimators:** 500 boosting rounds
- **Learning Rate:** 0.05
- **Max Depth:** 7
- **Objective:** reg:squarederror
- **Why it won:** Gradient boosting captures complex patterns, efficient handling of missing values (NDVI gaps)

---

## Design Rationale: Why Multi-Branch?

**Alternative 1:** Single MLP on concatenated features
```
[Climate + Geo] → FC(14 → 128) → FC(128 → 1)
```
**Problem:** Treats all features equally. Geographic features (2D) drown in climate features (12D).

**Alternative 2:** Separate models per modality, ensemble predictions
```
Climate_Model → Pred_1
Geo_Model → Pred_2
Final = 0.8×Pred_1 + 0.2×Pred_2
```
**Problem:** No learned interaction between modalities.

**Our Approach:** Multi-branch + attention fusion
```
Climate_Branch(128) + Geo_Branch(64) → Attention → Fused(192)
```
**Advantage:** 
- Each modality gets appropriate representation size
- Attention learns **which climate features matter for each location**
- Patent-defensible as novel system design (even if components are standard)

---

## Comparison to Prior Art

| System | Data Sources | Resolution | Architecture | Year |
|--------|--------------|------------|--------------|------|
| You et al. | Satellite only | County (US) | Deep Gaussian Process | 2017 |
| Kussul et al. | Satellite only | 500m grid | CNN | 2017 |
| Patel et al. | Climate + Satellite | State (India) | CNN-LSTM | 2023 |
| **This work** | **Climate + Satellite + Yield** | **District (India)** | **Multi-branch Fusion** | **2025** |

**Key novelty:** First system to harmonize ICRISAT + NASA + ISRO at district-level with multi-branch architecture.

---

## Future Architecture Improvements

1. **Spatial attention:** Incorporate district adjacency (neighboring districts have correlated yields)
2. **Temporal RNN:** Use monthly climate sequences (not annual aggregates)
3. **Transfer learning:** Pre-train on US/China datasets, fine-tune on India
4. **Uncertainty quantification:** Bayesian layers or dropout-based ensembles

---

**See Also:**
- MODELING_AND_EXPERIMENTS.md (training details)
- ERROR_ANALYSIS.md (why this architecture underperformed on limited data)