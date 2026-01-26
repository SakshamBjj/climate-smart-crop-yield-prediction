# Patent Information and Intellectual Property

## Published Patent Application

**Application Number:** 202541116475 A  
**Title:** Deep Fusion Neural Network System for Crop Yield Prediction  
**Publication Date:** December 12, 2025  
**Publication Journal:** The Patent Office Journal No. 50/2025  
**Status:** Published, Under Examination  
**Expected Grant:** 18-24 months from filing (typical timeline)

**Applicant:** Vellore Institute of Technology  
**Inventors:**
1. Dr. Jayakumar K (Supervisor, Associate Professor Sr., VIT SCOPE)
2. Saksham Bajaj (Lead Developer, BTech CSE 2021-2025)
3. Rishabhraj Srivastava (Co-Developer, BTech CSE 2021-2025)
4. Harshit Vijay Kumar (Co-Developer, BTech CSE 2021-2025)

**Jurisdiction:** India (International Classification: G01S 19/08, G06Q 30/0201, G01S 19/04, G01S 19/05, H04M 1/2753)

---

## What Is Patented?

### Patent Scope (From Abstract)

The patent covers a **system architecture** comprising:

1. **Data Acquisition Module**
   - Collects multi-source agricultural data:
     - Satellite imagery (vegetation indices: NDVI, VCI)
     - Climate parameters from meteorological APIs (NASA POWER)
     - Historical crop yield records at district level (ICRISAT)

2. **Data Preprocessing Pipeline**
   - Harmonizes heterogeneous data sources through:
     - Temporal alignment (daily → 16-day → annual)
     - Spatial harmonization (grids → district boundaries)
     - Feature scaling and normalization

3. **Deep Fusion Neural Network Architecture**
   - **Climate branch:** Processes meteorological and vegetation indices
   - **Geospatial branch:** Processes geographical coordinates
   - Separate processing branches for different data types
   - Multi-head attention mechanism combines outputs
   - Learns complex interactions between climate and location features

4. **Prediction Engine**
   - Generates district-level crop yield forecasts
   - Based on combined outputs from fusion architecture

---

## What Is NOT Patented (Prior Art)

To be clear about what the patent **does not claim**:

### Standard Machine Learning Components (Prior Art)
- ❌ **Multi-head attention mechanism** (Vaswani et al., 2017: "Attention Is All You Need")
- ❌ **Convolutional neural networks** (LeCun et al., 1998)
- ❌ **LSTM networks** (Hochreiter & Schmidhuber, 1997)
- ❌ **Random Forest / XGBoost** (Breiman, 2001; Chen & Guestrin, 2016)
- ❌ **Dropout, batch normalization** (Srivastava et al., 2014; Ioffe & Szegedy, 2015)

### Standard Agricultural Data Processing (Prior Art)
- ❌ **NDVI calculation** ((NIR - Red) / (NIR + Red)) — Rouse et al., 1974
- ❌ **Growing Degree Days (GDD)** — McMaster & Wilhelm, 1997
- ❌ **Zonal statistics** (raster → polygon aggregation) — GDAL standard operations

### Generic System Design Patterns (Prior Art)
- ❌ **Data preprocessing pipelines** (ETL patterns)
- ❌ **Modular software architecture** (separation of concerns)
- ❌ **API-based data acquisition** (REST API design)

---

## What Makes This Patent Novel?

### Novelty Claim 1: Multi-Source Integration at District Level

**Prior Work:**
- You et al. (2017): Satellite-only, county-level (US)
- Patel et al. (2023): Climate + Satellite, state-level (India)
- Jin et al. (2017): Multi-satellite fusion, county-level (US)

**Our Innovation:**
- **First system** to integrate ICRISAT (district, annual) + NASA POWER (grid, daily) + ISRO VEDAS (500m, 16-day)
- **District-level** resolution for India (administrative unit for policy)
- **Automated harmonization** (not manual data alignment)

**Patent Claim:** The specific combination and harmonization methodology at this resolution.

---

### Novelty Claim 2: Crop-Specific Temporal Alignment

**Standard Practice:** Use calendar year (Jan-Dec) for all crops

**Our Innovation:**
- **Kharif crops** (Rice, Maize): Align June Year_N to October Year_N
- **Rabi crops** (Wheat): Align November Year_N to March Year_(N+1)
- **Zaid crops** (Vegetables): Align March to June

**Why This Matters:**
- Wheat harvested in March 2016 is reported as "2015 yield" in ICRISAT
- Must align climate data to **crop year**, not calendar year
- System automates this alignment based on crop type

**Patent Claim:** Automated crop-specific temporal windowing methodology.

---

### Novelty Claim 3: Multi-Branch Fusion for Agricultural Data

**Prior Work in Multi-Modal Fusion:**
- Computer Vision: Image + Text fusion (CLIP, DALL-E) — OpenAI, 2021
- Medical Imaging: CT + MRI fusion — Litjens et al., 2017
- **But not applied to agricultural tabular + geospatial data**

**Our Innovation:**
- Separate branches for climate (dense, 12 features) vs. geospatial (sparse, 2 features)
- **Asymmetric branch sizes** (128 vs. 64) prevents feature imbalance
- Attention fusion learns **location-specific climate sensitivities**
  - Example: Coastal districts weight humidity higher than inland districts

**Patent Claim:** The specific architecture design for agricultural multi-modal data (tabular + geospatial).

---

### Novelty Claim 4: Scalable System Design for Nationwide Deployment

**Prior Work:**
- Academic research systems: Single-region, manual processing
- Commercial systems: Proprietary, not scalable

**Our Innovation:**
- **Modular architecture** (data acquisition, preprocessing, prediction are independent)
- **Scalable to 5,000+ districts** (current: 300, but system generalizes)
- **Automated data pipelines** (no manual intervention)
- **Documented reproducibility** (others can deploy)

**Patent Claim:** The system architecture as a whole, enabling operational deployment.

---

## Patent vs. Implementation: Important Distinction

### What the Patent Protects
**The method and system design:**
- How to integrate multi-source agricultural data at district level
- How to harmonize temporal and spatial resolutions automatically
- How to design a neural network for agricultural prediction
- How to scale this to nationwide coverage

### What the Patent Does NOT Protect
**The specific implementation:**
- Choice of Python vs. Java
- Use of TensorFlow vs. PyTorch
- Specific hyperparameters (learning rate, batch size)
- Exact number of layers or neurons

**Analogy:**
- Patent: "A recipe for integrating agricultural data using neural networks"
- Implementation: "A specific cake baked following that recipe"
- **Others can bake their own cake (different code), but must follow the recipe (patented method)**

---

## Why Code Is Not Publicly Released (Yet)

### Reason 1: Patent Examination in Progress

**Status:** Application published (Dec 2025), but **not yet granted**

**Timeline:**
- Nov 2025: Patent filed
- Dec 2025: Patent published (18-day early publication)
- Jan 2026: Current status (examination ongoing)
- Mid 2027 (estimated): Patent granted or rejected

---

**Risk if code released prematurely:**
- Competitors could argue "publicly disclosed before grant → invalidates claims"
- **Note:** This is a conservative interpretation. Many patents release code post-publication.
- VIT legal counsel recommends waiting until **grant** (not just publication)

---

### Reason 2: Dataset Licensing Restrictions

**ICRISAT TCI Database License:**
- **Available for:** Academic research, non-commercial use
- **Requires:** Institutional agreement (VIT has this)
- **Prohibits:** Redistribution without ICRISAT permission

**Implication:**
- We cannot release the **preprocessed dataset** publicly
- Code without data is not reproducible
- Must provide **data access protocol** for researchers

**Solution in Progress:**
- Negotiating with ICRISAT for data-sharing agreement
- Alternative: Release code + sample data (100 districts, anonymized)

---

### Reason 3: Institutional IP Policy Compliance

**VIT Technology Transfer Office Policy:**
- Faculty/student inventions during research = VIT owns IP
- Code release requires **approval from IP committee**
- Process: Submit request → Review (30 days) → Approval

**Current Status:**
- Patent filing approved (Nov 2025)
- Code release request submitted (Dec 2025)
- **Pending approval** (expected Jan 2026)

**Why the delay?**
- VIT exploring commercialization options (license to agritech startups)
- If commercialized → code remains proprietary
- If not → code will be open-sourced under Apache 2.0 license

---

## Code Availability Roadmap

### Phase 1: Verified Researchers (Current - Mid 2026)

**Who qualifies:**
- Academic researchers with institutional email
- Government agricultural departments
- Non-profit organizations

**What's provided:**
- Full source code (training + inference)
- Preprocessed sample dataset (100 districts, 5 years)
- Validation protocol (reproduce our RMSE results)
- Documentation (setup, usage, API reference)

**How to request:**
- Email: saksham.bajaj2021@vitstudent.ac.in
- Include: Affiliation, research purpose, expected publications
- Response time: 7-14 days
- Agreement: Non-disclosure, academic use only

---

### Phase 2: Open-Source Release (Post Patent Grant, ~Mid 2027)

**License:** Apache 2.0 + Patent Grant

**What this means:**
- ✅ Anyone can use the code for **any purpose** (commercial or non-commercial)
- ✅ No royalty fees
- ✅ Can modify and redistribute
- ⚠️ **With patent grant:** VIT grants you license to use the patented method
  - You won't get sued for patent infringement if you use this code
  - But: If you implement the method **independently** (without this code), you might need a license

**Example (similar to TensorFlow):**
- TensorFlow is Apache 2.0 licensed (open-source)
- But Google holds patents on some TensorFlow methods
- **Patent grant** in TensorFlow license means: "Use our code = patent license included"

**Release package:**
```
climate-smart-crop-yield/
├── src/                        # Full source code
├── data/                       # Sample datasets
│   ├── sample_300_districts.csv
│   └── data_access_guide.md    # How to get full ICRISAT data
├── models/                     # Pre-trained weights
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── deep_fusion.pth
├── notebooks/                  # Jupyter tutorials
│   ├── 01_data_loading.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── docs/                       # API documentation
├── tests/                      # Unit tests
├── requirements.txt
├── setup.py
├── LICENSE                     # Apache 2.0 + Patent Grant
└── README.md
```

---

### Phase 3: Commercial Licensing (Parallel Track)

**For companies wanting commercial deployment:**

**What's offered:**
- Exclusive license for specific regions (e.g., Maharashtra state)
- Customization support (adapt to company's data sources)
- Service-level agreements (SLA for model updates)
- Liability coverage (if model fails, VIT provides compensation)

**Pricing model (estimated):**
- One-time license fee: ₹10-50 lakhs (depends on region size)
- Annual maintenance: 10% of license fee
- Revenue sharing: 2-5% of product revenue (if applicable)

**Target customers:**
- Agritech startups (CropIn, Fasal, Ninjacart)
- Insurance companies (crop insurance products)
- Government agencies (state agricultural departments)
- International NGOs (FAO, CGIAR)

**Contact:** VIT Technology Transfer Office (patents@vit.ac.in)

---

## Frequently Asked Questions

### Q1: "I want to reproduce your results for my research. Can I get the code?"

**A:** Yes, if you're an academic researcher.

**Process:**
1. Email saksham.bajaj2021@vitstudent.ac.in with:
   - Your institutional affiliation
   - Research purpose (thesis, paper, project)
   - Expected publication venue
2. Sign academic use agreement (non-disclosure, cite our work)
3. Receive code + sample data within 14 days

**Alternative (no agreement needed):**
- Full methodology is documented in this repository
- You can **reimplement from scratch** using our specifications
- This is **not patent infringement** (you're not using our code, just the idea)
- **But:** If you publish, please cite our work

---

### Q2: "Can I use this for my startup/company?"

**A:** After open-source release (mid 2027), yes (under Apache 2.0 + Patent Grant).

**Before release:**
- Contact VIT Technology Transfer Office for commercial license
- Licensing process: 2-3 months
- Costs: Negotiable (depends on company size, region)

**Special case: Funded research projects**
- If your company has a research grant (ICAR, DBT, DST), we can collaborate
- Joint research agreements available
- Contact: Dr. Jayakumar K (jayakumar.k@vit.ac.in)

---

### Q3: "If I build my own crop yield model using similar data sources, will I infringe your patent?"

**A:** Depends on how similar your **system design** is.

**Probably NOT infringement:**
- Using ICRISAT + NASA data (publicly available)
- Using Random Forest or XGBoost (prior art)
- Using NDVI or GDD (standard agricultural metrics)

**Probably IS infringement:**
- Building a system with the **same architecture** (multi-branch fusion with climate + geo branches)
- Using the **same temporal alignment method** (crop-specific windowing)
- Replicating our **automated data harmonization pipeline** exactly

**Safe approach:**
- Design your own architecture (e.g., single-branch MLP, or ensemble of separate models)
- If unsure, consult a patent attorney

**Our stance:**
- We encourage research and innovation
- Patent is to protect VIT's IP, not to block academic research
- If you're a researcher, just cite our work — we won't sue

---

### Q4: "Why did you patent this if you believe in open science?"

**A:** Valid question. Here's our reasoning:

**Why we filed the patent:**
1. **VIT institutional requirement:** Faculty/student research outcomes must be reviewed for IP
2. **Protect against commercialization without attribution:** Without a patent, a company could take our work, commercialize it, and never cite us
3. **Enable open licensing:** Apache 2.0 + Patent Grant **requires a patent to exist first**
   - Without patent: We can open-source code, but can't prevent patent trolls from patenting our method
   - With patent: We hold the patent, grant it freely with open-source license

**Our commitment:**
- **No intent to sue academic researchers** (cite our work, you're fine)
- **No intent to charge unreasonable licensing fees** (commercial licenses will be affordable for startups)
- **Will open-source post-grant** (target: mid 2027)

**Inspiration:**
- Tesla open-sourced their patents (2014) to accelerate EV adoption
- Google open-sources TensorFlow (Apache 2.0 + Patent Grant)
- We're following the same model: **Patent to protect, then open-source to enable**

---

### Q5: "What if I implemented this before your patent was filed (Nov 2025)?"

**A:** You have **prior use rights** (in patent law).

**If you can prove:**
- You implemented a similar system before Nov 24, 2025
- You did so independently (not by copying our work)
- You have documentation (code commits, research notes)

**Then:**
- You can **continue using your implementation** (even if our patent is granted)
- **But:** You cannot expand it commercially without a license

**Example:**
- You built a district-level yield model in 2024 → You're safe
- You want to scale it to 5,000 districts in 2026 → Might need a license (depends on similarity)

**Advice:** Consult a patent attorney if you're in this situation.

---

## Summary: Who Can Use This Work?

| User Type | Current (Pre-Grant) | Post-Grant (Mid 2027+) |
|-----------|---------------------|------------------------|
| **Academic Researchers** | ✅ Request code via email | ✅ Open-source (Apache 2.0) |
| **Students (thesis/projects)** | ✅ Request code via email | ✅ Open-source (Apache 2.0) |
| **Government Agencies** | ✅ Contact VIT for collaboration | ✅ Open-source (Apache 2.0) |
| **Startups (non-commercial)** | ⚠️ Request code (case-by-case) | ✅ Open-source (Apache 2.0) |
| **Companies (commercial)** | ❌ License required | ✅ Open-source (Apache 2.0 + Patent Grant) |
| **NGOs** | ✅ Contact VIT for partnership | ✅ Open-source (Apache 2.0) |

**Key Takeaway:**
- **Research use:** Always allowed (just cite our work)
- **Commercial use:** Will be freely allowed post-grant (Apache 2.0)
- **Before grant:** Contact us for code access

---

## Citation

If you use this work (data, methodology, or code), please cite:

**Conference Paper (Recommended):**
```
@inproceedings{bajaj2025crop,
  title={Deep Learning Approach for Crop Yield Prediction using Intelligent Climate Change Prediction},
  author={Bajaj, Saksham and Srivastava, Rishabhraj and Kumar, Harshit Vijay and Jayakumar, K},
  booktitle={Capstone Project, VIT Vellore},
  year={2025},
  note={Patent Application No. 202541116475 A}
}
```

**Patent Reference:**
```
@misc{bajaj2025patent,
  title={Deep Fusion Neural Network System for Crop Yield Prediction},
  author={Jayakumar, K and Bajaj, Saksham and Srivastava, Rishabhraj and Kumar, Harshit Vijay},
  year={2025},
  note={Indian Patent Application No. 202541116475 A, Published Dec 12, 2025},
  applicant={Vellore Institute of Technology}
}
```

---

## Contact Information

**For Research Collaboration:**
📧 saksham.bajaj2021@vitstudent.ac.in  
📧 jayakumar.k@vit.ac.in (Faculty Supervisor)

**For Commercial Licensing:**
🏛️ VIT Technology Transfer Office  
📧 patents@vit.ac.in  
📞 +91-416-220-2108

**For Patent Inquiries:**
🔗 Indian Patent Office: https://ipindiaonline.gov.in/patentsearch/  
🔍 Search Application No.: 202541116475 A

---

**Last Updated:** January 2026  
**Next Update:** Post patent examination (estimated mid 2027)

---
