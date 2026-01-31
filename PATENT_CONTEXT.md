# Patent & IP Context

## Patent Status

**Application Number:** 202541116475 A  
**Title:** Deep Fusion Neural Network System for Crop Yield Prediction  
**Status:** Published (December 12, 2025) – Under Examination  
**Jurisdiction:** India  
**Applicant:** Vellore Institute of Technology  
**Inventors:** Dr. Jayakumar K, Saksham Bajaj, Rishabhraj Srivastava, Harshit Vijay Kumar

**Public Record:** [IP India Patent Search](https://ipindiaonline.gov.in/patentsearch/)

---

## Code & Data Availability

### What's Public (This Repository)

✅ **System Architecture** – High-level design and component overview  
✅ **Methodology** – Data integration, feature engineering, model selection  
✅ **Evaluation** – Complete results, error analysis, failure modes  
✅ **Visualizations** – Performance plots and reproducible generation scripts  
✅ **Engineering Decisions** – Design rationale and tradeoff analysis  

### What's Protected

🔒 **Full Implementation Code** – Dataset ETL, model training pipelines  
🔒 **DeepFusionNN Architecture** – Layer-by-layer specifications (patent-protected)  
🔒 **Proprietary Datasets** – ICRISAT requires institutional access agreement  
🔒 **Production Infrastructure** – Deployment configurations and API keys  

---

## Why This Approach?

1. **Patent Compliance:** Application under examination – avoiding premature disclosure of protected claims
2. **Dataset Licensing:** ICRISAT TCI database requires institutional agreements (not publicly redistributable)
3. **Institutional Policy:** VIT technology transfer guidelines for student-led research

---

## What the Patent Covers

### Novel System Components (Patent-Protected)

- **Multi-source integration framework** for agricultural data harmonization
- **Crop-specific temporal alignment** methodology (Kharif vs. Rabi windowing)
- **Adaptive spatial aggregation** (500m satellite pixels → irregular district boundaries)
- **DeepFusionNN architecture** with multi-head attention for climate-location interaction

### Standard Practice (Not Patent Claims)

- Zonal statistics for spatial aggregation (GDAL standard)
- Growing Degree Days (GDD) calculation (established agronomic method)
- Maximum Value Composite (MVC) for cloud masking (common in remote sensing)
- Random Forest and XGBoost algorithms (open-source implementations)

**Key Distinction:** Patent protects the **system integration and automation**, not individual ML techniques.

---

## Collaboration & Access

### For Recruiters

**Available:**
- Private walkthrough of implementation (screen share or code review)
- Detailed technical discussion of design decisions
- Live demo of data pipeline and model evaluation
- References from project supervisor (Dr. Jayakumar K)

**Contact:** saksham.bjj@gmail.com

### For Academic Researchers

**Available:**
- Methodology documentation and best practices
- Preprocessed sample datasets (100 districts, non-commercial use)
- Collaboration on dataset expansion (5,000+ districts)
- Joint research opportunities

**Contact:**  
Academic inquiries – saksham.bjj@gmail.com  
Institutional partnerships – patents@vit.ac.in

### For Commercial Licensing

**Available:**
- Technology transfer agreements through VIT
- Production deployment support
- Custom model development for specific crops/regions

**Contact:** patents@vit.ac.in

---

## Post-Patent Grant Roadmap

**Upon patent grant (estimated 18-24 months):**

1. **Reference Implementation** (Apache 2.0 + Patent Grant License)
   - Open-source data integration pipeline
   - Model training framework
   - Evaluation toolkit

2. **Sample Datasets** (Research License)
   - 100-district preprocessed subset
   - Synthetic test data for validation
   - Non-commercial use permitted

3. **Trained Model Weights** (Research License)
   - Random Forest ensemble (production-ready)
   - DeepFusionNN checkpoints (experimental)
   - Inference API examples

**Target Release:** Q3 2026 (subject to patent office timeline)

---

## Frequently Asked Questions

### Q: Can I reproduce your results?

**A:** Methodology is fully documented. Researchers can:
- Access ICRISAT database via institutional agreement
- Use NASA POWER API (publicly available)
- Download ISRO VEDAS data (registration required)
- Follow preprocessing steps in PIPELINE_OVERVIEW.md

Full reproduction requires institutional ICRISAT access. Partial validation possible with public datasets.

### Q: Why not use a permissive open-source license?

**A:** Three constraints:
1. **Patent under examination** – premature code release could jeopardize claims
2. **Dataset licensing** – ICRISAT data not redistributable without agreement
3. **Institutional policy** – VIT requires controlled disclosure for student research with patent potential

This is standard practice for academic research with commercial potential.

### Q: Is the patent defensive or commercial?

**A:** **Primarily defensive.** Protects the system design for:
- Academic publication without fear of patent trolls
- Future open-source release with clear IP ownership
- Potential commercial partnerships (agricultural tech companies, government agencies)

VIT's policy supports eventual open-sourcing post-grant.

### Q: Can I use your methodology in my own project?

**A:** Yes, **methodology is not patented**. You can:
- ✅ Use the documented approach for your own agricultural ML projects
- ✅ Implement similar data integration pipelines
- ✅ Apply feature engineering techniques (GDD, NDVI aggregation, etc.)
- ✅ Cite this work in academic publications

Patent covers the **specific automated system**, not general ML techniques.

### Q: How do I cite this work?

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

## Compliance Statement

This repository complies with:
- **VIT IP Policy** (Section 4.2: Student Research Disclosure)
- **India Patent Act 1970** (Section 10: Provisional vs. Complete Specification)
- **ICRISAT Data Usage Agreement** (No redistribution of raw TCI database)
- **GitHub Terms of Service** (Public repository with protected IP notice)

All public materials have been reviewed by VIT Technology Transfer Office.

---

## Updates & Timeline

| Date | Event | Status |
|------|-------|--------|
| Nov 24, 2025 | Patent application filed | ✅ Complete |
| Dec 12, 2025 | Patent published | ✅ Complete |
| Jan 2026 | GitHub repository public | ✅ Current |
| Q2 2026 | First examination report | ⏳ Pending |
| Q3 2026 | Patent grant (estimated) | ⏳ Pending |
| Q4 2026 | Open-source release (target) | ⏳ Pending |

**Stay updated:** Follow this repository for announcements on code release timeline.

---

**Last Updated:** January 2026