# Project Planning Documents
## AI for Trustworthy Decision Making - COVIDx CXR-4 Project

---

## Quick Navigation

| Document | Description |
|----------|-------------|
| [00_PROJECT_OVERVIEW.md](./00_PROJECT_OVERVIEW.md) | Complete project overview, repository structure, checklists |
| [01_MEMBER1_DATA_LEAD.md](./01_MEMBER1_DATA_LEAD.md) | Detailed plan for Data & Experiment Design Lead |
| [02_MEMBER2_MODELING_LEAD.md](./02_MEMBER2_MODELING_LEAD.md) | Detailed plan for Modeling & Privacy Lead |
| [03_MEMBER3_EVALUATION_LEAD.md](./03_MEMBER3_EVALUATION_LEAD.md) | Detailed plan for Evaluation & Interpretability Lead |
| [04_GANTT_CHART.md](./04_GANTT_CHART.md) | Visual timeline, dependencies, meeting schedule |

---

## Project at a Glance

```
PROJECT: AI for Trustworthy Decision Making
DATASET: COVIDx CXR-4 (Chest X-ray classification)
TEAM: 3 Members
DURATION: 5 Weeks
TOTAL POINTS: 6 (Stage 1: 2pts, Stage 2: 4pts)

TRUST DIMENSIONS:
  - Privacy: Differential Privacy (DP-SGD)
  - Interpretability: Grad-CAM, SHAP
```

---

## Team Roles Summary

| Role | Primary Responsibilities | Trust Dimension |
|------|-------------------------|-----------------|
| **Member 1: Data Lead** | Dataset, preprocessing, federated splits, attacks | Data robustness |
| **Member 2: Modeling Lead** | Model architecture, FL implementation, DP | Privacy |
| **Member 3: Evaluation Lead** | Metrics, evaluation, interpretability | Interpretability |

---

## Timeline Summary

```
Week 1: Foundation (Data pipeline, Model setup)
Week 2: Stage 1 Delivery (Training, Evaluation, Report)
Week 3: Stage 2 Start (DP, Interpretability implementations)
Week 4: Cross-Evaluation (Model exchange, testing)
Week 5: Final Delivery (Report, Presentation)
```

---

## Key Deliverables

### Stage 1 (2 points)
- [ ] Technical Report (3-4 pages)
- [ ] Centralized baseline model
- [ ] Federated model (3 clients)
- [ ] Evaluation comparison
- [ ] Short presentation

### Stage 2 (4 points)
- [ ] Final Report (18-20 pages)
- [ ] DP-enhanced models (ε = 1, 5, 10)
- [ ] Interpretability analysis (Grad-CAM, SHAP)
- [ ] Robustness evaluation
- [ ] Cross-evaluation results
- [ ] Final presentation

---

## Critical Dependencies

```
M1 Data → M2 Training → M3 Evaluation
              ↓
         M2 DP Models → M3 Final Evaluation
              ↓
     M1 Cross-Eval ← M2 Models → M3 Trust Analysis
```

---

## Getting Started

1. **Read your role document** (01, 02, or 03)
2. **Review the Gantt chart** for timing and dependencies
3. **Set up your development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch torchvision flwr opacus scikit-learn matplotlib
   ```
4. **Check the project overview** for repository structure
5. **Mark your first task as started** in the checklist

---

## Communication

- **Primary channel**: [Your team's preferred channel]
- **Repository**: [GitHub/GitLab link]
- **Meeting schedule**: See [04_GANTT_CHART.md](./04_GANTT_CHART.md)

---

## Estimated Effort per Member

| Member | Stage 1 | Stage 2 | Total |
|--------|---------|---------|-------|
| M1: Data Lead | ~25h | ~30h | ~55h |
| M2: Modeling Lead | ~30h | ~35h | ~65h |
| M3: Evaluation Lead | ~25h | ~40h | ~65h |
