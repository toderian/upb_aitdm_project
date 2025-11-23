# Project Gantt Chart & Timeline
## COVIDx CXR-4 Federated Learning Project

---

## Visual Timeline Overview

```
PROJECT TIMELINE (5 WEEKS)
═══════════════════════════════════════════════════════════════════════════════════════════════════

WEEK 1                          WEEK 2                          WEEK 3
Day: 1   2   3   4   5   6   7  │  1   2   3   4   5   6   7  │  1   2   3   4   5   6   7
────────────────────────────────┼──────────────────────────────┼──────────────────────────────
                                │                              │
 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │                              │
      STAGE 1 (2 points)        │                              │
                                │  ████████████████████████████│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                                │      STAGE 1 DELIVERY        │     STAGE 2 (4 points)
                                │                              │
────────────────────────────────┴──────────────────────────────┴──────────────────────────────

WEEK 4                          WEEK 5
Day: 1   2   3   4   5   6   7  │  1   2   3   4   5   6   7
──────────────────────────────┼──────────────────────────────
                                │
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    STAGE 2 CONTINUED         │      FINAL DELIVERY
                                │
══════════════════════════════════════════════════════════════

LEGEND:
▓▓▓ Stage 1 Development
███ Stage 1 Delivery
░░░ Stage 2 Development
```

---

## Detailed Week-by-Week Breakdown

### WEEK 1: Foundation & Data Pipeline

```
WEEK 1
═══════════════════════════════════════════════════════════════════════════════════════════════
         │  MON      │  TUE      │  WED      │  THU      │  FRI      │  SAT      │  SUN      │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 1 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │           │
Data     │ Download  │ Download  │ Preproc   │ Preproc   │ Fed Split │ Fed Split │  Buffer   │
Lead     │ + EDA     │ + EDA     │ Pipeline  │ Dataset   │ non-IID   │ NPZ files │           │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 2 │ ████████  │ ████████  │ ░░░░░░░░  │ ░░░░░░░░  │ ████████  │ ████████  │           │
Modeling │ Model     │ Model     │ ▒▒WAIT▒▒  │ ▒▒WAIT▒▒  │ Server    │ Server    │  Buffer   │
Lead     │ Research  │ Implement │ for data  │ for data  │ Implement │ + Client  │           │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 3 │ ████████  │ ████████  │ ████████  │           │           │           │           │
Eval     │ Metrics   │ Metrics   │ Eval      │  Buffer/  │  Buffer/  │  Buffer/  │  Buffer   │
Lead     │ Research  │ Define    │ Framework │  Support  │  Support  │  Support  │           │
─────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘

KEY DELIVERABLES:
- M1: Downloaded dataset, preprocessing pipeline, federated splits (client_X_data.npz)
- M2: Model architecture, Flower server skeleton
- M3: Metrics module, evaluation framework skeleton

SYNC POINT: Thursday/Friday - Data handoff from M1 to M2
████ Active work   ░░░ Blocked   ▒▒ Waiting
```

### WEEK 2: Model Training & Evaluation

```
WEEK 2
═══════════════════════════════════════════════════════════════════════════════════════════════
         │  MON      │  TUE      │  WED      │  THU      │  FRI      │  SAT      │  SUN      │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 1 │ ████████  │ ████████  │           │           │           │ ████████  │ ████████  │
Data     │ Data      │ Document  │  Buffer/  │  Buffer/  │  Buffer/  │ Report    │ Report    │
Lead     │ Validate  │ Statistics│  Support  │  Support  │  Support  │ Writing   │ Writing   │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 2 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │
Modeling │ Central   │ Central   │ FL Client │ FL Client │ FL Exper  │ FL Exper  │ Report    │
Lead     │ Training  │ Training  │ Complete  │ Test      │ Run       │ Analyze   │ Writing   │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 3 │ ░░░░░░░░  │ ░░░░░░░░  │ ░░░░░░░░  │ ████████  │ ████████  │ ████████  │ ████████  │
Eval     │ ▒▒WAIT▒▒  │ ▒▒WAIT▒▒  │ ▒▒WAIT▒▒  │ Eval      │ Eval      │ Compare   │ Present + │
Lead     │ for model │ for model │ for model │ Central   │ Federated │ Results   │ Report    │
─────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘

KEY DELIVERABLES:
- M1: Data documentation, statistics for report
- M2: Trained centralized model, trained federated model, training logs
- M3: Evaluation results, comparison table, Stage 1 presentation

SYNC POINTS:
- Wednesday: M2 delivers centralized model to M3
- Friday: M2 delivers federated model to M3
- Sunday: Stage 1 report finalization meeting
```

### WEEK 3: Stage 2 Kickoff - Trust Mechanisms

```
WEEK 3
═══════════════════════════════════════════════════════════════════════════════════════════════
         │  MON      │  TUE      │  WED      │  THU      │  FRI      │  SAT      │  SUN      │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 1 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │           │
Data     │ Data      │ Data      │ Label     │ Label     │ Backdoor  │ Backdoor  │  Buffer   │
Lead     │ Variants  │ Variants  │ Flip Atk  │ Flip Atk  │ Attack    │ Attack    │           │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 2 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │           │
Modeling │ DP Study  │ DP-SGD    │ DP-SGD    │ DP Train  │ DP Train  │ DP Train  │  Buffer   │
Lead     │           │ Implement │ Implement │ ε=1       │ ε=5       │ ε=10      │           │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 3 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │           │
Eval     │ Grad-CAM  │ Grad-CAM  │ Grad-CAM  │ SHAP      │ SHAP      │ Adversar  │  Buffer   │
Lead     │ Implement │ Implement │ Test      │ Implement │ Implement │ Attacks   │           │
─────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘

KEY DELIVERABLES:
- M1: Data variants, poisoning attack implementations
- M2: DP training module, DP models at different epsilon
- M3: Grad-CAM implementation, SHAP implementation, adversarial attack module

SYNC POINT: Friday - Check progress on all trust mechanisms
```

### WEEK 4: Cross-Evaluation & Integration

```
WEEK 4
═══════════════════════════════════════════════════════════════════════════════════════════════
         │  MON      │  TUE      │  WED      │  THU      │  FRI      │  SAT      │  SUN      │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 1 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │           │
Data     │ Cross     │ Cross     │ Cross     │ Cross     │ Data Card │ Data Card │  Buffer   │
Lead     │ Eval M2   │ Eval M2   │ Eval M3   │ Eval M3   │ Document  │ Document  │           │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 2 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │           │
Modeling │ DP + FL   │ DP + FL   │ DP + FL   │ Share     │ Cross     │ Cross     │  Buffer   │
Lead     │ Integrate │ Integrate │ Test      │ Models    │ Eval      │ Eval      │           │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 3 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │           │
Eval     │ Robust    │ Robust    │ Robust    │ Receive   │ Eval All  │ Eval All  │  Buffer   │
Lead     │ Test      │ Poison    │ Poison    │ DP Models │ Models    │ Models    │           │
─────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘

KEY DELIVERABLES:
- M1: Cross-evaluation results, data card documentation
- M2: DP + FL model, all models shared, cross-evaluation from M2 perspective
- M3: Robustness results, evaluation of all DP models

SYNC POINTS:
- Wednesday: M2 shares all DP models with team
- Friday: Cross-evaluation results exchange
- Saturday: Integration meeting - compare all results
```

### WEEK 5: Final Integration & Delivery

```
WEEK 5
═══════════════════════════════════════════════════════════════════════════════════════════════
         │  MON      │  TUE      │  WED      │  THU      │  FRI      │  SAT      │  SUN      │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 1 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ░░░░░░░░  │ ░░░░░░░░  │
Data     │ Report    │ Report    │ Report    │ Report    │ Present   │ FINAL    │ FINAL    │
Lead     │ ~6 pages  │ Writing   │ Review    │ Finalize  │ Practice  │ DELIVERY │ DELIVERY │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 2 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ░░░░░░░░  │ ░░░░░░░░  │
Modeling │ Report    │ Report    │ Report    │ Report    │ Present   │ FINAL    │ FINAL    │
Lead     │ ~6 pages  │ Writing   │ Review    │ Finalize  │ Practice  │ DELIVERY │ DELIVERY │
─────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
MEMBER 3 │ ████████  │ ████████  │ ████████  │ ████████  │ ████████  │ ░░░░░░░░  │ ░░░░░░░░  │
Eval     │ Report    │ Report    │ Report    │ Compile   │ Present   │ FINAL    │ FINAL    │
Lead     │ ~6 pages  │ Writing   │ Present   │ Final Rep │ Practice  │ DELIVERY │ DELIVERY │
─────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘

KEY DELIVERABLES:
- ALL: Final report (18-20 pages combined)
- M3: Final presentation slides
- ALL: Code repository cleaned and documented

SYNC POINTS:
- Monday: Integration meeting - combine all results
- Wednesday: Report draft review meeting
- Thursday: Final review meeting
- Friday: Presentation practice
```

---

## Dependency Graph

```
                                    STAGE 1 FLOW
═══════════════════════════════════════════════════════════════════════════════════════════

                    ┌─────────────────────────────────────────────────────────────────┐
                    │                         MEMBER 1                                 │
                    │                      Data & Design                               │
                    └─────────────────────────────────────────────────────────────────┘
                                                │
        ┌───────────────────────────────────────┼───────────────────────────────────────┐
        │                                       │                                       │
        ▼                                       ▼                                       ▼
┌───────────────┐                      ┌───────────────┐                      ┌───────────────┐
│   Download    │                      │  Preprocess   │                      │  Fed Splits   │
│   Dataset     │                      │   Pipeline    │                      │   (non-IID)   │
│   Day 1-2     │                      │   Day 3-4     │                      │   Day 5-6     │
└───────────────┘                      └───────────────┘                      └───────┬───────┘
                                                                                      │
                                                                                      │
        ┌─────────────────────────────────────────────────────────────────────────────┘
        │                              BLOCKER
        │                    ╔═══════════════════════════╗
        │                    ║  M2 needs federated       ║
        │                    ║  data splits to          ║
        │                    ║  implement FL client     ║
        │                    ╚═══════════════════════════╝
        │
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                      MEMBER 2                                            │
│                                   Modeling & FL                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
        │
        ├───────────────────────────────┬───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│   Model       │              │   Centralized │              │   FL Server   │
│   Research    │              │   Training    │              │   + Client    │
│   Day 1-2     │              │   Day 5-7     │              │   Day 3-6     │
└───────────────┘              └───────┬───────┘              └───────┬───────┘
                                       │                              │
                                       │                              │
                                       ▼                              ▼
                              ┌───────────────┐              ┌───────────────┐
                              │  Centralized  │              │   Federated   │
                              │    Model      │              │     Model     │
                              │   Week 2      │              │    Week 2     │
                              └───────┬───────┘              └───────┬───────┘
                                      │                              │
                                      │         BLOCKER              │
                                      │    ╔═══════════════╗         │
                                      └────║  M3 needs     ║─────────┘
                                           ║  trained      ║
                                           ║  models       ║
                                           ╚═══════╤═══════╝
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                      MEMBER 3                                            │
│                                   Evaluation                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
        │
        ├───────────────────────────────┬───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│   Metrics     │              │   Evaluate    │              │   Compare     │
│   Framework   │              │   Both Models │              │   Results     │
│   Day 1-4     │              │   Week 2      │              │   Week 2      │
└───────────────┘              └───────────────┘              └───────────────┘
                                                                      │
                                                                      ▼
                                                            ┌───────────────────┐
                                                            │   STAGE 1 DONE    │
                                                            │  Report + Present │
                                                            │     Week 2 End    │
                                                            └───────────────────┘
```

---

## Stage 2 Parallel Workflow

```
                                    STAGE 2 FLOW
═══════════════════════════════════════════════════════════════════════════════════════════

WEEK 3-4: PARALLEL DEVELOPMENT (All members work independently)

        MEMBER 1                        MEMBER 2                        MEMBER 3
    ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
    │ Data Variants │              │   DP-SGD      │              │   Grad-CAM    │
    │ + Attacks     │              │   Training    │              │   + SHAP      │
    │   Week 3      │              │   Week 3      │              │   Week 3      │
    └───────┬───────┘              └───────┬───────┘              └───────┬───────┘
            │                              │                              │
            │                              │                              │
            ▼                              ▼                              ▼
    ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
    │ Poisoned      │              │   DP Models   │              │   Adversarial │
    │ Datasets      │              │   ε=1,5,10    │              │   Attacks     │
    └───────┬───────┘              └───────┬───────┘              └───────┬───────┘
            │                              │                              │
            │                              │                              │
            └──────────────────────────────┼──────────────────────────────┘
                                           │
                                           │
                          ┌────────────────┴────────────────┐
                          │                                 │
                          │  CROSS-EVALUATION PHASE         │
                          │      Week 4                     │
                          │                                 │
                          │  All members exchange:          │
                          │  - Models                       │
                          │  - Data variants                │
                          │  - Attack implementations       │
                          │                                 │
                          └─────────────────────────────────┘
                                           │
            ┌──────────────────────────────┼──────────────────────────────┐
            │                              │                              │
            ▼                              ▼                              ▼
    ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
    │    M1 tests   │              │    M2 tests   │              │    M3 tests   │
    │  M2's models  │              │  on M1's data │              │  all models   │
    │  on variants  │              │   variants    │              │  comprehensive│
    └───────────────┘              └───────────────┘              └───────────────┘
            │                              │                              │
            └──────────────────────────────┼──────────────────────────────┘
                                           │
                                           ▼
                              ┌───────────────────────┐
                              │   INTEGRATION WEEK 5  │
                              │                       │
                              │  - Combine results    │
                              │  - Final report       │
                              │  - Presentation       │
                              └───────────────────────┘
```

---

## Critical Path Analysis

```
CRITICAL PATH (items that, if delayed, will delay the entire project)
═══════════════════════════════════════════════════════════════════════════════════════════

STAGE 1 CRITICAL PATH:
──────────────────────

Day 1-2          Day 3-4          Day 5-6          Week 2
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Dataset │ ──► │ Preproc │ ──► │Fed Split│ ──► │FL Train │ ──► │ Evaluate│
│ Download│     │ Pipeline│     │         │     │         │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
    M1              M1              M1              M2              M3

    │               │               │               │               │
    ▼               ▼               ▼               ▼               ▼
   RISK           RISK           RISK           RISK           RISK
   Slow         Complex        non-IID         FL bugs       Metrics
  download       images        issues          /config        issues


STAGE 2 CRITICAL PATH:
──────────────────────

Week 3              Week 4              Week 5
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ All trust   │ ──►│  Exchange   │ ──►│  Integrate  │ ──►│   Final     │
│ mechanisms  │    │  & cross    │    │   results   │    │   Report    │
│ implemented │    │  evaluate   │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
   M1, M2, M3         ALL                 ALL                 ALL

POTENTIAL BLOCKERS:
───────────────────
1. Dataset too large  → Mitigation: Use subset or PneumoniaMNIST
2. Opacus errors      → Mitigation: Start simple, test incrementally
3. Team unavailable   → Mitigation: Clear communication, buffer time
4. GPU access issues  → Mitigation: Colab Pro, university cluster
```

---

## Meeting Schedule Summary

```
MEETING SCHEDULE
═══════════════════════════════════════════════════════════════════════════════════════════

WEEK 1
├── Mon: Kickoff Meeting (1hr)
│   - Role assignments
│   - Environment setup
│   - Repository structure
│
└── Thu: Data Handoff Check (30min)
    - M1 demos data pipeline
    - M2 confirms can integrate


WEEK 2
├── Wed: Stage 1 Progress Check (1hr)
│   - M2 demos centralized model
│   - M3 shows evaluation framework
│   - Identify any blockers
│
└── Fri/Sun: Stage 1 Final (2hr)
    - Finalize Stage 1 report
    - Practice presentation
    - Plan Stage 2


WEEK 3
└── Mon: Stage 2 Kickoff (1hr)
    - Confirm trust mechanisms
    - Distribute responsibilities
    - Set up collaboration workflow


WEEK 4
├── Mon: Cross-Evaluation Coordination (1hr)
│   - Share models and data
│   - Plan cross-evaluation
│   - Resolve integration issues
│
└── Sat: Integration Check (1hr)
    - Review cross-evaluation results
    - Identify missing pieces


WEEK 5
├── Mon: Final Integration (2hr)
│   - Combine all results
│   - Resolve discrepancies
│   - Assign report sections
│
├── Wed: Report Review (1hr)
│   - Review draft report
│   - Finalize figures/tables
│
├── Thu: Final Review (2hr)
│   - Complete report review
│   - Prepare presentation
│
└── Fri: Presentation Practice (1hr)
    - Run through presentation
    - Timing and transitions
```

---

## Quick Reference: Who Does What When

```
QUICK REFERENCE TABLE
═══════════════════════════════════════════════════════════════════════════════════════════

STAGE 1 (Week 1-2)
──────────────────
│ Task                  │ Owner  │ Deadline   │ Depends On        │
├───────────────────────┼────────┼────────────┼───────────────────┤
│ Download dataset      │ M1     │ Day 2      │ None              │
│ EDA notebook          │ M1     │ Day 3      │ Dataset           │
│ Preprocessing         │ M1     │ Day 4      │ EDA               │
│ Federated splits      │ M1     │ Day 6      │ Preprocessing     │
│ Model architecture    │ M2     │ Day 2      │ None              │
│ Flower server         │ M2     │ Day 4      │ None              │
│ Flower client         │ M2     │ Day 6      │ Fed splits (M1)   │
│ Centralized training  │ M2     │ Week 2 Day 2│ Preprocessing (M1)│
│ FL experiments        │ M2     │ Week 2 Day 5│ Client, splits    │
│ Metrics framework     │ M3     │ Day 4      │ None              │
│ Eval centralized      │ M3     │ Week 2 Day 4│ Central model (M2)│
│ Eval federated        │ M3     │ Week 2 Day 6│ Fed model (M2)    │
│ Stage 1 report        │ ALL    │ Week 2 Day 7│ All above         │
│ Stage 1 presentation  │ M3     │ Week 2 Day 7│ Report            │

STAGE 2 (Week 3-5)
──────────────────
│ Task                  │ Owner  │ Deadline   │ Depends On        │
├───────────────────────┼────────┼────────────┼───────────────────┤
│ Data variants         │ M1     │ Week 3     │ Stage 1 data      │
│ Data poisoning attacks│ M1     │ Week 3-4   │ Variants          │
│ Cross-eval (M1 side)  │ M1     │ Week 4-5   │ M2's models       │
│ DP-SGD implementation │ M2     │ Week 3     │ Stage 1 model     │
│ DP training ε=1,5,10  │ M2     │ Week 3-4   │ DP-SGD            │
│ DP + FL integration   │ M2     │ Week 4     │ DP training       │
│ Grad-CAM              │ M3     │ Week 3     │ Stage 1 model     │
│ SHAP explanations     │ M3     │ Week 3     │ Stage 1 model     │
│ Adversarial attacks   │ M3     │ Week 3-4   │ Grad-CAM          │
│ Robustness testing    │ M3     │ Week 4     │ M1's attacks      │
│ Final model eval      │ M3     │ Week 4-5   │ All M2's models   │
│ Final report          │ ALL    │ Week 5     │ All above         │
│ Final presentation    │ ALL    │ Week 5     │ Report            │
```
