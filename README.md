# Modeling Trust Calibration in Human-AI Financial Decision Systems
### A Behavioral and Interactional Perspective

**Abdulla Al Noman | Student ID: 2512710**
MSc Human-Computer Interaction | University for the Creative Arts, Farnham
Module: FGCT7024 — Final Major Project | Supervisor: Roderick Morgan

---

## Project Overview

This repository contains all research artefacts for the MSc HCI Final Major Project investigating **trust calibration in AI-mediated financial decisions**. The study examines the conditions under which users overtrust, undertrust, or appropriately calibrate their reliance on AI-generated credit recommendations across three explanation conditions.

**Research Question:** Under what conditions do users exhibit overtrust, undertrust, or appropriately calibrated trust in AI-generated financial recommendations, and what behavioural and interactional factors determine each outcome?

---

## Study Design

A between-subjects behavioural experiment with 678 participants across three explanation conditions:

| Condition | Description |
|-----------|-------------|
| **A — No Explanation** | AI recommendation and confidence score only |
| **B — Surface Explanation** | AI recommendation + one plain-English sentence |
| **C — Counterfactual Explanation** | AI recommendation + "what would change this decision" |

The AI was deliberately wrong on **scenarios 3, 7, and 10** to measure calibration trajectory and recovery.

---

## Key Findings

| Measure | Condition A | Condition B | Condition C | F | p | η² |
|---------|------------|------------|------------|---|---|-----|
| AI Follow Rate | 0.653 | 0.583 | 0.509 | 43.15 | <.001 | 0.122 |
| AI Errors Caught | 1.040 | 1.335 | 1.800 | 45.78 | <.001 | 0.128 |
| Appeal Intent | 2.342 | 3.018 | 3.927 | 138.88 | <.001 | 0.309 |
| Perceived Fairness | 4.087 | 4.656 | 5.289 | 90.73 | <.001 | 0.226 |
| Trust in Automation | 4.615 | 4.288 | 3.943 | 49.16 | <.001 | 0.136 |

**Analysis N = 625** (53 excluded via suspicion probe)

---

## Repository Structure

```
trust-calibration-fmp/
│
├── data/
│   ├── trust_calibration_FINAL_678.csv          # Wide format — one row per participant
│   └── trust_calibration_FINAL_scenarios_long.csv  # Long format — one row per scenario
│
├── analysis/
│   ├── analysis_pipeline.py                      # Full Python analysis script
│   ├── table1_descriptives.csv                   # Descriptive statistics by condition
│   ├── table2_anova.csv                          # One-way ANOVA results
│   ├── table3_tukey.csv                          # Tukey HSD post-hoc comparisons
│   ├── table4_ancova.csv                         # ANCOVA — individual differences
│   ├── table5_correlations.csv                   # Pearson correlations
│   ├── table6_trajectory.csv                     # Trust trajectory across 10 scenarios
│   └── table7_mental_model.csv                   # Mental model accuracy
│
├── figures/
│   ├── figure1_trajectory.png                    # Trust calibration trajectory plot
│   ├── figure2_bar_outcomes.png                  # Key outcomes bar chart
│   ├── figure3_calibration_dist.png              # Calibration score distributions
│   ├── figure4_correlation_heatmap.png           # Correlation matrix heatmap
│   └── figure5_trust_moderation.png              # Moderation by financial literacy
│
├── docs/
│   ├── FGCT7024_FMP_Proposal_WithIEEE.docx       # Full research proposal (40 refs)
│   └── Data_Collection_Form.docx                 # Session data collection form
│
└── README.md
```

---

## Dataset

**678 participants** | **6,780 scenario-level decisions** | **32 columns**

### Demographics
- 56% Bangladeshi, 11% British, 10% Indian, 7% Pakistani, 5% Chinese, 4% Nigerian, 3% Malaysian, 4% Other
- 84% university students (undergraduate, MSc, PhD, MBA)
- Age groups: 18–22 (29%), 23–27 (37%), 28–32 (17%), 33–40 (10%), 41+ (7%)
- 25 UK universities represented

### Key Variables
| Variable | Description |
|----------|-------------|
| `ai_follow_rate_0_1` | Proportion of AI recommendations accepted |
| `calibration_score` | Correlation of confidence with decision accuracy |
| `ai_errors_caught_0_3` | Number of AI errors detected and overridden |
| `trust_in_automation_1_7` | Jian et al. (2000) Trust in Automation scale |
| `perceived_fairness_1_7` | Perceived fairness of the AI system |
| `appeal_intent_1_5` | Intent to appeal a declined decision |
| `financial_literacy_score_0_3` | Lusardi & Mitchell (2011) objective scale |
| `need_for_cognition_1_5` | Cacioppo & Petty (1982) NFC scale |
| `technology_disposition_1_5` | Venkatesh et al. (2003) TAM items |

---

## Analysis Pipeline

Run the full analysis with:

```bash
pip install pandas numpy scipy pingouin matplotlib
python3 analysis/analysis_pipeline.py
```

Produces all 7 tables and 5 figures in an `analysis_outputs/` folder.

---

## Theoretical Grounding

- **Lee & See (2004)** — Trust calibration as appropriate reliance
- **Parasuraman & Riley (1997)** — Automation misuse taxonomy
- **Dietvorst et al. (2015)** — Algorithmic aversion
- **Logg et al. (2019)** — Algorithmic appreciation
- **Kahneman (2011)** — Dual-process theory
- **Mayer, Davis & Schoorman (1995)** — Trust antecedents
- **Wachter et al. (2017)** — Counterfactual explanations

---

## Regulatory Context

- **EU AI Act (2024)** — Credit scoring classified as high-risk AI
- **UK FCA Consumer Duty (2023)** — Firms must ensure customers understand AI decisions

---

## Contact

**Abdulla Al Noman**
Email: abdullaalnomanco@gmail.com
MSc HCI | University for the Creative Arts, Farnham
Final submission: 8 September 2026

---

## Bradley-Terry Model

In addition to ANOVA, a **Bradley-Terry pairwise dominance model** was fitted to validate the condition rankings. The BT model estimates a latent "strength" parameter for each condition based on pairwise comparisons across all participants.

### Aggregate BT Strengths

| Condition | Aggregate BT Strength |
|-----------|----------------------|
| A — No Explanation | 0.5631 |
| B — Surface Explanation | 0.9629 |
| C — Counterfactual Explanation | **2.0617** |

### Win Probabilities (Condition C)

| Comparison | Win Probability |
|------------|----------------|
| P(C beats A) | **0.785** |
| P(C beats B) | **0.682** |
| P(B beats A) | 0.631 |

**Conclusion:** Both ANOVA and Bradley-Terry converge on the same ranking — **Counterfactual > Surface > No Explanation** — across all 7 outcome measures. Condition C wins all 7 pairwise dominance comparisons.
