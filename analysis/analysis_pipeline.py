"""
Trust Calibration Study — Full Analysis Pipeline
Modeling Trust Calibration in Human-AI Financial Decision Systems
Abdulla Al Noman | 2512710 | MSc HCI | UCA Farnham | FGCT7024

Run this file once your real data is in the same folder as this script.
Replace the CSV filenames below if yours are named differently.
"""

import pandas as pd
import numpy as np
import warnings
import os
from scipy import stats
import pingouin as pg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── FILE PATHS — change these if needed ──────────────────────────────────────
WIDE_CSV = 'trust_calibration_FINAL_678.csv'      # one row per participant
LONG_CSV = 'trust_calibration_FINAL_scenarios_long.csv'  # one row per scenario
OUTPUT_DIR = 'analysis_outputs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv(WIDE_CSV)
dl = pd.read_csv(LONG_CSV)
print(f"Loaded: {len(df)} participants, {len(dl)} scenario rows")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA CLEANING & EXCLUSIONS
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Cleaning data...")

# Exclude participants who correctly identified the study hypothesis
susp = df[df['suspicion_probe'].str.contains('Yes — I thought', na=False)]
dfc  = df[~df.index.isin(susp.index)].copy()

# Condition subsets
A = dfc[dfc.condition == 'A_no_explanation']
B = dfc[dfc.condition == 'B_surface']
C = dfc[dfc.condition == 'C_counterfactual']

print(f"  Original N={len(df)}")
print(f"  Excluded (suspicion probe): {len(susp)}")
print(f"  Analysis N={len(dfc)}  |  A={len(A)}  B={len(B)}  C={len(C)}")

# Also filter long-form to clean participants
dl_clean = dl[dl['participant_id'].isin(dfc['participant_id'])]

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — DESCRIPTIVE STATISTICS
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Descriptive statistics...")

OUTCOMES = [
    ('ai_follow_rate_0_1',       'AI Follow Rate (0–1)'),
    ('calibration_score',         'Calibration Score'),
    ('ai_errors_caught_0_3',      'AI Errors Caught (0–3)'),
    ('trust_in_automation_1_7',   'Trust in Automation (1–7)'),
    ('perceived_fairness_1_7',    'Perceived Fairness (1–7)'),
    ('appeal_intent_1_5',         'Appeal Intent (1–5)'),
    ('retrust_organisation_1_5',  'Re-trust Intent (1–5)'),
    ('mental_model_estimate_pct', 'Mental Model Estimate (%)'),
    ('avg_confidence_1_5',        'Avg Confidence (1–5)'),
    ('avg_response_time_ms',      'Avg Response Time (ms)'),
]

desc_rows = []
for col, lbl in OUTCOMES:
    row = {'Measure': lbl}
    for tag, sub in [('A', A), ('B', B), ('C', C), ('Total', dfc)]:
        row[f'M_{tag}']   = round(sub[col].mean(), 3)
        row[f'SD_{tag}']  = round(sub[col].std(),  3)
        row[f'Mdn_{tag}'] = round(sub[col].median(), 3)
    desc_rows.append(row)

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(f'{OUTPUT_DIR}/table1_descriptives.csv', index=False)
print(f"  Saved: table1_descriptives.csv")
print(desc_df[['Measure','M_A','M_B','M_C']].to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — ASSUMPTION CHECKS
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Assumption checks...")

assump_rows = []
for col, lbl in OUTCOMES[:6]:
    # Levene's test — homogeneity of variance
    lev_stat, lev_p = stats.levene(A[col], B[col], C[col])
    # Shapiro-Wilk — normality (sample 50 per group)
    sw_ps = []
    for sub in [A, B, C]:
        samp = sub[col].sample(min(50, len(sub)), random_state=42)
        _, sw_p = stats.shapiro(samp)
        sw_ps.append(sw_p)
    assump_rows.append({
        'Measure':      lbl,
        'Levene_F':     round(lev_stat, 3),
        'Levene_p':     round(lev_p, 3),
        'Levene_OK':    'Yes' if lev_p > .05 else 'Violated',
        'SW_p_A':       round(sw_ps[0], 3),
        'SW_p_B':       round(sw_ps[1], 3),
        'SW_p_C':       round(sw_ps[2], 3),
        'Normality_OK': 'Yes' if all(p > .05 for p in sw_ps) else 'Violated',
    })

assump_df = pd.DataFrame(assump_rows)
assump_df.to_csv(f'{OUTPUT_DIR}/table_assumptions.csv', index=False)
print(f"  Saved: table_assumptions.csv")
print(assump_df[['Measure','Levene_OK','Normality_OK']].to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — ONE-WAY ANOVA + EFFECT SIZES
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] One-Way ANOVA...")

anova_rows = []
for col, lbl in OUTCOMES:
    a, b, c = A[col], B[col], C[col]
    f, p    = stats.f_oneway(a, b, c)

    # Eta-squared
    grand = dfc[col].mean()
    ssb   = sum(len(g) * (g.mean() - grand)**2 for g in [a, b, c])
    sst   = sum(((v - grand)**2).sum() for v in [a, b, c])
    eta2  = ssb / sst if sst > 0 else 0

    # Omega-squared (less biased)
    k     = 3  # number of groups
    n_tot = len(dfc)
    ss_within = sst - ssb
    ms_within = ss_within / (n_tot - k)
    omega2 = (ssb - (k - 1) * ms_within) / (sst + ms_within)

    sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'

    anova_rows.append({
        'Measure':  lbl,
        'M_A':      round(a.mean(), 3),
        'SD_A':     round(a.std(),  3),
        'M_B':      round(b.mean(), 3),
        'SD_B':     round(b.std(),  3),
        'M_C':      round(c.mean(), 3),
        'SD_C':     round(c.std(),  3),
        'F':        round(f,    2),
        'df_between': 2,
        'df_within':  len(dfc) - 3,
        'p':        '<.001' if p < .001 else round(p, 3),
        'eta2':     round(eta2,   3),
        'omega2':   round(omega2, 3),
        'sig':      sig,
    })

    eta_label = 'small' if eta2 < .06 else 'medium' if eta2 < .14 else 'large'
    print(f"  {lbl}: F={f:.2f}, p={'<.001' if p<.001 else f'{p:.3f}'} {sig}, "
          f"η²={eta2:.3f} ({eta_label})")

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(f'{OUTPUT_DIR}/table2_anova.csv', index=False)
print(f"  Saved: table2_anova.csv")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 — TUKEY HSD POST-HOC
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Tukey HSD post-hoc comparisons...")

tukey_all = []
for col, lbl in OUTCOMES[:7]:
    t = pg.pairwise_tukey(data=dfc, dv=col, between='condition')
    t.insert(0, 'Measure', lbl)
    tukey_all.append(t)

tukey_df = pd.concat(tukey_all, ignore_index=True)
tukey_df.to_csv(f'{OUTPUT_DIR}/table3_tukey.csv', index=False)
print(f"  Saved: table3_tukey.csv")

# Print AI Follow Rate comparisons
t_fr = tukey_df[tukey_df['Measure'] == 'AI Follow Rate (0–1)']
print("\n  AI Follow Rate — pairwise comparisons:")
print(t_fr[['A','B','diff','hedges','p-tukey']].to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════════
# STEP 6 — ANCOVA (individual differences as covariates)
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 6] ANCOVA — moderation by individual differences...")

ancova_results = []
for col, lbl in [('ai_follow_rate_0_1','AI Follow Rate'),
                 ('calibration_score','Calibration Score'),
                 ('ai_errors_caught_0_3','AI Errors Caught')]:
    anc = pg.ancova(
        data   = dfc,
        dv     = col,
        between= 'condition',
        covar  = ['financial_literacy_score_0_3',
                  'need_for_cognition_1_5',
                  'technology_disposition_1_5']
    )
    anc.insert(0, 'Outcome', lbl)
    ancova_results.append(anc)

ancova_df = pd.concat(ancova_results, ignore_index=True)
ancova_df.to_csv(f'{OUTPUT_DIR}/table4_ancova.csv', index=False)
print(f"  Saved: table4_ancova.csv")
print(ancova_df[['Outcome','Source','F','p-unc','np2']].to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════════
# STEP 7 — PEARSON CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 7] Pearson correlations...")

corr_rows = []
for v, vlbl in [
    ('financial_literacy_score_0_3', 'Financial Literacy'),
    ('need_for_cognition_1_5',       'Need for Cognition'),
    ('technology_disposition_1_5',   'Tech Disposition'),
]:
    for o, olbl in [
        ('ai_follow_rate_0_1',     'AI Follow Rate'),
        ('calibration_score',       'Calibration Score'),
        ('ai_errors_caught_0_3',    'AI Errors Caught'),
        ('trust_in_automation_1_7', 'Trust in Automation'),
    ]:
        r, p = stats.pearsonr(dfc[v], dfc[o])
        sig  = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
        corr_rows.append({'Predictor': vlbl, 'Outcome': olbl,
                          'r': round(r, 3), 'p': '<.001' if p < .001 else round(p, 3),
                          'sig': sig})
        print(f"  {vlbl} × {olbl}: r={r:.3f} {sig}")

pd.DataFrame(corr_rows).to_csv(f'{OUTPUT_DIR}/table5_correlations.csv', index=False)
print(f"  Saved: table5_correlations.csv")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 8 — TRUST TRAJECTORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 8] Trust trajectory across scenarios...")

traj = (dl_clean
        .groupby(['condition', 'scenario'])['followed_ai']
        .mean()
        .reset_index())

traj_pivot = traj.pivot(index='scenario', columns='condition', values='followed_ai')
traj_pivot.to_csv(f'{OUTPUT_DIR}/table6_trajectory.csv')
print("  Follow rate by scenario:")
print(traj_pivot.round(3).to_string())

# Trajectory drop at error scenarios
print("\n  Trust drop at AI error scenarios (3, 7, 10):")
for cond_name, cond_col in [
    ('A_no_explanation',  'A_no_explanation'),
    ('B_surface',         'B_surface'),
    ('C_counterfactual',  'C_counterfactual'),
]:
    c_traj = traj[traj['condition'] == cond_name].set_index('scenario')['followed_ai']
    for err_sc, prev_sc in [(3, 2), (7, 6), (10, 9)]:
        if prev_sc in c_traj.index and err_sc in c_traj.index:
            drop = c_traj[err_sc] - c_traj[prev_sc]
            print(f"  {cond_name} | Scenario {err_sc}: drop = {drop:+.3f}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 9 — MENTAL MODEL ACCURACY
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 9] Mental model accuracy...")

mm_rows = []
for tag, sub in [('A', A), ('B', B), ('C', C)]:
    mm_rows.append({
        'Condition':       f'Condition {tag}',
        'Mean_Estimate_%': round(sub['mental_model_estimate_pct'].mean(), 1),
        'SD_Estimate':     round(sub['mental_model_estimate_pct'].std(),  1),
        'Accurate_%':      round(sub['mental_model_accurate'].mean() * 100, 1),
        'True_value_%':    70,
        'Bias':            round(sub['mental_model_estimate_pct'].mean() - 70, 1),
    })

mm_df = pd.DataFrame(mm_rows)
mm_df.to_csv(f'{OUTPUT_DIR}/table7_mental_model.csv', index=False)
print(f"  Saved: table7_mental_model.csv")
print(mm_df.to_string(index=False))

# Chi-square on mental model accuracy between conditions
acc_counts = dfc.groupby('condition')['mental_model_accurate'].value_counts().unstack(fill_value=0)
chi2, p_chi, dof, _ = stats.chi2_contingency(acc_counts)
print(f"\n  Chi-square on mental model accuracy: χ²={chi2:.2f}, df={dof}, p={'<.001' if p_chi<.001 else round(p_chi,3)}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 10 — FIGURES
# ════════════════════════════════════════════════════════════════════════════════
print("\n[STEP 10] Generating figures...")

COLORS = {
    'A_no_explanation':  '#2a78d6',
    'B_surface':         '#eda100',
    'C_counterfactual':  '#1baf7a',
}
COND_LABELS = {
    'A_no_explanation':  'A — No Explanation',
    'B_surface':         'B — Surface',
    'C_counterfactual':  'C — Counterfactual',
}

# ── Figure 1: Trust Trajectory ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('#FAFBFC')
ax.set_facecolor('#FAFBFC')

for cond in ['A_no_explanation', 'B_surface', 'C_counterfactual']:
    d = traj[traj['condition'] == cond]
    ax.plot(d['scenario'], d['followed_ai'],
            marker='o', linewidth=2.5, markersize=7,
            color=COLORS[cond], label=COND_LABELS[cond], zorder=3)

for sc in [3, 7, 10]:
    ax.axvline(x=sc, color='#E53E3E', linestyle='--', alpha=0.5, linewidth=1.2, zorder=2)
    ax.annotate(f'AI Error\nScenario {sc}',
                xy=(sc, 0.38), ha='center', fontsize=9,
                color='#E53E3E', fontweight='500')

ax.set_xlabel('Scenario Number', fontsize=12, labelpad=8)
ax.set_ylabel('AI Follow Rate (proportion)', fontsize=12, labelpad=8)
ax.set_title('Figure 1. Trust Calibration Trajectory Across 10 Loan Decision Scenarios\n'
             'by Explanation Condition (N=630 after exclusions)',
             fontsize=12, fontweight='600', pad=12)
ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
ax.set_xticks(range(1, 11))
ax.set_ylim(0.28, 0.84)
ax.grid(axis='y', alpha=0.25, linewidth=0.7)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure1_trajectory.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Saved: figure1_trajectory.png")

# ── Figure 2: Bar chart — 4 key outcomes ─────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(15, 5.5))
fig.patch.set_facecolor('#FAFBFC')

bar_cfg = [
    ('ai_follow_rate_0_1',      'AI Follow Rate',         (0.38, 0.80)),
    ('ai_errors_caught_0_3',    'AI Errors Caught (0–3)', (0.70, 2.10)),
    ('appeal_intent_1_5',       'Appeal Intent (1–5)',    (1.40, 4.60)),
    ('perceived_fairness_1_7',  'Perceived Fairness (1–7)',(3.00, 6.20)),
]
bar_colors  = [COLORS['A_no_explanation'], COLORS['B_surface'], COLORS['C_counterfactual']]
bar_xlabels = ['A\n(No Expl.)', 'B\n(Surface)', 'C\n(Counterfact.)']

for ax, (col, title, ylim) in zip(axes, bar_cfg):
    ax.set_facecolor('#FAFBFC')
    means = [A[col].mean(), B[col].mean(), C[col].mean()]
    sems  = [A[col].sem(),  B[col].sem(),  C[col].sem()]
    bars  = ax.bar(bar_xlabels, means, yerr=sems,
                   color=bar_colors, width=0.55, capsize=5,
                   error_kw={'linewidth': 1.5}, zorder=3, alpha=0.88)
    for idx, (bar, m, se) in enumerate(zip(bars, means, sems)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + se + (ylim[1] - ylim[0]) * 0.015,
                f'{m:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='600')
    ax.set_title(title, fontsize=10, fontweight='600', pad=8)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.22, linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=9)

fig.suptitle('Figure 2. Key Outcome Measures by Explanation Condition (Mean ± SE)',
             fontsize=12, fontweight='600', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure2_bar_outcomes.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Saved: figure2_bar_outcomes.png")

# ── Figure 3: Calibration score distributions ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
fig.patch.set_facecolor('#FAFBFC')

for ax, (cond, sub, lbl) in zip(axes, [
    ('A_no_explanation',  A, 'Condition A\n(No Explanation)'),
    ('B_surface',         B, 'Condition B\n(Surface)'),
    ('C_counterfactual',  C, 'Condition C\n(Counterfactual)'),
]):
    ax.set_facecolor('#FAFBFC')
    ax.hist(sub['calibration_score'], bins=22,
            color=COLORS[cond], alpha=0.78, edgecolor='white', linewidth=0.5)
    ax.axvline(sub['calibration_score'].mean(),
               color='#111', linestyle='--', linewidth=1.8,
               label=f"M = {sub['calibration_score'].mean():.3f}")
    ax.set_title(f'{lbl}', fontsize=10, fontweight='600')
    ax.set_xlabel('Calibration Score', fontsize=9)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('Frequency', fontsize=10)
fig.suptitle('Figure 3. Distribution of Calibration Scores by Explanation Condition',
             fontsize=12, fontweight='600', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure3_calibration_dist.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Saved: figure3_calibration_dist.png")

# ── Figure 4: Individual differences heatmap ──────────────────────────────────
corr_matrix = dfc[['financial_literacy_score_0_3','need_for_cognition_1_5',
                    'technology_disposition_1_5','ai_follow_rate_0_1',
                    'calibration_score','ai_errors_caught_0_3',
                    'trust_in_automation_1_7','perceived_fairness_1_7']].corr()

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#FAFBFC')
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.75, label='Pearson r')

labels = ['Fin. Literacy','Need for\nCognition','Tech\nDisposition',
          'AI Follow\nRate','Calibration\nScore','Errors\nCaught',
          'Trust in\nAutomation','Perceived\nFairness']
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5, rotation=35, ha='right')
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.values[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7.5, color='white' if abs(val) > 0.35 else 'black')

ax.set_title('Figure 4. Correlation Matrix — Individual Differences and Outcome Measures',
             fontsize=11, fontweight='600', pad=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure4_correlation_heatmap.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Saved: figure4_correlation_heatmap.png")

# ── Figure 5: Trust in automation by condition + fin literacy ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#FAFBFC')

# Left: TiA by condition
ax = axes[0]
ax.set_facecolor('#FAFBFC')
means = [A['trust_in_automation_1_7'].mean(),
         B['trust_in_automation_1_7'].mean(),
         C['trust_in_automation_1_7'].mean()]
sems  = [A['trust_in_automation_1_7'].sem(),
         B['trust_in_automation_1_7'].sem(),
         C['trust_in_automation_1_7'].sem()]
ax.bar(bar_xlabels, means, yerr=sems, color=bar_colors,
       width=0.55, capsize=5, error_kw={'linewidth':1.5}, alpha=0.88, zorder=3)
for idx,(m,se) in enumerate(zip(means,sems)):
    ax.text(idx, m+se+0.04, f'{m:.2f}', ha='center', fontsize=10, fontweight='600')
ax.set_title('Trust in Automation\nby Explanation Condition', fontsize=11, fontweight='600')
ax.set_ylabel('Trust in Automation Scale (1–7)', fontsize=10)
ax.set_ylim(3.5, 5.4)
ax.grid(axis='y', alpha=0.22); ax.spines[['top','right']].set_visible(False)

# Right: errors caught by financial literacy score
ax2 = axes[1]
ax2.set_facecolor('#FAFBFC')
for cond, col in [('A_no_explanation','#2a78d6'),('B_surface','#eda100'),('C_counterfactual','#1baf7a')]:
    sub = dfc[dfc.condition==cond].groupby('financial_literacy_score_0_3')['ai_errors_caught_0_3'].mean()
    ax2.plot(sub.index, sub.values, marker='o', linewidth=2, markersize=7,
             color=col, label=COND_LABELS[cond])
ax2.set_title('AI Errors Caught\nby Financial Literacy', fontsize=11, fontweight='600')
ax2.set_xlabel('Financial Literacy Score (0–3)', fontsize=10)
ax2.set_ylabel('Mean AI Errors Caught', fontsize=10)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.set_xticks([0,1,2,3])
ax2.grid(alpha=0.22); ax2.spines[['top','right']].set_visible(False)

fig.suptitle('Figure 5. Trust in Automation and Moderation by Financial Literacy',
             fontsize=12, fontweight='600', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure5_trust_moderation.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Saved: figure5_trust_moderation.png")

# ════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY PRINT
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ANALYSIS COMPLETE")
print("="*65)
print(f"\nAnalysis N = {len(dfc)} (excluded {len(susp)} suspicious)")
print(f"Condition split: A={len(A)}, B={len(B)}, C={len(C)}")

print("\nKEY RESULTS:")
print("-"*65)
for col, lbl in OUTCOMES[:7]:
    a,b,c = A[col].mean(), B[col].mean(), C[col].mean()
    f,p   = stats.f_oneway(A[col], B[col], C[col])
    grand = dfc[col].mean()
    ssb   = sum(len(g)*(g.mean()-grand)**2 for g in [A[col],B[col],C[col]])
    sst   = sum(((v-grand)**2).sum() for v in [A[col],B[col],C[col]])
    eta2  = ssb/sst
    sig   = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
    print(f"  {lbl}")
    print(f"    A={a:.3f}  B={b:.3f}  C={c:.3f}  |  F={f:.2f} p={'<.001' if p<.001 else f'{p:.3f}'} {sig}  η²={eta2:.3f}")

print("\nOUTPUT FILES:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(f'{OUTPUT_DIR}/{f}')
    print(f"  {f:<45} {size:>8,} bytes")
print(f"\nAll saved to: ./{OUTPUT_DIR}/")
