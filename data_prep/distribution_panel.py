# "Fake but faithful" proof: synthetic bank's vital distributions vs. the real low-acuity
# subpopulation's. Admit-rate is plotted separately and explicitly labeled as an
# intentional enrichment (50% in the bank vs. the real ~1.9% base rate) — a disclosed
# design choice, not something the sampling targets or claims to match.
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ED = r"C:\Users\zjian\Downloads\mimic-iv-ed-2.2\mimic-iv-ed-2.2"
VITALS = [("sbp", "SBP (mmHg)"), ("heartrate", "HR (bpm)"), ("o2sat", "SpO2 (%)"),
          ("resprate", "RR (breaths/min)"), ("temperature", "Temp (°F)")]
PLAUSIBLE = {
    "temperature": (90.0, 106.0), "heartrate": (30, 220), "resprate": (6, 60),
    "o2sat": (50, 100), "sbp": (60, 250),
}

stays = pd.read_csv(f"{ED}/edstays.csv")
triage = pd.read_csv(f"{ED}/triage.csv")
df = triage.merge(stays[["stay_id", "disposition"]], on="stay_id", how="inner")
low = df[df["acuity"].isin([4, 5])].copy()
real_admit_rate = low["disposition"].isin(["ADMITTED", "EXPIRED"]).mean()

with open("backend/cases.json") as f:  # canonical file — same one the backend serves
    bank = json.load(f)
bank_df = pd.DataFrame([c["vitals"] for c in bank])
bank_admit_rate = sum(1 for c in bank if c["truth"]["subpopulation"] == "under_triaged") / len(bank)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (col, label) in zip(axes, VITALS):
    lo, hi = PLAUSIBLE[col]
    real_vals = pd.to_numeric(low[col], errors="coerce").dropna()
    real_vals = real_vals[(real_vals >= lo) & (real_vals <= hi)]
    ax.hist(real_vals, bins=30, density=True, alpha=0.5, label=f"real low-acuity (n={len(real_vals):,})", color="#4b7bbf")
    ax.hist(bank_df[col], bins=12, density=True, alpha=0.6, label=f"synthetic bank (n={len(bank_df)})", color="#d97b29")
    ax.set_title(label)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

ax = axes[5]
bars = ax.bar(["Real low-acuity\npopulation", "Synthetic bank\n(INTENTIONAL)"],
              [real_admit_rate * 100, bank_admit_rate * 100],
              color=["#4b7bbf", "#d97b29"])
ax.set_ylabel("admit/died rate (%)")
ax.set_title("Admit rate — NOT distribution-matched")
for bar, val in zip(bars, [real_admit_rate * 100, bank_admit_rate * 100]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%", ha="center", fontsize=9)
ax.annotate(
    "Bank is deliberately enriched 8/8 under-triaged\nfor a usable demo session — disclosed design\nchoice, not a claim about real prevalence.",
    xy=(0.5, 0.5), xycoords="axes fraction", ha="center", fontsize=8, style="italic", color="#555",
)

fig.suptitle("Distribution-match panel: synthetic case bank vs. real MIMIC-IV-ED low-acuity (ESI 4-5) population", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

import os
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/distribution_match.png", dpi=150)
print(f"real admit/died rate: {real_admit_rate:.3%}")
print(f"synthetic bank admit rate: {bank_admit_rate:.0%} (intentional, disclosed)")
print("wrote figures/distribution_match.png")
