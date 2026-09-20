# Headline-numbers artifact, same family as eval_undertriage.py: offline, read-only
# scoring of the production flagger (loaded from artifacts/flagger.joblib AS-IS, not
# retrained) over the held-out non-urgent (ESI 4-5) MIMIC population. Reproduces that
# script's exact preprocessing (same split, same scaler) so the saved distribution is
# honestly comparable to what backend/scoring.py computes for a live case at serve time.
#
# Purpose: lets the reveal show a case's risk as a real percentile ("top 7% risk among
# non-urgent patients") instead of inventing one. The output is a sorted array of
# anonymous risk SCORES only (no patient records, no identifiers) — an aggregate
# statistic, same governance category as eval_summary.json.
#
# DUA: reads MIMIC CSVs from local disk only. Writes only a list of scores, no records.
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split

ED = r"C:\Users\zjian\Downloads\mimic-iv-ed-2.2\mimic-iv-ed-2.2"

artifact = joblib.load("artifacts/flagger.joblib")
model, tfidf, vitals, medians, scaler = (
    artifact["model"], artifact["tfidf"], artifact["vitals"], artifact["medians"], artifact["scaler"]
)

stays = pd.read_csv(f"{ED}/edstays.csv")
triage = pd.read_csv(f"{ED}/triage.csv")
df = triage.merge(stays[["stay_id", "disposition"]], on="stay_id", how="inner")

low = df[df["acuity"].isin([4, 5])].copy()
low["under"] = low["disposition"].isin(["ADMITTED", "EXPIRED"]).astype(int)

cc = low["chiefcomplaint"].fillna("").str.lower()
for leak in ["transfer", "intubated", "admit"]:
    cc = cc.str.replace(leak, "", regex=False)

X_num = low[vitals].apply(pd.to_numeric, errors="coerce")
X_num = X_num.fillna(pd.Series(medians))

# same random_state=0 split as eval_undertriage.py — the held-out test half is the
# "non-urgent population" a live case's percentile is measured against
_, Xte_n, _, yte, _, cc_te = train_test_split(
    X_num, low["under"], cc, test_size=0.25, stratify=low["under"], random_state=0
)

Xte_n_s = scaler.transform(Xte_n)
Xte_t = tfidf.transform(cc_te)
Xte = hstack([csr_matrix(Xte_n_s), Xte_t])

proba = model.predict_proba(Xte)[:, 1]
sorted_scores = sorted(round(float(p), 6) for p in proba)

with open("artifacts/risk_percentiles.json", "w") as f:
    json.dump({"n": len(sorted_scores), "sorted_scores": sorted_scores}, f)

print(f"wrote artifacts/risk_percentiles.json — {len(sorted_scores)} held-out risk scores, "
      f"range [{sorted_scores[0]:.4f}, {sorted_scores[-1]:.4f}]")
