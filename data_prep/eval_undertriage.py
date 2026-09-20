# Headline-numbers artifact: offline, read-only evaluation of the production flagger on
# held-out MIMIC. Does NOT retrain the production model (loaded from artifacts/flagger.joblib
# as-is) — the only fitting done here is the small vitals-only comparison baseline, matching
# the methodology train_flagger.py already used for that same sub-metric.
#
# DUA: reads MIMIC CSVs from local disk only. Writes only aggregate numbers, no records.
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

Xtr_n, Xte_n, ytr, yte, cc_tr, cc_te = train_test_split(
    X_num, low["under"], cc, test_size=0.25, stratify=low["under"], random_state=0
)

Xtr_n_s = scaler.transform(Xtr_n)
Xte_n_s = scaler.transform(Xte_n)
Xte_t = tfidf.transform(cc_te)
Xte = hstack([csr_matrix(Xte_n_s), Xte_t])

auc_text_vitals = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])

vitals_only_m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_n_s, ytr)
auc_vitals_only = roc_auc_score(yte, vitals_only_m.predict_proba(Xte_n_s)[:, 1])

proba = model.predict_proba(Xte)[:, 1]
p20 = float(np.quantile(proba, 1 - 0.20))
flagged = proba >= p20
yt = yte.values
caught, total = int((flagged & (yt == 1)).sum()), int((yt == 1).sum())
fp = int((flagged & (yt == 0)).sum())
catch_rate = caught / total
ffpc = fp / max(caught, 1)

summary = {
    "auc_text_vitals": round(float(auc_text_vitals), 4),
    "auc_vitals_only": round(float(auc_vitals_only), 4),
    "p20_threshold": round(p20, 4),
    "n_under_triaged_heldout": total,
    "n_caught": caught,
    "catch_rate": round(catch_rate, 4),
    "n_false_positives": fp,
    "false_flags_per_catch": round(ffpc, 2),
}

print(json.dumps(summary, indent=2))

with open("artifacts/eval_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("wrote artifacts/eval_summary.json")
