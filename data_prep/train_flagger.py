# Canonical training script for the under-triage flagger.
# Mirrors recon.py's pipeline exactly (same data, same held-out split, same leak-token
# stripping, same text pipeline, same model class/hyperparameters) with exactly one change:
# numeric vital features are standardized (StandardScaler fit on the TRAIN split only)
# before being fed to the model, so raw magnitude (e.g. SBP 118, temp 98.4) no longer
# drives risk score by scale alone.
#
# DUA: reads MIMIC CSVs from local disk only; nothing here leaves the machine.
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ED = r"C:\Users\zjian\Downloads\mimic-iv-ed-2.2\mimic-iv-ed-2.2"

stays = pd.read_csv(f"{ED}/edstays.csv")
triage = pd.read_csv(f"{ED}/triage.csv")
df = triage.merge(stays[["stay_id", "disposition"]], on="stay_id", how="inner")

low = df[df["acuity"].isin([4, 5])].copy()
low["under"] = low["disposition"].isin(["ADMITTED", "EXPIRED"]).astype(int)

cc = low["chiefcomplaint"].fillna("").str.lower()
for leak in ["transfer", "intubated", "admit"]:
    cc = cc.str.replace(leak, "", regex=False)

vitals = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
X_num = low[vitals].apply(pd.to_numeric, errors="coerce")
medians = X_num.median()
X_num = X_num.fillna(medians)

Xtr_n, Xte_n, ytr, yte, cc_tr, cc_te = train_test_split(
    X_num, low["under"], cc, test_size=0.25, stratify=low["under"], random_state=0
)

scaler = StandardScaler().fit(Xtr_n)
Xtr_n_s = scaler.transform(Xtr_n)
Xte_n_s = scaler.transform(Xte_n)

tfidf = TfidfVectorizer(min_df=20, ngram_range=(1, 2))
Xtr_t = tfidf.fit_transform(cc_tr)
Xte_t = tfidf.transform(cc_te)

Xtr = hstack([csr_matrix(Xtr_n_s), Xtr_t])
Xte = hstack([csr_matrix(Xte_n_s), Xte_t])

m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)

vitals_only_m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_n_s, ytr)
auc_vitals_only = roc_auc_score(yte, vitals_only_m.predict_proba(Xte_n_s)[:, 1])
auc_text_vitals = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
print(f"AUC (vitals only, standardized): {auc_vitals_only:.3f}")
print(f"AUC (vitals+text, standardized): {auc_text_vitals:.3f}")

proba = m.predict_proba(Xte)[:, 1]
p20 = float(np.quantile(proba, 1 - 0.20))
flagged = proba >= p20
yt = yte.values
caught, total = int((flagged & (yt == 1)).sum()), int((yt == 1).sum())
fp = int((flagged & (yt == 0)).sum())
print(
    f"p20 threshold = {p20:.4f}  ->  catch {caught/total:.0%} of missed-sick "
    f"({caught}/{total}) | {fp/max(caught,1):.1f} false flags per real catch"
)

joblib.dump(
    {
        "model": m,
        "tfidf": tfidf,
        "vitals": vitals,
        "medians": medians.to_dict(),
        "scaler": scaler,
    },
    "artifacts/flagger.joblib",
)
with open("artifacts/thresholds.json", "w") as f:
    json.dump({"p20": p20}, f, indent=2)

print("wrote artifacts/flagger.joblib and artifacts/thresholds.json")
