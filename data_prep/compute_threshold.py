# One-time, read-only-against-MIMIC calibration step. Does NOT retrain the model:
# loads the already-fitted model/tfidf from flagger.joblib and reproduces recon.py's
# exact preprocessing + held-out split (same random_state=0) to derive the two things
# the original dump never persisted: per-vital medians (for filling missing vitals on
# live synthetic cases) and the p20 risk threshold (the fixed operating point).
#
# DUA: reads MIMIC CSVs from local disk only; nothing here leaves the machine.
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

ED = r"C:\Users\zjian\Downloads\mimic-iv-ed-2.2\mimic-iv-ed-2.2"

artifact = joblib.load("flagger.joblib")
model, tfidf, vitals = artifact["model"], artifact["tfidf"], artifact["vitals"]

stays = pd.read_csv(f"{ED}/edstays.csv")
triage = pd.read_csv(f"{ED}/triage.csv")
df = triage.merge(stays[["stay_id", "disposition"]], on="stay_id", how="inner")

low = df[df["acuity"].isin([4, 5])].copy()
low["under"] = low["disposition"].isin(["ADMITTED", "EXPIRED"]).astype(int)

cc = low["chiefcomplaint"].fillna("").str.lower()
for leak in ["transfer", "intubated", "admit"]:
    cc = cc.str.replace(leak, "", regex=False)

X_num = low[vitals].apply(pd.to_numeric, errors="coerce")
medians = X_num.median()
X_num = X_num.fillna(medians)

_, Xte_n, _, yte, _, cc_te = train_test_split(
    X_num, low["under"], cc, test_size=0.25, stratify=low["under"], random_state=0
)

Xte_t = tfidf.transform(cc_te)
Xte = hstack([csr_matrix(Xte_n.values), Xte_t])

proba = model.predict_proba(Xte)[:, 1]
p20 = float(np.quantile(proba, 1 - 0.20))

flagged = proba >= p20
yt = yte.values
caught, total = int((flagged & (yt == 1)).sum()), int((yt == 1).sum())
print(f"p20 threshold = {p20:.4f}  ->  catches {caught}/{total} ({caught/total:.0%}) of under-triaged in held-out set")

joblib.dump(
    {"model": model, "tfidf": tfidf, "vitals": vitals, "medians": medians.to_dict()},
    "artifacts/flagger.joblib",
)
with open("artifacts/thresholds.json", "w") as f:
    json.dump({"p20": p20}, f, indent=2)

print("wrote artifacts/flagger.joblib and artifacts/thresholds.json")
