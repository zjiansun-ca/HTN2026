import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
import joblib

ED = r"C:\Users\zjian\Downloads\mimic-iv-ed-2.2\mimic-iv-ed-2.2"   # adjust to your path
stays  = pd.read_csv(f"{ED}/edstays.csv")
triage = pd.read_csv(f"{ED}/triage.csv")

df = triage.merge(stays[["stay_id", "disposition"]], on="stay_id", how="inner")

# 1. Sanity check: acuity should look like ESI (~3/34/55/8/0.3%)
print(df["acuity"].value_counts(normalize=True, dropna=False).sort_index())

# 2. THE number — your under-triage base rate:
low = df[df["acuity"].isin([4, 5])]
under = low["disposition"].isin(["ADMITTED", "EXPIRED"]).mean()
print(f"Low-acuity (4–5) admitted/died: {under:.1%}  (n={len(low):,})")

# 3. Peek at the free-text Gemini will have to parse:
print(df["chiefcomplaint"].dropna().sample(10).tolist())

# temp is FAHRENHEIT in MIMIC (BIDMC, US) — don't use Celsius thresholds
low = df[df["acuity"].isin([4, 5])].copy()
low["under"] = low["disposition"].isin(["ADMITTED", "EXPIRED"])

def redflag_vitals(r):
    f = 0
    if pd.notna(r.sbp) and r.sbp < 90:          f += 1   # hypotension
    if pd.notna(r.o2sat) and r.o2sat < 92:      f += 1   # hypoxia
    if pd.notna(r.heartrate) and r.heartrate > 110: f += 1
    if pd.notna(r.resprate) and r.resprate > 24:    f += 1
    if pd.notna(r.temperature) and r.temperature > 100.4: f += 1  # °F
    return f

low["nflags"] = low.apply(redflag_vitals, axis=1)
print(low.groupby("under")["nflags"].mean())   # do the under-triaged carry more red-flag vitals?

terms = ["chest pain","hypotension","short of breath","sob","syncope",
         "altered","unresponsive","suicid","overdose","stroke","weakness"]
cc = low.loc[low["under"], "chiefcomplaint"].fillna("").str.lower()
print(f"Under-triaged with a red-flag complaint token: {cc.str.contains('|'.join(terms)).mean():.1%}")

low = df[df["acuity"].isin([4, 5])].copy()
low["under"] = low["disposition"].isin(["ADMITTED", "EXPIRED"]).astype(int)

# strip outcome-leaking tokens — these encode what happened AFTER the door
cc = low["chiefcomplaint"].fillna("").str.lower()
for leak in ["transfer", "intubated", "admit"]:
    cc = cc.str.replace(leak, "", regex=False)

vitals = ["temperature","heartrate","resprate","o2sat","sbp","dbp","pain"]
X_num = low[vitals].apply(pd.to_numeric, errors="coerce")
X_num = X_num.fillna(X_num.median())

Xtr_n, Xte_n, ytr, yte, cc_tr, cc_te = train_test_split(
    X_num, low["under"], cc, test_size=0.25, stratify=low["under"], random_state=0)

tfidf = TfidfVectorizer(min_df=20, ngram_range=(1,2))
Xtr_t = tfidf.fit_transform(cc_tr); Xte_t = tfidf.transform(cc_te)
Xtr = hstack([csr_matrix(Xtr_n.values), Xtr_t])
Xte = hstack([csr_matrix(Xte_n.values), Xte_t])

m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
print(f"AUC (vitals only): {roc_auc_score(yte, LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr_n, ytr).predict_proba(Xte_n)[:,1]):.3f}")
print(f"AUC (vitals+text): {roc_auc_score(yte, m.predict_proba(Xte)[:,1]):.3f}")

proba = m.predict_proba(Xte)[:,1]
yt = yte.values
for budget in [0.10, 0.15, 0.20, 0.30]:
    thr = np.quantile(proba, 1 - budget)      # flag the top `budget` by risk
    flagged = proba >= thr
    caught = int((flagged & (yt == 1)).sum())
    total  = int((yt == 1).sum())
    fp     = int((flagged & (yt == 0)).sum())
    print(f"flag {budget:.0%} of lows -> catch {caught/total:.0%} of missed-sick "
          f"({caught}/{total}) | {fp/max(caught,1):.1f} false flags per real catch")


    
joblib.dump({"model": m, "tfidf": tfidf, "vitals": vitals}, "flagger.joblib")