"""Flagger scoring — the live decision path. No web framework dependency, so this can be
imported by the self-check gate (or anything else) without pulling in FastAPI or booting
a server. Faithful reason = the model's real top contributing features (coef x value),
namespaced by kind so a vital and a same-named text token can never collide.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parent.parent

artifact = joblib.load(ROOT / "artifacts" / "flagger.joblib")
MODEL = artifact["model"]
TFIDF = artifact["tfidf"]
VITALS = artifact["vitals"]  # ordered list, matches training column order
MEDIANS = artifact["medians"]
SCALER = artifact["scaler"]

with open(ROOT / "artifacts" / "thresholds.json") as f:
    P20 = json.load(f)["p20"]

# training stripped these outcome-leaking tokens before fitting tfidf; live text
# must get the same treatment or the vectorizer sees an out-of-distribution string
LEAK_TOKENS = ["transfer", "intubated", "admit"]


def _clean_text(text: str) -> str:
    text = (text or "").lower()
    for tok in LEAK_TOKENS:
        text = text.replace(tok, "")
    return text


def _score_case(case: dict) -> tuple[float, bool, list[dict]]:
    vitals_row = case["vitals"]
    x_num_raw = np.array(
        [[float(vitals_row.get(v)) if vitals_row.get(v) is not None else MEDIANS[v] for v in VITALS]]
    )
    # DataFrame with training column names avoids a spurious sklearn feature-name warning
    x_num = SCALER.transform(pd.DataFrame(x_num_raw, columns=VITALS))  # medians-imputed values must be scaled too
    x_text = TFIDF.transform([_clean_text(case["chief_complaint"])])
    x = hstack([csr_matrix(x_num), x_text])

    proba = float(MODEL.predict_proba(x)[0, 1])
    flag = proba >= P20

    coefs = MODEL.coef_[0]
    contributions = [
        {"feature": name, "kind": "vital", "contribution": float(coefs[i] * x_num[0, i])}
        for i, name in enumerate(VITALS)
    ]

    term_names = TFIDF.get_feature_names_out()
    x_text_row = x_text.tocoo()
    for col, val in zip(x_text_row.col, x_text_row.data):
        coef_i = coefs[len(VITALS) + col]
        contributions.append({"feature": term_names[col], "kind": "text", "contribution": float(coef_i * val)})

    # sort by magnitude only — kind (not name) disambiguates a vital from a same-named text token
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    top_features = [
        {"feature": c["feature"], "kind": c["kind"], "contribution": round(c["contribution"], 4)}
        for c in contributions[:5]
    ]
    return proba, flag, top_features
