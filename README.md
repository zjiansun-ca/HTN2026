## Kairos

**Some ER patients are triaged "non-urgent," sent to wait — and are actually sick.** Kairos is a triage *second reader* that catches them.

Canada logs ~16 million ER visits a year; ~500,000 people leave before a doctor ever sees them, and longer waits are associated with worsening conditions and adverse events. The most dangerous error in that system isn't the patient who's obviously critical — it's the one triaged low-acuity who quietly deteriorates in the waiting room. Kairos is built to catch that specific, high-cost, hard-to-spot failure.

## What it does

You sit in the triage nurse's seat. For a patient already triaged non-urgent, you decide: keep them in the waiting room, or pull them back for reassessment? Then Kairos reveals its own call — an under-triage flag with a faithful, feature-level explanation — alongside **what actually happened to that patient.** Across everyone who plays, a live leaderboard shows how often humans vs. the model catch the quietly-sick.

## How we built it — and why it's rigorous

Kairos is trained on **425,000 real emergency-department visits** (MIMIC-IV-ED). The core design choice: it's validated against **real patient outcomes** — hospital admission and death — **not against the triage nurse's own label.** Training a model to predict the nurse's label just launders the human bias we're trying to correct; anchoring on outcomes measures whether we catch patients the current process *missed*.

- **Model:** logistic regression on standardized vitals + TF-IDF of the chief complaint. AUC **0.786** (text + vitals) vs **0.633** (vitals alone) — the chief-complaint text nearly doubles discrimination over vitals, which is why simple vital thresholds miss these patients.
- **Operating point:** flagging the top 20% of low-acuity patients by risk catches **~59% of the ones who were actually admitted or died.** False flags are cheap (a 30-second second look); a missed catch can be fatal — an asymmetry we designed around deliberately.
- **Interpretability:** every flag shows its real feature contributions (vitals vs. text, positive vs. protective) — no black box.
- **Synthetic demo cases** are drawn from the real low-acuity subpopulation using per-stratum multivariate distributions conditioned on complaint type, so they're distributionally faithful without exposing any real record.

## Data governance

MIMIC-IV-ED is credentialed data under a PhysioNet Data Use Agreement. **No raw patient record ever leaves the machine** — not to an API, a cloud database, or the repo. The model is trained locally; only aggregate statistics and fully synthetic cases are used in the live app. Gemini and Tiger Data only ever see model outputs or synthetic data.

## Tech

FastAPI backend; vanilla single-file frontend styled as a clinical console. **Tiger Data** (hypertable + continuous aggregate) powers the live human-vs-model leaderboard. **Gemini** turns the model's real feature contributions into plain-language clinical explanations at reveal time. **Sentry** provides tracing and logging on the decision endpoint — because in a clinical tool, a silent wrong answer is the dangerous failure.

## Challenges & what we learned

We caught and fixed our own biases: an unstandardized-feature bug where normal-but-large vitals inflated risk; an independent-sampling flaw that made synthetic cases artificially easy; a graceful-degradation path that hung on filtered ports (fixed with a circuit breaker). The recurring lesson: in clinical ML, the honest failure modes matter more than the headline accuracy.

## Limitations

MIMIC uses ESI, not Canada's CTAS; it's a single US hospital, so external validity is unproven. Some model coefficients are counterintuitive (class-imbalance confounds), and the model over-weights certain tokens (e.g. "fall"). Kairos is a **research prototype and decision-support tool — not a diagnosis, not medical advice, and never a replacement for the triage nurse.**

## What's next

A real clinical path: silent-mode pilot (log flags without changing care) → nurse-override feedback to retrain → integration into eCTAS → automated reassessment alerts for boarded patients.
