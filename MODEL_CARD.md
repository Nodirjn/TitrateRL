# Model card — TitrateRL

## Overview

TitrateRL is a simulation environment and a set of reinforcement-learning and rule-based policies
that adjust a once-daily basal insulin dose for **virtual** adults with Type 2 diabetes.

| | |
|---|---|
| Developers | Team TitrateRL, Central Asian University (course 4-SRC-10-804) |
| Version | 0.1.0.dev0 (pre-M1) |
| License | MIT |
| Type | Research software; Gymnasium environment; PPO/DQN agents (Stable-Baselines3) |

## Intended use

* **In silico research** on basal-insulin titration algorithms: benchmarking RL against clinical
  titration rules, studying reward design and personalization.
* Teaching and demonstration of safe-RL methodology.

## Out-of-scope and prohibited uses

* **Any dosing decision for a real person.** TitrateRL is not a medical device, is not validated
  or cleared for clinical use, and its outputs are not medical advice.
* Inferring real-world efficacy or safety of any insulin regimen.
* Use with real patient data. The dashboard accepts only virtual-patient identifiers and offers
  no input for real glucose values or doses.

## Data

No human data. 150 virtual patients generated from published model parameters and
distributions approximating the baseline of the Treat-to-Target trial (see
[docs/model.md](docs/model.md)).

## Evaluation

Policies are compared on 30 held-out virtual patients × 10 scenarios with common random numbers,
using consensus CGM metrics (time in range, time below range) computed from noise-free glucose and
pre-specified statistical tests ([docs/evaluation.md](docs/evaluation.md)).
**No trained-agent results exist yet** (status: pre-M1).

## Known limitations

* **No glucagon / counter-regulation** in the model: hypoglycemia depth and duration can be
  exaggerated; hypoglycemia results are comparative, not absolute risks.
* Model parameters were identified from single-meal data of a small group of subjects; 24-week
  simulation is an extrapolation.
* The virtual cohort does not yet reproduce all Treat-to-Target trial outcomes (check V4: the
  week-24 fasting glucose is lower and more patients reach the target than in the trial; see
  [docs/validation.md](docs/validation.md)).
* Perfect adherence; no exercise, illness, stress, missed injections, background-therapy effects
  or disease progression.
* A policy that performs well in simulation may fail in reality (sim-to-real gap).
* The safety shield is rule-based and gives no formal safety guarantee.

## Ethical considerations

Only simulated patients are used; no human subjects, personal data or clinical records are
involved. Every output of the dashboard and the evaluation pipeline is labelled as a simulation
result.
