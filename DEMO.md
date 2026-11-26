# Week 12 demo — System Testing and Evaluation: plan and outcome

Live demonstration of TitrateRL for the Week 12 progress report (instructor and industry
evaluator). Duration about 10 minutes plus questions. Every member presents one part.

> In silico research software. Every number shown is a simulation result, not medical advice.

## 1. Prepare the laptop (the day before)

```bash
git clone https://github.com/Nodirjn/TitrateRL.git
cd TitrateRL
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # small CPU-only build
pip install -e ".[dev,dashboard]"

# warm-up: compiles the simulator once (Numba cache), ~30 s
titraterl simulate --policy b2_treat_to_target
# full system test, ~3-5 min; produces reports/system_test.md
titraterl system-test
# final evaluation results are already in the repository: docs/results/full_protocol.md
```

Checklist before the session:

- [ ] `titraterl system-test` shows all PASS except ST-03 (known calibration gap V4)
- [ ] `titraterl dashboard` opens in the browser and lists the 20 final agents (pilot agents marked "(pilot)")
- [ ] GitHub repository open in a browser tab: latest CI run green, pull requests visible
- [ ] Laptop charged, notifications off, terminal font large, browser zoom 125 %
- [ ] Fallback: `docs/course/` PDF and `docs/results/test_full/figures/` open in case of a failure

## 2. Demo script

| Time | Presenter | What to show | Command / screen |
|---|---|---|---|
| 0:00–1:30 | Nodirjon Khabibullaev | Goal, research questions, roadmap and the plan for System Testing and Evaluation (four steps) | slides 1–3 |
| 1:30–3:30 | Ibrohim Abralov | Simulator verification: integrator, simglucose cross-check, glargine profile; V4 sensitivity analysis: missed injections explain the gap to the trial | `titraterl validate --quick` |
| 3:30–5:30 | Boburmirzo Uchqunov | System tests against the MVP acceptance criteria; CI on GitHub; installable package | `titraterl system-test --skip-unit`, GitHub Actions tab |
| 5:30–7:30 | Abdurashid Abdusattarov | RL agents: rewards R1–R3, safety shield, full training (20 runs), reward selection, robustness with missed doses; live comparison PPO vs Treat-to-Target on one virtual patient | `titraterl dashboard` → patient T2D-005, policies `b2_treat_to_target`, `ppo_rstar_shield_s0` |
| 7:30–9:30 | Nodirjon Khabibullaev | Final evaluation on 30 test patients × 10 scenarios: TIR/TBR with CIs, H1–H3, limitations, plan for Weeks 13–14 | `docs/results/full_protocol.md`, slides 7–10 |
| 9:30– | all | Questions | |

## 3. Likely questions from the evaluator

| Question | Short answer |
|---|---|
| Is this safe to use on patients? | No. It is in silico research software; the dashboard accepts only virtual patients and every output is labelled as a simulation. |
| Why simulation and not real data? | Testing untrained dosing policies on people is unethical; no public patient-level T2D titration data exist; simulators are the accepted preclinical step. |
| How do you know the simulator is correct? | Test-driven development: RK4 matches SciPy, the right-hand side matches simglucose to 1e-9, equilibrium and dose-response tests, glargine profile check (V1, V2). V4 (Treat-to-Target trial) fails in the main cohort and passes once 15 % missed injections are simulated; that world is used as a robustness test. |
| Are the RL results final? | Yes for the protocol: 5 seeds × 10⁶ days per configuration, 20 runs. H1 supported; H2 and H3 not supported (gain 2–4 pp, below the 5 pp threshold). |
| Why keep a failing system test (ST-03)? | The criterion was fixed in advance; we explain the gap (non-adherence) instead of loosening it. |
| What stops the agent from overdosing? | Weekly rate limit (≤ 8 U per 7 days) and dose bound (≤ 1 U/kg) for every policy, plus the safety shield for RL agents; tested in ST-08 and ST-09. |
| How is the comparison fair? | Same 30 unseen patients, same 10 scenarios, identical meals and sensor noise (common random numbers), pre-specified statistics. |
