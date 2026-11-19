# Week 11 demo — System Testing and Evaluation

Live demonstration of TitrateRL for the Week 11 progress report (instructor and industry
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
# pilot evaluation results are already in the repository: results/week11_pilot/report.md
```

Checklist before the session:

- [ ] `titraterl system-test` shows all PASS except ST-03 (known calibration gap V4)
- [ ] `titraterl dashboard` opens in the browser and lists the five pilot agents
- [ ] GitHub repository open in a browser tab: latest CI run green, pull requests visible
- [ ] Laptop charged, notifications off, terminal font large, browser zoom 125 %
- [ ] Fallback: `docs/course/` PDF and `results/week11_pilot/figures/` open in case of a failure

## 2. Demo script

| Time | Presenter | What to show | Command / screen |
|---|---|---|---|
| 0:00–1:30 | Nodirjon Khabibullaev | Goal, research questions, where we are on the roadmap (Week 11: system testing and evaluation) | slides 1–3 |
| 1:30–3:30 | Ibrohim Abralov | Simulator verification: integrator, simglucose cross-check, glargine profile; honest status of the calibration (V4) | `titraterl validate --quick` |
| 3:30–5:30 | Boburmirzo Uchqunov | System tests against the MVP acceptance criteria; CI on GitHub; installable package | `titraterl system-test --skip-unit`, GitHub Actions tab |
| 5:30–7:30 | Abdurashid Abdusattarov | RL agents: rewards R1–R3, safety shield, training and checkpoint selection; live comparison PPO vs Treat-to-Target on one virtual patient | `titraterl dashboard` → patient T2D-005, policies `b2_treat_to_target`, `ppo_rstar_shield_s0` |
| 7:30–9:30 | Nodirjon Khabibullaev | Pilot evaluation on 30 test patients × 10 scenarios: TIR/TBR with CIs, H1–H3 preliminary, limitations, plan for Weeks 12–14 | `results/week11_pilot/report.md`, slides 6–8 |
| 9:30– | all | Questions | |

## 3. Likely questions from the evaluator

| Question | Short answer |
|---|---|
| Is this safe to use on patients? | No. It is in silico research software; the dashboard accepts only virtual patients and every output is labelled as a simulation. |
| Why simulation and not real data? | Testing untrained dosing policies on people is unethical; no public patient-level T2D titration data exist; simulators are the accepted preclinical step. |
| How do you know the simulator is correct? | Test-driven development: RK4 matches SciPy, the right-hand side matches simglucose to 1e-9, equilibrium and dose-response tests, glargine profile check (V1, V2). Calibration to the Treat-to-Target trial (V4) is still open and reported. |
| Are the RL results final? | No. The pilot uses 3×10⁵ training days and one seed (protocol: 10⁶ days, five seeds). Weeks 11–12 run the full protocol. |
| What stops the agent from overdosing? | Weekly rate limit (≤ 8 U per 7 days) and dose bound (≤ 1 U/kg) for every policy, plus the safety shield for RL agents; tested in ST-08 and ST-09. |
| How is the comparison fair? | Same 30 unseen patients, same 10 scenarios, identical meals and sensor noise (common random numbers), pre-specified statistics. |
