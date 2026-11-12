# Physiological model

The virtual patient is the meal simulation model of **Dalla Man, Rizza and Cobelli (IEEE TBME
54(10), 2007)** with its published Type 2 diabetes parameters, extended in this project with
subcutaneous long-acting (glargine) insulin. Source: `src/titraterl/simulation/`.

## State vector (15 states)

| # | Symbol | Meaning | Unit |
|---|---|---|---|
| 0 | `Gp` | glucose mass in plasma | mg/kg |
| 1 | `Gt` | glucose mass in slowly equilibrating tissues | mg/kg |
| 2 | `Gs` | interstitial glucose (read by the CGM) | mg/kg |
| 3 | `Il` | insulin mass in the liver | pmol/kg |
| 4 | `Ip` | insulin mass in plasma | pmol/kg |
| 5 | `X` | insulin action on utilization | pmol/L |
| 6 | `I1` | delayed insulin signal, compartment 1 | pmol/L |
| 7 | `Id` | delayed insulin signal acting on production | pmol/L |
| 8 | `Ipo` | portal-vein insulin | pmol/kg |
| 9 | `Y` | static beta-cell secretion component | pmol/kg/min |
| 10–12 | `Qsto1`, `Qsto2`, `Qgut` | stomach (solid, liquid), intestine | mg |
| 13–14 | `Isc1`, `Isc2` | subcutaneous insulin: depot, soluble | pmol/kg |

## Equations

Glucose, insulin, EGP, utilization, secretion and hepatic extraction (proposal eqs. 1–10):

```
dGp/dt  = EGP + Ra − Uii − E − k1·Gp + k2·Gt                 G = Gp / VG
dGt/dt  = −Uid + k1·Gp − k2·Gt
dIl/dt  = −(m1 + m3)·Il + m2·Ip + S
dIp/dt  = −(m2 + m4)·Ip + m1·Il + Ra_I                       I = Ip / VI
EGP     = max(0, kp1 − kp2·Gp − kp3·Id − kp4·Ipo)
Uid     = (Vm0 + Vmx·X)·Gt / (Km0 + Gt)
dX/dt   = −p2U·X + p2U·(I − Ib)
S       = γ·Ipo         dIpo/dt = −γ·Ipo + Spo     Spo = Y + K·max(0, dG/dt) + Sb
dY/dt   = −α·[Y − β(G − Gb)]          if β(G − Gb) ≥ −Sb
        = −α·Y − α·Sb                 otherwise
m3      = HE·m1 / (1 − HE)            HE = −m5·S + m6
dIsc1/dt = −(kd + ka1)·Isc1 + u(t)    dIsc2/dt = kd·Isc1 − ka2·Isc2    Ra_I = ka1·Isc1 + ka2·Isc2
```

Gut, renal excretion, delay chain and CGM compartment (Appendix A, eqs. A1–A9):

```
dQsto1/dt = −kmax·Qsto1 + CHO(t)
dQsto2/dt = −kempt(Qsto)·Qsto2 + kmax·Qsto1
dQgut/dt  = −kabs·Qgut + kempt(Qsto)·Qsto2           Ra = f·kabs·Qgut / BW
kempt     = kmin + (kmax − kmin)/2 · {tanh[αe(Qsto − bD)] − tanh[βe(Qsto − cD)] + 2}
Uii = Fcns          E = ke1·max(0, Gp − ke2)
dI1/dt = −ki(I1 − I)            dId/dt = −ki(Id − I1)
dGs/dt = −ksc·Gs + ksc·Gp       CGM = Gs / VG + ε_CGM
```

The bedtime injection of `d` units enters as an impulse `u = 6000·d / BW` pmol/kg into `Isc1` at
21:00 (1 U = 6000 pmol, as in simglucose). The system is integrated with a fixed-step RK4 scheme
with a 1-minute step.

## Parameters

Nominal T2D values (`NominalParameters` in `parameters.py`). Units follow Dalla Man et al. (2007).

| Group | Parameter values |
|---|---|
| Glucose kinetics | VG = 1.49 dL/kg, k1 = 0.042, k2 = 0.071 min⁻¹ |
| Insulin kinetics | VI = 0.04 L/kg, m1 = 0.379, m2 = 0.673, m4 = 0.269 min⁻¹, m5 = 0.0526, HEb = 0.6 |
| Gut | kmax = 0.0465, kmin = 0.0076, kabs = 0.023 min⁻¹, b = 0.68, c = 0.09, f = 0.9 |
| EGP | kp2 = 0.0007, kp3 = 0.005, kp4 = 0.0786, ki = 0.0066 |
| Utilization | Fcns = 1.0, Vmx = 0.034, Km0 = 466.21, p2U = 0.084 |
| Secretion | K = 0.99, α = 0.013, β = 0.05, γ = 0.5 |
| Renal | ke1 = 0.0007, ke2 = 269 |
| Interstitial | ksc = 0.1 min⁻¹ |
| Basal (all patients) | Ib = 50 pmol/L, EGPb = 2.0 mg/kg/min |
| Glargine (project) | kd = 4.1·10⁻⁴ min⁻¹, ka1 = 0, ka2 = 0.015 min⁻¹ |

> **To verify before M1.** The nominal T2D values were transcribed for the first implementation
> and must be checked line by line against Table I of Dalla Man et al. (2007) by the simulator
> owner. The basal insulin `Ib` and basal EGP `EGPb` are nominal values kept equal for all
> patients (proposal IV-C2); their sources must be documented.

### Patient-specific parameters

For each patient, `derive_basal()` solves the steady state without meals or exogenous insulin so
that the patient is in equilibrium at its own pre-insulin fasting glucose `Gb`:

```
m3b = HEb·m1/(1 − HEb);  Ipb = Ib·VI;  Ilb = (m2 + m4)·Ipb / m1
Sb  = (m1 + m3b)·Ilb − m2·Ipb;   m6 = HEb + m5·Sb;   Ipob = Sb / γ
Gpb = Gb·VG;  Eb = ke1·max(0, Gpb − ke2)
Gtb = (Fcns + Eb − EGPb + k1·Gpb) / k2;   Vm0 = (EGPb − Fcns − Eb)(Km0 + Gtb) / Gtb
kp1 = EGPb + kp2·Gpb + kp3·Ib + kp4·Ipob
```

Heterogeneity: `ρ_SI` scales `Vmx` and `kp3` (insulin sensitivity); `ρ_β` scales `β` and `K`
(beta-cell responsivity). The secretion threshold equals each patient's `Gb`.

## Glargine absorption

The two-compartment chain of eq. (10) gives, for one dose, `Ra_I(t) ∝ e^(−kd·t) − e^(−ka2·t)`.
`kd` and `ka2` were chosen as the values that satisfy both parts of check V2 (flat profile and
steady state by the 4th dose):

* after a single dose, `Ra_I` stays within ±25 % of its mean between 2 and 24 h (23.7 %);
* with once-daily dosing, 90.3 % of the steady-state daily exposure is reached by the 4th dose.

The margins are small. The published glargine model of Schiavon et al. (precipitate and soluble
compartments) should replace these constants if it can be adapted (proposal IV-C1).

## Meals and measurements

* Breakfast 45 g (07:30–08:30), lunch 70 g (12:00–13:00), dinner 80 g (18:30–19:30), snack 10 g
  (15:00–16:00, probability 0.3); amounts ±20 %; each meal eaten over 10 min.
* CGM every 5 min: Breton–Kovatchev error model (AR(1), Johnson SU) as in simglucose, clipped to
  39–400 mg/dL.
* Fasting fingerstick at 07:00: Gaussian error, SD 7.5 mg/dL below 100 mg/dL and 7.5 % above
  (≈ 95 % within ISO 15197:2013 limits; tested).

## Known limitations

* **No glucagon or counter-regulation.** Apart from the weak dependence of EGP on glucose and the
  fall in secretion, nothing opposes a fall in glucose, so the depth and duration of hypoglycemia
  can be exaggerated. Hypoglycemia is interpreted **comparatively** between policies.
* T2D parameters were identified from single-meal data of a small group; 24-week multi-meal
  simulation is an extrapolation.
* Background therapy (metformin, sulfonylureas, GLP-1 agonists), exercise, illness, missed
  injections and disease progression are not modeled.
