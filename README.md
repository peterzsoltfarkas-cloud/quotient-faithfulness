# Quotient-Faithful Representation Learning — Code Repository

Companion code for the paper series on quotient faithfulness in representation learning.

**Peter Farkas** — Åbo Akademi University, DAZE Project  
LinkedIn: [Paper I](https://www.linkedin.com/in/peter-farkas-data/)

---

## Paper series

| Paper | Title | Status |
|-------|-------|--------|
| I (2026a) | *Task Sufficiency Does Not Certify Quotient Faithfulness* | Published |
| II (2026b) | *Admissibility Sources and Stochastic Quotient Recovery* | Draft |
| III (2026c) | *Architecture Bounds for Quotient-Faithful Representation* | Draft |
| IV (2026d) | *Six Traditions, Fourteen Reframings* | Draft |
| V (2026e) | *Generator Specifications and the GCD Scheme* | Draft |
| VI (2026f) | *Preliminary Empirical Investigation of the GCD Framework* | Draft |

---

## Repository structure

```
gcd/                        Core GCD framework
  model.py                  Encoder, LieGen, FiLM decoder architectures
  generators.py             Lie generator bank and group action utilities
  losses.py                 L_eq, L_cl, L_phys loss functions
  topology.py               Persistent homology utilities (ripser wrapper)
  homology.py               H1/H2 bar computation helpers

examples/                   Paper experiments I–V
  train_phase_cylinder.py   Experiment 1: Phase cylinder (S¹)
  train_hopf.py             Experiment 2: Hopf fibration (S²)
  train_double_pendulum.py  Experiment 3 (earlier): Double pendulum baseline
  run_all_experiments.py    Batch runner for experiments 1–3 + PH evaluation
  run_extrapolation_culled.py  Structural tests: extrapolation and culled data
  run_ph_certification.py   Persistent homology measurements (Table 1)
  make_split_loss_figures.py   Training dynamics figures
  test_circle_ph.py         PH unit test: circle
  test_sphere_ph.py         PH unit test: sphere

  paper6_double_pendulum/   Paper VI — Experiment 3 (canonical variables)
    run_dp_canonical.py     Lagrangian-derived canonical variables,
                            learned Hamiltonian head, three-condition benchmark

  paper6_color_fantasy/     Paper VI — Experiment 4 (ship hotel system)
    cf_covariance_analysis.py   Multi-scale cross-covariance (raw)
    cf_partial_covariance.py    Partial correlations (seasonal+diurnal removed)
    cf_input_covariance.py      Input×input covariance + PCA
    cf_physics_analysis.py      Thermal states, windchill, impulse responses
    cf_delay_estimation.py      FIR delay estimation per orthogonal state
    cf_hierarchical_gcd.py      Arch-B vs Arch-E hierarchical GCD benchmark
    cf_fourblock.py             Four-block architecture + Block-2 ablation
```

---

## Installation

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib ripser
```

The Color Fantasy scripts additionally require:
```bash
pip install openpyxl  # for reading the IDA ICE Excel data files
```

Data files (IDA ICE simulation + weather, provided separately):
- `Color_Fantasy_IDA_ICE_results_-_August_Brækken.xlsx`
- `Color_Fantasy_weather_data_-_August_Brækken.xlsx`

Source: Brækken, Gabrielii & Nord (2023). Place in the working directory or
update the path constants at the top of each `cf_*.py` script.

---

## Quick start

**Phase cylinder (S¹ recovery):**
```bash
python examples/train_phase_cylinder.py
```

**Double pendulum (canonical variables, Hamiltonian physics loss):**
```bash
python examples/paper6_double_pendulum/run_dp_canonical.py
```

**Color Fantasy analysis pipeline (run in order):**
```bash
python examples/paper6_color_fantasy/cf_covariance_analysis.py
python examples/paper6_color_fantasy/cf_partial_covariance.py
python examples/paper6_color_fantasy/cf_input_covariance.py
python examples/paper6_color_fantasy/cf_physics_analysis.py
python examples/paper6_color_fantasy/cf_delay_estimation.py
python examples/paper6_color_fantasy/cf_hierarchical_gcd.py
python examples/paper6_color_fantasy/cf_fourblock.py
```

Figures are written to `figures/` (created automatically).

---

## Key design principles

All experiments follow the same derivation-first workflow:

1. Derive the manifold structure from the Lagrangian/Hamiltonian or physical model
2. Identify generators from the topology of the configuration/state space
3. Establish the embedding floor from the product structure
4. Define losses from the manifold (not imposed ad hoc)
5. Train and evaluate

The physics loss `L_phys` in the double pendulum uses a **learned head**
`Ĥ(fθ(x))` regressing to the analytically known Hamiltonian `H(x)`, not
batch variance. See `run_dp_canonical.py` and Paper VI §5.1 for the
batch-learning analysis explaining why batch variance is incorrect for
pooled multi-energy data.

---

## AI attribution

Code developed collaboratively with Claude (Anthropic) using the `+` attribution
level under the series' AI disclosure framework. Author retains full scientific
and intellectual responsibility for all design decisions and interpretations.

---

## Citation

```bibtex
@misc{farkas2026f,
  author = {Farkas, Peter},
  title  = {Preliminary Empirical Investigation of the {GCD} Framework},
  year   = {2026},
  note   = {Manuscript. Åbo Akademi University, DAZE Project.}
}
```
