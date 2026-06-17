# Physics-Driven Hybrid Digital Twin (LG-HDT) for PHM

Reproducibility package for the DCETIP 2026 paper *"A Physics-Driven Hybrid
Digital Twin for Prognostics and Health Management: a Landau–Ginzburg
Approach"* (D. Rocha, R. Pinto, G. Gonçalves; FEUP/SYSTEC, University of Porto).

The LG-HDT models equipment health as a **cubic Landau–Ginzburg free energy**
with an **external stress field**, fuses multiple PLC sensor channels into a
single control parameter, and reads a remaining-useful-life estimate off the
**Kramers escape rate**. It extends the conceptual quartic model of our ETFA
2025 paper into a working prognostic architecture with a Just-in-Time (JIT)
maintenance trigger.

## What this method is — and is not

This package is written to be honest about what the method does. On a synthetic
test-bed informed by the SMC SIF-400 topology, the LG-HDT **does not beat**
purely data-driven baselines (MLP, Random Forest) on nominal accuracy: a
well-tuned Random Forest is a strong baseline on clean, stationary data. What
the physical prior buys instead, and what this code demonstrates, is:

1. **Stability** — the LG-HDT degrades the slowest and shows the **lowest
   run-to-run variance** under severe noise (Table II).
2. **Noise-invariant RUL** — its RUL error stays approximately flat as noise
   grows, where the baselines deteriorate (Table III).
3. **Tiny footprint** — ~20 parameters / ~0.5 KB vs. hundreds of KB for the
   baselines, making edge/PLC deployment realistic (Table IV).
4. **Interpretability** — every output is a physical quantity (phase, energy
   barrier, RUL), not a bare label.

**Honesty note on the test-bed.** The SIF-400 *station topology and sensor
types* are taken from the manufacturer's public documentation. The *degradation
dynamics are not measured data* — there were no run-to-failure logs of the
SIF-400 available. Every failure-mode profile in `src/simulate.py` is an
explicit engineering assumption, stated as such in the paper. This is a
synthetic test-bed informed by the real topology, not a data-driven twin of a
physical unit.

## Repository layout

```
physics-hdt-phm/
├── src/
│   ├── simulate.py     # SIF-400 topology-informed synthetic test-bed
│   └── evaluate.py     # LG-HDT model + MLP/RF baselines + evaluation
├── scripts/
│   └── reproduce_paper.py   # reproduces Tables II, III, IV + interpretability
├── results/            # output of reproduce_paper.py is written here
├── requirements.txt
├── LICENSE
└── README.md
```

## Quickstart

```bash
git clone https://github.com/diogrocha/physics-hdt-phm.git
cd physics-hdt-phm
pip install -r requirements.txt

# Reproduce the paper tables (measured, ~3-5 min):
python scripts/reproduce_paper.py

# Or run a single noise level and see all five models:
python src/evaluate.py
```

## Expected output (matches the paper)

| Noise | MLP | Random Forest | LG-HDT |
|------:|:---:|:-------------:|:------:|
|  8 %  | 95.1 ± 0.5 | 95.7 ± 0.3 | 89.9 ± 2.3 |
| 75 %  | 80.8 ± 4.6 | 85.1 ± 2.7 | **82.4 ± 1.5** |

At 75 % noise the LG-HDT has the lowest variance of the three models and
overtakes the MLP. Footprint: LG-HDT ~20 parameters (~0.5 KB) vs. Random Forest
~3 400 nodes (~296 KB).

## Citation

If you use this code, please cite the DCETIP 2026 paper and the ETFA 2025
predecessor:



@inproceedings{rocha2025etfa,
  author    = {Rocha, Diogo and Pinto, Rui and Gon\c{c}alves, Gil},
  title     = {A Hybrid Digital Twin Framework for Stability Analysis in
               Dynamic Systems},
  booktitle = {Proc. 30th IEEE Int. Conf. Emerging Technologies and
               Factory Automation (ETFA)},
  year      = {2025},
  doi       = {10.1109/ETFA65518.2025.11205759}
}
```

## License

MIT — see [LICENSE](LICENSE).
