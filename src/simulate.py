"""
Synthetic degradation simulator for the SMC SIF-400 assembly line.

IMPORTANT -- HONESTY NOTE
-------------------------
The SIF-400 station topology and sensor types reproduced here are taken from
the manufacturer's public documentation (SMC International Training). The
DEGRADATION DYNAMICS, however, are NOT measured data: no run-to-failure logs of
the SIF-400 were available to us. Every failure-mode profile below is an
explicit ENGINEERING ASSUMPTION made by the authors, stated in the paper. This
module is therefore a synthetic test-bed *informed by* the real SIF-400
topology, not a data-driven digital twin of a specific physical unit.

Stations and sensors modelled (assembly block, per the paper):
  SIF-401  Pallet/container feeding    -> electric actuator current
  SIF-402  Solid filling               -> dosing position / fill check
  SIF-403  Liquid filling              -> flow rate
  SIF-405  Capping                     -> pneumatic / vacuum pressure
  SIF-406  Container warehouse         -> insertion/extraction motor current

Assumed failure modes (AUTHORS' ASSUMPTIONS, declared in the paper):
  A1  electric actuators: gradual current rise from friction/wear (monotone)
  A2  pneumatic/vacuum: pressure decay from seal leakage, with partial
      recoveries when a cycle reseats the seal (intermittent)
  A3  flow regulation: drift with occasional clogging bursts (abrupt-ish)
  A4  position/dosing: slow loss of repeatability (variance growth)
All modes share severe sensor noise; the JIT label is defined on the hidden
structural state, not on the noisy reading.
"""

from __future__ import annotations
import numpy as np

# Real SIF-400 assembly stations and their dominant sensor channel.
STATIONS = [
    ("SIF-401", "actuator_current"),   # electric actuator
    ("SIF-402", "dosing_position"),    # solid fill check
    ("SIF-403", "flow_rate"),          # liquid flow
    ("SIF-405", "vacuum_pressure"),    # pneumatic capping
    ("SIF-406", "motor_current"),      # warehouse motor
]
N_CH = len(STATIONS)

# Nominal sensor levels and full-degradation shifts (authors' assumptions).
# These set the SCALE of each channel; they are not measured values.
NOMINAL = np.array([1.20, 5.00, 2.50, -0.65, 1.10])   # e.g. A, mm, L/min, bar, A
FAULT_SHIFT = np.array([0.90, -1.50, -1.20, 0.40, 0.85])
NOISE_SCALE = np.array([1.20, 5.00, 2.50, 0.65, 1.10])


def _structural_eta(t, rng):
    """Hidden structural degradation eta(t) in [0,1], irreversible, with a
    random final severity so the end-of-horizon state spans the threshold."""
    eta_final = rng.uniform(0.2, 1.0)
    onset = rng.uniform(0.2, 0.6)
    ramp = np.clip((t - onset) / (1.0 - onset), 0.0, 1.0)
    eta = eta_final * ramp ** rng.uniform(0.8, 1.5)
    return np.maximum.accumulate(eta)


class SIF400Simulator:
    def __init__(self, n_lifecycles=1000, noise_level=0.08, cycle_length=100,
                 seed=42):
        self.n = n_lifecycles
        self.noise_level = noise_level
        self.L = cycle_length
        self.rng = np.random.default_rng(seed)

    def generate(self):
        t = np.linspace(0.0, 1.0, self.L)
        X = np.empty((self.n, self.L, N_CH))
        eta_struct = np.empty((self.n, self.L))
        y = np.empty(self.n, dtype=int)

        for k in range(self.n):
            eta = _structural_eta(t, self.rng)
            eta_struct[k] = eta

            for c, (_, kind) in enumerate(STATIONS):
                # Channel mean follows the structural degradation...
                mean = NOMINAL[c] + FAULT_SHIFT[c] * eta

                # ...modulated by the ASSUMED failure mode for that sensor type.
                if kind in ("actuator_current", "motor_current"):
                    # A1: monotone friction rise; clean monotone signature
                    extra = 0.0
                elif kind == "vacuum_pressure":
                    # A2: seal leakage with partial reseating recoveries
                    extra = np.zeros(self.L)
                    for _ in range(self.rng.integers(2, 5)):
                        cpos = self.rng.uniform(0.4, 0.95)
                        w = self.rng.uniform(0.02, 0.05)
                        extra += self.rng.uniform(0.10, 0.25) * \
                            np.exp(-((t - cpos) ** 2) / (2 * w ** 2))
                    extra = -FAULT_SHIFT[c] * extra  # recoveries oppose the fault
                elif kind == "flow_rate":
                    # A3: drift plus occasional clogging bursts
                    burst = self.rng.choice([0.0, 1.0], size=self.L,
                                            p=[0.97, 0.03])
                    extra = burst * self.rng.uniform(-0.8, 0.8, self.L)
                else:  # dosing_position
                    # A4: loss of repeatability -> variance grows with eta
                    extra = self.rng.normal(0.0, 0.4 * eta, self.L)

                noise = self.rng.normal(0.0, self.noise_level * NOISE_SCALE[c],
                                        self.L)
                X[k, :, c] = mean + extra + noise

            # JIT label on the hidden structural state near end of horizon.
            y[k] = 1 if eta[-15] > 0.6 else 0

        return X, eta_struct, y


if __name__ == "__main__":
    sim = SIF400Simulator(n_lifecycles=500, seed=0)
    X, eta, y = sim.generate()
    print(f"X shape = {X.shape}  (lifecycles, steps, channels)")
    print(f"channels = {[s for s, _ in STATIONS]}")
    print(f"positives (JIT due) = {y.mean():.1%}")
    for c, (st, kind) in enumerate(STATIONS):
        print(f"  {st} [{kind:18s}] mean={X[:,:,c].mean():.2f} std={X[:,:,c].std():.2f}")
