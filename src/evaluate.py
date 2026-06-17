"""
Evaluation on the SIF-400 topology-informed synthetic test-bed.

Multi-channel version: the LG-HDT fuses the 5 station sensor channels into a
single control parameter eta (nonlinear fusion), then applies the cubic
Landau-Ginzburg landscape with an external stress field H (Bertolami coupling)
and the Kramers reading. Baselines (MLP, Random Forest) use per-channel window
statistics. All models are measured on the same held-out test set; nothing is
hard-coded.
"""

from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from simulate import SIF400Simulator, STATIONS, N_CH


# ---- shared smoothing ----
def ema(signal, alpha=0.25):
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]
    return out


# =====================================================================
# LG-HDT: nonlinear fusion -> cubic LGT (+external field H) -> Kramers
# =====================================================================
class LGHDT:
    def __init__(self, a0=1.0, alpha=0.8, gamma=2.2, b=1.0, h_coupling=0.5):
        self.a0, self.alpha, self.gamma, self.b = a0, alpha, gamma, b
        self.h_coupling = h_coupling
        self.w = None          # fusion weights (per channel)
        self.mu = None         # channel normalisation
        self.sd = None
        self.threshold = None

    # nonlinear multisensor fusion -> eta in [0,1] per timestep
    def _fuse_series(self, Xk):
        # Xk: (L, N_CH). Normalise channels, take |.|, weighted sum -> squashed.
        z = (Xk - self.mu) / self.sd
        lin = np.abs(z) @ self.w
        return 1.0 / (1.0 + np.exp(-lin))      # logistic squash to [0,1]

    def _coeffs(self, eta):
        return self.a0 - self.alpha * eta, self.gamma * eta

    def _F(self, psi, a, c, H):
        return (0.5*a*psi**2 - (1/3)*c*psi**3 + 0.25*self.b*psi**4 - H*psi)

    def _fpp(self, psi, a, c):
        return a - 2*c*psi + 3*self.b*psi**2

    def barrier(self, eta, H):
        a, c = self._coeffs(eta)
        roots = np.roots([self.b, -c, a, -H])
        real = sorted(r.real for r in roots if abs(r.imag) < 1e-8)
        if len(real) < 3:
            psi_h = real[0] if real else 0.0
            fh = max(self._fpp(psi_h, a, c), 1e-3)
            return 0.0, fh, fh
        psi_h, psi_s, _ = real
        dF = self._F(psi_s, a, c, H) - self._F(psi_h, a, c, H)
        return (max(0.0, dF), max(self._fpp(psi_h, a, c), 1e-3),
                max(abs(self._fpp(psi_s, a, c)), 1e-3))

    def _score(self, Xk):
        sm = np.column_stack([ema(Xk[:, c]) for c in range(N_CH)])
        eta_series = self._fuse_series(sm)
        eta_hat = float(np.clip(eta_series[-15:].mean(), 0.0, 1.0))
        sigma = float(np.std(eta_series[-15:] - eta_series[-15:].mean()) + 1e-3)
        H = self.h_coupling * eta_hat
        dF, fh, fs = self.barrier(eta_hat, H)
        intr = 1.0 if dF > 0 else 0.0
        escape = np.exp(-dF / (sigma**2 + 1e-6))
        return eta_hat + intr * escape

    def fit(self, Xtr, ytr):
        # channel normalisation from training data
        flat = Xtr.reshape(-1, N_CH)
        self.mu = flat.mean(0)
        self.sd = flat.std(0) + 1e-9
        # fusion weights: ridge of |z| end-window means onto the label proxy
        feats, targ = [], []
        for k in range(len(Xtr)):
            z = np.abs((Xtr[k, -15:] - self.mu) / self.sd).mean(0)
            feats.append(z)
            targ.append(ytr[k])
        A = np.array(feats); yv = np.array(targ, float)
        self.w = np.linalg.solve(A.T@A + 1.0*np.eye(N_CH), A.T@yv)
        # calibrate decision threshold on train
        s = np.array([self._score(Xtr[k]) for k in range(len(Xtr))])
        cand = np.quantile(s, np.linspace(0.02, 0.98, 97))
        best, bacc = cand[0], -1
        for thr in cand:
            acc = ((s >= thr).astype(int) == ytr).mean()
            if acc > bacc:
                bacc, best = acc, thr
        self.threshold = float(best)
        return self

    def predict(self, X):
        return np.array([1 if self._score(X[k]) >= self.threshold else 0
                         for k in range(len(X))])


def ml_features(X):
    # per-channel window statistics
    feats = []
    for k in range(len(X)):
        row = []
        for c in range(N_CH):
            w = X[k, -15:, c]
            row += [w.mean(), w.std(), w.min(), w.max()]
        feats.append(row)
    return np.array(feats)


def run(noise_level=0.08, seed=42, verbose=True):
    sim = SIF400Simulator(n_lifecycles=1000, noise_level=noise_level, seed=seed)
    X, _, y = sim.generate()
    Xml = ml_features(X)

    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import xgboost as xgb

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
    rf.fit(Xml[tr], y[tr]); pr = rf.predict(Xml[te])

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
    mlp.fit(Xml[tr], y[tr]); pm = mlp.predict(Xml[te])

    # XGBoost: the de-facto strong baseline for tabular PHM features
    xgbc = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             subsample=0.9, eval_metric="logloss",
                             random_state=seed, verbosity=0)
    xgbc.fit(Xml[tr], y[tr]); px = xgbc.predict(Xml[te])

    # SVM with RBF kernel (standardised features)
    svm = make_pipeline(StandardScaler(),
                        SVC(C=2.0, gamma="scale", random_state=seed))
    svm.fit(Xml[tr], y[tr]); ps = svm.predict(Xml[te])

    hdt = LGHDT().fit(X[tr], y[tr]); ph = hdt.predict(X[te])

    def m(yp):
        return (accuracy_score(y[te], yp), precision_score(y[te], yp, zero_division=0),
                recall_score(y[te], yp, zero_division=0), f1_score(y[te], yp, zero_division=0))
    rows = [("MLP Neural Network", *m(pm)),
            ("Random Forest", *m(pr)),
            ("XGBoost", *m(px)),
            ("SVM (RBF)", *m(ps)),
            ("LG-HDT (proposed)", *m(ph))]
    if verbose:
        print(f" positives={y.mean():.1%}")
        print(f"{'Model':24s}{'Acc':>8s}{'Prec':>8s}{'Recall':>8s}{'F1':>8s}")
        for n, a, p, r, f in rows:
            print(f"{n:24s}{a:8.1%}{p:8.1%}{r:8.1%}{f:8.2f}")
    return rows


if __name__ == "__main__":
    print("=" * 64)
    print(" SIF-400 topology-informed test-bed | 8% noise | measured")
    print("=" * 64)
    run(noise_level=0.08, seed=42)
