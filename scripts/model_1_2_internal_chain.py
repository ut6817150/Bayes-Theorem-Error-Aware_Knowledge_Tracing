"""Internal ablation of the M1.2 cell-chain emission: which noise channels
(guess, slip) should the per-KC chains carry, and at which bounds?

Nine arms. The four standard arms use the classic 0.3 clamps on every open
channel; the five extended arms free one or both channels to 0.85. Closed
channels are pinned at the floor (1e-4). Chains train and evaluate on cell
correctness (all realized cells); question correctness never enters. Same
class surface as Model_1_1: plug straight into Evaluator. evaluate() rows
carry a kc column for per-chain breakdowns.

Standard (clamped):
  Model_1_2_Internal_Both        g<=0.3, s<=0.3   (classic BKT emission)
  Model_1_2_Internal_GuessOnly   g<=0.3, s=0      (doctrine emission, clamped)
  Model_1_2_Internal_SlipOnly    g=0,    s<=0.3   (mirror emission)
  Model_1_2_Internal_Neither     g=0,    s=0      (deterministic emission)
Extended (freed to 0.85):
  Model_1_2_Internal_GuessOnlyFree   g<=0.85, s=0
  Model_1_2_Internal_SlipOnlyFree    g=0,     s<=0.85
  Model_1_2_Internal_BothGFree       g<=0.85, s<=0.3
  Model_1_2_Internal_BothSFree       g<=0.3,  s<=0.85
  Model_1_2_Internal_BothFree        g<=0.85, s<=0.85
"""

import numpy as np
import pandas as pd
import os
import json

KC_COLS = ["kc1_sample_space", "kc2_conditioning", "kc3_joint_chain",
           "kc4_total_probability", "kc5_bayes_update"]

FLOOR = 1e-4


class Chain:
    """Two-state chain with switchable emission channels."""

    def __init__(self, L0, T, g, s):
        self.L0, self.T, self.g, self.s = L0, T, g, s

    def predict(self, m):
        return m * (1 - self.s) + (1 - m) * self.g

    def update(self, m, label):
        p = self.predict(m)
        if label == 1:
            post = m * (1 - self.s) / p if p > 0 else m
        else:
            post = m * self.s / (1 - p) if p < 1 else m
        return post + (1 - post) * self.T


def _forward_backward(seq, L0, T, g, s):
    n = len(seq)
    e1 = lambda y: (1 - s) if y == 1 else s
    e0 = lambda y: g if y == 1 else (1 - g)
    a = np.zeros((n, 2)); c = np.zeros(n)
    a[0, 1] = L0 * e1(seq[0]); a[0, 0] = (1 - L0) * e0(seq[0])
    c[0] = max(a[0].sum(), 1e-12); a[0] /= c[0]
    for t in range(1, n):
        a[t, 1] = (a[t-1, 1] + a[t-1, 0] * T) * e1(seq[t])
        a[t, 0] = a[t-1, 0] * (1 - T) * e0(seq[t])
        c[t] = max(a[t].sum(), 1e-12); a[t] /= c[t]
    b = np.ones((n, 2))
    for t in range(n - 2, -1, -1):
        b[t, 1] = e1(seq[t+1]) * b[t+1, 1] / c[t+1]
        b[t, 0] = (T * e1(seq[t+1]) * b[t+1, 1] + (1 - T) * e0(seq[t+1]) * b[t+1, 0]) / c[t+1]
    gamma = a * b
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
    xi01 = np.zeros(max(n - 1, 0))
    for t in range(n - 1):
        xi01[t] = a[t, 0] * T * e1(seq[t+1]) * b[t+1, 1] / c[t+1]
    return gamma, xi01, np.log(c).sum()


def fit_chain(sequences, use_g, use_s, g_max=0.3, s_max=0.3,
              n_restarts=5, tol=1e-4, max_iter=200, seed=42):
    """EM over pooled cell sequences. Open channels re-estimated and clamped
    at their bound; closed channels pinned at the floor."""
    seqs = [x for x in sequences if len(x) > 0]
    rng = np.random.default_rng(seed)
    best_ll, best = -np.inf, None
    for _ in range(n_restarts):
        L0 = rng.uniform(0.2, 0.6); T = rng.uniform(0.05, 0.3)
        g = rng.uniform(0.05, 0.25) if use_g else FLOOR
        s = rng.uniform(0.05, 0.25) if use_s else FLOOR
        prev = -np.inf
        for _ in range(max_iter):
            G1 = Xi = Upre = G0a = G0c = Ma = Mw = LL = 0.0
            for seq in seqs:
                gamma, xi01, ll = _forward_backward(seq, L0, T, g, s)
                LL += ll
                G1 += gamma[0, 1]
                Xi += xi01.sum()
                Upre += gamma[:-1, 0].sum() if len(seq) > 1 else 0.0
                G0a += gamma[:, 0].sum()
                G0c += sum(gamma[t, 0] for t in range(len(seq)) if seq[t] == 1)
                Ma += gamma[:, 1].sum()
                Mw += sum(gamma[t, 1] for t in range(len(seq)) if seq[t] == 0)
            L0 = min(max(G1 / len(seqs), 1e-3), 0.999)
            T = min(max(Xi / max(Upre, 1e-9), 1e-4), 0.5)
            if use_g:
                g = min(max(G0c / max(G0a, 1e-9), 1e-4), g_max)
            if use_s:
                s = min(max(Mw / max(Ma, 1e-9), 1e-4), s_max)
            if abs(LL - prev) < tol:
                prev = LL
                break
            prev = LL
        if prev > best_ll:
            best_ll, best = prev, Chain(L0, T, g, s)
    return best


class Model_1_2_Internal_Base:
    """Cell-level chains; evaluate on every realized cell, predict-before-
    update, qc never consulted."""
    USE_G = True
    USE_S = True
    G_MAX = 0.3
    S_MAX = 0.3

    def __init__(self, train_df, n_restarts=5, seed=42):
        self.train_df = train_df
        self.n_restarts = n_restarts
        self.seed = seed
        self.chains = {}

    def fit(self):
        for kc in KC_COLS:
            seqs = []
            for pid, grp in self.train_df.groupby("participant_id"):
                grp = grp.sort_values("question_number")
                seq = [1 if getattr(r, kc) == "correct" else 0
                       for r in grp.itertuples()
                       if getattr(r, kc) in ("correct", "wrong")]
                seqs.append(seq)
            self.chains[kc] = fit_chain(seqs, self.USE_G, self.USE_S,
                                        g_max=self.G_MAX, s_max=self.S_MAX,
                                        n_restarts=self.n_restarts,
                                        seed=self.seed)
        return self

    def evaluate(self, heldout_df):
        """Walk the held-out participant cell by cell. Returns rows of
        participant, question, kc, p_pred, y_true."""
        out = []
        states = {kc: self.chains[kc].L0 for kc in KC_COLS}
        for r in heldout_df.sort_values("question_number").itertuples():
            for kc in KC_COLS:
                cell = getattr(r, kc)
                if cell in ("correct", "wrong"):
                    p = float(self.chains[kc].predict(states[kc]))
                    y = 1 if cell == "correct" else 0
                    out.append(dict(participant_id=r.participant_id,
                                    question_number=r.question_number,
                                    kc=kc, p_pred=p, y_true=y))
                    states[kc] = self.chains[kc].update(states[kc], y)
        return pd.DataFrame(out)


# Standard arms (classic clamps)

class Model_1_2_Internal_Slip_And_Guess(Model_1_2_Internal_Base):
    """Classic emission: guess and slip open, clamped 0.3/0.3."""
    USE_G, USE_S, G_MAX, S_MAX = True, True, 0.3, 0.3


class Model_1_2_Internal_GuessOnly(Model_1_2_Internal_Base):
    """Doctrine emission, clamped: g open (0.3), s pinned."""
    USE_G, USE_S, G_MAX, S_MAX = True, False, 0.3, 0.3


class Model_1_2_Internal_SlipOnly(Model_1_2_Internal_Base):
    """Mirror emission, clamped: s open (0.3), g pinned."""
    USE_G, USE_S, G_MAX, S_MAX = False, True, 0.3, 0.3


class Model_1_2_Internal_Neither(Model_1_2_Internal_Base):
    """Deterministic emission: both channels pinned."""
    USE_G, USE_S = False, False


# Extended arms (freed to 0.85)

class Model_1_2_Internal_GuessOnly_G_Free(Model_1_2_Internal_Base):
    """Doctrine emission, freed: g open to 0.85, s pinned."""
    USE_G, USE_S, G_MAX, S_MAX = True, False, 0.85, 0.3


class Model_1_2_Internal_SlipOnly_S_Free(Model_1_2_Internal_Base):
    """Mirror emission, freed: s open to 0.85, g pinned."""
    USE_G, USE_S, G_MAX, S_MAX = False, True, 0.3, 0.85


class Model_1_2_Internal_Slip_And_Guess_G_Free(Model_1_2_Internal_Base):
    """Both channels, g freed: g<=0.85, s<=0.3."""
    USE_G, USE_S, G_MAX, S_MAX = True, True, 0.85, 0.3


class Model_1_2_Internal_Slip_And_Guess_S_Free(Model_1_2_Internal_Base):
    """Both channels, s freed: g<=0.3, s<=0.85."""
    USE_G, USE_S, G_MAX, S_MAX = True, True, 0.3, 0.85


class Model_1_2_Internal_Slip_And_Guess_S_And_G_Free(Model_1_2_Internal_Base):
    """Both channels freed: g<=0.85, s<=0.85."""
    USE_G, USE_S, G_MAX, S_MAX = True, True, 0.85, 0.85


# Save internal chain helper

def save_internal_chain_from_evaluator(ev, out_dir):
    """Dump a completed Evaluator run for one internal-chain arm.

    Writes, under out_dir/<ModelClassName>/:
      fold_<pid>.json   one per fold: the five chains' L0/T/g/s
      predictions.csv   the pooled LOPO prediction rows
      metrics.json      the run's metric dict
      index.json        class name, seed, channel config, bounds, fold list

    Pure dump, no refitting: reads ev.fold_models, ev.predictions,
    ev.metrics. Returns the class directory path.
    """

    if not getattr(ev, "fold_models", None):
        raise ValueError("evaluator has no fold_models; call ev.run() first")

    cls = ev.model_class
    cdir = os.path.join(out_dir, cls.__name__)
    os.makedirs(cdir, exist_ok=True)

    folds = []
    for pid in sorted(ev.fold_models):
        m = ev.fold_models[pid]
        rec = dict(heldout=pid,
                   chains={kc: dict(L0=float(c.L0), T=float(c.T),
                                    g=float(c.g), s=float(c.s))
                           for kc, c in m.chains.items()})
        fname = f"fold_{pid}.json"
        with open(os.path.join(cdir, fname), "w") as f:
            json.dump(rec, f, indent=1)
        folds.append(fname)

    ev.predictions.to_csv(os.path.join(cdir, "predictions.csv"), index=False)

    with open(os.path.join(cdir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in ev.metrics.items()}, f, indent=1)

    index = dict(model=cls.__name__, seed=ev.seed,
                 use_g=cls.USE_G, use_s=cls.USE_S,
                 g_max=cls.G_MAX, s_max=cls.S_MAX, floor=FLOOR,
                 n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)

    return cdir