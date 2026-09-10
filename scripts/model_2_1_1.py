"""M2.1.1: typed flags as evidence, observation view (Feng), static dynamics.

The mounted M1.2 stack unchanged (adopted Both chains, MIX2 routes, partition
link). One addition: on each turn, every flag the annotation presents (non-NA,
fired or quiet) contributes a typed likelihood factor to its home KC's chain
update. Fires are the loudest bad news (mastered fire rate pinned), quiets
positive evidence. NA flags contribute nothing; flag-free turns collapse to
M1.2 exactly.

Two-stage flag fitting, matching the outer chain's two-stage precedent:
stage one fits the correctness chains (cached or fresh); the flag tables are
then estimated from training walks, u0 = responsibility-weighted fire rate
among unmastered mass at the flag's presented turns, shrunk toward the
published written-format calibration (pseudo-count kappa), clipped to
[0.1, 0.9]. u1 is pinned, never fitted. Conjunction (zero fires) therefore
sits at its calibration anchor by construction.

Same surface as the other models: plug into Evaluator, evaluate() returns
qc-level predictions (the mounted secondary target).
"""
import numpy as np
import pandas as pd
import os
import json

from scripts.model_1_2_outer_chain import Model_1_2_MIX2, load_internal_chains

KC_COLS = ["kc1_sample_space", "kc2_conditioning", "kc3_joint_chain",
           "kc4_total_probability", "kc5_bayes_update"]

FLAGS = ["conjunction", "inverse", "time_axis",
         "denominator_neglect", "base_rate_neglect"]

# Mechanism-based homing per the report: conjunction KC1, inverse KC2,
# time-axis KC2, denominator neglect KC4, base-rate neglect KC5.
FLAG_HOME = {
    "conjunction": "kc1_sample_space",
    "inverse": "kc2_conditioning",
    "time_axis": "kc2_conditioning",
    "denominator_neglect": "kc4_total_probability",
    "base_rate_neglect": "kc5_bayes_update",
}

# Published written-format calibration anchors (approximate error rates).
CALIB = {
    "conjunction": 0.85,
    "inverse": 0.65,
    "time_axis": 0.63,
    "denominator_neglect": 0.70,
    "base_rate_neglect": 0.67,
}

U1_PIN = 0.01      # mastered fire rate, pinned, never fitted
KAPPA = 5.0        # shrinkage pseudo-count toward the calibration anchor
U0_LO, U0_HI = 0.1, 0.9


class Model_2_1_1(Model_1_2_MIX2):
    """MIX2 outer chain whose per-KC updates also consume typed flag
    evidence on the home KC (observation view, constant tau)."""

    def __init__(self, train_df, n_restarts=5, seed=42, chain_cache=None):
        super().__init__(train_df, n_restarts=n_restarts, seed=seed,
                         chain_cache=chain_cache)
        self.u0 = {}

    # flag likelihood factors for one row, grouped by home KC
    def _flag_factors(self, row):
        out = {}
        for f in FLAGS:
            v = getattr(row, f)
            if v in ("fired", "quiet"):
                kc = FLAG_HOME[f]
                u0 = self.u0.get(f, CALIB[f])
                if v == "fired":
                    l1, l0 = U1_PIN, u0
                else:
                    l1, l0 = 1.0 - U1_PIN, 1.0 - u0
                a, b = out.get(kc, (1.0, 1.0))
                out[kc] = (a * l1, b * l0)
        return out

    # the update: correctness and flag factors jointly, then drift
    def _update_states(self, row, states):
        ff = self._flag_factors(row)
        for k in KC_COLS:
            cell = getattr(row, k)
            ch = self.chains[k]
            has_cell = cell in ("correct", "wrong")
            l1, l0 = ff.get(k, (1.0, 1.0))
            if not has_cell and (l1, l0) == (1.0, 1.0):
                continue
            if has_cell:
                y = 1 if cell == "correct" else 0
                e1 = (1 - ch.s) if y == 1 else ch.s
                e0 = ch.g if y == 1 else (1 - ch.g)
                l1, l0 = l1 * e1, l0 * e0
            m = states[k]
            num = m * l1
            den = num + (1 - m) * l0
            post = num / den if den > 0 else m
            states[k] = post + (1 - post) * ch.T

    # stage 1.5: flag tables from training walks (correctness-only
    # responsibilities, so the estimate does not consume its own output)
    def _fit_flag_tables(self):
        num = {f: 0.0 for f in FLAGS}
        den = {f: 0.0 for f in FLAGS}
        for _, grp in self.train_df.groupby("participant_id"):
            states = {kc: self.chains[kc].L0 for kc in KC_COLS}
            for r in grp.sort_values("question_number").itertuples():
                for f in FLAGS:
                    v = getattr(r, f)
                    if v in ("fired", "quiet"):
                        w = 1.0 - states[FLAG_HOME[f]]
                        den[f] += w
                        if v == "fired":
                            num[f] += w
                # correctness-only update for the responsibility walk
                for k in KC_COLS:
                    cell = getattr(r, k)
                    if cell in ("correct", "wrong"):
                        states[k] = self.chains[k].update(
                            states[k], 1 if cell == "correct" else 0)
        for f in FLAGS:
            u = (num[f] + KAPPA * CALIB[f]) / (den[f] + KAPPA)
            self.u0[f] = float(np.clip(u, U0_LO, U0_HI))

    def fit(self):
        self._fit_chains()
        self._fit_flag_tables()
        best = (np.inf, None, None, None)
        for wv in self.GRID:
            self.shape = {"w_mix": wv}
            x, y = self._walk_xy(self.train_df)
            v, s0, g0 = self._fit_anchors(x, y)
            if v < best[0]:
                best = (v, s0, g0, wv)
        _, self.s0, self.g0, wv = best
        self.shape = {"w_mix": wv}
        return self

# Save Helper

def save_model_2_1_1_from_evaluator(ev, out_dir):
    """Dump a completed Evaluator run for the flag-observation model.

    Writes, under out_dir/<ModelClassName>/:
      fold_<pid>.json   one per fold: bridge (s0, g0), shape, fitted u0 table
      predictions.csv   the pooled LOPO prediction rows
      metrics.json      the run's metric dict
      index.json        class name, seed, pins and homing, fold list

    Pure dump, no refitting. Inner chains are not saved (they live in the
    internal-chain cache). Returns the class directory path.
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
                   s0=float(m.s0), g0=float(m.g0),
                   shape={k: (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in getattr(m, "shape", {}).items()},
                   u0={f: float(v) for f, v in m.u0.items()})
        fname = f"fold_{pid}.json"
        with open(os.path.join(cdir, fname), "w") as f:
            json.dump(rec, f, indent=1)
        folds.append(fname)

    ev.predictions.to_csv(os.path.join(cdir, "predictions.csv"), index=False)

    with open(os.path.join(cdir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in ev.metrics.items()}, f, indent=1)

    index = dict(model=cls.__name__, seed=ev.seed,
                 u1_pin=U1_PIN, kappa=KAPPA, calib=CALIB, homing=FLAG_HOME,
                 n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)

    return cdir