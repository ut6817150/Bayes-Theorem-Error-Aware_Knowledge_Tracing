"""M2.1.2: typed flags as evidence, state view (Liu), static dynamics.

The mounted M1.2 stack (adopted Both chains, MIX2 routes, partition link)
with a parallel disposition node B beside mastery L on every flag-hosting
KC. The belief per KC is the four-cell joint pi(l, d). Flags read B only
(fired testifies the habit is present); correctness reads both, with a
fitted trap penalty delta on the mastered-but-trapped cell that applies
only when the turn triggers the habit (the DESIGNED trap schedule, per the
trigger-map ruling; flag observations themselves follow the realized-
applicability ruling and are consumed wherever the annotation presents
them, off-schedule included). KC2 carries one shared B for its two faces
(inverse, time-axis), each face keeping its own fitted detector pair.

Estimation is two-stage like M2.1.1: chains cached or fresh; disposition
priors and detector rates from training rows with pseudo-count shrinkage;
then w_mix and delta on sequential stage-two grids with the anchors.

Same surface as the other models: plug into Evaluator, evaluate() returns
qc-level predictions.
"""
import os
import json
import numpy as np
import pandas as pd

from scripts.model_1_2_outer_chain import Model_1_2_MIX2, ROUTES, load_internal_chains

KC_COLS = ["kc1_sample_space", "kc2_conditioning", "kc3_joint_chain",
           "kc4_total_probability", "kc5_bayes_update"]

FLAGS = ["conjunction", "inverse", "time_axis",
         "denominator_neglect", "base_rate_neglect"]

FLAG_HOME = {
    "conjunction": "kc1_sample_space",
    "inverse": "kc2_conditioning",
    "time_axis": "kc2_conditioning",
    "denominator_neglect": "kc4_total_probability",
    "base_rate_neglect": "kc5_bayes_update",
}

# Designed trap schedule (the instrument's matrix): which questions trigger
# which habit. Off-schedule flag observations still update B; the schedule
# only gates the delta penalty in the correctness emission.
SCHEDULE = {
    "conjunction": {2},
    "inverse": {3, 10, 12},
    "time_axis": {4, 6},
    "denominator_neglect": {7, 8, 10, 12},
    "base_rate_neglect": {1, 8, 10, 11, 12},
}

V0_PIN = 0.01      # clean-disposition fire rate, pinned
KAPPA_V = 2.0      # shrinkage pseudo-count for the detector rates
V1_CENTER = 0.5    # shrinkage center for P(fired | trapped)
KAPPA_B = 1.0      # Laplace-style shrinkage for the disposition priors
DELTA_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

KC_FLAGS = {}
for f, kc in FLAG_HOME.items():
    KC_FLAGS.setdefault(kc, []).append(f)


def kc_triggered(kc, q):
    return any(q in SCHEDULE[f] for f in KC_FLAGS.get(kc, []))


class Model_2_1_2(Model_1_2_MIX2):
    """MIX2 outer chain over four-cell (mastery, disposition) beliefs.
    Flags heat B; a fitted trap penalty converts a hot B into lowered
    correctness on triggered turns only."""

    def __init__(self, train_df, n_restarts=5, seed=42, chain_cache=None):
        super().__init__(train_df, n_restarts=n_restarts, seed=seed,
                         chain_cache=chain_cache)
        self.v1 = {}
        self.b0 = {}
        self.delta = 0.2

    def _init_states(self):
        states = {}
        for kc in KC_COLS:
            L0 = self.chains[kc].L0
            B0 = self.b0.get(kc, 0.0)
            states[kc] = np.array([[(1-L0)*(1-B0), (1-L0)*B0],
                                   [L0*(1-B0),     L0*B0]])
        return states

    def _act(self, kc, pi, q):
        ch = self.chains[kc]
        trig = kc_triggered(kc, q)
        e = np.array([[ch.g, ch.g],
                      [1-ch.s, max(1-ch.s-(self.delta if trig else 0.0), ch.g+0.01)]])
        return float((pi * e).sum()), e

    def _pooled(self, row, states):
        q = int(row.question_number)
        def act(k):
            return self._act(k, states[k], q)[0]
        if q in ROUTES:
            w = self.shape.get("w_mix", 0.5)
            outs = []
            for route in ROUTES[q]:
                f = np.array([act(k) for k in route if k in self.chains])
                outs.append(float(np.prod(f)) if len(f) else 1.0)
            return w * outs[0] + (1 - w) * outs[1]
        ks = [k for k in row.designed_kcs if k in self.chains]
        f = np.array([act(k) for k in ks])
        return float(np.prod(f)) if len(f) else 1.0

    def _update_states(self, row, states):
        q = int(row.question_number)
        for kc in KC_COLS:
            cell = getattr(row, kc)
            has_cell = cell in ("correct", "wrong")
            fl1 = fl0 = 1.0
            for f in KC_FLAGS.get(kc, []):
                v = getattr(row, f)
                if v in ("fired", "quiet"):
                    v1 = self.v1.get(f, V1_CENTER)
                    if v == "fired":
                        fl1, fl0 = fl1 * v1, fl0 * V0_PIN
                    else:
                        fl1, fl0 = fl1 * (1 - v1), fl0 * (1 - V0_PIN)
            if not has_cell and fl1 == 1.0 and fl0 == 1.0:
                continue
            pi = states[kc]
            like = np.ones((2, 2))
            like[:, 1] *= fl1
            like[:, 0] *= fl0
            if has_cell:
                _, e = self._act(kc, pi, q)
                eo = e if cell == "correct" else 1 - e
                like = like * eo
            post = pi * like
            tot = post.sum()
            if tot > 0:
                post = post / tot
            T = self.chains[kc].T
            drift = post[0] * T
            post = np.array([post[0] - drift, post[1] + drift])
            states[kc] = post

    def _walk_xy(self, df):
        xs, ys = [], []
        for _, grp in df.groupby("participant_id"):
            states = self._init_states()
            for r in grp.sort_values("question_number").itertuples():
                if r.question_correct in ("correct", "wrong"):
                    xs.append(self._pooled(r, states))
                    ys.append(1 if r.question_correct == "correct" else 0)
                self._update_states(r, states)
        return np.array(xs), np.array(ys)

    def evaluate(self, heldout_df):
        out = []
        states = self._init_states()
        for r in heldout_df.sort_values("question_number").itertuples():
            if r.question_correct in ("correct", "wrong"):
                out.append(dict(participant_id=r.participant_id,
                                question_number=r.question_number,
                                p_pred=float(self.predict_row(r, states)),
                                y_true=1 if r.question_correct == "correct" else 0))
            self._update_states(r, states)
        return pd.DataFrame(out)

    def _fit_flag_tables(self):
        by_flag_fires = {f: 0.0 for f in FLAGS}
        by_flag_pres = {f: 0.0 for f in FLAGS}
        firers = {f: set() for f in FLAGS}
        for r in self.train_df.itertuples():
            for f in FLAGS:
                v = getattr(r, f)
                if v == "fired":
                    firers[f].add(r.participant_id)
        for r in self.train_df.itertuples():
            for f in FLAGS:
                v = getattr(r, f)
                if v in ("fired", "quiet") and r.participant_id in firers[f]:
                    by_flag_pres[f] += 1
                    if v == "fired":
                        by_flag_fires[f] += 1
        for f in FLAGS:
            v = (by_flag_fires[f] + KAPPA_V * V1_CENTER) / (by_flag_pres[f] + KAPPA_V)
            self.v1[f] = float(np.clip(v, 0.05, 0.95))
        n_part = self.train_df.participant_id.nunique()
        for kc, fs in KC_FLAGS.items():
            kc_firers = set().union(*[firers[f] for f in fs])
            self.b0[kc] = float((len(kc_firers) + KAPPA_B) / (n_part + 2 * KAPPA_B))

    def fit(self):
        self._fit_chains()
        self._fit_flag_tables()
        self.delta = 0.2
        best = (np.inf, None, None, None)
        for wv in self.GRID:
            self.shape = {"w_mix": wv}
            x, y = self._walk_xy(self.train_df)
            v, s0, g0 = self._fit_anchors(x, y)
            if v < best[0]:
                best = (v, s0, g0, wv)
        _, self.s0, self.g0, wv = best
        self.shape = {"w_mix": wv}
        best = (np.inf, None, None, None)
        for dv in DELTA_GRID:
            self.delta = dv
            x, y = self._walk_xy(self.train_df)
            v, s0, g0 = self._fit_anchors(x, y)
            if v < best[0]:
                best = (v, s0, g0, dv)
        _, self.s0, self.g0, dv = best
        self.delta = dv
        self.shape = {"w_mix": wv, "delta": dv}
        return self


def save_model_2_1_2_from_evaluator(ev, out_dir):
    """Dump a completed Evaluator run for the disposition-state model.
    Writes per-fold jsons (bridge, shape, detector rates, disposition
    priors), the pooled predictions, the metrics, and an index."""
    if not getattr(ev, "fold_models", None):
        raise ValueError("evaluator has no fold_models; call ev.run() first")
    cls = ev.model_class
    cdir = os.path.join(out_dir, cls.__name__)
    os.makedirs(cdir, exist_ok=True)
    folds = []
    for pid in sorted(ev.fold_models):
        m = ev.fold_models[pid]
        rec = dict(heldout=pid, s0=float(m.s0), g0=float(m.g0),
                   shape={k: float(v) for k, v in m.shape.items()},
                   v1={f: float(v) for f, v in m.v1.items()},
                   b0={k: float(v) for k, v in m.b0.items()})
        fname = f"fold_{pid}.json"
        with open(os.path.join(cdir, fname), "w") as f:
            json.dump(rec, f, indent=1)
        folds.append(fname)
    ev.predictions.to_csv(os.path.join(cdir, "predictions.csv"), index=False)
    with open(os.path.join(cdir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in ev.metrics.items()}, f, indent=1)
    index = dict(model=cls.__name__, seed=ev.seed, v0_pin=V0_PIN,
                 kappa_v=KAPPA_V, v1_center=V1_CENTER, kappa_b=KAPPA_B,
                 schedule={f: sorted(q) for f, q in SCHEDULE.items()},
                 homing=FLAG_HOME, n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return cdir
