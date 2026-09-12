"""M2.3: typed flags with fitted per-class persistence, the exploratory dial.

M2.2 with the hard gate replaced by two fitted tau-multipliers, one per
flag class, applied to the home KC's learn rate after that student's
first fire of that class:

    tau_k(t) = m_class * tau_k   after the first fire, else tau_k

with m_bias and m_skill each on a coarse grid in [0, 1]. The earlier
rungs are recovered as corners: both multipliers one is M2.1.1,
bias-zero-skill-one is M2.2, both-zero is the uniform freeze. The rung
is exploratory by registration: a fitted dial cannot be rejected and
therefore tests nothing; it never substitutes for the M2.2 contrast.
Read fitted multipliers near the legislated poles as confirmation from
inside a freer model, elsewhere as a finding.

Identifiability note, stated before fitting: the skill dial acts only
on kc4 after denominator fires, and kc4's fitted learn rate is ~0.011,
so the likelihood over m_skill is expected flat; report it as
unidentified at this n rather than as a value.

Same surface as the other models: plug into Evaluator, evaluate()
returns qc-level predictions.
"""
import os
import json

import numpy as np

from scripts.model_2_1_1 import (Model_2_1_1, KC_COLS, FLAGS, FLAG_HOME,
                                 CALIB, U1_PIN, KAPPA)
from scripts.model_2_2 import BIAS_FLAGS, SKILL_FLAGS

M_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


class Model_2_3(Model_2_1_1):
    """M2.2's chassis with the hard gate softened to two fitted
    per-class tau-multipliers."""

    def __init__(self, train_df, n_restarts=5, seed=42, chain_cache=None):
        super().__init__(train_df, n_restarts=n_restarts, seed=seed,
                         chain_cache=chain_cache)
        self.m_bias = 0.0
        self.m_skill = 1.0

    def _update_states(self, row, states):
        hit = states.setdefault("_hit", {"bias": set(), "skill": set()})
        for f in BIAS_FLAGS:
            if getattr(row, f) == "fired":
                hit["bias"].add(FLAG_HOME[f])
        for f in SKILL_FLAGS:
            if getattr(row, f) == "fired":
                hit["skill"].add(FLAG_HOME[f])
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
            mult = 1.0
            if k in hit["bias"]:
                mult = min(mult, self.m_bias)
            if k in hit["skill"]:
                mult = min(mult, self.m_skill)
            states[k] = post + (1 - post) * ch.T * mult

    def fit(self):
        self._fit_chains()
        self._fit_flag_tables()
        best = (np.inf, None, None, None)
        self.m_bias, self.m_skill = 0.0, 1.0
        for wv in self.GRID:
            self.shape = {"w_mix": wv}
            x, y = self._walk_xy(self.train_df)
            v, s0, g0 = self._fit_anchors(x, y)
            if v < best[0]:
                best = (v, s0, g0, wv)
        _, self.s0, self.g0, wv = best
        self.shape = {"w_mix": wv}
        best = (np.inf, None, None, None, None)
        for mb in M_GRID:
            for ms in M_GRID:
                self.m_bias, self.m_skill = mb, ms
                x, y = self._walk_xy(self.train_df)
                v, s0, g0 = self._fit_anchors(x, y)
                if v < best[0]:
                    best = (v, s0, g0, mb, ms)
        _, self.s0, self.g0, self.m_bias, self.m_skill = best
        self.shape = {"w_mix": wv, "m_bias": self.m_bias, "m_skill": self.m_skill}
        return self


def save_model_2_3_from_evaluator(ev, out_dir):
    """Dump a completed Evaluator run for the persistence-dial model.
    Writes per-fold jsons (bridge, shape with the fitted multipliers,
    u-tables), the pooled predictions, the metrics, and an index."""
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
                   u0={f: float(v) for f, v in m.u0.items()})
        fname = f"fold_{pid}.json"
        with open(os.path.join(cdir, fname), "w") as f:
            json.dump(rec, f, indent=1)
        folds.append(fname)
    ev.predictions.to_csv(os.path.join(cdir, "predictions.csv"), index=False)
    with open(os.path.join(cdir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in ev.metrics.items()}, f, indent=1)
    index = dict(model=cls.__name__, seed=ev.seed, u1_pin=U1_PIN, kappa=KAPPA,
                 calib=CALIB, homing=FLAG_HOME, bias_class=BIAS_FLAGS,
                 skill_class=SKILL_FLAGS, m_grid=M_GRID,
                 n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return cdir
