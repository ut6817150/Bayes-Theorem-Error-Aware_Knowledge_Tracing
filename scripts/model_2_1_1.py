"""M2.1.1: typed flags as evidence on the mounted stack, Feng-faithful.

The emission follows Feng et al. (2019): misconceptions enter as
observations by enlarging the outcome alphabet. Per home KC per turn
the outcome is one symbol from a mutually exclusive set, correct,
wrong-with-flag-j-fired, wrong-all-quiet, with correct-and-fired a
structural zero (a fire is the home cell's failure signature). Writing
q_l = P(fired | wrong, L=l):

    P(correct | l)     = e_l(correct)
    P(wrong, f_j | l)  = e_l(wrong) * q_l_j
    P(wrong, none | l) = e_l(wrong) * prod_j (1 - q_l_j)

Cell and flag are one joint observation: the wrong is counted once,
inside e_l(wrong), and the flag adds which-kind-of-wrong. q_1 is
pinned at Q_PIN_1 (a mastered student's rare wrongs are ordinary
slips, not signature blunders); q_0 = P(fired | wrong, presented,
L=0) is fitted per flag from fire counts among presented wrongs,
each weighted by the wrong-conditioned responsibility
P(L=0 | history, wrong) = (1-b)(1-g) / ((1-b)(1-g) + b s), the
E-step weight matching the estimand's own conditioning (the flag
stays out of the weight; the walk stays causal), shrunk toward a
neutral center (Q_KAPPA pseudo-observations at Q_CENTER, the
published calibrations being marginals with no counterpart for this
conditional), clipped to [0.01, 0.99]. Flag-only
turns (cell NA) use the implied marginal r_l = q_l * e_l(wrong). On
flag-free turns the per-turn update is M1.2's exactly. A turn's flags
never enter that turn's forecast; they reach later forecasts only
through the state.

Structure: Model_2_1_1 is the shared emission-agnostic chassis (the
mounted M1.2 stack, the homing map, the responsibility walk, the
per-KC update skeleton with likelihood/bookkeeping/drift hooks, the
two-stage fit). Model_2_1_1_Joint, the canonical model, supplies the
alphabet likelihood and the q tables. Dynamics rungs and ablation
arms subclass one or the other and override hooks only.

Same surface as the other models: plug into Evaluator, evaluate()
returns qc-level predictions (the mounted secondary target).
"""
import os
import json

import numpy as np
import pandas as pd

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

KC_FLAGS = {}
for _f, _kc in FLAG_HOME.items():
    KC_FLAGS.setdefault(_kc, []).append(_f)


class Model_2_1_1(Model_1_2_MIX2):
    """Shared chassis for the typed-flag family: the mounted M1.2
    stack, the responsibility walk, the update skeleton, the two-stage
    fit. Emission-agnostic; subclasses supply _fit_flag_tables and
    _turn_likelihood, and may override _before_update and _T."""

    # ---- the responsibility walk: correctness-only updates, so table
    #      estimates never consume their own output. count(r, states)
    #      is called per row before that row's correctness updates.
    def _responsibility_walk(self, count):
        for _, grp in self.train_df.groupby("participant_id"):
            states = {kc: self.chains[kc].L0 for kc in KC_COLS}
            for r in grp.sort_values("question_number").itertuples():
                count(r, states)
                for k in KC_COLS:
                    cell = getattr(r, k)
                    if cell in ("correct", "wrong"):
                        states[k] = self.chains[k].update(
                            states[k], 1 if cell == "correct" else 0)

    # ---- hooks
    def _fit_flag_tables(self):
        raise NotImplementedError

    def _turn_likelihood(self, row, k, cell, has_cell, present):
        """Return (l1, l0), the turn's likelihood for KC k under
        mastered and unmastered."""
        raise NotImplementedError

    def _before_update(self, row, states):
        """Per-row bookkeeping before the KC loop (dynamics rungs
        record fires here)."""
        return None

    def _T(self, k, states):
        """The drift rate for KC k this turn (dynamics rungs gate it)."""
        return self.chains[k].T

    # ---- the update skeleton
    def _update_states(self, row, states):
        self._before_update(row, states)
        for k in KC_COLS:
            cell = getattr(row, k)
            has_cell = cell in ("correct", "wrong")
            present = [f for f in KC_FLAGS.get(k, [])
                       if getattr(row, f) in ("fired", "quiet")]
            if not has_cell and not present:
                continue
            l1, l0 = self._turn_likelihood(row, k, cell, has_cell, present)
            m = states[k]
            num = m * l1
            den = num + (1 - m) * l0
            post = num / den if den > 0 else m
            states[k] = post + (1 - post) * self._T(k, states)

    # ---- two-stage fit: chains, tables, then w_mix and anchors
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


class Model_2_1_1_Joint(Model_2_1_1):
    """The canonical model: the Feng-faithful joint alphabet on the
    shared chassis."""

    Q_PIN_1 = 0.01     # P(fired | wrong, mastered), pinned, never fitted
    Q_KAPPA = 1.0      # neutral-center shrink pseudo-count
    Q_CENTER = 0.5

    def __init__(self, train_df, n_restarts=5, seed=42, chain_cache=None):
        super().__init__(train_df, n_restarts=n_restarts, seed=seed,
                         chain_cache=chain_cache)
        self.q0 = {}

    def _fit_flag_tables(self):
        num = {f: 0.0 for f in FLAGS}
        den = {f: 0.0 for f in FLAGS}

        def count(r, states):
            for f in FLAGS:
                v = getattr(r, f)
                kc = FLAG_HOME[f]
                if v in ("fired", "quiet") and getattr(r, kc) == "wrong":
                    # wrong-conditioned responsibility, the E-step weight
                    # matching q0's own conditioning: P(L=0 | history, wrong),
                    # Bayes over the cell, the flag still unseen.
                    b = states[kc]
                    ch = self.chains[kc]
                    w = (1.0 - b) * (1.0 - ch.g)
                    w = w / (w + b * ch.s) if (w + b * ch.s) > 0 else 1.0 - b
                    den[f] += w
                    if v == "fired":
                        num[f] += w

        self._responsibility_walk(count)
        self.q0 = {}
        for f in FLAGS:
            q = (num[f] + self.Q_KAPPA * self.Q_CENTER) / (den[f] + self.Q_KAPPA)
            self.q0[f] = float(np.clip(q, 0.01, 0.99))

    def _turn_likelihood(self, row, k, cell, has_cell, present):
        ch = self.chains[k]
        e1w, e0w = ch.s, 1 - ch.g
        if has_cell and present:
            fired = [f for f in present if getattr(row, f) == "fired"]
            if cell == "correct":
                if fired:
                    raise ValueError(
                        f"structural zero violated: correct cell with fired "
                        f"{fired} on {k} (participant {row.participant_id}, "
                        f"Q{int(row.question_number)})")
                return 1 - ch.s, ch.g
            if fired:
                if len(fired) > 1:
                    raise NotImplementedError(
                        f"multiple same-KC fires {fired} on {k}: the alphabet "
                        f"has no such symbol in this data; extend it before use")
                f = fired[0]
                return e1w * self.Q_PIN_1, e0w * self.q0[f]
            p1 = np.prod([1 - self.Q_PIN_1 for _ in present])
            p0 = np.prod([1 - self.q0[f] for f in present])
            return e1w * p1, e0w * p0
        if has_cell:
            if cell == "correct":
                return 1 - ch.s, ch.g
            return ch.s, 1 - ch.g
        l1 = l0 = 1.0
        for f in present:
            r1 = self.Q_PIN_1 * e1w
            r0 = self.q0[f] * e0w
            if getattr(row, f) == "fired":
                l1, l0 = l1 * r1, l0 * r0
            else:
                l1, l0 = l1 * (1 - r1), l0 * (1 - r0)
        return l1, l0


def save_model_2_1_1_from_evaluator(ev, out_dir):
    """Dump a completed Evaluator run for the canonical model.

    Writes, under out_dir/<ModelClassName>/: per-fold jsons (bridge,
    shape, q-tables), the pooled predictions, the metrics, and an
    index. Pure dump, no refitting. Returns the class directory path.
    """
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
                   q0={f: float(v) for f, v in m.q0.items()})
        fname = f"fold_{pid}.json"
        with open(os.path.join(cdir, fname), "w") as f:
            json.dump(rec, f, indent=1)
        folds.append(fname)
    ev.predictions.to_csv(os.path.join(cdir, "predictions.csv"), index=False)
    with open(os.path.join(cdir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in ev.metrics.items()}, f, indent=1)
    index = dict(model=cls.__name__, seed=ev.seed, emission="joint",
                 q_pin_1=cls.Q_PIN_1, q_kappa=cls.Q_KAPPA,
                 q_center=cls.Q_CENTER, homing=FLAG_HOME,
                 n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return cdir