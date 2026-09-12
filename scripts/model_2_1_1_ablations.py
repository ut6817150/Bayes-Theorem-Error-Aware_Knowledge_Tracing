"""M2.1.1 ablations: the emission-form-by-priors square.

Four cells, the registered M2.1.1 (factorized emission, calibrated
shrinkage) serving as the reference corner, plus three arms:

Model_2_1_1_Joint: the faithful-Feng emission. Per home KC per turn
the outcome is partitioned into mutually exclusive symbols, correct,
wrong-with-flag-j fired, wrong-all-quiet, with the correct-and-fired
cell a structural zero, so cell and flag are modeled jointly and the
conditional-independence assumption is removed. q0 is P(fired given
wrong, unmastered), fitted with a light neutral-center shrink; the
mastered counterpart is pinned at 0.01. Flag-only turns use the
implied marginal fire probability.

Model_2_1_1_No_Literature: the factorized form with the calibration
centers deleted. Flag rates fitted from the data alone, kappa zero,
clip floor 0.01, so no prior knowledge enters through the bounds
either. Channels with no training fires collapse toward the floor.

Model_2_1_1_Joint_No_Literature: both deletions at once, the joint
form with every prior removed, u-table and q0 both kappa-free at the
0.01 floor. The harshest cell: the primary contrast's survival here
is the strongest robustness statement, and the thin-conditional
saturation it exposes (q0 near one on small wrong-denominators) is
why the faithful form needs its shrink more than the factorized form
needs its calibration.
"""
import os
import json

import numpy as np

from scripts.model_2_1_1 import (Model_2_1_1, KC_COLS, FLAGS, FLAG_HOME,
                                 CALIB, U1_PIN)
from scripts.model_1_1 import Model_1_1

KC_FLAGS = {}
for f, kc in FLAG_HOME.items():
    KC_FLAGS.setdefault(kc, []).append(f)


class Model_2_1_1_Joint(Model_2_1_1):
    """Faithful-Feng joint emission: per home KC per turn, mutually
    exclusive symbols with the correct-and-fired cell a structural
    zero. q0 is P(fired_j given wrong, unmastered), fitted; the
    mastered counterpart is pinned at 0.01."""

    Q_PIN_1 = 0.01
    Q_KAPPA = 1.0
    Q_CENTER = 0.5

    def _fit_flag_tables(self):
        super()._fit_flag_tables()
        num = {f: 0.0 for f in FLAGS}
        den = {f: 0.0 for f in FLAGS}
        for _, grp in self.train_df.groupby("participant_id"):
            states = {kc: self.chains[kc].L0 for kc in KC_COLS}
            for r in grp.sort_values("question_number").itertuples():
                for f in FLAGS:
                    v = getattr(r, f)
                    kc = FLAG_HOME[f]
                    cell = getattr(r, kc)
                    if v in ("fired", "quiet") and cell == "wrong":
                        w = 1.0 - states[kc]
                        den[f] += w
                        if v == "fired":
                            num[f] += w
                for k in KC_COLS:
                    cell = getattr(r, k)
                    if cell in ("correct", "wrong"):
                        states[k] = self.chains[k].update(
                            states[k], 1 if cell == "correct" else 0)
        self.q0 = {}
        for f in FLAGS:
            q = (num[f] + self.Q_KAPPA * self.Q_CENTER) / (den[f] + self.Q_KAPPA)
            self.q0[f] = float(np.clip(q, 0.01, 0.99))

    def _update_states(self, row, states):
        for k in KC_COLS:
            cell = getattr(row, k)
            has_cell = cell in ("correct", "wrong")
            present = [f for f in KC_FLAGS.get(k, [])
                       if getattr(row, f) in ("fired", "quiet")]
            if not has_cell and not present:
                continue
            ch = self.chains[k]
            e1w, e0w = ch.s, 1 - ch.g
            if has_cell and present:
                fired = [f for f in present if getattr(row, f) == "fired"]
                if cell == "correct":
                    l1, l0 = 1 - ch.s, ch.g
                elif fired:
                    f = fired[0]
                    l1 = e1w * self.Q_PIN_1
                    l0 = e0w * self.q0[f]
                else:
                    p1 = np.prod([1 - self.Q_PIN_1 for _ in present])
                    p0 = np.prod([1 - self.q0[f] for f in present])
                    l1, l0 = e1w * p1, e0w * p0
            elif has_cell:
                y = 1 if cell == "correct" else 0
                l1 = (1 - ch.s) if y == 1 else ch.s
                l0 = ch.g if y == 1 else (1 - ch.g)
            else:
                l1 = l0 = 1.0
                for f in present:
                    r1 = self.Q_PIN_1 * e1w
                    r0 = self.q0[f] * e0w
                    if getattr(row, f) == "fired":
                        l1, l0 = l1 * r1, l0 * r0
                    else:
                        l1, l0 = l1 * (1 - r1), l0 * (1 - r0)
            m = states[k]
            numr = m * l1
            denr = numr + (1 - m) * l0
            post = numr / denr if denr > 0 else m
            states[k] = post + (1 - post) * ch.T


class Model_2_1_1_No_Literature(Model_2_1_1):
    """Calibration centers deleted: kappa zero, clip floor 0.01, the
    flag rates fitted from the data alone."""

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
                for k in KC_COLS:
                    cell = getattr(r, k)
                    if cell in ("correct", "wrong"):
                        states[k] = self.chains[k].update(
                            states[k], 1 if cell == "correct" else 0)
        for f in FLAGS:
            u = num[f] / den[f] if den[f] > 0 else 0.01
            self.u0[f] = float(np.clip(u, 0.01, 0.99))


class Model_2_1_1_Joint_No_Literature(Model_2_1_1_Joint):
    """The joint form with every prior deleted: u-table kappa-free at
    the 0.01 floor, q0 kappa-free at the 0.01 floor."""

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
                for k in KC_COLS:
                    cell = getattr(r, k)
                    if cell in ("correct", "wrong"):
                        states[k] = self.chains[k].update(
                            states[k], 1 if cell == "correct" else 0)
        for f in FLAGS:
            u = num[f] / den[f] if den[f] > 0 else 0.01
            self.u0[f] = float(np.clip(u, 0.01, 0.99))
        num = {f: 0.0 for f in FLAGS}
        den = {f: 0.0 for f in FLAGS}
        for _, grp in self.train_df.groupby("participant_id"):
            states = {kc: self.chains[kc].L0 for kc in KC_COLS}
            for r in grp.sort_values("question_number").itertuples():
                for f in FLAGS:
                    v = getattr(r, f)
                    kc = FLAG_HOME[f]
                    cell = getattr(r, kc)
                    if v in ("fired", "quiet") and cell == "wrong":
                        w = 1.0 - states[kc]
                        den[f] += w
                        if v == "fired":
                            num[f] += w
                for k in KC_COLS:
                    cell = getattr(r, k)
                    if cell in ("correct", "wrong"):
                        states[k] = self.chains[k].update(
                            states[k], 1 if cell == "correct" else 0)
        self.q0 = {}
        for f in FLAGS:
            q = num[f] / den[f] if den[f] > 0 else 0.01
            self.q0[f] = float(np.clip(q, 0.01, 0.99))


class Model_2_1_1_Classic_BKT(Model_1_1):
    """M2.1.1's flag mechanism integrated into the user's Model_1_1,
    classic BKT: qc labels only, pseudo-turn training, compensatory-
    mean recombination, the per-cell annotation unused. The flags are
    the sole cell-layer information admitted: each presented flag
    multiplies its typed factor (registered kappa-5 tables, u1 pinned)
    into its home chain's Bayes update during the walk; prediction is
    untouched. A flag whose home KC is outside the question's designed
    set updates that chain alone (flag-only Bayes plus drift), per the
    realized-applicability ruling. Against Model_1_1 this asks whether
    the typed channel adds signal to the classic chassis that never
    sees cells."""

    U1 = U1_PIN
    KAPPA_C = 5.0

    def fit(self):
        super().fit()
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
                if r.question_correct in ("correct", "wrong"):
                    y = 1 if r.question_correct == "correct" else 0
                    for k in r.designed_kcs:
                        if k in self.chains:
                            states[k] = self.chains[k].update(states[k], y)
        self.u0 = {}
        for f in FLAGS:
            u = (num[f] + self.KAPPA_C * CALIB[f]) / (den[f] + self.KAPPA_C)
            self.u0[f] = float(np.clip(u, 0.1, 0.9))
        return self

    def evaluate(self, heldout_df):
        import pandas as pd
        out = []
        states = {kc: self.chains[kc].L0 for kc in KC_COLS}
        for r in heldout_df.sort_values("question_number").itertuples():
            ff = {}
            for f in FLAGS:
                v = getattr(r, f)
                if v in ("fired", "quiet"):
                    kc = FLAG_HOME[f]
                    u0 = self.u0.get(f, CALIB[f])
                    l1, l0 = (self.U1, u0) if v == "fired" else (1 - self.U1, 1 - u0)
                    a, b = ff.get(kc, (1.0, 1.0))
                    ff[kc] = (a * l1, b * l0)
            has_q = r.question_correct in ("correct", "wrong")
            kcs = [k for k in r.designed_kcs if k in self.chains] if has_q else []
            if has_q and kcs:
                y = 1 if r.question_correct == "correct" else 0
                p = float(np.mean([self.chains[k].predict(states[k]) for k in kcs]))
                out.append(dict(participant_id=r.participant_id,
                                question_number=r.question_number,
                                p_pred=p, y_true=y))
                for k in kcs:
                    ch = self.chains[k]
                    l1, l0 = ff.pop(k, (1.0, 1.0))
                    e1 = (1 - ch.s) if y == 1 else ch.s
                    e0 = ch.g if y == 1 else (1 - ch.g)
                    m = states[k]
                    numr = m * l1 * e1
                    denr = numr + (1 - m) * l0 * e0
                    post = numr / denr if denr > 0 else m
                    states[k] = post + (1 - post) * ch.T
            for k, (l1, l0) in ff.items():
                ch = self.chains[k]
                m = states[k]
                numr = m * l1
                denr = numr + (1 - m) * l0
                post = numr / denr if denr > 0 else m
                states[k] = post + (1 - post) * ch.T
        return pd.DataFrame(out)