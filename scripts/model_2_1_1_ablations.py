"""M2.1.1 ablations: sibling emission forms on the shared chassis,
plus the classic chassis.

Model_2_1_1_Factorized: the conditional-independence form, the
registered original, retained as the predictive benchmark. Cell and
flags multiplied as independent witnesses given L,
P(o, F | L) = P(o | L) * prod_j P(F_j | L), u1 pinned at U1_PIN, u0
the responsibility-weighted marginal fire rate shrunk toward the
published written-format calibration centers (kappa = KAPPA), clipped
to [U0_LO, U0_HI]. Its small edge over the canonical rides the
bundle of differences, emission form, marginal against conditional
parameterization, prior center, and shrink weight; the double-count
(a fire structurally entails the home cell's failure, so the product
counts the same bad news twice) is one component whose isolated
share is not identified at this n.

Model_2_1_1_Factorized_No_Literature: the factorized form with the
calibration deleted, kappa zero, clip floor 0.01.

Model_2_1_1_Joint_No_Shrink: the canonical joint form with its
neutral-center shrink deleted, q0 kappa-free at the 0.01 floor. The
thin-conditional saturation arm; what it removes is stabilization,
not literature, none having entered the joint form.

Model_2_1_1_Classic_BKT: the flag mechanism on the user's Model_1_1
(qc labels only, pseudo-turn training, compensatory mean, the
per-cell annotation unused). The joint alphabet needs the home cell
and this chassis has none, so the integration is necessarily the
marginal-factor form; off-designed homes update flag-only.
"""
import numpy as np

from scripts.model_2_1_1 import (Model_2_1_1, Model_2_1_1_Joint,
                                 KC_COLS, FLAGS, FLAG_HOME, KC_FLAGS)
from scripts.model_1_1 import Model_1_1

# Published written-format calibration centers (error rates), per the
# report's cited figures (Diaz and Batanero 2009): used by the
# factorized benchmark and the classic arm only.
CALIB = {
    "conjunction": 0.79,
    "inverse": 0.65,
    "time_axis": 0.63,
    "denominator_neglect": 0.82,
    "base_rate_neglect": 0.67,
}

U1_PIN = 0.01      # factorized: mastered fire rate, pinned
KAPPA = 5.0        # factorized: pre-registered shrinkage pseudo-count
                   # toward the calibration anchor; retained after
                   # sensitivity sweeps showed a flat surface
                   # (kappa_sweep_m2_1_1.csv)
U0_LO, U0_HI = 0.1, 0.9


class Model_2_1_1_Factorized(Model_2_1_1):
    """The registered conditional-independence form, the predictive
    benchmark: marginal u tables, product likelihood."""

    U_KAPPA = KAPPA
    U_CLIP = (U0_LO, U0_HI)

    def __init__(self, train_df, n_restarts=5, seed=42, chain_cache=None):
        super().__init__(train_df, n_restarts=n_restarts, seed=seed,
                         chain_cache=chain_cache)
        self.u0 = {}

    def _fit_flag_tables(self):
        num = {f: 0.0 for f in FLAGS}
        den = {f: 0.0 for f in FLAGS}

        def count(r, states):
            for f in FLAGS:
                v = getattr(r, f)
                if v in ("fired", "quiet"):
                    w = 1.0 - states[FLAG_HOME[f]]
                    den[f] += w
                    if v == "fired":
                        num[f] += w

        self._responsibility_walk(count)
        self.u0 = {}
        for f in FLAGS:
            u = (num[f] + self.U_KAPPA * CALIB[f]) / (den[f] + self.U_KAPPA)
            self.u0[f] = float(np.clip(u, *self.U_CLIP))

    def _turn_likelihood(self, row, k, cell, has_cell, present):
        ch = self.chains[k]
        l1 = l0 = 1.0
        for f in present:
            u0 = self.u0.get(f, CALIB[f])
            if getattr(row, f) == "fired":
                l1, l0 = l1 * U1_PIN, l0 * u0
            else:
                l1, l0 = l1 * (1.0 - U1_PIN), l0 * (1.0 - u0)
        if has_cell:
            y = 1 if cell == "correct" else 0
            e1 = (1 - ch.s) if y == 1 else ch.s
            e0 = ch.g if y == 1 else (1 - ch.g)
            l1, l0 = l1 * e1, l0 * e0
        return l1, l0


class Model_2_1_1_Factorized_No_Literature(Model_2_1_1_Factorized):
    """The factorized form, calibration deleted: kappa zero, clip
    floor 0.01, rates from the data alone."""

    U_KAPPA = 0.0
    U_CLIP = (0.01, 0.99)

    def _fit_flag_tables(self):
        num = {f: 0.0 for f in FLAGS}
        den = {f: 0.0 for f in FLAGS}

        def count(r, states):
            for f in FLAGS:
                v = getattr(r, f)
                if v in ("fired", "quiet"):
                    w = 1.0 - states[FLAG_HOME[f]]
                    den[f] += w
                    if v == "fired":
                        num[f] += w

        self._responsibility_walk(count)
        self.u0 = {}
        for f in FLAGS:
            u = num[f] / den[f] if den[f] > 0 else 0.01
            self.u0[f] = float(np.clip(u, *self.U_CLIP))


class Model_2_1_1_Joint_No_Shrink(Model_2_1_1_Joint):
    """The canonical joint form with the neutral-center shrink
    deleted: q0 kappa-free at the 0.01 floor."""

    Q_KAPPA = 0.0


class Model_2_1_1_Classic_BKT(Model_1_1):
    """The flag mechanism on the user's Model_1_1, qc labels only, the
    per-cell annotation unused. Marginal-factor integration by
    necessity (no cells for the joint alphabet); u tables via qc-walk
    responsibilities, kappa = KAPPA toward CALIB; off-designed homes
    update flag-only."""

    U1 = U1_PIN
    KAPPA_C = KAPPA

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