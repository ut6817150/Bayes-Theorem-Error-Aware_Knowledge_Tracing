"""M2.2 ablations: three arms decomposing the class-aware freeze.

Arm 1, Model_2_2_Gate_Only: the M1.2 base with the freeze and nothing
else. No flag likelihood factors anywhere; fires are consumed only as
gate triggers, a bias-class fire zeroing the home KC's learn rate for
the session. Asks whether the dynamics value needs the evidence
channel at all (the third corner of the evidence-by-dynamics square).

Arm 2, Model_2_2_On_State_View: the gate mounted on the M2.1.2
chassis. The disposition machinery runs exactly as in M2.1.2, and
additionally a bias-class fire freezes the home KC's mastery drift
(the skill axis; the habit axis was frozen already). The fork's other
body gets the dynamics test.

Arm 3, Model_2_2_Unratcheted: M2.2 with forgiveness on counter-
evidence. A bias-class fire freezes the home KC as before, but a
later quiet from any bias-class flag homed there unfreezes it; a
later fire freezes it again. Tests whether the ratchet's permanence
earns its keep.

All arms share the mounted stack, the registered kappa, and the
evaluator surface.
"""
import os
import json

import numpy as np

from scripts.model_1_2_outer_chain import Model_1_2_MIX2
from scripts.model_2_1_1 import KC_COLS, FLAG_HOME
from scripts.model_2_2 import Model_2_2, BIAS_FLAGS
from scripts.model_2_1_2 import Model_2_1_2, KC_FLAGS, V0_PIN


class Model_2_2_Gate_Only(Model_1_2_MIX2):
    """M1.2 emissions, no flag factors; fires only trip the class-aware
    freeze on the home KC's learn rate."""

    def _update_states(self, row, states):
        frozen = states.setdefault("_frozen", set())
        for f in BIAS_FLAGS:
            if getattr(row, f) == "fired":
                frozen.add(FLAG_HOME[f])
        for k in KC_COLS:
            cell = getattr(row, k)
            if cell not in ("correct", "wrong"):
                continue
            ch = self.chains[k]
            y = 1 if cell == "correct" else 0
            e1 = (1 - ch.s) if y == 1 else ch.s
            e0 = ch.g if y == 1 else (1 - ch.g)
            m = states[k]
            num = m * e1
            den = num + (1 - m) * e0
            post = num / den if den > 0 else m
            T = 0.0 if k in frozen else ch.T
            states[k] = post + (1 - post) * T


class Model_2_2_On_State_View(Model_2_1_2):
    """The M2.1.2 chassis plus the class-aware freeze on the mastery
    drift. The habit axis was frozen already; a bias-class fire now
    freezes the skill axis of the home KC too. The parent drifts
    inline, so the update is overridden in full with the gated T."""

    def _update_states(self, row, states):
        frozen = states.setdefault("_frozen", set())
        for f in BIAS_FLAGS:
            if getattr(row, f) == "fired":
                frozen.add(FLAG_HOME[f])
        q = int(row.question_number)
        for kc in KC_COLS:
            cell = getattr(row, kc)
            has_cell = cell in ("correct", "wrong")
            fl1 = fl0 = 1.0
            for f in KC_FLAGS.get(kc, []):
                v = getattr(row, f)
                if v in ("fired", "quiet"):
                    v1 = self.v1.get(f, 0.5)
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
            T = 0.0 if kc in frozen else self.chains[kc].T
            drift = post[0] * T
            states[kc] = np.array([post[0] - drift, post[1] + drift])


class Model_2_2_Unratcheted(Model_2_2):
    """M2.2 with forgiveness on counter-evidence: a bias-class quiet on
    the frozen home KC unfreezes it; a later fire freezes it again."""

    def _update_states(self, row, states):
        frozen = states.setdefault("_frozen", set())
        for f in BIAS_FLAGS:
            v = getattr(row, f)
            if v == "quiet":
                frozen.discard(FLAG_HOME[f])
        for f in BIAS_FLAGS:
            if getattr(row, f) == "fired":
                frozen.add(FLAG_HOME[f])
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
            T = 0.0 if k in frozen else ch.T
            states[k] = post + (1 - post) * T


def save_ablation_from_evaluator(ev, out_dir):
    """Dump a completed Evaluator run for one ablation arm. Writes
    per-fold jsons (bridge, shape, and whatever fitted tables the arm
    carries), the pooled predictions, the metrics, and an index."""
    if not getattr(ev, "fold_models", None):
        raise ValueError("evaluator has no fold_models; call ev.run() first")
    cls = ev.model_class
    cdir = os.path.join(out_dir, cls.__name__)
    os.makedirs(cdir, exist_ok=True)
    folds = []
    for pid in sorted(ev.fold_models):
        m = ev.fold_models[pid]
        rec = dict(heldout=pid, s0=float(m.s0), g0=float(m.g0),
                   shape={k: float(v) for k, v in getattr(m, "shape", {}).items()})
        if getattr(m, "u0", None):
            rec["u0"] = {f: float(v) for f, v in m.u0.items()}
        if getattr(m, "v1", None):
            rec["v1"] = {f: float(v) for f, v in m.v1.items()}
            rec["b0"] = {k: float(v) for k, v in m.b0.items()}
        fname = f"fold_{pid}.json"
        with open(os.path.join(cdir, fname), "w") as f:
            json.dump(rec, f, indent=1)
        folds.append(fname)
    ev.predictions.to_csv(os.path.join(cdir, "predictions.csv"), index=False)
    with open(os.path.join(cdir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in ev.metrics.items()}, f, indent=1)
    index = dict(model=cls.__name__, seed=ev.seed, bias_class=BIAS_FLAGS,
                 n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return cdir
