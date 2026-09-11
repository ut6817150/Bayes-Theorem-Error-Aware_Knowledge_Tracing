"""M2.2: typed flags with the class-aware freeze, the tested secondary.

M2.1.1 verbatim plus one rule on the transition. Bias-class fires
(conjunction, inverse, time-axis, base-rate neglect) freeze their home
KC's learn rate for the rest of the session, first fire, ratcheted,
never unfrozen. The skill-class fire (denominator neglect) leaves its
home KC's drift running. Evidence still updates belief through Bayes,
quiets and corrects still lift it; what dies post-fire is the free
upward drift, improvement the model was never shown.

Grounding: the CPR instruction-resistance table (biases flat under two
weeks of teaching, the denominator-hosting skill jumping 18 to 69) and
the impasse account of self-repair (skill gaps are felt and repaired,
biases are walked away from confidently). Zero new parameters: the
class assignment is legislated from the literature, the freeze is a
hard zero, so M2.2 minus M2.1.1 is a pure test of the persistence
claim.

Same surface as the other models: plug into Evaluator, evaluate()
returns qc-level predictions.
"""
import os
import json

from scripts.model_2_1_1 import (Model_2_1_1, KC_COLS, FLAGS, FLAG_HOME,
                                 CALIB, U1_PIN, KAPPA)

BIAS_FLAGS = ["conjunction", "inverse", "time_axis", "base_rate_neglect"]
SKILL_FLAGS = ["denominator_neglect"]


class Model_2_2(Model_2_1_1):
    """M2.1.1 plus the class-aware gate: a bias-class fire zeroes the
    home KC's learn rate for the rest of the session."""

    def _update_states(self, row, states):
        frozen = states.setdefault("_frozen", set())
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


def save_model_2_2_from_evaluator(ev, out_dir):
    """Dump a completed Evaluator run for the class-aware gate model.
    Writes per-fold jsons (bridge, shape, fitted u0 table), the pooled
    predictions, the metrics, and an index carrying the class table."""
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
                 skill_class=SKILL_FLAGS, n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return cdir

# Save Helper

def save_model_2_2_from_evaluator(ev, out_dir):
    """Dump a completed Evaluator run for the class-aware gate model.
    Writes per-fold jsons (bridge, shape, fitted u0 table), the pooled
    predictions, the metrics, and an index carrying the class table."""
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
                 skill_class=SKILL_FLAGS, n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return cdir
