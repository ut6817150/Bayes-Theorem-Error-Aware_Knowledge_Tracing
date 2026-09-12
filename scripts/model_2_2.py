"""M2.2: typed flags with the class-aware freeze, the tested secondary.

Mounted on the joint canonical chassis per the declared revision.

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

from scripts.model_2_1_1 import Model_2_1_1_Joint, FLAG_HOME

BIAS_FLAGS = ["conjunction", "inverse", "time_axis", "base_rate_neglect"]
SKILL_FLAGS = ["denominator_neglect"]


class Model_2_2(Model_2_1_1_Joint):
    """The class-aware freeze on the canonical chassis: bias-class
    fires zero the home KC's drift for the session (first fire,
    ratcheted); the skill-class flag never gates. Emission untouched."""

    def _before_update(self, row, states):
        frozen = states.setdefault("_frozen", set())
        for f in BIAS_FLAGS:
            if getattr(row, f) == "fired":
                frozen.add(FLAG_HOME[f])

    def _T(self, k, states):
        if k in states.get("_frozen", set()):
            return 0.0
        return self.chains[k].T


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
                   q0={f: float(v) for f, v in m.q0.items()})
        fname = f"fold_{pid}.json"
        with open(os.path.join(cdir, fname), "w") as f:
            json.dump(rec, f, indent=1)
        folds.append(fname)
    ev.predictions.to_csv(os.path.join(cdir, "predictions.csv"), index=False)
    with open(os.path.join(cdir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in ev.metrics.items()}, f, indent=1)
    index = dict(model=cls.__name__, seed=ev.seed, emission='joint',
                 homing=FLAG_HOME, bias_class=BIAS_FLAGS,
                 skill_class=SKILL_FLAGS, n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return cdir