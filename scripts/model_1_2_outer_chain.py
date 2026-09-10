"""M1.2 outer chain: KC -> qc pooling over the adopted inner chains.

Stack (adopted): inner chains = classic Both emission (g, s clamped 0.3,
fit_chain from scripts.model_1_2_internal_chain), trained on all realized
cells; outer link = the partition form

    P(qc correct) = x * (1 - s0) + (1 - x) * g0

(DINA marginalized over the chain-intact event; endpoints match the
mechanism censuses, s0 0.077 and g0 0.061). x pools the DESIGNED-set
act-probabilities by each variant's method; MIX2 replaces the designed set
with its two-route table on Q1/Q4/Q6/Q7. Updates run on realized cells;
qc never trains chains. Anchors (and any shape parameter) are fitted in
stage two by qc likelihood on training walks.

Adopted variant: Model_1_2_MIX2. LOGIT and ISO are references.

With chain_cache, each fold's inner chains load from the saved jsons (the
fold whose participant is absent from train_df); without it, stage one
fits fresh.
"""
import os
import json
import numpy as np
import pandas as pd

from scripts.model_1_2_internal_chain import Chain, fit_chain

KC_COLS = ["kc1_sample_space", "kc2_conditioning", "kc3_joint_chain",
           "kc4_total_probability", "kc5_bayes_update"]


def load_internal_chains(folder):
    """Load saved internal-chain fold models from a class directory
    (fold_<pid>.json files written by save_internal_chain_from_evaluator).
    Returns {pid: {kc: Chain}}."""
    cache = {}
    for fname in sorted(os.listdir(folder)):
        if fname.startswith("fold_") and fname.endswith(".json"):
            rec = json.load(open(os.path.join(folder, fname)))
            cache[rec["heldout"]] = {kc: Chain(**p)
                                     for kc, p in rec["chains"].items()}
    return cache


class Model_1_2_Outer_Base:
    """Stage one: inner chains (cached or fresh). Stage two: anchor fit on
    training walks under the partition link. Held-out walk: predict-before-
    update, updates on realized cells."""

    def __init__(self, train_df, n_restarts=5, seed=42, chain_cache=None):
        self.train_df = train_df
        self.n_restarts = n_restarts
        self.seed = seed
        self.chain_cache = chain_cache
        self.chains = {}
        self.s0 = self.g0 = None
        self.shape = {}

    def _fit_chains(self):
        if self.chain_cache is not None:
            missing = set(self.chain_cache) - set(self.train_df.participant_id.unique())
            if len(missing) == 1:
                self.chains = self.chain_cache[missing.pop()]
                return
        for kc in KC_COLS:
            seqs = []
            for _, grp in self.train_df.groupby("participant_id"):
                grp = grp.sort_values("question_number")
                seqs.append([1 if getattr(r, kc) == "correct" else 0
                             for r in grp.itertuples()
                             if getattr(r, kc) in ("correct", "wrong")])
            self.chains[kc] = fit_chain(seqs, use_g=True, use_s=True,
                                        g_max=0.3, s_max=0.3,
                                        n_restarts=self.n_restarts,
                                        seed=self.seed)

    def _factors(self, row, states):
        ks = [k for k in row.designed_kcs if k in self.chains]
        return ks, np.array([self.chains[k].predict(states[k]) for k in ks])

    def _pooled(self, row, states):
        raise NotImplementedError

    def _update_states(self, row, states):
        for k in KC_COLS:
            cell = getattr(row, k)
            if cell in ("correct", "wrong"):
                states[k] = self.chains[k].update(states[k], 1 if cell == "correct" else 0)

    def _walk_xy(self, df):
        xs, ys = [], []
        for _, grp in df.groupby("participant_id"):
            states = {kc: self.chains[kc].L0 for kc in KC_COLS}
            for r in grp.sort_values("question_number").itertuples():
                if r.question_correct in ("correct", "wrong"):
                    xs.append(self._pooled(r, states))
                    ys.append(1 if r.question_correct == "correct" else 0)
                self._update_states(r, states)
        return np.array(xs), np.array(ys)

    @staticmethod
    def _nll(y, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    def _fit_anchors(self, x, y, grid=41, bound=0.3):
        """Partition link: P = x*(1-s0) + (1-x)*g0."""
        best = (np.inf, 0.05, 0.05)
        for s0 in np.linspace(1e-3, bound, grid):
            for g0 in np.linspace(1e-3, bound, grid):
                v = self._nll(y, x * (1 - s0) + (1 - x) * g0)
                if v < best[0]:
                    best = (v, s0, g0)
        return best

    def fit(self):
        self._fit_chains()
        x, y = self._walk_xy(self.train_df)
        _, self.s0, self.g0 = self._fit_anchors(x, y)
        return self

    def predict_row(self, row, states):
        x = self._pooled(row, states)
        return x * (1 - self.s0) + (1 - x) * self.g0

    def evaluate(self, heldout_df):
        out = []
        states = {kc: self.chains[kc].L0 for kc in KC_COLS}
        for r in heldout_df.sort_values("question_number").itertuples():
            if r.question_correct in ("correct", "wrong"):
                out.append(dict(participant_id=r.participant_id,
                                question_number=r.question_number,
                                p_pred=float(self.predict_row(r, states)),
                                y_true=1 if r.question_correct == "correct" else 0))
            self._update_states(r, states)
        return pd.DataFrame(out)


def _fit_with_shape_grid(model, pname, pvals):
    model._fit_chains()
    best = (np.inf, None, None, None)
    for pv in pvals:
        model.shape = {pname: pv}
        x, y = model._walk_xy(model.train_df)
        v, s0, g0 = model._fit_anchors(x, y)
        if v < best[0]:
            best = (v, s0, g0, pv)
    _, model.s0, model.g0, pv = best
    model.shape = {pname: pv}
    return model


# MODEL_1_2_AND
# The conjunctive product. Every designed KC is treated as a strict
# requirement: x is the product of the required act-probabilities, so one
# weak factor drags the whole prediction down and each additional factor
# can only shrink x (the length penalty).

class Model_1_2_AND(Model_1_2_Outer_Base):
    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        return float(np.prod(f)) if len(f) else 1.0


# MODEL_1_2_MEAN
# The compensatory average. x is the plain mean of the required
# act-probabilities: strengths substitute for weaknesses and no single
# broken concept can veto the answer. Kept as the no-veto contrast arm.

class Model_1_2_MEAN(Model_1_2_Outer_Base):
    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        return float(np.mean(f)) if len(f) else 1.0


# MODEL_1_2_MIN
# The weakest link. x is the smallest required act-probability: the answer
# is only as strong as the shakiest concept, the other factors contribute
# nothing, and risk does not compound with more requirements.

class Model_1_2_MIN(Model_1_2_Outer_Base):
    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        return float(np.min(f)) if len(f) else 1.0


# MODEL_1_2_MAX
# The disjunctive pole. x is the largest required act-probability: one
# strong concept carries the answer regardless of the rest. Kept as a
# bracket closer for the conjunctive-versus-disjunctive question.

class Model_1_2_MAX(Model_1_2_Outer_Base):
    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        return float(np.max(f)) if len(f) else 1.0


# MODEL_1_2_GENMEAN
# The generalized mean with a fitted exponent p: x is the mean of the
# factors each raised to the power p, then the p-th root. p is a dial
# across the whole symmetric family (p = 1 the mean, p = 0 the geometric
# mean, p -> -inf the min, p -> +inf the max); stage two grids p and lets
# the data locate itself on that axis.

class Model_1_2_GENMEAN(Model_1_2_Outer_Base):
    GRID = list(np.concatenate([np.linspace(-8, -1, 8), np.linspace(-0.5, 3, 8)]))

    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        if not len(f):
            return 1.0
        p = self.shape.get("p", 1.0)
        f = np.clip(f, 1e-6, 1)
        if abs(p) < 1e-3:
            return float(np.exp(np.mean(np.log(f))))
        return float(np.mean(f ** p) ** (1.0 / p))

    def fit(self):
        return _fit_with_shape_grid(self, "p", self.GRID)


# MODEL_1_2_TEMPAND
# The tempered product: x is the AND product raised to a fitted power
# theta. Theta below one softens the product's compounding, conceding that
# the factors are not fully independent while keeping the conjunctive
# ordering; stage two grids theta.

class Model_1_2_TEMPAND(Model_1_2_Outer_Base):
    GRID = list(np.linspace(0.2, 1.5, 14))

    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        return float(np.prod(f) ** self.shape.get("theta", 1.0)) if len(f) else 1.0

    def fit(self):
        return _fit_with_shape_grid(self, "theta", self.GRID)


# MODEL_1_2_LEAKY
# The noisy-AND with a uniform per-factor leak: each factor f becomes
# leak + (1 - leak) * f before multiplying, a bypass wire on every gate.
# With probability leak a requirement is circumvented regardless of the
# act; stage two grids the shared leak. The anonymous-uniform cousin of
# MIX2's structured routes.

class Model_1_2_LEAKY(Model_1_2_Outer_Base):
    GRID = list(np.linspace(0.0, 0.6, 13))

    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        if not len(f):
            return 1.0
        leak = self.shape.get("leak", 0.0)
        return float(np.prod(leak + (1 - leak) * f))

    def fit(self):
        return _fit_with_shape_grid(self, "leak", self.GRID)


# MODEL_1_2_DINO
# The disjunctive noisy-OR: x = 1 - prod(1 - f), the answer fails only if
# every required concept fails. Kept as the second bracket closer.

class Model_1_2_DINO(Model_1_2_Outer_Base):
    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        return float(1.0 - np.prod(1.0 - f)) if len(f) else 1.0


# MODEL_1_2_OWA
# The ordered weighted average: the factors are sorted ascending (weakest
# first) and combined with position weights from a small grid, so the model
# can attend mostly to the weakest one or two acts without ignoring the
# rest. Sits between MIN and MEAN.

class Model_1_2_OWA(Model_1_2_Outer_Base):
    GRID = [[4,1,1,1,1],[2,1,1,1,1],[1,1,1,1,1],[1,1,1,1,4],[3,2,1,1,1]]

    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        if not len(f):
            return 1.0
        w = self.shape.get("w")
        fs = np.sort(f)
        if w is None or len(w) < len(fs):
            return float(np.mean(fs))
        w = np.asarray(w[:len(fs)], dtype=float); w = w / w.sum()
        return float((w * fs).sum())

    def fit(self):
        return _fit_with_shape_grid(self, "w", self.GRID)


# MODEL_1_2_MIX2 (ADOPTED)
# The route mixture. On four items the review documented two genuinely
# different solution routes needing different KCs, so the model computes an
# AND product per route and blends them with a fitted weight w (the prior
# that a student takes the minimal route), marginalizing over the
# unobserved route choice. All other items are single-route AND over the
# designed set. Routes (route two = the realized rich path, which on
# Q1/Q6/Q7 coincides with the designed pair and on Q4 exceeds it):
#   Q1 kc1 | kc1+kc5;  Q4 kc2 | kc2+kc3+kc4+kc5;  Q6 kc2 | kc2+kc3;
#   Q7 kc4 | kc3+kc4.
# Q7's minimal entry rests on one witness (P12) - audit pending.

ROUTES = {
    1: [["kc1_sample_space"], ["kc1_sample_space", "kc5_bayes_update"]],
    4: [["kc2_conditioning"],
        ["kc2_conditioning", "kc3_joint_chain", "kc4_total_probability",
         "kc5_bayes_update"]],
    6: [["kc2_conditioning"], ["kc2_conditioning", "kc3_joint_chain"]],
    7: [["kc4_total_probability"], ["kc3_joint_chain", "kc4_total_probability"]],
}


class Model_1_2_MIX2(Model_1_2_Outer_Base):
    GRID = list(np.linspace(0.0, 1.0, 11))

    def _pooled(self, row, states):
        q = int(row.question_number)
        if q in ROUTES:
            w = self.shape.get("w_mix", 0.5)
            outs = []
            for route in ROUTES[q]:
                f = np.array([self.chains[k].predict(states[k])
                              for k in route if k in self.chains])
                outs.append(float(np.prod(f)) if len(f) else 1.0)
            return w * outs[0] + (1 - w) * outs[1]
        ks, f = self._factors(row, states)
        return float(np.prod(f)) if len(f) else 1.0

    def fit(self):
        return _fit_with_shape_grid(self, "w_mix", self.GRID)


# MODEL_1_2_LOGIT (REFERENCE)
# The learned bridge: a logistic regression over the chains' outputs
# instead of a mechanism. Features per question are the centered log
# act-probabilities of the designed KCs plus a per-KC absence indicator;
# one global weight vector is Newton-fitted with a light ridge on the
# training walks, and the sigmoid absorbs the anchors. Bounds what any
# pooling shape could achieve; not a candidate for adoption.

class Model_1_2_LOGIT(Model_1_2_Outer_Base):

    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        return float(np.prod(f)) if len(f) else 1.0

    def _feats(self, row, states):
        nk = len(KC_COLS)
        v = np.zeros(1 + 2 * nk)
        v[0] = 1.0
        for i, k in enumerate(KC_COLS):
            if k in row.designed_kcs:
                fkc = self.chains[k].predict(states[k])
                v[1 + i] = np.log(np.clip(fkc, 1e-6, 1)) + 0.25
            else:
                v[1 + nk + i] = 1.0
        return v

    def fit(self):
        self._fit_chains()
        X, Y = [], []
        for _, grp in self.train_df.groupby("participant_id"):
            states = {kc: self.chains[kc].L0 for kc in KC_COLS}
            for r in grp.sort_values("question_number").itertuples():
                if r.question_correct in ("correct", "wrong"):
                    X.append(self._feats(r, states))
                    Y.append(1 if r.question_correct == "correct" else 0)
                self._update_states(r, states)
        X = np.array(X); Y = np.array(Y).astype(float)
        d = X.shape[1]
        b = np.zeros(d); lam = 0.05
        prev = np.inf
        it = 0
        for it in range(100):
            z = X @ b
            p = 1 / (1 + np.exp(-z))
            nll = self._nll(Y, p) + 0.5 * lam * (b[1:] ** 2).sum()
            grad = X.T @ (p - Y) + lam * np.r_[0, b[1:]]
            H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(d)
            H[0, 0] -= lam
            b -= np.linalg.solve(H, grad)
            if abs(prev - nll) < 1e-8:
                break
            prev = nll
        assert it < 99, "LOGIT Newton failed to converge"
        self.b = b
        self.s0 = self.g0 = 0.0
        return self

    def predict_row(self, row, states):
        return float(1 / (1 + np.exp(-(self._feats(row, states) @ self.b))))


# MODEL_1_2_ISO (REFERENCE)
# The isotonic calibration reference: pooling is the AND product, but the
# two-parameter partition line is replaced by a monotone step function
# fitted on the training walks (pool-adjacent-violators), i.e. the best
# non-decreasing map from product to correctness rate. A monotone map
# cannot reorder rows, so it isolates calibration value from ranking value
# for the AND family.

class Model_1_2_ISO(Model_1_2_Outer_Base):
    def _pooled(self, row, states):
        ks, f = self._factors(row, states)
        return float(np.prod(f)) if len(f) else 1.0

    def fit(self):
        self._fit_chains()
        x, y = self._walk_xy(self.train_df)
        order = np.argsort(x)
        xs, ys = x[order], y[order].astype(float)
        v, w = [], []
        for j in range(len(ys)):
            v.append(ys[j]); w.append(1.0)
            while len(v) > 1 and v[-2] > v[-1]:
                nv = (v[-2] * w[-2] + v[-1] * w[-1]) / (w[-2] + w[-1])
                nw = w[-2] + w[-1]
                v = v[:-2] + [nv]; w = w[:-2] + [nw]
        fitted = []
        for nv, nw in zip(v, w):
            fitted += [nv] * int(round(nw))
        self._iso_x = xs
        self._iso_y = np.clip(np.array(fitted[:len(xs)]), 1e-3, 1 - 1e-3)
        self.s0 = self.g0 = 0.0
        return self

    def predict_row(self, row, states):
        x = self._pooled(row, states)
        i = min(max(np.searchsorted(self._iso_x, x), 0), len(self._iso_y) - 1)
        return float(self._iso_y[i])
    

# Save model and predictions helper

def save_outer_chain_from_evaluator(ev, out_dir):
    """Dump a completed Evaluator run for one outer-chain variant.

    Writes, under out_dir/<ModelClassName>/:
      fold_<pid>.json   one per fold: the bridge (s0, g0) and any shape params
      predictions.csv   the pooled LOPO prediction rows
      metrics.json      the run's metric dict
      index.json        class name, seed, fold list

    Pure dump, no refitting: reads ev.fold_models, ev.predictions,
    ev.metrics. Inner chains are not saved (they live in the internal-chain
    cache). Returns the class directory path.
    """
    import os
    import json

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
                          for k, v in getattr(m, "shape", {}).items()})
        fname = f"fold_{pid}.json"
        with open(os.path.join(cdir, fname), "w") as f:
            json.dump(rec, f, indent=1)
        folds.append(fname)

    ev.predictions.to_csv(os.path.join(cdir, "predictions.csv"), index=False)

    with open(os.path.join(cdir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in ev.metrics.items()}, f, indent=1)

    index = dict(model=cls.__name__, seed=ev.seed,
                 n_folds=len(folds), folds=folds)
    with open(os.path.join(cdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)

    return cdir