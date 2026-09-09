"""Evaluator: leave-one-participant-out harness, model-agnostic."""

import numpy as np
import pandas as pd


def _auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((pos_i > neg).sum() + 0.5 * (pos_i == neg).sum() for pos_i in pos)
    return wins / (len(pos) * len(neg))


def _auprc_wrong(y, p):
    """Area under the precision-recall curve with WRONG (y = 0) as the
    positive class, scored by 1 - p. No-skill floor = wrong prevalence."""
    y, p = np.asarray(y), np.asarray(p)
    pos = (y == 0).astype(int)
    if pos.sum() == 0 or pos.sum() == len(y):
        return float("nan")
    score = 1 - p
    order = np.argsort(-score)
    pos = pos[order]
    tp = np.cumsum(pos)
    fp = np.cumsum(1 - pos)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / pos.sum()
    # step integration over recall (average precision)
    d_rec = np.diff(np.r_[0.0, rec])
    return float((prec * d_rec).sum())


def _metrics(y, p, thresh=0.5):
    y, p = np.asarray(y), np.asarray(p)
    yhat = (p >= thresh).astype(int)
    acc = (yhat == y).mean()
    tp = ((yhat == 1) & (y == 1)).sum()
    fp = ((yhat == 1) & (y == 0)).sum()
    fn = ((yhat == 0) & (y == 1)).sum()
    tn = ((yhat == 0) & (y == 0)).sum()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    rec_wrong = tn / (tn + fp) if tn + fp else float("nan")
    bal_acc = (rec + rec_wrong) / 2 if rec == rec else float("nan")
    eps = 1e-9
    ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    base = y.mean()
    ll_base = -np.mean(y * np.log(base + eps) + (1 - y) * np.log(1 - base + eps))
    return dict(auc=_auc(y, p), accuracy=acc, f1=f1, log_loss=ll,
                auprc_wrong=_auprc_wrong(y, p), bal_acc=bal_acc,
                base_rate=float(base), log_loss_base=float(ll_base),
                wrong_n=int((y == 0).sum()), n=len(y))


class Evaluator:
    """Takes a model class and the full 26-participant dataframe.
    Runs the 26-fold leave-one-participant-out loop, calling each model's
    own fit and evaluate, and pools the standardized prediction rows."""

    def __init__(self, model_class, df, model_kwargs=None, seed=0):
        self.model_class = model_class
        self.df = df
        self.model_kwargs = model_kwargs or {}
        self.seed = seed
        self.predictions = None
        self.metrics = None
        self.fold_models = {}

    def run(self):
        parts = sorted(self.df["participant_id"].unique())
        folds = []
        self.fold_models = {}
        for pid in parts:
            train = self.df[self.df.participant_id != pid]
            hold = self.df[self.df.participant_id == pid]
            model = self.model_class(train, seed=self.seed, **self.model_kwargs)
            model.fit()
            self.fold_models[pid] = model
            folds.append(model.evaluate(hold))
        self.predictions = pd.concat(folds, ignore_index=True)
        self.metrics = _metrics(self.predictions.y_true, self.predictions.p_pred)
        return self

    def bootstrap(self, n=2000, seed=0):
        """Participant-clustered bootstrap CIs for the metric set."""
        rng = np.random.default_rng(seed)
        parts = self.predictions.participant_id.unique()
        stats = []
        for _ in range(n):
            take = rng.choice(parts, size=len(parts), replace=True)
            sub = pd.concat([self.predictions[self.predictions.participant_id == p]
                             for p in take])
            stats.append(_metrics(sub.y_true, sub.p_pred))
        out = {}
        for k in ("auc", "accuracy", "f1", "log_loss"):
            vals = np.array([s_[k] for s_ in stats])
            out[k] = (float(np.nanpercentile(vals, 2.5)),
                      float(np.nanpercentile(vals, 97.5)))
        return out

    def run_record(self, data_path):
        return dict(model=self.model_class.__name__,
                    seed=self.seed, 
                    n_predictions=len(self.predictions),
                    metrics=self.metrics)
