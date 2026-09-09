"""M1.1: classic BKT baseline. qc labels only, pseudo-turn expansion,
compensatory-mean recombination. The per-cell annotation is unused by design."""
import numpy as np
import pandas as pd

KC_COLS = ["kc1_sample_space", "kc2_conditioning", "kc3_joint_chain",
           "kc4_total_probability", "kc5_bayes_update"]


class BKTChain:
    """One KC chain: prior L0, learn T, guess g, slip s. No forgetting."""

    def __init__(self, L0, T, g, s):
        self.L0 = L0
        self.T = T
        self.g = g
        self.s = s

    def predict(self, m):
        return m * (1 - self.s) + (1 - m) * self.g

    def update(self, m, label):
        p = self.predict(m)
        if label == 1:
            post = m * (1 - self.s) / p if p > 0 else m
        else:
            post = m * self.s / (1 - p) if p < 1 else m
        return post + (1 - post) * self.T


def _forward_backward(seq, L0, T, g, s):
    n = len(seq)
    e1 = lambda y: (1 - s) if y == 1 else s
    e0 = lambda y: g if y == 1 else (1 - g)
    a = np.zeros((n, 2)); c = np.zeros(n)
    a[0, 1] = L0 * e1(seq[0]); a[0, 0] = (1 - L0) * e0(seq[0])
    c[0] = a[0].sum(); a[0] /= c[0]
    for t in range(1, n):
        a[t, 1] = (a[t-1, 1] + a[t-1, 0] * T) * e1(seq[t])
        a[t, 0] = a[t-1, 0] * (1 - T) * e0(seq[t])
        c[t] = a[t].sum(); a[t] /= c[t]
    b = np.ones((n, 2))
    for t in range(n - 2, -1, -1):
        b[t, 1] = e1(seq[t+1]) * b[t+1, 1] / c[t+1]
        b[t, 0] = (T * e1(seq[t+1]) * b[t+1, 1] + (1 - T) * e0(seq[t+1]) * b[t+1, 0]) / c[t+1]
    gamma = a * b
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi01 = np.zeros(max(n - 1, 0))
    for t in range(n - 1):
        xi01[t] = a[t, 0] * T * e1(seq[t+1]) * b[t+1, 1] / c[t+1]
    ll = np.log(c).sum()
    return gamma, xi01, ll


def fit_bkt(sequences, n_restarts=5, tol=1e-4, max_iter=200,
            g_max=0.3, s_max=0.3, seed=42):
    """EM over pooled sequences with restarts and guess/slip clamps."""
    seqs = [s_ for s_ in sequences if len(s_) > 0]
    rng = np.random.default_rng(seed)
    best_ll, best = -np.inf, None
    for r in range(n_restarts):
        L0 = rng.uniform(0.2, 0.6); T = rng.uniform(0.05, 0.3)
        g = rng.uniform(0.05, 0.25); s = rng.uniform(0.05, 0.25)
        prev_ll = -np.inf
        for it in range(max_iter):
            G1 = Xi = Upre = G0a = G0c = M_a = Mw = LL = 0.0
            for seq in seqs:
                gamma, xi01, ll = _forward_backward(seq, L0, T, g, s)
                LL += ll
                G1 += gamma[0, 1]
                Xi += xi01.sum()
                Upre += gamma[:-1, 0].sum() if len(seq) > 1 else 0.0
                G0a += gamma[:, 0].sum()
                G0c += sum(gamma[t, 0] for t in range(len(seq)) if seq[t] == 1)
                M_a += gamma[:, 1].sum()
                Mw += sum(gamma[t, 1] for t in range(len(seq)) if seq[t] == 0)
            L0 = min(max(G1 / len(seqs), 1e-3), 0.999)
            T = min(max(Xi / max(Upre, 1e-9), 1e-4), 0.5)
            g = min(max(G0c / max(G0a, 1e-9), 1e-4), g_max)
            s = min(max(Mw / max(M_a, 1e-9), 1e-4), s_max)
            if abs(LL - prev_ll) < tol:
                prev_ll = LL
                break
            prev_ll = LL
        if prev_ll > best_ll:
            best_ll, best = prev_ll, BKTChain(L0, T, g, s)
    return best


class Model_1_1:
    """Classic BKT: five chains trained on pseudo-turns carrying qc labels."""

    def __init__(self, train_df, n_restarts=5, seed=42):
        self.train_df = train_df
        self.n_restarts = n_restarts
        self.seed = seed
        self.chains = {}

    def fit(self):
        for kc in KC_COLS:
            seqs = []
            for pid, grp in self.train_df.groupby("participant_id"):
                grp = grp.sort_values("question_number")
                seq = [1 if r.question_correct == "correct" else 0
                       for r in grp.itertuples()
                       if kc in r.designed_kcs and r.question_correct in ("correct", "wrong")]
                seqs.append(seq)
            self.chains[kc] = fit_bkt(seqs, n_restarts=self.n_restarts, seed=self.seed)
        return self

    def evaluate(self, heldout_df):
        """Walk the held-out participant: predict-before-update per question.
        Returns rows of participant, question, p_pred, y_true."""
        out = []
        states = {kc: self.chains[kc].L0 for kc in KC_COLS}
        for r in heldout_df.sort_values("question_number").itertuples():
            if r.question_correct not in ("correct", "wrong"):
                continue
            kcs = [k for k in r.designed_kcs if k in self.chains]
            if not kcs:
                continue
            p = float(np.mean([self.chains[k].predict(states[k]) for k in kcs]))
            y = 1 if r.question_correct == "correct" else 0
            out.append(dict(participant_id=r.participant_id,
                            question_number=r.question_number,
                            p_pred=p, y_true=y))
            for k in kcs:
                states[k] = self.chains[k].update(states[k], y)
        return pd.DataFrame(out)
