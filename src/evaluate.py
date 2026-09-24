"""
Evaluation metrics for the inspection-prioritisation problem.

Main metric: recall at the top 20% WITHIN each borough, averaged across boroughs.
"If each council inspects the riskiest 20% of its own list, what share of its
failing businesses does it find?"

All "within" metrics are computed per borough and then averaged, so a model gets
no credit for knowing that one council rates more strictly than another.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import GROUP, TARGET


def _by_borough(data, score, seed=0):
    """Yield one DataFrame (target, score) per borough, rows shuffled so ties break randomly."""
    frame = pd.DataFrame({
        "group": data[GROUP].to_numpy(),
        "target": data[TARGET].to_numpy(),
        "score": np.asarray(score, dtype=float),
    }).sample(frac=1, random_state=seed)
    for _, borough in frame.groupby("group"):
        if borough["target"].sum() > 0:
            yield borough


def recall_at_top(data, score, top=0.2):
    """Average (over boroughs) share of failures found in each borough's top `top` fraction."""
    recalls = []
    for borough in _by_borough(data, score):
        k = int(np.ceil(top * len(borough)))
        top_k = borough.sort_values("score", ascending=False, kind="stable").head(k)
        recalls.append(top_k["target"].sum() / borough["target"].sum())
    return float(np.mean(recalls))


def pr_auc_within(data, score):
    """Average (over boroughs) PR-AUC computed inside each borough."""
    return float(np.mean([
        average_precision_score(b["target"], b["score"]) for b in _by_borough(data, score)
    ]))


def evaluate(name, data, score):
    """One row of the results table."""
    return {
        "model": name,
        "recall@20%": recall_at_top(data, score),
        "recall@10%": recall_at_top(data, score, top=0.1),
        "PR-AUC (within)": pr_auc_within(data, score),
        "PR-AUC (pooled)": average_precision_score(data[TARGET], score),
        "ROC-AUC (pooled)": roc_auc_score(data[TARGET], score),
    }