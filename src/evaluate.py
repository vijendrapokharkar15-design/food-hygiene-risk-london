"""
Evaluation metrics for the inspection-prioritisation problem.

Main metric: recall at the top 20% WITHIN each borough, averaged across boroughs.
"If each council inspects the riskiest 20% of its own list, what share of its
failing businesses does it find?"
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import GROUP, TARGET


def recall_at_top(data, score, top=0.2, seed=0):
    """Average (over boroughs) share of failures found in each borough's top `top` fraction."""
    frame = pd.DataFrame({
        "group": data[GROUP].to_numpy(),
        "target": data[TARGET].to_numpy(),
        "score": np.asarray(score, dtype=float),
    })
    # Shuffle first so that tied scores end up in random order, not file order
    frame = frame.sample(frac=1, random_state=seed)

    recalls = []
    for _, borough in frame.groupby("group"):
        n_fail = borough["target"].sum()
        if n_fail == 0:
            continue
        k = int(np.ceil(top * len(borough)))
        top_k = borough.sort_values("score", ascending=False, kind="stable").head(k)
        recalls.append(top_k["target"].sum() / n_fail)
    return float(np.mean(recalls))


def evaluate(name, data, score):
    """One row of the results table."""
    return {
        "model": name,
        "recall@20%": recall_at_top(data, score),
        "PR-AUC": average_precision_score(data[TARGET], score),
        "ROC-AUC": roc_auc_score(data[TARGET], score),
    }