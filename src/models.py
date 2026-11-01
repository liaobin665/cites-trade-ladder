# -*- coding: utf-8 -*-
# Author: Liao Bin (廖彬)
# Created: 2026-10-15
# Updated: 2026-11-01
"""Ranking models and small evaluation helpers.

LightGBM LambdaRank is the main model; the other learners are only used as
comparison arms. Everything is trained on the same query blocks so that the
comparison isolates the loss function rather than the batching scheme.
"""
from __future__ import annotations

import numpy as np


def chunk_year(X, y, size=9000, seed=0):
    """Shuffle one test year and cut it into LambdaRank query blocks.

    LightGBM needs explicit query groups. Blocks of ~9k rows keep the pairwise
    gradient computation tractable while keeping a similar positive rate in
    every block.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    chunks = [idx[i:i + size] for i in range(0, len(idx), size)]
    Xc = np.vstack([X[c] for c in chunks])
    yc = np.concatenate([y[c] for c in chunks])
    return Xc, yc, [len(c) for c in chunks]


def build_train(cache, years, seed=0, size=9000):
    """Concatenate several years of query blocks into one training matrix."""
    Xs, ys, groups = [], [], []
    for t in years:
        X, y, _ = cache[t]
        Xc, yc, g = chunk_year(X, y, size=size, seed=seed * 1000 + t)
        Xs.append(Xc); ys.append(yc); groups.extend(g)
    return np.vstack(Xs), np.concatenate(ys), groups


def ndcg_at_k(y, s, k):
    """Binary-relevance NDCG at k (used for reporting, not for training)."""
    order = np.argsort(-s)[:k]
    rel = y[order].astype(np.float64)
    disc = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    ideal = np.sort(y.astype(np.float64))[::-1][:k]
    idcg = (ideal * disc[:len(ideal)]).sum()
    return float((rel * disc).sum() / idcg) if idcg > 0 else 0.0


def zscore_avg(preds):
    """Average seeds after per-seed z-scoring.

    Raw scores from different seeds are not on the same scale; z-scoring first
    stops a seed with a wider score range from dominating the ensemble.
    """
    z = [(p - p.mean()) / (p.std() + 1e-9) for p in preds]
    return np.mean(z, axis=0)


def metrics_at(y, s, B):
    order = np.argsort(-s)[:B]
    hit = int(y[order].sum())
    npos = int(y.sum())
    return dict(hit=hit, recall=hit / npos if npos else 0.0,
                precision=hit / B, ndcg=ndcg_at_k(y, s, B))


def random_expectation(y, B, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    p = y.mean()
    idx = np.array([rng.choice(n, size=B, replace=False) for _ in range(draws)])
    hits = y[idx].sum(1)
    return float(hits.mean()), float(np.quantile(hits, 0.025)), float(np.quantile(hits, 0.975))


def fit_lgb_rank(X, y, groups, seed=0, **over):
    import lightgbm as lgb
    p = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[100],
             boosting="gbdt", learning_rate=0.05, num_leaves=31,
             min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, num_threads=8, seed=seed,
             bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    p.update(over)
    n = p.pop("num_boost_round", 300)
    model = lgb.train(p, lgb.Dataset(X, label=y, group=groups), num_boost_round=n)
    return model


def fit_lgb_binary(X, y, groups=None, seed=0, **over):
    import lightgbm as lgb
    pos = max(int(y.sum()), 1)
    # The positive class is around 0.1% of the candidates. Without
    # scale_pos_weight the binary model would simply predict "no event"
    # almost everywhere and the comparison would not be meaningful.
    p = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8,
             seed=seed, bagging_seed=seed + 1, scale_pos_weight=(len(y) - pos) / pos)
    p.update(over)
    n = p.pop("num_boost_round", 300)
    return lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=n)


def fit_xgb_rank(X, y, groups, seed=0, **over):
    import xgboost as xgb
    qid = np.repeat(np.arange(len(groups)), groups)
    p = dict(objective="rank:pairwise", n_estimators=300, learning_rate=0.05,
             max_depth=6, subsample=0.8, colsample_bytree=0.8,
             tree_method="hist", n_jobs=8, random_state=seed, eval_metric="ndcg")
    p.update(over)
    model = xgb.XGBRanker(**p)
    model.fit(X, y, qid=qid, verbose=False)
    return model


def fit_cat_rank(X, y, groups, seed=0, **over):
    from catboost import CatBoostRanker, Pool
    gid = np.repeat(np.arange(len(groups)), groups)
    # YetiRank was too slow on 9k-row query blocks (it could not finish the
    # 8-year rolling run), so the comparison arm uses QuerySoftMax with a
    # deliberately smaller budget: 100 iterations, depth 4.
    p = dict(loss_function="QuerySoftMax", iterations=100, learning_rate=0.1,
             depth=4, boosting_type="Plain", verbose=False, thread_count=8,
             random_seed=seed, allow_writing_files=False)
    p.update(over)
    model = CatBoostRanker(**p)
    model.fit(Pool(X, label=y, group_id=gid))
    return model


def fit_logreg(X, y, groups=None, seed=0, **over):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    p = dict(C=1.0, class_weight="balanced", max_iter=1000, n_jobs=8,
             random_state=seed, solver="lbfgs")
    p.update(over)
    model = LogisticRegression(**p)
    model.fit(Xs, y)
    return (model, sc)


def predict(model, X):
    if isinstance(model, tuple):
        m, sc = model
        return m.predict_proba(sc.transform(X))[:, 1]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)
