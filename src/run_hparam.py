# -*- coding: utf-8 -*-
# Author: Liao Bin (廖彬)
# Created: 2026-10-21
# Updated: 2026-11-01
"""Hyperparameter sensitivity: two 2-D grids, run on 2018 and 2024."""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_year, make_labels  # noqa: E402
from models import build_train, fit_lgb_rank, ndcg_at_k  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
BS = (100, 500, 1000)


def load_all():
    return {T: (lambda X, m: (X, make_labels(T, m, 2, 1)[0], m))(*load_year(T))
            for T in range(2016, 2025)}


def run(cache, T, **over):
    Xte, yte, _ = cache[T]
    Xtr, ytr, groups = build_train(cache, range(2016, T), seed=0)
    m = fit_lgb_rank(Xtr, ytr, groups, seed=0, **over)
    p = m.predict(Xte)
    return {f"hit{B}": int(yte[np.argsort(-p)[:B]].sum()) for B in BS} | \
           {f"ndcg{B}": ndcg_at_k(yte, p, B) for B in BS}


def main():
    t0 = time.time()
    cache = load_all()
    rows = []
    for T in (2018, 2024):
        for trees in (100, 200, 300, 500, 800):
            for lr in (0.01, 0.03, 0.05, 0.1):
                r = run(cache, T, num_boost_round=trees, learning_rate=lr)
                rows.append(dict(T=T, knob="trees_lr", trees=trees, lr=lr,
                                 leaves=31, sampling=0.8, **r))
                print("grid", T, trees, lr, r, flush=True)
        for leaves in (15, 31, 63, 127):
            for sampling in (0.6, 0.8, 1.0):
                r = run(cache, T, num_leaves=leaves,
                        feature_fraction=sampling, bagging_fraction=sampling)
                rows.append(dict(T=T, knob="leaves_sampling", trees=300, lr=0.05,
                                 leaves=leaves, sampling=sampling, **r))
                print("grid", T, leaves, sampling, r, flush=True)
    pd.DataFrame(rows).to_csv(RES / "hparam_grid.csv", index=False, encoding="utf-8-sig")
    print("elapsed", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
