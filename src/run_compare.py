# -*- coding: utf-8 -*-
# Author: Liao Bin (廖彬)
# Created: 2026-10-20
# Updated: 2026-11-01
"""Model comparison and feature/loss ablations.

Both parts use a single seed by default; the numbers here are meant to isolate
the loss function and the feature groups, not to estimate seed variance.
"""
from __future__ import annotations

import argparse
import sys
import time
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_year, make_labels, FEATURES  # noqa: E402
from models import (build_train, fit_lgb_rank, fit_lgb_binary, fit_xgb_rank,  # noqa: E402
                    fit_cat_rank, fit_logreg, predict, ndcg_at_k)

ROOT = HERE.parent
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
BS = (20, 50, 100, 200, 500, 1000)
YEARS = list(range(2016, 2025))
TEST = list(range(2017, 2025))


def load_all():
    cache = {}
    for T in YEARS:
        X, meta = load_year(T)
        y, _ = make_labels(T, meta, K=2, horizon=1)
        cache[T] = (X, y, meta)
    return cache


def metric_rows(y, s, T, name, extra, fit_s, n_train):
    npos = int(y.sum())
    out = []
    for B in BS:
        order = np.argsort(-s)[:B]
        hit = int(y[order].sum())
        out.append(dict(T=T, name=name, B=B, hit=hit, n_pos=npos,
                        n_units=len(y), recall=hit / max(npos, 1),
                        precision=hit / B, ndcg=ndcg_at_k(y, s, B),
                        fit_seconds=fit_s, n_train=n_train, **extra))
    return out


def run_models(cache, seeds):
    models = {"lgb_rank": fit_lgb_rank, "lgb_bce": fit_lgb_binary,
              "xgb_rank": fit_xgb_rank, "cat_rank": fit_cat_rank,
              "logreg": fit_logreg}
    rows = []
    for T in TEST:
        Xte, yte, _ = cache[T]
        Xtr, ytr, groups = build_train(cache, range(2016, T), seed=0)
        for name, fn in models.items():
            preds, t0 = [], time.time()
            for seed in seeds:
                model = fn(Xtr, ytr, groups, seed=seed)
                preds.append(predict(model, Xte))
            fit_s = (time.time() - t0) / len(seeds)
            p = np.mean([(v - v.mean()) / (v.std() + 1e-9) for v in preds], axis=0)
            rows.extend(metric_rows(yte, p, T, name, {}, fit_s, len(ytr)))
            print(f"model {name} T={T} B100={rows[-4]['hit']} {fit_s:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(RES / "model_compare.csv", index=False, encoding="utf-8-sig")


ABLATIONS = {
    "full": None,
    "no_trade": ("drop", {"trade"}),
    "no_graph": ("drop", {"graph"}),
    "no_form": ("drop", {"form"}),
    "no_context": ("drop", {"context"}),
    "unit_only": ("keep", {"trade", "context"}),
    "no_next": ("names", ("opp_", "next_")),
    "form_unit_only": ("names", ("opp_", "next_", "cand_")),
}


def select(config, featnames):
    if config is None:
        return np.arange(len(featnames))
    kind, arg = config
    groups = [g for _, g in FEATURES]
    if kind == "drop":
        return np.array([i for i, g in enumerate(groups) if g not in arg])
    if kind == "keep":
        return np.array([i for i, g in enumerate(groups) if g in arg])
    return np.array([i for i, n in enumerate(featnames) if not n.startswith(arg)])


def run_ablation(cache, seeds):
    rows = []
    for T in TEST:
        Xte, yte, meta = cache[T]
        Xtr, ytr, groups = build_train(cache, range(2016, T), seed=0)
        for name, cfg in ABLATIONS.items():
            cols = select(cfg, meta["featnames"])
            preds, t0 = [], time.time()
            for seed in seeds:
                model = fit_lgb_rank(Xtr[:, cols], ytr, groups, seed=seed)
                preds.append(model.predict(Xte[:, cols]))
            fit_s = (time.time() - t0) / len(seeds)
            p = np.mean([(v - v.mean()) / (v.std() + 1e-9) for v in preds], axis=0)
            rows.extend(metric_rows(yte, p, T, name, {"n_feat": len(cols)},
                                    fit_s, len(ytr)))
            print(f"ablate {name} T={T} nfeat={len(cols)} B100={rows[-4]['hit']}",
                  flush=True)
    pd.DataFrame(rows).to_csv(RES / "ablation.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["models", "ablation"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    t0 = time.time()
    cache = load_all()
    if args.part == "models":
        run_models(cache, args.seeds)
    else:
        run_ablation(cache, args.seeds)
    print("elapsed", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
