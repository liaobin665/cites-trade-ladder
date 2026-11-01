# -*- coding: utf-8 -*-
# Author: Liao Bin (廖彬)
# Created: 2026-10-21
# Updated: 2026-11-01
"""Robustness: five-seed ensembles and label threshold/window sensitivity.

The seed part reuses the seed scores saved by run_main.py, so run_main must
finish before this script.
"""
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
BS = (20, 50, 100, 200, 500, 1000)
YEARS = list(range(2016, 2025))
TEST = list(range(2017, 2025))


def load_all(K=2, horizon=1):
    cache = {}
    for T in YEARS:
        X, meta = load_year(T)
        y, _ = make_labels(T, meta, K=K, horizon=horizon)
        cache[T] = (X, y, meta)
    return cache


def eval_rows(y, s, T, cfg, extra, vol_order=None):
    npos = int(y.sum())
    rows = []
    for B in BS:
        hit = int(y[np.argsort(-s)[:B]].sum())
        rec = dict(T=T, cfg=cfg, B=B, hit=hit, n_pos=npos, n_units=len(y),
                   recall=hit / max(npos, 1), precision=hit / B,
                   ndcg=ndcg_at_k(y, s, B), **extra)
        if vol_order is not None:
            rec["vol_hit"] = int(y[vol_order[:B]].sum())
        rows.append(rec)
    return rows


def seeds_part(cache):
    rows = []
    for T in TEST:
        Xte, yte, _ = cache[T]
        preds = [np.load(RES / f"test_T{T}.npz")["seed_scores"][i] for i in range(3)]
        for seed in (3, 4):
            Xtr, ytr, groups = build_train(cache, range(2016, T), seed=seed)
            m = fit_lgb_rank(Xtr, ytr, groups, seed=seed)
            preds.append(m.predict(Xte))
            rows.extend(eval_rows(yte, preds[-1], T, f"single_s{seed}",
                                  {"seed": seed}))
            print(f"seeds T={T} seed={seed} B100="
                  f"{int(yte[np.argsort(-preds[-1])[:100]].sum())}", flush=True)
        for seed in (0, 1, 2):
            rows.extend(eval_rows(yte, preds[seed], T, f"single_s{seed}",
                                  {"seed": seed}))
        p = np.mean([(v - v.mean()) / (v.std() + 1e-9) for v in preds], axis=0)
        rows.extend(eval_rows(yte, p, T, "ens5", {"seed": -1}))
        print(f"seeds T={T} ens5 B100={int(yte[np.argsort(-p)[:100]].sum())}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "robust_seeds.csv", index=False, encoding="utf-8-sig")


def label_part():
    rows = []
    for cfg, K, h in [("K1", 1, 1), ("K5", 5, 1), ("w2", 2, 2)]:
        cache = load_all(K=K, horizon=h)
        for T in TEST:
            Xte, yte, meta = cache[T]
            Xtr, ytr, groups = build_train(cache, range(2016, T), seed=0)
            m = fit_lgb_rank(Xtr, ytr, groups, seed=0)
            p = m.predict(Xte)
            vol_order = np.lexsort((-meta["u_rec"], -meta["u_w3"]))
            rows.extend(eval_rows(yte, p, T, cfg, {"K": K, "horizon": h},
                                  vol_order=vol_order))
            print(f"label {cfg} T={T} pos={int(yte.sum())} "
                  f"B100={int(yte[np.argsort(-p)[:100]].sum())}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "robust_label.csv", index=False, encoding="utf-8-sig")


def main():
    t0 = time.time()
    cache = load_all()
    seeds_part(cache)
    label_part()
    print("elapsed", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
