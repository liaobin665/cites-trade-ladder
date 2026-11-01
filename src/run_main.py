# -*- coding: utf-8 -*-
# Author: Liao Bin (廖彬)
# Created: 2026-10-18
# Updated: 2026-11-01
"""Main experiment: rolling-origin evaluation, 2017-2024.

For every test year T the model is trained on 2016..T-1 only. The headline
scores are 8-year means over a fixed review budget B. Outputs feed the tables
and most of the figures, so run this before the other analysis scripts.
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
from models import build_train, fit_lgb_rank, zscore_avg, ndcg_at_k  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
BS = (20, 50, 100, 200, 500, 1000)
YEARS = list(range(2016, 2025))
SEEDS = (0, 1, 2)


def load_cache(K=2, horizon=1):
    cache = {}
    for T in YEARS:
        X, meta = load_year(T)
        y, rec = make_labels(T, meta, K=K, horizon=horizon)
        cache[T] = (X, y, meta)
        print(f"cache T={T} n={len(y)} pos={int(y.sum())}", flush=True)
    return cache


def main():
    t0 = time.time()
    cache = load_cache(K=2, horizon=1)
    rows, seed_rows = [], []
    for T in YEARS[1:]:
        Xte, yte, mt = cache[T]
        preds = []
        for seed in SEEDS:
            Xtr, ytr, groups = build_train(cache, range(2016, T), seed=seed)
            model = fit_lgb_rank(Xtr, ytr, groups, seed=seed)
            p = model.predict(Xte)
            preds.append(p)
            for B in BS:
                seed_rows.append(dict(T=T, seed=seed, B=B,
                                      hits=int(yte[np.argsort(-p)[:B]].sum())))
            print(f"T={T} seed={seed} train={len(ytr)} done {time.time()-t0:.0f}s", flush=True)
        s = zscore_avg(preds)
        vol = mt["u_w3"]
        form = mt["s23_w3"]
        # Baselines: primary key is the score, secondary key is the record
        # count. lexsort makes the tie-breaking explicit and reproducible;
        # plain argsort would leave tied units in row order.
        vol_order = np.lexsort((-mt["u_rec"], -vol))
        form_order = np.lexsort((-vol, -form))
        n = len(yte)
        for B in BS:
            mh = int(yte[np.argsort(-s)[:B]].sum())
            vh = int(yte[vol_order[:B]].sum())
            fh = int(yte[form_order[:B]].sum())
            # Analytic chance level: drawing B units without replacement from
            # a pool with n_pos positives.
            p = yte.mean()
            exp_hits = B * p
            sd = np.sqrt(B * p * (1 - p) * max(n - B, 0) / max(n - 1, 1))
            rows.append(dict(T=T, B=B, n_units=n, n_pos=int(yte.sum()),
                             model=mh, volume=vh, form=fh,
                             random=exp_hits, random_lo=max(0, exp_hits - 1.96 * sd),
                             random_hi=exp_hits + 1.96 * sd,
                             recall=mh / max(int(yte.sum()), 1), precision=mh / B,
                             ndcg=ndcg_at_k(yte, s, B)))
        np.savez_compressed(
            RES / f"test_T{T}.npz",
            keys=mt["keys"], cu=mt["cu"], cv=mt["cv"], ts=mt["ts"],
            y=yte, score=s, volume=vol, formhist=form,
            u_rec=mt["u_rec"], best=mt["best"], n_cand=mt["n_cand"],
            n_cand_s2=mt["n_cand_s2"], n_cand_s3=mt["n_cand_s3"],
            seed_scores=np.vstack(preds),
        )
        r100 = [r for r in rows if r["T"] == T and r["B"] == 100][0]
        print(f"T={T} B100 model={r100['model']} vol={r100['volume']} "
              f"form={r100['form']} random={r100['random']:.2f} pos={int(yte.sum())} "
              f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "main_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seed_rows).to_csv(RES / "per_seed_hits.csv", index=False, encoding="utf-8-sig")
    avg = df.groupby("B")[["model", "volume", "form", "random", "recall", "precision", "ndcg"]].mean()
    sem = df.groupby("B")[["model", "volume", "form", "random"]].sem()
    avg.to_csv(RES / "main_average.csv", encoding="utf-8-sig")
    sem.to_csv(RES / "main_sem.csv", encoding="utf-8-sig")
    print(avg.round(2).to_string())
    print("elapsed", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
