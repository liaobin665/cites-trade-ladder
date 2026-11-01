# -*- coding: utf-8 -*-
# Author: Liao Bin (廖彬)
# Created: 2026-10-22
# Updated: 2026-11-01
"""SHAP attribution, permutation importance, and failure analysis (2024).

TreeSHAP comes from LightGBM's pred_contrib, so no extra dependency is needed.
The failure statistics use the 3-seed ensemble saved by run_main.py, while the
attribution uses the single seed shown in the corresponding figure.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_year, make_labels, FEATNAMES, FEATURES  # noqa: E402
from models import build_train, fit_lgb_rank  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
T = 2024


def main():
    t0 = time.time()
    cache = {}
    for t in range(2016, 2025):
        X, meta = load_year(t)
        y, _ = make_labels(t, meta, 2, 1)
        cache[t] = (X, y, meta)
    Xte, yte, meta = cache[T]
    Xtr, ytr, groups = build_train(cache, range(2016, T), seed=0)
    model = fit_lgb_rank(Xtr, ytr, groups, seed=0)
    score = model.predict(Xte)
    print("fit", round(time.time() - t0, 1), flush=True)

    contrib = model.predict(Xte, pred_contrib=True)
    sv = contrib[:, :-1].astype(np.float32)
    base = contrib[:, -1].astype(np.float32)
    np.savez_compressed(RES / "shap_values.npz", values=sv, base=base,
                        data=Xte.astype(np.float32), score=score, y=yte)
    imp = pd.DataFrame({"feature": FEATNAMES,
                        "mean_abs": np.abs(sv).mean(0),
                        "group": [g for _, g in FEATURES]})
    imp.sort_values("mean_abs", ascending=False).to_csv(
        RES / "shap_importance.csv", index=False, encoding="utf-8-sig")

    # permutation importance on Hit@100
    rng = np.random.default_rng(0)
    base_hit = int(yte[np.argsort(-score)[:100]].sum())
    nrep = 3
    rows = []
    for j, name in enumerate(FEATNAMES):
        hits = []
        for _ in range(nrep):
            Xp = Xte.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            s = model.predict(Xp)
            hits.append(int(yte[np.argsort(-s)[:100]].sum()))
        rows.append(dict(feature=name, hit=float(np.mean(hits)),
                         drop=base_hit - float(np.mean(hits))))
        if j % 10 == 0:
            print("perm", j, name, round(time.time() - t0, 1), flush=True)
    pd.DataFrame(rows).sort_values("drop", ascending=False).to_csv(
        RES / "perm_importance.csv", index=False, encoding="utf-8-sig")

    # failure statistics use the 3-seed ensemble (fig13), not the single seed
    # used for the SHAP attribution above
    ens = np.load(RES / f"test_T{T}.npz", allow_pickle=True)["score"]
    assert len(ens) == len(yte)
    order = np.argsort(-ens)
    ranks = np.empty(len(yte), dtype=np.int64)
    ranks[order] = np.arange(1, len(yte) + 1)
    pos_rank = ranks[yte == 1]
    summary = dict(n_pos=int(yte.sum()), hit100=int((pos_rank <= 100).sum()),
                   hit500=int((pos_rank <= 500).sum()),
                   hit1000=int((pos_rank <= 1000).sum()),
                   median_rank=float(np.median(pos_rank)))
    pd.Series(summary).to_csv(RES / "failure_summary.csv",
                              encoding="utf-8-sig")

    idx = np.argsort(-ens)
    top_fp = [i for i in idx if yte[i] == 0][:25]
    missed = [i for i in np.argsort(ens) if yte[i] == 1][:25]
    cols = {n: Xte[:, j] for j, n in enumerate(FEATNAMES)}
    rec = []
    for tag, ids in (("top_non_event", top_fp), ("missed_event", missed)):
        for i in ids:
            rec.append(dict(kind=tag, rank=int(ranks[i]), y=int(yte[i]),
                            score=float(ens[i]),
                            exporter=int(meta["cu"][i]), importer=int(meta["cv"][i]),
                            taxon=int(meta["ts"][i]), best=int(meta["best"][i]),
                            u_w3=float(meta["u_w3"][i]),
                            s23_w3=float(meta["s23_w3"][i]),
                            n_cand=int(meta["n_cand"][i]),
                            opp_peer_r3=float(cols["opp_peer_r3"][i]),
                            next_gap=float(cols["next_gap"][i])))
    pd.DataFrame(rec).to_csv(RES / "failure_cases.csv", index=False,
                             encoding="utf-8-sig")
    print(summary, flush=True)
    print("elapsed", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
