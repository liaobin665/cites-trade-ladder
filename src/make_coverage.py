# -*- coding: utf-8 -*-
# Author: Liao Bin (廖彬)
# Created: 2026-10-16
# Updated: 2026-11-01
"""Universe coverage, term summary, and the headline dataset statistics."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)


def coverage():
    B = D.load_base(); L = B["L"]; rk_code = B["rank_of_code"]
    NC, NS = B["NC"], B["NS"]
    rows = []
    for T in range(2017, 2025):
        H = L[L.Year < T]
        keys, inv = np.unique(H.key.to_numpy(), return_inverse=True)
        nU = len(keys)
        best = np.full(nU, -1, dtype=np.int8)
        rk = rk_code[H.tk.to_numpy()]
        ok = rk >= 0
        np.maximum.at(best, inv[ok], rk[ok].astype(np.int8))
        active = np.bincount(inv[H.Year.to_numpy() >= T - 3], minlength=nU) > 0
        tar = L[L.Year == T]
        tk = tar.key.to_numpy()
        p = np.searchsorted(keys, tk)
        valid = (p < nU) & (keys[np.clip(p, 0, nU - 1)] == tk)
        p, rkt, nrt = p[valid], rk_code[tar.tk.to_numpy()[valid]], \
            tar.n_rec.to_numpy()[valid].astype(float)
        good = (rkt >= 2) & (rkt > best[p])
        rec = np.bincount(p[good], weights=nrt[good], minlength=nU)
        y = rec >= 2
        rows.append(dict(T=T, n_hist_units=nU, n_pos_all=int(y.sum()),
                         n_pos_active=int((y & active).sum()),
                         share=float((y & active).sum() / max(y.sum(), 1)),
                         n_active=int(active.sum())))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "coverage.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False), flush=True)


def summary():
    B = D.load_base(); L = B["L"]
    lad = pd.read_csv(D.LADDER)
    top = (L.groupby("Term").n_rec.sum().sort_values(ascending=False).head(15)
           .rename("records").reset_index())
    top["ladder"] = top.Term.map(dict(zip(lad.term, lad.ladder)))
    top.to_csv(RES / "top_terms.csv", index=False, encoding="utf-8-sig")
    meta = dict(n_rows=int(len(L)), year_min=int(L.Year.min()),
                year_max=int(L.Year.max()), n_records=int(L.n_rec.sum()),
                n_terms=int(L.Term.nunique()), n_taxa=int(L.Taxon.nunique()),
                n_countries=int(len(B["CV"])),
                ladder_counts=lad.ladder.value_counts().to_dict(),
                s23_terms=int((B["rank_of_code"] >= 2).sum()),
                n_ladder_terms=int(len(lad)))
    pd.Series(meta).to_json(RES / "data_summary.json", force_ascii=False)
    print(meta, flush=True)


if __name__ == "__main__":
    summary()
    coverage()
