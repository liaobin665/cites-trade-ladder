# -*- coding: utf-8 -*-
# Author: Liao Bin (廖彬)
# Created: 2026-10-15
# Updated: 2026-11-01
"""Data layer: review units, processing-form transition labels, and features.

Inputs
------
``data/links_l4.parquet``
    Aggregated CITES trade records with columns
    Year, Exporter, Importer, Taxon, Term, n_rec, n_appI, n_appII,
    n_srcW, n_purT, n_repE, n_repI.
``config/term_ladder.csv``
    Mapping from CITES term values to four processing levels (S0-S3).
    Terms without a specified form are left out of the ladder.

The raw CITES data are not distributed with this repository; see docs/DATA.md
for the official source and the aggregation recipe.

Features only use records with Year < T; labels use records in [T, T+w-1].
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = pathlib.Path(os.environ.get("CITES_DATA_DIR", ROOT / "data"))
LINKS = DATA_DIR / "links_l4.parquet"
LADDER = pathlib.Path(os.environ.get("CITES_LADDER",
                                     ROOT / "config" / "term_ladder.csv"))
CACHE = pathlib.Path(os.environ.get("CITES_CACHE_DIR", ROOT / "cache"))
CACHE.mkdir(parents=True, exist_ok=True)

RANK = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}

FEATURES = [
    # trade volume (12)
    ("u_rec", "trade"), ("u_years", "trade"), ("u_w1", "trade"),
    ("u_w3", "trade"), ("u_w5", "trade"),
    ("exp_rec", "trade"), ("imp_rec", "trade"), ("tax_rec", "trade"),
    ("exp_w3", "trade"), ("imp_w3", "trade"), ("tax_w3", "trade"),
    ("u_share_tax", "trade"),
    # graph structure (8)
    ("pair_rec", "graph"), ("pair_years", "graph"), ("pair_taxa", "graph"),
    ("pair_cn", "graph"), ("pair_aa", "graph"),
    ("exp_deg", "graph"), ("imp_deg", "graph"), ("same_country", "graph"),
    # form, unit-level (9)
    ("best_rank", "form"), ("n_form", "form"), ("form_entropy", "form"),
    ("s0_ratio", "form"), ("s1_ratio", "form"), ("s2_ratio", "form"),
    ("s23_w3", "form"), ("n_cand", "form"), ("cand_ratio", "form"),
    # form, allowed higher-form means (8)
    ("cand_term_rec_mean", "form"), ("cand_term_w3_mean", "form"),
    ("cand_term_trend_mean", "form"), ("cand_species_used_mean", "form"),
    ("cand_exp_used_mean", "form"), ("cand_imp_used_mean", "form"),
    ("cand_term_expdeg_mean", "form"), ("cand_term_recency_mean", "form"),
    # form, next-rung opportunity (11)
    ("next_gap", "form"),
    ("opp_peer_r2", "form"), ("opp_species_r2", "form"),
    ("opp_exp_r2", "form"), ("opp_imp_r2", "form"), ("opp_pop_r2", "form"),
    ("opp_peer_r3", "form"), ("opp_species_r3", "form"),
    ("opp_exp_r3", "form"), ("opp_imp_r3", "form"),
    # context (9)
    ("exp_appI", "context"), ("exp_srcW", "context"), ("exp_purT", "context"),
    ("tax_appI", "context"), ("tax_srcW", "context"), ("tax_purT", "context"),
    ("recency", "context"), ("span", "context"), ("u_rec_per_year", "context"),
]
FEATNAMES = [n for n, _ in FEATURES]
GROUPS = {g: [i for i, (_, gg) in enumerate(FEATURES) if gg == g]
          for g in ("trade", "graph", "form", "context")}
CAND_COLS = [i for i, n in enumerate(FEATNAMES) if n.startswith("cand_")]
NEXT_COLS = [i for i, n in enumerate(FEATNAMES) if n.startswith("next_")]
MEAN_COLS = [i for i, n in enumerate(FEATNAMES) if n.endswith("_mean")]


_BASE = None


def load_base():
    global _BASE
    if _BASE is not None:
        return _BASE
    if not LINKS.exists():
        raise FileNotFoundError(
            f"{LINKS} not found. Put the aggregated trade table under data/ or "
            "set CITES_DATA_DIR; see docs/DATA.md.")
    L = pd.read_parquet(LINKS)
    # Country / taxon / term codes are kept as integers so that units can be
    # identified by a single int64 key instead of a tuple of strings.
    for c in ("Exporter", "Importer", "Taxon", "Term"):
        L[c] = L[c].astype(str)
    L["Year"] = L["Year"].astype(int)
    CV = {v: i for i, v in enumerate(sorted(set(L.Exporter) | set(L.Importer)))}
    TV = {v: i for i, v in enumerate(sorted(L.Taxon.unique()))}
    KV = {v: i for i, v in enumerate(sorted(L.Term.unique()))}
    L["cu"] = L.Exporter.map(CV).astype("int32")
    L["cv"] = L.Importer.map(CV).astype("int32")
    L["ts"] = L.Taxon.map(TV).astype("int32")
    L["tk"] = L.Term.map(KV).astype("int32")
    NC, NS, NK = len(CV), len(TV), len(KV)
    L["key"] = (L.cu.astype(np.int64) * NC + L.cv) * NS + L.ts

    lad = pd.read_csv(LADDER)
    rank_of_code = np.full(NK, -1, dtype=np.int8)
    for term, ladder in zip(lad.term, lad.ladder):
        if term in KV and ladder in RANK:
            rank_of_code[KV[term]] = RANK[ladder]
    s23 = np.where((rank_of_code == 2) | (rank_of_code == 3))[0].astype(np.int64)
    _BASE = dict(L=L, CV=CV, TV=TV, KV=KV, NC=NC, NS=NS, NK=NK,
                 rank_of_code=rank_of_code, S23=s23, S23R=rank_of_code[s23].astype(np.int64))
    return _BASE


def _entropy(inv, tk, w, nU):
    df = pd.DataFrame({"u": inv, "k": tk, "w": w.astype(np.float64)})
    g = df.groupby(["u", "k"], sort=False).w.sum().reset_index()
    tot = np.bincount(g.u.to_numpy(), weights=g.w.to_numpy(), minlength=nU)
    p = g.w.to_numpy() / np.maximum(tot[g.u.to_numpy()], 1e-12)
    ent = np.bincount(g.u.to_numpy(), weights=-p * np.log(np.maximum(p, 1e-12)), minlength=nU)
    nform = np.bincount(g.u.to_numpy(), minlength=nU)
    return ent / np.maximum(np.log(np.maximum(nform, 1)), 1e-9)


def _group_sum(inv, values, nU, mask=None):
    if mask is None:
        return np.bincount(inv, weights=values, minlength=nU)
    return np.bincount(inv[mask], weights=values[mask], minlength=nU)


def _distinct_count(inv, other, nU, mask=None):
    if mask is None:
        a, b = inv, other
    else:
        a, b = inv[mask], other[mask]
    if len(a) == 0:
        return np.zeros(nU)
    k = np.unique(a.astype(np.int64) * (other.max() + 1) + b.astype(np.int64))
    return np.bincount(k // (other.max() + 1), minlength=nU)


def build_features(T: int, recent: int = 3):
    """Feature matrix for all units with trade history before T.

    Candidate set (main): active in [T-recent, T-1], best rank <= S2,
    and at least one unused S2/S3 form above the historical best.
    """
    B = load_base()
    L = B["L"]; NC, NS, NK = B["NC"], B["NS"], B["NK"]
    S23, S23R, rk_code = B["S23"], B["S23R"], B["rank_of_code"]
    nS23 = len(S23)

    H = L[L.Year < T]
    keys, inv = np.unique(H.key.to_numpy(), return_inverse=True)
    nU = len(keys)
    n_hist = nU
    cu = (keys // (NC * NS)).astype(np.int32)
    cv = ((keys // NS) % NC).astype(np.int32)
    ts = (keys % NS).astype(np.int32)
    yr = H.Year.to_numpy(); nr = H.n_rec.to_numpy().astype(np.float64)
    tk = H.tk.to_numpy(); rk = rk_code[tk]

    u_rec = _group_sum(inv, nr, nU)
    u_w1 = _group_sum(inv, nr, nU, yr >= T - 1)
    u_w3 = _group_sum(inv, nr, nU, yr >= T - 3)
    u_w5 = _group_sum(inv, nr, nU, yr >= T - 5)
    u_years = _distinct_count(inv, yr, nU)
    u_first = np.full(nU, 9999, dtype=np.int32); np.minimum.at(u_first, inv, yr)
    u_last = np.zeros(nU, dtype=np.int32); np.maximum.at(u_last, inv, yr)
    u_nform = _distinct_count(inv, tk, nU)
    u_entropy = _entropy(inv, tk, nr, nU)
    best = np.full(nU, -1, dtype=np.int8)
    ok = rk >= 0
    np.maximum.at(best, inv[ok], rk[ok].astype(np.int8))
    lad_rec = [np.zeros(nU) for _ in range(4)]
    for r in range(4):
        lad_rec[r] = _group_sum(inv, nr, nU, rk == r)
    lad_tot = np.maximum(sum(lad_rec), 1e-12)
    s23_w3 = _group_sum(inv, nr, nU, (rk >= 2) & (yr >= T - 3))

    # candidate matrix: unused S2/S3 forms above the historical best
    s23pos = np.full(NK, -1, dtype=np.int32); s23pos[S23] = np.arange(nS23)
    used = np.zeros((nU, nS23), dtype=bool)
    m = s23pos[tk] >= 0
    used[inv[m], s23pos[tk[m]]] = True
    allowed = ~used & (S23R[None, :] > best[:, None])
    n_cand = allowed.sum(1).astype(np.int32)
    active = (u_rec > 0) & (u_w3 > 0) & (best <= 2) & (n_cand > 0)
    keep = np.where(active)[0]
    keys, cu, cv, ts = keys[keep], cu[keep], cv[keep], ts[keep]
    best, n_cand, allowed = best[keep], n_cand[keep], allowed[keep]
    u_rec, u_w1, u_w3, u_w5 = u_rec[keep], u_w1[keep], u_w3[keep], u_w5[keep]
    u_years, u_first, u_last = u_years[keep], u_first[keep], u_last[keep]
    u_nform, u_entropy, s23_w3 = u_nform[keep], u_entropy[keep], s23_w3[keep]
    lad_rec = [a[keep] for a in lad_rec]
    lad_tot = np.maximum(sum(lad_rec), 1e-12)
    nU = len(keys)
    n_cand_s2 = (allowed & (S23R[None, :] == 2)).sum(1).astype(np.float32)
    n_cand_s3 = (allowed & (S23R[None, :] == 3)).sum(1).astype(np.float32)

    # country / species / pair histories
    exp_rec = np.bincount(H.cu.to_numpy(), weights=nr, minlength=NC)
    imp_rec = np.bincount(H.cv.to_numpy(), weights=nr, minlength=NC)
    tax_rec = np.bincount(H.ts.to_numpy(), weights=nr, minlength=NS)
    exp_years = _distinct_count(H.cu.to_numpy(), yr, NC)
    imp_years = _distinct_count(H.cv.to_numpy(), yr, NC)
    tax_years = _distinct_count(H.ts.to_numpy(), yr, NS)
    exp_w3 = np.bincount(H.cu.to_numpy()[yr >= T - 3], weights=nr[yr >= T - 3], minlength=NC)
    imp_w3 = np.bincount(H.cv.to_numpy()[yr >= T - 3], weights=nr[yr >= T - 3], minlength=NC)
    tax_w3 = np.bincount(H.ts.to_numpy()[yr >= T - 3], weights=nr[yr >= T - 3], minlength=NS)

    pair = H.cu.to_numpy().astype(np.int64) * NC + H.cv.to_numpy()
    pair_rec = np.bincount(pair, weights=nr, minlength=NC * NC)
    pair_years = _distinct_count(pair, yr, NC * NC)
    pair_taxa = _distinct_count(pair, H.ts.to_numpy(), NC * NC)

    A = np.zeros((NC, NC), dtype=np.float32)
    pidx = np.unique(pair)
    A[pidx // NC, pidx % NC] = 1.0
    A = np.maximum(A, A.T)
    deg = A.sum(0)
    wgt = 1.0 / np.log(np.maximum(deg, 2))
    CN = A @ A.T
    AA = (A * wgt[None, :]) @ A.T

    et = np.zeros((NC, NS), dtype=bool)
    q = np.unique(H.cu.to_numpy().astype(np.int64) * NS + H.ts.to_numpy())
    et[q // NS, q % NS] = True
    it = np.zeros((NC, NS), dtype=bool)
    q = np.unique(H.cv.to_numpy().astype(np.int64) * NS + H.ts.to_numpy())
    it[q // NS, q % NS] = True
    ef = np.zeros((NC, NK), dtype=bool)
    q = np.unique(H.cu.to_numpy().astype(np.int64) * NK + H.tk.to_numpy())
    ef[q // NK, q % NK] = True
    inf = np.zeros((NC, NK), dtype=bool)
    q = np.unique(H.cv.to_numpy().astype(np.int64) * NK + H.tk.to_numpy())
    inf[q // NK, q % NK] = True
    sf = np.zeros((NS, NK), dtype=bool)
    q = np.unique(H.ts.to_numpy().astype(np.int64) * NK + H.tk.to_numpy())
    sf[q // NK, q % NK] = True
    # distinct exporters per species-form, for peer-diffusion features
    q = np.unique((H.ts.to_numpy().astype(np.int64) * NC + H.cu.to_numpy()) * NK
                  + H.tk.to_numpy())
    esf = np.bincount((q // NK) // NC * NK + q % NK,
                      minlength=NS * NK).reshape(NS, NK).astype(np.float32)
    q = np.unique(H.ts.to_numpy().astype(np.int64) * NC + H.cu.to_numpy())
    n_exp_sp = np.bincount(q // NC, minlength=NS).astype(np.float32)

    # context ratios
    exp_appI = np.bincount(H.cu.to_numpy(), weights=H.n_appI.to_numpy(), minlength=NC) / np.maximum(exp_rec, 1)
    exp_srcW = np.bincount(H.cu.to_numpy(), weights=H.n_srcW.to_numpy(), minlength=NC) / np.maximum(exp_rec, 1)
    exp_purT = np.bincount(H.cu.to_numpy(), weights=H.n_purT.to_numpy(), minlength=NC) / np.maximum(exp_rec, 1)
    tax_appI = np.bincount(H.ts.to_numpy(), weights=H.n_appI.to_numpy(), minlength=NS) / np.maximum(tax_rec, 1)
    tax_srcW = np.bincount(H.ts.to_numpy(), weights=H.n_srcW.to_numpy(), minlength=NS) / np.maximum(tax_rec, 1)
    tax_purT = np.bincount(H.ts.to_numpy(), weights=H.n_purT.to_numpy(), minlength=NS) / np.maximum(tax_rec, 1)

    # term-level global statistics (for candidate aggregation)
    term_rec = np.bincount(tk, weights=nr, minlength=NK)
    term_w3 = np.bincount(tk[yr >= T - 3], weights=nr[yr >= T - 3], minlength=NK)
    term_last = np.zeros(NK, dtype=np.int32); np.maximum.at(term_last, tk, yr)
    term_expdeg = np.bincount(tk, weights=np.ones_like(nr), minlength=NK)
    q = np.unique(H.tk.to_numpy().astype(np.int64) * NC + H.cu.to_numpy())
    term_expdeg = np.bincount(q // NC, minlength=NK)

    # assemble unit features
    def lg(x):
        # Counts are heavy-tailed; log1p keeps them well behaved without
        # dropping zeros. Ratio features are left on their natural scale.
        return np.log1p(np.maximum(x, 0)).astype(np.float32)

    feats = [
        lg(u_rec), lg(u_years), lg(u_w1), lg(u_w3), lg(u_w5),
        lg(exp_rec[cu]), lg(imp_rec[cv]), lg(tax_rec[ts]),
        lg(exp_w3[cu]), lg(imp_w3[cv]), lg(tax_w3[ts]),
        (u_rec / np.maximum(tax_rec[ts], 1)).astype(np.float32),
        lg(pair_rec[pairvec := (cu.astype(np.int64) * NC + cv)]),
        pair_years[pairvec].astype(np.float32), lg(pair_taxa[pairvec]),
        CN[cu, cv].astype(np.float32), AA[cu, cv].astype(np.float32),
        deg[cu].astype(np.float32), deg[cv].astype(np.float32),
        (cu == cv).astype(np.float32),
        best.astype(np.float32), lg(u_nform), u_entropy.astype(np.float32),
        (lad_rec[0] / lad_tot).astype(np.float32), (lad_rec[1] / lad_tot).astype(np.float32),
        (lad_rec[2] / lad_tot).astype(np.float32),
        lg(s23_w3), lg(n_cand), (n_cand / nS23).astype(np.float32),
    ]

    # allowed higher-form aggregates and next-rung profile
    def lg_s23(v):
        return lg(v[S23])

    sf_s23 = sf[np.ix_(ts, S23)].astype(np.float32)
    ef_s23 = ef[np.ix_(cu, S23)].astype(np.float32)
    inf_s23 = inf[np.ix_(cv, S23)].astype(np.float32)
    peer_s23 = (esf[np.ix_(ts, S23)]
                / np.maximum(n_exp_sp[ts][:, None], 1)).astype(np.float32)
    term_rec_s23 = lg_s23(term_rec)
    term_w3_s23 = lg_s23(term_w3)
    trend_s23 = (term_w3[S23] / (term_rec[S23] + 1)).astype(np.float32)
    expdeg_s23 = lg_s23(term_expdeg)
    recency_s23 = np.minimum(T - term_last[S23], 999).astype(np.float32)

    for mat in (term_rec_s23, term_w3_s23, trend_s23, sf_s23, ef_s23, inf_s23,
                expdeg_s23, recency_s23):
        sm = (mat[None, :] * allowed).sum(1) if mat.ndim == 1 else (mat * allowed).sum(1)
        feats.append((sm / np.maximum(n_cand, 1)).astype(np.float32))

    allowed_r = np.where(allowed, S23R[None, :], np.int64(99))
    min_r = allowed_r.min(1)
    feats.append((min_r - best).astype(np.float32))
    for r in (2, 3):
        msk = allowed & (S23R[None, :] == r)
        feats.extend([
            (peer_s23 * msk).max(1).astype(np.float32),
            (sf_s23 * msk).max(1).astype(np.float32),
            (ef_s23 * msk).max(1).astype(np.float32),
            (inf_s23 * msk).max(1).astype(np.float32),
        ])
        if r == 2:
            pop = np.where(msk, term_w3[S23][None, :], 0.0).max(1)
            feats.append(lg(pop).astype(np.float32))

    feats.extend([
        exp_appI[cu].astype(np.float32), exp_srcW[cu].astype(np.float32), exp_purT[cu].astype(np.float32),
        tax_appI[ts].astype(np.float32), tax_srcW[ts].astype(np.float32), tax_purT[ts].astype(np.float32),
        np.minimum(T - u_last, 999).astype(np.float32),
        (u_last - u_first + 1).astype(np.float32),
        lg(u_rec / np.maximum(u_years, 1)),
    ])
    X = np.stack(feats, axis=1).astype(np.float32)
    assert X.shape[1] == len(FEATNAMES), (X.shape[1], len(FEATNAMES))
    for i, n in enumerate(FEATNAMES):
        if n.startswith(("opp_peer", "opp_species", "opp_exp", "opp_imp")):
            assert X[:, i].min() >= 0 and X[:, i].max() <= 1.0001, (n, X[:, i].max())
    active = np.ones(len(keys), dtype=bool)
    meta = dict(keys=keys, cu=cu, cv=cv, ts=ts, best=best, n_cand=n_cand,
                u_rec=u_rec, u_w3=u_w3, u_w1=u_w1, s23_w3=s23_w3,
                n_cand_s2=n_cand_s2, n_cand_s3=n_cand_s3, n_hist=n_hist,
                allowed=allowed, active=active, T=T, recent=recent,
                featnames=FEATNAMES)
    return X, meta


def make_labels(T: int, meta: dict, K: int = 2, horizon: int = 1):
    """Binary label: at least K records of an unused S2/S3 form in [T, T+horizon).

    Requiring K >= 2 filters out one-off records (samples, typos, re-exports)
    that would otherwise dominate the positive set.
    """
    B = load_base(); L = B["L"]; rk_code = B["rank_of_code"]
    keys = meta["keys"]; best = meta["best"]
    nU = len(keys)
    tar = L[(L.Year >= T) & (L.Year < T + horizon)]
    if len(tar) == 0:
        return np.zeros(nU, dtype=np.int8), np.zeros(nU, dtype=np.int32)
    tar_key = tar.key.to_numpy()
    inv = np.searchsorted(keys, tar_key)
    valid = (inv < nU) & (keys[np.clip(inv, 0, nU - 1)] == tar_key)
    inv = inv[valid]
    rk = rk_code[tar.tk.to_numpy()[valid]]
    nr = tar.n_rec.to_numpy()[valid].astype(np.float64)
    ok = (rk >= 2) & (rk > best[inv])
    rec = np.bincount(inv[ok], weights=nr[ok], minlength=nU)
    y = (rec >= K).astype(np.int8)
    return y, rec.astype(np.int32)


def cache_path(T: int, recent: int):
    return CACHE / f"feat_T{T}_r{recent}.npz"


def load_year(T: int, recent: int = 3, rebuild: bool = False):
    fp = cache_path(T, recent)
    if fp.exists() and not rebuild:
        d = np.load(fp, allow_pickle=True)
        meta = {k: d[k] for k in d.files if k not in ("X",)}
        meta["T"] = int(meta["T"]); meta["recent"] = int(meta["recent"])
        meta["featnames"] = list(meta["featnames"])
        return d["X"], meta
    X, meta = build_features(T, recent=recent)
    np.savez_compressed(fp, X=X, **{k: v for k, v in meta.items() if k != "featnames"},
                        featnames=np.array(meta["featnames"], dtype=object))
    return X, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2025)))
    ap.add_argument("--recent", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    for T in args.years:
        t0 = time.time()
        X, meta = load_year(T, recent=args.recent, rebuild=args.rebuild)
        y2, rec2 = make_labels(T, meta, K=2, horizon=1)
        print(f"T={T} units={len(X)} active={int(meta['active'].sum())} "
              f"posK2={int(y2[meta['active']].sum())} dim={X.shape[1]} "
              f"time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
