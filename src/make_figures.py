# -*- coding: utf-8 -*-
# Author: Liao Bin (廖彬)
# Created: 2026-10-24
# Updated: 2026-11-01
"""All figures (English labels, Okabe-Ito colour-blind-safe palette).

Requires the result files produced by run_main / run_compare / run_hparam /
run_robust / run_shap / make_coverage. Some panels also read the per-year test
splits written by run_main; those files are not committed.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

OK = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00",
          purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#8C8C8C")
GC = {"trade": OK["blue"], "graph": OK["green"], "form": OK["orange"],
      "context": OK["purple"]}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")


def load_year_npz(T):
    return np.load(RES / f"test_T{T}.npz", allow_pickle=True)


# ---------------------------------------------------------------- fig 1
def fig_data():
    main = pd.read_csv(RES / "main_results.csv")
    cov = pd.read_csv(RES / "coverage.csv")
    years = sorted(main["T"].unique())
    units = [main[main["T"] == y].n_units.iloc[0] for y in years]
    npos = [main[main["T"] == y].n_pos.iloc[0] for y in years]

    fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.6))
    ax = axs[0, 0]
    ax.bar(years, np.array(units) / 1000, color=OK["sky"], width=0.65,
           label="candidate units")
    ax.set_ylabel("candidate units (thousand)")
    ax2 = ax.twinx()
    ax2.plot(years, npos, "o-", color=OK["verm"], lw=1.2)
    ax2.set_ylabel("positive transitions", color=OK["verm"])
    ax2.tick_params(axis="y", colors=OK["verm"])
    ax2.spines["right"].set_visible(True)
    ax.set_title("Candidate set and realised transitions")
    panel(ax, "a")

    ax = axs[0, 1]
    z = load_year_npz(2018)
    v = z["volume"]; y = z["y"]
    bins = np.linspace(0, np.log1p(v.max()), 40)
    ax.hist(np.log1p(v[y == 0]), bins=bins, density=True, color=OK["grey"],
            alpha=0.55, label="no transition")
    ax.hist(np.log1p(v[y == 1]), bins=bins, density=True, color=OK["verm"],
            alpha=0.6, label="transition")
    ax.set_xlabel("log(1 + records in past 3 years)")
    ax.set_ylabel("density")
    ax.set_title("Unit trade volume, test year 2018")
    ax.legend(frameon=False)
    panel(ax, "b")

    ax = axs[1, 0]
    ax.bar(cov["T"], cov.share, color=OK["green"], width=0.65)
    ax.axhline(cov.share.mean(), color=OK["verm"], ls="--", lw=0.9)
    ax.text(2017.2, cov.share.mean() + 0.05,
            f"mean {cov.share.mean():.2f}", color=OK["verm"], fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("share of transitions covered")
    ax.set_title("Coverage of the active candidate universe")
    panel(ax, "c")

    ax = axs[1, 1]
    for K, col in ((1, OK["blue"]), (2, OK["orange"]), (5, OK["purple"])):
        cnt = []
        for T in years:
            _, meta = D.load_year(T)
            yy, _ = D.make_labels(T, meta, K=K, horizon=1)
            cnt.append(int(yy.sum()))
        ax.plot(years, cnt, "o-", color=col, label=f"$K$={K}")
    ax.set_ylabel("positive units")
    ax.set_title("Label sensitivity to the record threshold")
    ax.legend(frameon=False)
    panel(ax, "d")
    for ax in axs.ravel():
        ax.set_xlabel("test year") if ax is axs[1, 0] or ax is axs[1, 1] else None
    save(fig, "fig01_data")


# ---------------------------------------------------------------- fig 2
def fig_label():
    lad = pd.read_csv(D.LADDER)
    top = pd.read_csv(RES / "top_terms.csv").head(15)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.9),
                            gridspec_kw={"width_ratios": [1.05, 1.25]})
    ax = axs[0]
    lc = {"S0": OK["sky"], "S1": OK["green"], "S2": OK["orange"], "S3": OK["verm"]}
    order = top.iloc[::-1]
    ax.barh(order.Term, order.records / 1e6,
            color=[lc.get(l, OK["grey"]) for l in order.ladder], height=0.72)
    ax.set_xlabel("records 1975-2024 (million)")
    ax.set_title("Most traded terms and their ladder")
    handles = [Patch(color=lc[k], label=f"{k} (n={int((lad.ladder==k).sum())})")
               for k in lc]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    panel(ax, "a")

    ax = axs[1]
    z = load_year_npz(2024)
    y = z["y"]; score = z["score"]; best = z["best"]; u = z["u_rec"]
    cand = np.where((y == 1) & (best == 1))[0]
    if len(cand) == 0:
        cand = np.where(y == 1)[0]
    i = cand[np.argmax(u[cand])]
    cu, cv, ts = int(z["cu"][i]), int(z["cv"][i]), int(z["ts"][i])
    B = D.load_base(); L = B["L"]; rk = B["rank_of_code"]
    sel = L[(L.cu == cu) & (L.cv == cv) & (L.ts == ts)].copy()
    sel["rank"] = rk[sel.tk.to_numpy()]
    g = (sel[sel["rank"] >= 0].groupby(["Year", "rank"]).n_rec.sum()
         .unstack(fill_value=0))
    years = np.arange(int(g.index.min()), 2025)
    g = g.reindex(years, fill_value=0)
    # 逐年堆叠柱：每根柱是该年记录按阶梯的构成，柱高即记录数
    bottom = np.zeros(len(years))
    for r, col, lab in ((0, OK["sky"], "S0 live"), (1, OK["green"], "S1 raw"),
                        (2, OK["orange"], "S2 semi"), (3, OK["verm"], "S3 finished")):
        v = g[r].to_numpy() if r in g.columns else np.zeros(len(years))
        if v.sum() == 0:
            continue
        ax.bar(years, v, bottom=bottom, width=0.78, color=col, label=lab,
               edgecolor="white", linewidth=0.3, zorder=3)
        bottom += v
    top = float(bottom.max())
    ax.axvspan(2023.6, 2024.6, color="0.90", zorder=0)
    ax.axvline(2023.5, color=OK["grey"], ls="--", lw=0.8, zorder=4)
    ax.set_ylim(0, top * 1.78)
    n24 = int(bottom[-1])
    n2_24 = int(g.loc[2024, 2]) if 2 in g.columns else 0
    n0_24 = n24 - n2_24 if 2 in g.columns else n24
    ax.text(1987, top * 1.66, "dashed line: features use Year < 2024", color="0.35",
            fontsize=6.9, ha="left", va="center")
    ax.annotate(f"first S2 record in 2024\n{n0_24} live + {n2_24} extract (K=2)",
                xy=(2024, n24 - n2_24 / 2), xytext=(1987, top * 1.40),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=OK["verm"],
                                connectionstyle="arc3,rad=-0.22"),
                color=OK["verm"], fontsize=7.1, ha="left", va="center")
    ax.set_xticks([1990, 2000, 2010, 2024])
    ax.set_xlim(1986, 2025.2)
    name = list(B["TV"])[ts]
    exp, imp = list(B["CV"])[cu], list(B["CV"])[cv]
    ax.set_title(f"{name}  {exp}->{imp}", fontsize=7.5)
    ax.set_xlabel("year")
    ax.set_ylabel("records per year")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.33, 0.64),
              ncol=1, fontsize=6.3, handlelength=1.1, labelspacing=0.22)
    panel(ax, "b")
    save(fig, "fig02_label")


# ---------------------------------------------------------------- fig 3
def fig_main():
    main = pd.read_csv(RES / "main_results.csv")
    avg = main.groupby("B")[["model", "volume", "form"]].mean()
    sem = main.groupby("B")[["model", "volume", "form"]].sem()
    rnd = main.groupby("B")["random"].mean()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for name, col, lab in (("model", OK["blue"], "TopB ranker"),
                           ("form", OK["orange"], "form history"),
                           ("volume", OK["grey"], "trade volume")):
        ax.plot(avg.index, avg[name], "o-", color=col, label=lab)
        ax.fill_between(avg.index, avg[name] - sem[name], avg[name] + sem[name],
                        color=col, alpha=0.16, lw=0)
    ax.plot(rnd.index, rnd.values, "s--", color=OK["verm"], label="random expectation")
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1000"])
    ax.set_xlabel("audit budget $B$")
    ax.set_ylabel("transitions found (Hit@$B$)")
    ax.set_title("Mean over 2017-2024, error band $\\pm$1 SEM")
    ax.legend(frameon=False)
    for B in (100, 1000):
        ax.annotate(f"{avg.loc[B,'model']:.1f}", (B, avg.loc[B, "model"]),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=7, color=OK["blue"])
    save(fig, "fig03_main")


# ---------------------------------------------------------------- fig 4
def fig_recall_precision():
    main = pd.read_csv(RES / "main_results.csv")
    avg = main.groupby("B")[["recall", "precision", "model"]].mean()
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, col, lab, letter in ((axs[0], "recall", "Recall@$B$", "a"),
                                 (axs[1], "precision", "Precision@$B$", "b")):
        ax.plot(avg.index, avg[col], "o-", color=OK["blue"])
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 200, 500, 1000])
        ax.set_xticklabels(["20", "50", "100", "200", "500", "1000"])
        ax.set_xlabel("audit budget $B$")
        ax.set_title(lab)
        panel(ax, letter)
    axs[0].set_ylabel("mean share of transitions found")
    axs[1].set_ylabel("mean hit rate")
    save(fig, "fig04_recall_precision")


# ---------------------------------------------------------------- fig 5
def fig_shortlist():
    budgets = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    main = pd.read_csv(RES / "main_results.csv")
    nu = main.groupby("T").n_units.first()
    cap = {"model": [], "volume": [], "form": []}
    pct = {"model": [], "volume": []}
    for T in range(2017, 2025):
        z = load_year_npz(T)
        y = z["y"]; npos = y.sum()
        score = z["score"]; vol = z["volume"]; form = z["formhist"]; rec = z["u_rec"]
        orders = {"model": np.argsort(-score),
                  "volume": np.lexsort((-rec, -vol)),
                  "form": np.lexsort((-vol, -form))}
        for k, order in orders.items():
            for b in budgets:
                cap[k].append(y[order[:b]].sum() / max(npos, 1))
        for k in ("model", "volume"):
            order = orders[k]
            ranks = np.empty(len(y), dtype=np.int64)
            ranks[order] = np.arange(1, len(y) + 1)
            pct[k].append(ranks[y == 1] / len(y))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
    ax = axs[0]
    for k, col, lab in (("model", OK["blue"], "TopB ranker"),
                        ("form", OK["orange"], "form history"),
                        ("volume", OK["grey"], "trade volume")):
        v = np.array(cap[k]).reshape(len(range(2017, 2025)), len(budgets)).mean(0)
        ax.plot(budgets, v, "o-", color=col, label=lab)
    ax.axvline(100, color=OK["verm"], ls=":", lw=0.8)
    ax.text(106, 0.62, "$B$=100", color=OK["verm"], fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlim(15, 25000)
    ax.set_xlabel("audit budget $B$")
    ax.set_ylabel("share of transitions captured")
    ax.set_title("Capture curve")
    ax.legend(frameon=False, loc="upper left")
    panel(ax, "a")
    ax = axs[1]
    for k, col, lab in (("model", OK["blue"], "TopB ranker"),
                        ("volume", OK["grey"], "trade volume")):
        v = np.concatenate(pct[k])
        xs = np.sort(v)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, label=lab)
    for q, ytxt in ((500, 0.88), (1000, 0.70), (5000, 0.52)):
        frac = float(np.mean([q / n for n in nu.values]))
        ax.axvline(frac, color=OK["grey"], ls=":", lw=0.7)
        ax.text(frac * 1.06, ytxt, f"$B$={q}", color=OK["grey"], fontsize=6.2,
                rotation=90, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    ax.set_xscale("log")
    ax.set_xlabel("percentile rank of a realised transition")
    ax.set_ylabel("ECDF")
    ax.set_title("Where transitions sit in each ranking")
    ax.legend(frameon=False, loc="upper left")
    panel(ax, "b")
    save(fig, "fig05_shortlist")


# ---------------------------------------------------------------- fig 6
def fig_ablation():
    df = pd.read_csv(RES / "ablation.csv")
    names = ["full", "no_trade", "no_graph", "no_form", "no_context",
             "unit_only", "no_next", "form_unit_only"]
    labels = ["Full", "No trade", "No graph", "No form", "No context",
              "Unit only", "No next-rung", "Form unit only"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    xs = np.arange(len(names))
    for j, (B, col) in enumerate(((100, OK["blue"]), (500, OK["orange"]),
                                  (1000, OK["green"]))):
        m = [df[(df.name == n) & (df.B == B)].hit.mean() for n in names]
        s = [df[(df.name == n) & (df.B == B)].hit.sem() for n in names]
        ax.bar(xs + (j - 1) * 0.26, m, width=0.24, color=col, label=f"$B$={B}",
               yerr=s, capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel("mean transitions found")
    ax.set_ylim(0, 70)
    ax.set_title("Feature-group and aggregation ablations")
    ax.legend(frameon=False, ncol=3, fontsize=5.8, loc="upper left")
    panel(ax, "a")
    ax = axs[1]
    nf = [df[df.name == n].n_feat.iloc[0] if len(df[df.name == n]) else 0
          for n in names]
    ax.barh(np.arange(len(names)), nf, color=OK["sky"], height=0.62)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("number of features")
    ax.set_title("Input dimension")
    panel(ax, "b")
    save(fig, "fig06_ablation")


# ---------------------------------------------------------------- fig 7
def fig_models_hparam():
    df = pd.read_csv(RES / "model_compare.csv")
    hp = pd.read_csv(RES / "hparam_grid.csv")
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 3.1),
                            gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axs[0]
    names = ["logreg", "lgb_bce", "xgb_rank", "cat_rank", "lgb_rank"]
    labels = ["LogReg", "LGB BCE", "XGB rank", "CatBoost", "TopB ranker"]
    xs = np.arange(len(names))
    for j, (B, col) in enumerate(((100, OK["blue"]), (500, OK["orange"]),
                                  (1000, OK["green"]))):
        m = [df[(df.name == n) & (df.B == B)].hit.mean() for n in names]
        s = [df[(df.name == n) & (df.B == B)].hit.sem() for n in names]
        ax.bar(xs + (j - 1) * 0.26, m, width=0.24, color=col, label=f"$B$={B}",
               yerr=s, capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("mean transitions found")
    ax.set_ylim(0, 68)
    ax.set_title("Model comparison")
    ax.legend(frameon=False, fontsize=6)
    panel(ax, "a")

    for ax, knob, row, col, xl, yl in (
            (axs[1], "trees_lr", "trees", "lr", "trees", "learning rate"),
            (axs[2], "leaves_sampling", "leaves", "sampling", "leaves", "sampling")):
        sub = hp[(hp.knob == knob) & (hp["T"] == 2018)]
        piv = sub.pivot(index=col, columns=row, values="hit100")
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"Hit@100, 2018 ({knob.replace('_', ' x ')})", fontsize=7)
        for a in range(piv.shape[0]):
            for b in range(piv.shape[1]):
                val = piv.values[a, b]
                tc = "black" if im.norm(val) > 0.62 else "white"
                ax.text(b, a, int(val), ha="center", va="center",
                        fontsize=6, color=tc)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    panel(axs[1], "b", x=-0.22, y=1.12); panel(axs[2], "c", x=-0.22, y=1.12)
    save(fig, "fig07_models_hparam")


# ---------------------------------------------------------------- fig 8
def fig_shap():
    imp = pd.read_csv(RES / "shap_importance.csv")
    z = np.load(RES / "shap_values.npz")
    sv, base, data, score, y = z["values"], z["base"], z["data"], z["score"], z["y"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.6),
                            gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axs[0]
    top = imp.sort_values("mean_abs").tail(12)
    ax.barh(top.feature, top.mean_abs, color=[GC[g] for g in top.group], height=0.7)
    ax.set_xlabel("mean |SHAP| (raw score)")
    ax.set_title("Global attribution")
    handles = [Patch(color=GC[g], label=g) for g in ("trade", "graph", "form", "context")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=6)
    panel(ax, "a")

    ax = axs[1]
    pos = np.where(y == 1)[0]
    i = pos[np.argmax(score[pos])] if len(pos) else int(np.argmax(score))
    v = sv[i]; order = np.argsort(-np.abs(v))[:9]
    names = np.array(D.FEATNAMES)[order].tolist()
    vals = v[order].tolist()
    # 其余维合并成一根残差条，保证条形累加恰好落到最终分数
    names.append(f"other {len(v) - len(order)} features")
    vals.append(float(v.sum() - v[order].sum()))
    names, vals = np.array(names), np.array(vals)
    left = base[i]
    for j, (n, vv) in enumerate(zip(names[::-1], vals[::-1])):
        col = OK["grey"] if str(n).startswith("other ") else (
            OK["verm"] if vv > 0 else OK["blue"])
        ax.barh(j, vv, left=left, color=col, height=0.66)
        left += vv
    ax.axvline(base[i], color=OK["grey"], ls="--", lw=0.7)
    ax.axvline(score[i], color="black", lw=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1], fontsize=6.5)
    ax.set_xlabel("score contribution")
    ax.set_title("Top-scored realised transition (2024)")
    panel(ax, "b", x=-0.22, y=1.12)
    save(fig, "fig08_shap")


# ---------------------------------------------------------------- fig 9
def fig_robust():
    seeds = pd.read_csv(RES / "robust_seeds.csv")
    lab = pd.read_csv(RES / "robust_label.csv")
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.9))
    ax = axs[0]
    cfgs = ["single_s0", "single_s1", "single_s2", "single_s3", "single_s4", "ens5"]
    labels = ["s0", "s1", "s2", "s3", "s4", "5-seed"]
    m = [seeds[(seeds.cfg == c) & (seeds.B == 100)].hit.mean() for c in cfgs]
    s = [seeds[(seeds.cfg == c) & (seeds.B == 100)].hit.sem() for c in cfgs]
    ax.bar(range(len(cfgs)), m, color=[OK["sky"]] * 5 + [OK["blue"]], yerr=s,
           capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(labels)
    ax.set_ylabel("Hit@100 (mean over years)")
    ax.set_title("Seed robustness")
    panel(ax, "a")
    ax = axs[1]
    for j, (cfg, lab_) in enumerate((("K1", "$K$=1"), ("K2", "$K$=2"),
                                     ("K5", "$K$=5"))):
        if cfg == "K2":
            main = pd.read_csv(RES / "main_results.csv")
            m = main[main.B == 100].model.mean()
            s = main[main.B == 100].model.sem()
            np_ = main[main.B == 100].n_pos.mean()
        else:
            m = lab[(lab.cfg == cfg) & (lab.B == 100)].hit.mean()
            s = lab[(lab.cfg == cfg) & (lab.B == 100)].hit.sem()
            np_ = lab[(lab.cfg == cfg) & (lab.B == 100)].n_pos.mean()
        ax.bar(j, m, color=OK["orange"], yerr=s, capsize=1.5,
               error_kw=dict(lw=0.6))
        ax.text(j, m + s + 0.4, f"{np_:.0f}", ha="center", fontsize=6,
                color=OK["grey"],
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax.set_ylim(0, 18)
    ax.set_xticks(range(3)); ax.set_xticklabels(["$K$=1", "$K$=2", "$K$=5"])
    ax.set_ylabel("Hit@100")
    ax.set_title("Record threshold")
    panel(ax, "b", x=-0.24, y=1.10)
    ax = axs[2]
    m1 = pd.read_csv(RES / "main_results.csv")
    v1 = m1[m1.B == 100].model.mean()
    v2 = lab[(lab.cfg == "w2") & (lab.B == 100)].hit.mean()
    s2 = lab[(lab.cfg == "w2") & (lab.B == 100)].hit.sem()
    ax.bar([0, 1], [v1, v2], color=[OK["green"], OK["purple"]],
           yerr=[m1[m1.B == 100].model.sem(), s2], capsize=1.5,
           error_kw=dict(lw=0.6))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["1-year", "2-year"])
    ax.set_ylabel("Hit@100")
    ax.set_title("Label window")
    panel(ax, "c", x=-0.24, y=1.10)
    save(fig, "fig09_robust")


# ---------------------------------------------------------------- appendix
def fig_complexity():
    df = pd.read_csv(RES / "model_compare.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for name, col, lab in (("logreg", OK["grey"], "LogReg"),
                           ("lgb_bce", OK["orange"], "LGB BCE"),
                           ("lgb_rank", OK["blue"], "TopB ranker"),
                           ("xgb_rank", OK["green"], "XGB rank"),
                           ("cat_rank", OK["purple"], "CatBoost")):
        sub = df[df.name == name].groupby("T").fit_seconds.mean()
        n = df[df.name == name].groupby("T").n_train.mean()
        ax.plot(n, sub, "o-", color=col, label=lab)
    ax.set_xlabel("training rows")
    ax.set_ylabel("fit time (s)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("Training cost by model")
    save(fig, "fig10_complexity")


def fig_perm():
    df = pd.read_csv(RES / "perm_importance.csv").sort_values("drop").tail(15)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    g = pd.read_csv(RES / "shap_importance.csv").set_index("feature")
    ax.barh(df.feature, df["drop"], color=[GC.get(g.loc[f, "group"], OK["grey"])
                                           for f in df.feature], height=0.7)
    ax.set_xlabel("Hit@100 drop after permutation")
    ax.set_title("Permutation importance (2024)")
    save(fig, "fig12_permutation")


def fig_failure():
    z = load_year_npz(2024)
    y, score = z["y"], z["score"]
    ranks = np.empty(len(y), dtype=np.int64)
    ranks[np.argsort(-score)] = np.arange(1, len(y) + 1)
    pr = ranks[y == 1]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axs[0]
    ax.hist(np.log10(pr), bins=25, color=OK["blue"], alpha=0.85)
    for b in (100, 1000):
        ax.axvline(np.log10(b), color=OK["verm"], ls="--", lw=0.8)
        ax.text(np.log10(b) + 0.03, ax.get_ylim()[1] * 0.85, f"B={b}",
                color=OK["verm"], fontsize=6.5)
    ax.set_xlabel("log10 rank of realised transition")
    ax.set_ylabel("count")
    ax.set_title("Rank distribution of 2024 positives")
    panel(ax, "a")
    ax = axs[1]
    top = np.argsort(-score)[:100]
    ax.hist(score[top][y[top] == 0], bins=20, color=OK["grey"], alpha=0.7,
            label="top-100 non-events")
    ax.hist(score[top][y[top] == 1], bins=20, color=OK["verm"], alpha=0.7,
            label="top-100 events")
    ax.set_xlabel("raw score")
    ax.set_ylabel("count")
    ax.set_title("Score overlap in the audit list")
    ax.legend(frameon=False, fontsize=6)
    panel(ax, "b")
    save(fig, "fig13_failure")


def fig_pca():
    from sklearn.decomposition import PCA
    z = load_year_npz(2024)
    X, meta = D.load_year(2024)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(20000, len(X)), replace=False)
    Z = PCA(n_components=2, random_state=0).fit_transform(
        (X[idx] - X[idx].mean(0)) / (X[idx].std(0) + 1e-9))
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.scatter(Z[:, 0], Z[:, 1], s=2, c=OK["grey"], alpha=0.25, label="no transition")
    y = z["y"][idx]
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=9, c=OK["verm"], alpha=0.85,
               label="transition")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Feature space, 2024 (20k sample)")
    ax.legend(frameon=False, fontsize=6, markerscale=2)
    save(fig, "fig14_pca")


def fig_species_form():
    B = D.load_base(); L = B["L"]; rk = B["rank_of_code"]; TV = B["TV"]
    s23codes = np.where(rk >= 2)[0]
    rows = []
    for T in range(2017, 2025):
        z = load_year_npz(T)
        for i in np.where(z["y"] == 1)[0]:
            rows.append((T, int(z["ts"][i])))
    d = pd.DataFrame(rows, columns=["T", "ts"]).groupby("ts").size().sort_values(
        ascending=False).head(14)
    taxa = d.index.to_numpy()
    terms = (L[L.tk.isin(s23codes) & (L.Year >= 2017)]
             .groupby("Term").n_rec.sum().sort_values(ascending=False).head(12).index)
    M = np.zeros((len(taxa), len(terms)))
    for a, ts in enumerate(taxa):
        sub = L[(L.ts == ts) & (L.Year >= 2017)]
        for b, tm in enumerate(terms):
            M[a, b] = sub[sub.Term == tm].n_rec.sum()
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    im = ax.imshow(np.log1p(M), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=40, ha="right", fontsize=6)
    ax.set_yticks(range(len(taxa)))
    ax.set_yticklabels([list(TV)[t] for t in taxa], fontsize=6, style="italic")
    ax.set_title("Higher-ladder trade of frequent transition taxa")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="log(1 + records)")
    save(fig, "fig11_species_form")


def fig_cases():
    T = 2024
    X, meta = D.load_year(T)
    y, rec = D.make_labels(T, meta, 2, 1)
    z = np.load(RES / "shap_values.npz")
    sv, base, score = z["values"], z["base"], z["score"]
    assert len(score) == len(y)

    B = D.load_base(); L = B["L"]; rk = B["rank_of_code"]; TV, CV = B["TV"], B["CV"]
    order = np.argsort(-score)
    rank = np.empty(len(score), dtype=np.int64)
    rank[order] = np.arange(1, len(score) + 1)
    pos = order[y[order] == 1]
    missed = [i for i in np.argsort(score) if y[i] == 1]
    cases = [("hit", int(pos[1])), ("hit", int(pos[2])), ("miss", int(missed[0]))]

    fig, axs = plt.subplots(3, 1, figsize=(6.6, 4.9))
    for ax, (kind, i), letter in zip(axs, cases, "abc"):
        cu, cv, ts = int(meta["cu"][i]), int(meta["cv"][i]), int(meta["ts"][i])
        sel = L[(L.cu == cu) & (L.cv == cv) & (L.ts == ts) & (L.Year == T)].copy()
        sel["rank"] = rk[sel.tk.to_numpy()]
        new = int(sel["rank"].max()) if len(sel) else -1
        v = sv[i]
        o = np.argsort(-np.abs(v))[:8][::-1]
        names = [D.FEATNAMES[j] for j in o]
        vals = v[o]
        ax.barh(range(len(vals)), vals, height=0.66,
                color=[OK["verm"] if t > 0 else OK["blue"] for t in vals])
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=6.8)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("SHAP contribution (raw score)" if letter == "c" else "")
        ax.set_title(f"{kind}: {list(TV)[ts]} {list(CV)[cu]}->{list(CV)[cv]}   "
                     f"rank {int(rank[i])}   S{int(meta['best'][i])}->S{new}   "
                     f"score {float(score[i]):+.2f} (base {float(base[i]):+.2f})",
                     fontsize=7.6, loc="left")
        panel(ax, letter, x=-0.30, y=1.05)
    handles = [Patch(color=OK["verm"], label="raises the score"),
               Patch(color=OK["blue"], label="lowers the score")]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    save(fig, "fig15_cases")


def fig_seed_yearly():
    df = pd.read_csv(RES / "per_seed_hits.csv")
    df = df[df.B == 100]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for seed, col in zip(sorted(df.seed.unique()),
                         (OK["sky"], OK["green"], OK["purple"])):
        sub = df[df.seed == seed].sort_values("T")
        ax.plot(sub["T"], sub.hits, "o-", color=col, lw=0.9, ms=3,
                label=f"seed {seed}")
    mean = df.groupby("T").hits.mean()
    ax.plot(mean.index, mean.values, "s--", color=OK["verm"], lw=1.1,
            label="3-seed mean")
    ax.set_xlabel("test year")
    ax.set_ylabel("Hit@100")
    ax.set_title("Per-seed variability by test year")
    ax.legend(frameon=False, fontsize=6.2, ncol=2)
    save(fig, "fig16_seed_yearly")


def fig_universe():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    years = list(range(2017, 2025))
    comp = {r: [] for r in (0, 1, 2)}
    for T in years:
        z = load_year_npz(T)
        b = z["best"]
        for r in comp:
            comp[r].append(int((b == r).sum()))
    bottom = np.zeros(len(years))
    cols = {0: OK["sky"], 1: OK["green"], 2: OK["orange"]}
    for r in (0, 1, 2):
        ax.bar(years, np.array(comp[r]) / 1000, bottom=bottom,
               color=cols[r], width=0.65, label=f"historical best S{r}")
        bottom += np.array(comp[r]) / 1000
    ax.set_ylabel("candidate units (thousand)")
    ax.set_xlabel("test year")
    ax.set_title("Composition of the eligible universe")
    ax.legend(frameon=False, fontsize=6.2)
    save(fig, "fig17_universe")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"data": fig_data, "label": fig_label, "main": fig_main,
           "rp": fig_recall_precision, "short": fig_shortlist,
           "abl": fig_ablation, "models": fig_models_hparam, "shap": fig_shap,
           "robust": fig_robust, "complex": fig_complexity, "perm": fig_perm,
           "failure": fig_failure, "pca": fig_pca, "species": fig_species_form,
           "cases": fig_cases, "seedyear": fig_seed_yearly,
           "universe": fig_universe}
    for k, f in fns.items():
        if which in ("all", k):
            try:
                f()
            except Exception as e:  # keep going, report which one failed
                print("FAILED", k, type(e).__name__, e, flush=True)
