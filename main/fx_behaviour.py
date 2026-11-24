"""
fx_behaviour.py — functions of notebook 03_behaviour

Code moved out of the notebook cells without changing what it computes. Nothing here saves anything.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import fx_common as fxc

OUTS = ["correct", "early", "late"]
PHASES = ["start", "middle", "end"]


def sign_flip_p(values, n_perm=10000, seed=0):
    """Mean of the values and its two-sided sign-flip p (each value's sign flipped at random). Same as functions.sign_flip_p."""
    import functions as fn
    return fn.sign_flip_p(values, n_perm=n_perm, seed=seed)


# ── helper used by B1–B3 ──────────────────────────────────────────────────────────────────
def paired_test(df, value, cond_col, a, b, min_n=5, agg="median"):
    """Per session: median (or mean) of `value` in condition a and b (>= min_n trials each);
    Wilcoxon signed-rank across sessions, per rat and all rats. Returns (table, per-session values)."""
    g = df.groupby(["rat", "session", cond_col])[value].agg([agg, "size"]).reset_index()
    w = g.pivot_table(index=["rat", "session"], columns=cond_col, values=[agg, "size"])
    ok = (w[("size", a)] >= min_n) & (w[("size", b)] >= min_n)
    m = w.loc[ok, agg][[a, b]].reset_index()
    out = []
    for rat, d in list(m.groupby("rat")) + [("all", m)]:
        diff = d[b] - d[a]
        p = wilcoxon(d[a], d[b]).pvalue if len(d) >= 5 else np.nan
        out.append({"rat": rat, "n_sessions": len(d), f"median {a}": round(d[a].median(), 3), f"median {b}": round(d[b].median(), 3),
                    "median diff (b-a)": round(diff.median(), 3), "sessions b>a": int((diff > 0).sum()),
                    "wilcoxon p": round(p, 4) if p == p else np.nan})
    return pd.DataFrame(out), m


def _mixed(formula, d):
    import statsmodels.formula.api as smf
    return smf.mixedlm(formula, d, groups=d["session"]).fit(reml=True, method=["lbfgs", "powell", "nm"], maxiter=2000)


# ── B1 post-error slowing ─────────────────────────────────────────────────────────────────
def b1_post_error_slowing(K):
    """Correct trials; RT (ms) ~ after_late + after_early + long (+ rat), random intercept per session (main);
    per-session median RT after correct vs after error, Wilcoxon (check). Returns (lmm table, wilcoxon table)."""
    c = K[(K["label"] == "correct") & K["prev_outcome"].isin(["correct", "early", "late"])].copy()
    c["rt_ms"] = 1000 * c["rt_s"]
    c["after_late"] = (c["prev_outcome"] == "late").astype(int)
    c["after_early"] = (c["prev_outcome"] == "early").astype(int)
    c["long"] = (c["type"] == "long").astype(int)
    print("correct trials by previous outcome:", c["prev_outcome"].value_counts().to_dict())
    rows = []
    for rat, d in list(c.groupby("rat")) + [("all", c)]:
        m = _mixed("rt_ms ~ after_late + after_early + long" + (" + C(rat)" if rat == "all" else ""), d)
        print(f"rat {rat}: converged = {m.converged} | session variance = {float(m.cov_re.iloc[0, 0]):.1f}")
        ci = m.conf_int()
        for term in ["after_late", "after_early"]:
            rows.append({"rat": rat, "effect": term, "n_trials": len(d), "n_sessions": d["session"].nunique(),
                         "delta_ms": round(m.params[term], 1), "ci_low": round(ci.loc[term, 0], 1), "ci_high": round(ci.loc[term, 1], 1),
                         "p": m.pvalues[term]})
    lmm = pd.DataFrame(rows)
    print("\nB1 mixed model (difference to 'after correct', ms):"); print(lmm.to_string(index=False, formatters={"p": "{:.2g}".format}))
    c["prev_err"] = np.where(c["prev_outcome"] == "correct", "after correct", "after error")
    wil, _ = paired_test(c, "rt_s", "prev_err", "after correct", "after error")
    print("\nB1 check, per-session median RT (s), Wilcoxon:"); print(wil.to_string(index=False))
    return lmm, wil


# ── B2 early releases after an error ──────────────────────────────────────────────────────
def b2_early_after_error(K):
    """Per-session share of early releases after correct vs after error, Wilcoxon (main);
    logistic GEE early ~ after_late + after_early + long (+ rat), sessions as clusters (check). Returns (wilcoxon, gee)."""
    import statsmodels.api as sm, statsmodels.formula.api as smf
    d = K[K["prev_outcome"].isin(["correct", "early", "late"])].copy()
    d["is_early"] = (d["label"] == "early").astype(int)
    d["prev_err"] = np.where(d["prev_outcome"] == "correct", "after correct", "after error")
    d["after_late"] = (d["prev_outcome"] == "late").astype(int)
    d["after_early"] = (d["prev_outcome"] == "early").astype(int)
    d["long"] = (d["type"] == "long").astype(int)
    print("trials by previous outcome:", d["prev_outcome"].value_counts().to_dict())
    print("share early after correct / after error (all trials pooled): "
          f"{d.loc[d.prev_err == 'after correct', 'is_early'].mean():.3f} / {d.loc[d.prev_err == 'after error', 'is_early'].mean():.3f}")
    wil, _ = paired_test(d, "is_early", "prev_err", "after correct", "after error", min_n=10, agg="mean")
    print("\nB2 share of early releases per session (median over sessions), after correct vs after error:"); print(wil.to_string(index=False))
    rows = []
    for rat, x in list(d.groupby("rat")) + [("all", d)]:
        g = smf.gee("is_early ~ after_late + after_early + long" + (" + C(rat)" if rat == "all" else ""), groups="session", data=x,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = g.conf_int()
        for term in ["after_late", "after_early"]:
            rows.append({"rat": rat, "effect": term, "n_trials": len(x), "n_sessions": x["session"].nunique(),
                         "odds_ratio": round(np.exp(g.params[term]), 2), "ci_low": round(np.exp(ci.loc[term, 0]), 2),
                         "ci_high": round(np.exp(ci.loc[term, 1]), 2), "p": g.pvalues[term]})
    gee = pd.DataFrame(rows)
    print("\nB2 check, trial-level GEE (odds ratio vs after correct; < 1 = fewer early releases):")
    print(gee.to_string(index=False, formatters={"p": "{:.2g}".format}))
    return wil, gee


# ── B3 latency to press ───────────────────────────────────────────────────────────────────
def b3_latency_to_press(K):
    """Latency to press before early vs before correct: per-session medians, Wilcoxon (main);
    mixed model log10(latency) ~ early + long (+ rat) (check). Returns (wilcoxon, lmm)."""
    d = K[K["label"].isin(["correct", "early"]) & K["latency_to_press_s"].notna()].copy()
    print(f"trials: {len(d)} | latency (s) min {d['latency_to_press_s'].min():.2f}, median {d['latency_to_press_s'].median():.2f}, "
          f"95th pct {d['latency_to_press_s'].quantile(.95):.1f}, max {d['latency_to_press_s'].max():.0f} | <= 0: {(d['latency_to_press_s'] <= 0).sum()}")
    d = d[d["latency_to_press_s"] > 0]
    wil, _ = paired_test(d, "latency_to_press_s", "label", "correct", "early")
    print("\nB3 median latency to press before correct vs before early trials (s):"); print(wil.to_string(index=False))
    d["log_lat"] = np.log10(d["latency_to_press_s"]); d["early"] = (d["label"] == "early").astype(int); d["long"] = (d["type"] == "long").astype(int)
    rows = []
    for rat, x in list(d.groupby("rat")) + [("all", d)]:
        m = _mixed("log_lat ~ early + long" + (" + C(rat)" if rat == "all" else ""), x); ci = m.conf_int()
        rows.append({"rat": rat, "n_trials": len(x), "n_sessions": x["session"].nunique(), "converged": m.converged,
                     "ratio_early_vs_correct": round(10 ** m.params["early"], 3), "ci_low": round(10 ** ci.loc["early", 0], 3),
                     "ci_high": round(10 ** ci.loc["early", 1], 3), "p": m.pvalues["early"]})
    lmm = pd.DataFrame(rows)
    print("\nB3 check, trial-level mixed model (ratio of latency before early / before correct; < 1 = faster before early):")
    print(lmm.to_string(index=False, formatters={"p": "{:.2g}".format}))
    return wil, lmm


# ── label checks ──────────────────────────────────────────────────────────────────────────
def label_checks(TR, K, fast_rt=0.12):
    """Six consistency checks of the trial labels (all trials and kept trials) + the number of very fast correct responses."""
    A = TR.copy(); A["hold_s"] = A["release"] - A["press"]
    print("1. max |hold_duration − (release − press)| (s):", round((A["hold_duration"] - A["hold_s"]).abs().max(), 4))
    touch = A["is_touch"].astype(bool)
    print("\n2. touch trials:", touch.sum(), "| of them labelled early in trials.csv:", (touch & (A["early"] == 1)).sum())
    print("   early label (ours) with hold < 0.1 s:", ((A["label"] == "early") & (A["hold_s"] < 0.1)).sum())
    print(pd.crosstab(touch, A["hold_s"] < 0.1, rownames=["is_touch"], colnames=["hold < 0.1 s"]).to_string())
    L = pd.DataFrame({"touch": touch, "correct": A["correct"] == 1, "early": A["early"] == 1, "late": A["late"] == 1})
    combo = L.apply(lambda r: " + ".join(c for c in L.columns if r[c]) or "(none)", axis=1)
    print("\n3. label combinations in trials.csv:"); print(combo.value_counts().to_string())
    cl = K[K["label"].isin(["correct", "late"])]
    print("\n4. rt_s (s) of kept correct / late trials:")
    print(cl.groupby(["label", "type"])["rt_s"].describe()[["count", "min", "50%", "max"]].round(3).to_string())
    print("   correct with rt outside 0–0.5 s:", (~cl.loc[cl["label"] == "correct", "rt_s"].between(0, 0.5)).sum(),
          "| late with rt <= 0.5 s:", (cl.loc[cl["label"] == "late", "rt_s"] <= 0.5).sum())
    w = cl.groupby(["session", "label"])["rt_s"].agg(["max", "min"]).unstack("label")
    print("   sessions where correct and late rt overlap:", int((w[("max", "correct")] >= w[("min", "late")]).sum()))
    print("\n5. kept trials, full_hold == 0 by label:"); print(pd.crosstab(K["label"], K["full_hold"] == 0, colnames=["full_hold == 0"]).to_string())
    print("\n6. kept trials with press from TTL:", K["press_from_ttl"].sum(), "| from press_np_s:", (~K["press_from_ttl"]).sum())
    noTTL = K[~K["press_from_ttl"]].groupby(["rat", "session"]).size().sort_values(ascending=False)
    print("   sessions with presses from press_np_s:", len(noTTL)); print(noTTL.head(10).to_string())
    print(f"\n7. kept correct trials with rt < {fast_rt} s:", (K.loc[K["label"] == "correct", "rt_s"] < fast_rt).sum(), "of", (K["label"] == "correct").sum())


# ── performance along the session ─────────────────────────────────────────────────────────
def trials_with_position(TR, final=None):
    """Kept trials with position in the session (rank of the trial / number of trials, computed on ALL trials, before 'keep')."""
    PT = TR.sort_values(["session", "trial_index"]).copy()
    PT["position"] = PT.groupby("session")["trial_index"].rank(pct=True)
    if "keep" in PT.columns: PT = PT[PT["keep"] == True]
    if final is not None: PT = PT[PT["session"].isin(final["session"])]
    PT["rat"] = PT["rat"].astype(str)
    return PT


def outcome_share_phases(TR, cuts=(0.25, 0.85), n_bins=20, blocks_as_6c=False):
    """Long trials: % correct / early / late (each of correct + early + late) per session phase, and start − middle,
    end − middle per session with a sign-flip test. Returns (table, trials used)."""
    PT = trials_with_position(TR)
    A_ = PT[(PT["type"] == "long") & PT["label"].isin(OUTS)].copy()
    if blocks_as_6c: A_ = A_[(A_["long_prob"] == 0.5) | (A_["long_prob"] >= 0.75)]
    A_["pbin"] = pd.cut(A_["position"], np.linspace(0, 1, n_bins + 1), labels=False, include_lowest=True)
    A_["phase"] = pd.cut(A_["position"], [-0.001, *cuts, 1.001], labels=PHASES).astype(str)
    print(f"sessions: {A_.session.nunique()} | long trials: {len(A_)} | {'0.5 and ≥0.75 blocks' if blocks_as_6c else 'all blocks'}")
    other = {"correct": "E+L", "early": "C+L", "late": "C+E"}
    rows = []
    for outc in OUTS:
        X = A_.assign(y=lambda d: 100 * (d.label == outc))
        SPc = X.groupby(["rat", "session", "phase"])["y"].mean().unstack()[PHASES]
        for rat in ["all", "134", "135", "136"]:
            sp = SPc if rat == "all" else SPc[SPc.index.get_level_values("rat") == rat]
            x = X if rat == "all" else X[X.rat == rat]
            r = dict(comparison=outc, rat=rat, sessions=len(sp))
            for ph in PHASES:
                xp = x[x.phase == ph]; r[ph] = sp[ph].mean()
                r[f"n_{ph}"] = f"{int((xp.label == outc).sum())} / {int((xp.label != outc).sum())} {other[outc]}"
            for ph in ("start", "end"):
                m, p, n = sign_flip_p((sp[ph] - sp["middle"]).dropna().values)
                r[f"{ph}_minus_middle"], r[f"p_{ph}"] = m, p
            rows.append(r)
    OS = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        for outc in OUTS:
            print(f"\n===== % {outc} of all long trials (correct + early + late) =====")
            print(OS[OS.comparison == outc].drop(columns="comparison").to_string(index=False))
    return OS, A_


def plot_outcome_share_phases(OS, A_, cuts=(0.25, 0.85), n_bins=20):
    """Rows: % correct, % early, % late, all three together; columns: all rats + each rat (mean ± SEM over sessions)."""
    ylab = {o: f"% {o}\n({o} / correct + early + late)" for o in OUTS}
    ymax = {"late": 50, "early": 90, "correct": 100}
    rats = ["all", "134", "135", "136"]
    fig, axs = plt.subplots(4, 4, figsize=(15, 13), sharex=True, facecolor="white")
    for i, outc in enumerate(OUTS):
        col = fxc.COND[outc]; X = A_.assign(y=lambda d: 100 * (d.label == outc))
        for j, rat in enumerate(rats):
            x = X if rat == "all" else X[X.rat == rat]
            c = x.groupby(["session", "pbin"])["y"].mean().groupby("pbin").agg(["mean", "sem"]).reindex(range(n_bins))
            xc = (c.index + 0.5) / n_bins
            for ax, alone in ((axs[i, j], True), (axs[3, j], False)):
                ax.fill_between(xc, c["mean"] - c["sem"], c["mean"] + c["sem"], color=col, alpha=0.18 if alone else 0.12, lw=0)
                ax.plot(xc, c["mean"], "o-", color=col, lw=2, ms=3, label=outc)
            ax = axs[i, j]; o = OS[(OS.comparison == outc) & (OS.rat == rat)].iloc[0]
            for (a, b), ph in zip([(0, cuts[0]), cuts, (cuts[1], 1)], PHASES):
                ax.text((a + b) / 2, 1.0, f"{ph}\n{o[ph]:.0f} %\n{o[f'n_{ph}']}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="0.3")
            ax.set_ylim(0, ymax[outc])
            if j == 0: ax.set_ylabel(ylab[outc])
            if i == 0: ax.set_title(("all rats" if rat == "all" else fxc.RAT_LABEL.get(rat, f"rat {rat}")) + f" ({x.session.nunique()} sessions)",
                                    fontsize=10, fontweight="bold", pad=40)
    for j in range(4):
        axs[3, j].set_ylim(0, 100); axs[3, j].set_xlabel("Position in session")
        for (a, b), ph in zip([(0, cuts[0]), cuts, (cuts[1], 1)], PHASES):
            axs[3, j].text((a + b) / 2, 1.0, ph, transform=axs[3, j].get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="0.3")
    axs[3, 0].set_ylabel("% of all long trials\n(all three outcomes)")
    axs[3, 3].legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    for ax in axs.flat:
        ax.set_xlim(0, 1)
        for cut in cuts: ax.axvline(cut, color="k", ls="--", lw=1)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.suptitle("Long trials along the session: % correct, % early, % late (each of all long trials); bottom row = all three together",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── outcome counts ────────────────────────────────────────────────────────────────────────
def _counts(y, **keys):
    n3 = int(y["label"].isin(OUTS).sum()); r = dict(**keys, sessions=y["session"].nunique(), total_c_e_l=n3)
    for o in OUTS:
        k = int((y["label"] == o).sum()); r[o] = k; r[f"% {o}"] = 100 * k / max(n3, 1)
    return r


def outcome_counts(TR, final, cuts=(0.25, 0.85)):
    """Number and % of correct / early / late: A per rat and type, B per phase, rat and type, C per phase pooled."""
    PT = trials_with_position(TR, final)
    PT["phase"] = pd.cut(PT["position"], [-0.001, *cuts, 1.001], labels=PHASES).astype(str)
    print(f"sessions: {PT['session'].nunique()} | trials: {len(PT)}")
    gr = ["134", "135", "136", "all"]
    OC = pd.DataFrame([_counts(PT[PT["type"] == t] if r == "all" else PT[(PT["type"] == t) & (PT["rat"] == r)], type=t, rat=r)
                       for t in ("long", "short") for r in gr])
    OCP = pd.DataFrame([_counts((PT if r == "all" else PT[PT.rat == r]).query("type == @t and phase == @ph"), rat=r, type=t, phase=ph)
                        for r in gr for t in ("long", "short") for ph in PHASES])
    OP = pd.DataFrame([_counts(PT[PT["type"].isin(["long", "short"]) & (PT.phase == ph)], phase=ph) for ph in PHASES]).set_index("phase")
    with pd.option_context("display.float_format", lambda x: f"{x:.1f}"):
        for t in ("long", "short"):
            print(f"\n===== A. {t} trials, whole session ====="); print(OC[OC["type"] == t].drop(columns="type").to_string(index=False))
        for t in ("long", "short"):
            print(f"\n===== B. {t} trials per phase ====="); print(OCP[OCP["type"] == t].drop(columns="type").to_string(index=False))
        print("\n===== C. long + short pooled, all rats, per phase ====="); print(OP.to_string())
    return OC, OCP, OP, PT["session"].nunique()


def _stacked(ax, d, labels, fs=8, min_pct=5):
    bottom = np.zeros(len(d))
    for o in OUTS:
        v = d[f"% {o}"].values
        ax.bar(range(len(d)), v, bottom=bottom, color=fxc.COND[o], edgecolor="w", width=0.7, label=o)
        for kk, (vv, nn, bb) in enumerate(zip(v, d[o].values, bottom)):
            if vv >= min_pct: ax.text(kk, bb + vv / 2, f"{vv:.0f} %\n({nn})", ha="center", va="center", fontsize=fs, color="w")
        bottom += v
    ax.set_xticks(range(len(d))); ax.set_xticklabels(labels, fontsize=fs + 1); ax.set_ylim(0, 100)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


def plot_outcome_counts(OC, OCP, OP, n_sessions):
    """Three figures: A per rat (long | short), B per phase (rows long/short, columns rats + all), C per phase pooled."""
    gr = ["134", "135", "136", "all"]; lab = lambda r: "All rats" if r == "all" else fxc.RAT_LABEL.get(r, f"Rat {r}")
    figA, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, facecolor="white")
    for ax, t in zip(axs, ("long", "short")):
        d = OC[OC["type"] == t].set_index("rat").loc[gr]
        _stacked(ax, d, [f"{lab(r)}\n(n={d.loc[r, 'total_c_e_l']})" for r in gr]); ax.set_title(f"{t.capitalize()} trials", fontsize=11, fontweight="bold")
    axs[0].set_ylabel("% of trials (correct + early + late)"); axs[1].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figA.suptitle(f"Trial outcomes per rat ({n_sessions} sessions)", fontsize=12, fontweight="bold"); figA.tight_layout()
    figB, axs = plt.subplots(2, 4, figsize=(15, 8), sharey=True, facecolor="white")
    for i, t in enumerate(("long", "short")):
        for j, r in enumerate(gr):
            d = OCP[(OCP["type"] == t) & (OCP.rat == r)].set_index("phase").loc[PHASES]
            _stacked(axs[i, j], d, [f"{ph.capitalize()}\n(n={d.loc[ph, 'total_c_e_l']})" for ph in PHASES], fs=7, min_pct=6)
            if i == 0: axs[i, j].set_title(lab(r), fontsize=10, fontweight="bold")
            if j == 0: axs[i, j].set_ylabel(f"{t.capitalize()} trials\n% (correct + early + late)")
    axs[0, -1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figB.suptitle("Trial outcomes per session phase (start 0–0.25, middle 0.25–0.85, end 0.85–1)", fontsize=12, fontweight="bold"); figB.tight_layout()
    figC, ax = plt.subplots(figsize=(5, 4.6), facecolor="white")
    _stacked(ax, OP.loc[PHASES], [f"{ph.capitalize()}\n(n={OP.loc[ph, 'total_c_e_l']})" for ph in PHASES])
    ax.set_ylabel("% of trials (correct + early + late)"); ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"Trial outcomes per session phase\nlong + short, all rats ({n_sessions} sessions)", fontsize=10, fontweight="bold"); figC.tight_layout()
    return figA, figB, figC
