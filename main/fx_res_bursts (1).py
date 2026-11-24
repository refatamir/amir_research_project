"""
fx_res_bursts.py — beta (and gamma) bursts: cells of 07 section 3, and the burst-along-the-session cells of section 4
(Step 0, Step 3, Step 4, Gamma Step 1). Burst detection as Karvat et al. 2021 (06 steps 2–4, 6–7).
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fx_results import cell


@cell()
def load_burst_tables():
    """6a. LOAD BURST TRIAL TABLES (all sessions) — used by 6c onwards"""
    global BETA_DIR, BURST_BANDS_R, E, FAR, FAR_UM, GROUPS6, L, MIN_RT, MIN_TR, T, TABS, TESTS6, band, files, s, z
    # ── 6a. LOAD BURST TRIAL TABLES (all sessions) — used by 6c onwards ──
    BURST_BANDS_R = ["beta 15-30", "gamma 30-100"]
    GROUPS6 = ("mPFC pooled", "PL", "IL", "Cg1")
    MIN_RT, MIN_TR = None, 1
    FAR_UM = 300
    TESTS6 = [("late long",  "late",  "long",  "correct", "late"),
              ("late short", "late",  "short", "correct", "late"),
              ("early long (time-matched)", "early", "long", "correct long, time-matched", "early long"),
              ("early long (own release)",  "early", "long", "correct long, own release",  "early long")]
    files = [s for s in final["session"] if os.path.exists(os.path.join(BURST_DIR, s + ".npz"))]
    TABS, FAR = {bd: {"late": [], "early": []} for bd in BURST_BANDS_R}, set()
    for s in files:
        z = np.load(os.path.join(BURST_DIR, s + ".npz")); T = pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))
        FAR |= {(s, k.split("|")[0], k.split("|")[1]) for k in z.files if k.endswith("|distance_um") and z[k][0] > FAR_UM}
        for band in BURST_BANDS_R:
            L, E = fn.burst_trial_tables(z, T, band, min_rt=MIN_RT)
            TABS[band]["late"].append(L.assign(session=s, rat=RAT[s])); TABS[band]["early"].append(E.assign(session=s, rat=RAT[s]))
    TABS = {bd: {w: pd.concat(v, ignore_index=True) for w, v in d.items()} for bd, d in TABS.items()}
    BETA_DIR = os.path.join(AMIR, "beta bursts"); os.makedirs(BETA_DIR, exist_ok=True)
    print(f"burst files: {len(files)} | far channel-regions: {len(FAR)} | beta late-table rows: {len(TABS['beta 15-30']['late'])}")


@cell()
def burst_channels_brain():
    """P4c. Selected channel per region (beta bursts) in brain coordinates, coronal; style as Fig 1C → supplementary"""
    global AMIR_, CC, LONG, Line2D, REG_COL4, REP, RNAME, SEL, SHORT, TABLES_, _, ax, axes, bad, d, df, fig, g, handles, j
    global miss, rat, rats, reg, rr, shown, sp_, sup_dir
    # ── P4c. Selected channel per region (beta bursts) in brain coordinates, coronal; style as Fig 1C → supplementary ──
    from matplotlib.lines import Line2D
    REG_COL4 = {"Cingulate area 1": "#9467bd", "Prelimbic area": "#1f77b4", "Infralimbic area": "#ff7f0e", "Basal forebrain region": "#8c564b"}
    SHORT = {"Cingulate area 1": "Cg1", "Prelimbic area": "PL", "Infralimbic area": "IL"}
    LONG = {v: k for k, v in SHORT.items()}
    RNAME = {"134": "Rat 1", "135": "Rat 2", "136": "Rat 3"}
    AMIR_ = r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data"
    TABLES_ = globals().get("TABLES", r"PASTE_THE_TABLES_PATH_HERE")          # folder with representative_channels.csv
    sup_dir = os.path.join(AMIR_, "supplementary"); os.makedirs(sup_dir, exist_ok=True)

    REP = pd.read_csv(os.path.join(TABLES_, "representative_channels.csv")); REP["rat"] = REP["rat"].astype(str)
    CC = pd.read_csv(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\Probe\electrode_positions_brain.csv")
    CC["rat"] = CC["rat"].astype(str)
    CC = CC.dropna(subset=["ml_mm", "dv_mm"]).drop_duplicates(["rat", "x", "y"])
    for df in (CC, REP):
        df["xr"], df["yr"] = df["x"].astype(float).round(1), df["y"].astype(float).round(1)
    SEL = REP.merge(CC[["rat", "xr", "yr", "ml_mm", "dv_mm", "region"]], on=["rat", "xr", "yr"], how="left", suffixes=("", "_brain"))
    miss = SEL[SEL["ml_mm"].isna()]
    if len(miss):
        print("selected channels without brain coordinates (check x/y):\n", miss[["rat", "region", "x", "y"]].to_string(index=False))
    bad = SEL.dropna(subset=["region_brain"])
    bad = bad[bad["region_brain"].map(SHORT) != bad["region"]]
    if len(bad):
        print("selected channel lies in another area by histology:\n", bad[["rat", "region", "region_brain", "x", "y"]].to_string(index=False))

    rats = sorted(CC["rat"].unique())
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, len(rats), figsize=(2.6 * len(rats), 4.0), squeeze=False, facecolor="white")
    for j, rat in enumerate(rats):
        ax, d = axes[0, j], CC[CC.rat == rat]
        for reg in REG_COL4:
            g = d[d["region"] == reg]
            if len(g):
                ax.scatter(g["ml_mm"], g["dv_mm"], s=6, color=REG_COL4[reg], lw=0)
        for _, rr in SEL[(SEL.rat == rat) & SEL["ml_mm"].notna()].iterrows():
            ax.scatter(rr.ml_mm, rr.dv_mm, s=70, marker="o", facecolor=REG_COL4[LONG[rr.region]], edgecolor="k", lw=1.2, zorder=5)

        ax.axvline(0, color="k", lw=0.6, ls="--")                                   # midline
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(RNAME.get(rat, f"Rat {rat}"), fontsize=9, fontweight="bold")
        ax.set_xlabel("ML (mm)"); ax.set_ylabel("DV (mm)" if j == 0 else "")
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    shown = [r for r in REG_COL4 if (CC["region"] == r).any()]
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=REG_COL4[r], label=r) for r in shown] + \
              [Line2D([], [], marker="o", ls="", ms=7, markerfacecolor="white", markeredgecolor="k", mew=1.2, label="Selected channel")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _savefig(fig, os.path.join(sup_dir, "Fig S14 selected channel per region.png"))
    plt.show()
    print(SEL[["rat", "region", "x", "y", "ml_mm", "dv_mm", "region_brain"]].to_string(index=False))


@cell()
def beta_power_karvat():
    """beta_power_karvat"""
    global A, BIN, BURST_DIR, Bn, DATA_OUT, F, FIG_DIR, GROUPS, G_, KARVAT_2D, M, MAIN, P, PW_DIR, STYLE, T, WIN, Y, ax, axes
    global bcols, c, centers, ci, col, d1, d2, edges, f, fdist, fig, g, glob, k, keep, lab, mu, multipletests, np, os, p
    global p_bin, pairwise_tukeyhsd, pd, plt, regs, res, rm_anova, rows, s, tdist, tm, y, z
    import os, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from scipy.stats import f as fdist, t as tdist
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests

    MAIN      = r"C:\Users\ar687\Documents\GitHub\AmirAnalysis\main"
    DATA_OUT  = r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\data"
    BURST_DIR = os.path.join(DATA_OUT, "Bursts detection")
    PW_DIR    = os.path.join(DATA_OUT, "Beta power p90 LOCAL")
    FIG_DIR   = os.path.join(MAIN, "figures", "07_beta_power"); os.makedirs(FIG_DIR, exist_ok=True)
    WIN, BIN  = (-0.5, 0.5), 0.025
    GROUPS    = ["mPFC pooled", "IL", "PL", "Cg1"]
    STYLE     = {"correct": ("correct (≈ hit)", "0.3"), "late": ("late (≈ miss)", "#1565c0")}

    # 1. power per trial in 25-ms bins, correct and late long trials, all sessions
    edges = np.round(np.arange(WIN[0], WIN[1] + 1e-9, BIN), 3); centers = edges[:-1] + BIN / 2
    rows = []
    for f in sorted(glob.glob(os.path.join(PW_DIR, "*.npz"))):
        if f.endswith(".tmp.npz"): continue
        s = os.path.basename(f)[:-4]
        z = np.load(f); tm = z["times"]
        T = pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))
        regs = [g for g in ("IL", "PL", "Cg1") if f"{g}|tone" in z.files]
        M = {g: z[f"{g}|tone"] for g in regs}
        M["mPFC pooled"] = np.nanmean(np.stack([M[g] for g in regs]), axis=0)
        keep = np.where((T["type"] == "long") & T["label"].isin(["correct", "late"]))[0]
        for g, A in M.items():
            Bn = np.stack([np.nanmean(A[:, (tm >= a) & (tm < a + BIN)], axis=1) for a in edges[:-1]], axis=1)
            for k in keep:
                if np.isfinite(Bn[k]).all():
                    rows.append((s, g, T.loc[k, "label"], *Bn[k]))
    P = pd.DataFrame(rows, columns=["session", "group", "cond", *[f"b{j}" for j in range(len(centers))]])
    bcols = [c for c in P.columns if c.startswith("b")]

    # 2. two-way repeated-measures ANOVA: outcome (between trials) × time (within trials), trials as units
    def rm_anova(Ys):
        Y = np.vstack(Ys); g = np.repeat(np.arange(len(Ys)), [len(y) for y in Ys]); N, Tn = Y.shape; GM = Y.mean()
        ng = np.bincount(g); mg = np.array([Y[g == k].mean() for k in range(len(Ys))])
        mgt = np.array([Y[g == k].mean(0) for k in range(len(Ys))]); mt, mi = Y.mean(0), Y.mean(1)
        ss_g = Tn * np.sum(ng * (mg - GM) ** 2); ss_s = Tn * np.sum((mi - mg[g]) ** 2); ss_t = N * np.sum((mt - GM) ** 2)
        ss_gt = np.sum(ng[:, None] * (mgt - mg[:, None] - mt[None, :] + GM) ** 2)
        ss_e = np.sum((Y - GM) ** 2) - ss_g - ss_s - ss_t - ss_gt; G = len(Ys); df_s, df_e = N - G, (N - G) * (Tn - 1)
        out = {}
        for name, ss_, d1, den, d2 in [("outcome", ss_g, G - 1, ss_s, df_s), ("time", ss_t, Tn - 1, ss_e, df_e),
                                       ("time × outcome", ss_gt, (G - 1) * (Tn - 1), ss_e, df_e)]:
            F = (ss_ / d1) / (den / d2); out[name] = (F, d1, d2, fdist.sf(F, d1, d2))
        return out

    # 3. test + figure per region
    res = []
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, g in zip(axes, GROUPS):
        G_ = P[P["group"] == g]
        Y = {c: G_.loc[G_.cond == c, bcols].values for c in ("correct", "late")}
        print(f"\n=== {g}: {len(Y['correct'])} correct, {len(Y['late'])} late trials ({G_.session.nunique()} sessions)")
        if min(len(Y["correct"]), len(Y["late"])) < 2:
            print("   not enough trials in one outcome, skipped"); continue
        for k, (F, d1, d2, p) in rm_anova([Y["correct"], Y["late"]]).items():
            print(f"   {k:15s} F({d1},{d2}) = {F:.2f}, p = {p:.3g}")
            res.append(dict(group=g, term=k, F=F, df1=d1, df2=d2, p=p))
        p_bin = np.array([pairwise_tukeyhsd(np.r_[Y["correct"][:, j], Y["late"][:, j]],
                                            np.r_[["correct"] * len(Y["correct"]), ["late"] * len(Y["late"])]).pvalues[0]
                          for j in range(len(centers))])
        print("   bins with p < 0.01 (Tukey):", [f"{c:+.3f}" for c, p in zip(centers, p_bin) if p < 0.01] or "none")
        for c, p in zip(centers, p_bin):
            if p < 0.01: ax.axvspan(c - BIN / 2, c + BIN / 2, color="0.88", zorder=0)
        for c in ("correct", "late"):
            y = Y[c]; mu = y.mean(0); ci = tdist.ppf(0.975, len(y) - 1) * y.std(0, ddof=1) / np.sqrt(len(y))
            lab, col = STYLE[c]
            ax.plot(centers, mu, color=col, label=f"{lab}, n={len(y)}"); ax.fill_between(centers, mu - ci, mu + ci, color=col, alpha=0.25)
        ax.axvline(0, color="k", lw=0.8); ax.set_title(g); ax.set_xlabel("Time from tone (s)")
    axes[0].set_ylabel("Beta power (fraction of 90th percentile)"); axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); _savefig(plt.gcf(), os.path.join(FIG_DIR, "beta_power_correct_vs_late_KarvatFig2D.png")); plt.show()
    KARVAT_2D = pd.DataFrame(res)

    # 4. FDR (Benjamini–Hochberg) over the four groups (IL, PL, Cg1, pooled) within each term, as stated in the methods
    KARVAT_2D["q"] = KARVAT_2D.groupby("term")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    print("\nFDR over the four groups within each term:")
    print(KARVAT_2D.round({"F": 2, "p": 4, "q": 4}).to_string(index=False))


@cell(needs={
    'beta_power_karvat': ['BIN', 'P', 'bcols', 'centers'],
})
def beta_power_pooled():
    """Beta power, pooled mPFC: correct vs late long trials around the tone (paper style; shown only)"""
    global COLS, G_, Y, ax, c, ci, col, fig, idx, lab, mu, p_bin, pairwise_tukeyhsd, segs, sg, sig, sp, tdist, y, y0, y1, yb
    # ── Beta power, pooled mPFC: correct vs late long trials around the tone (paper style; shown only) ──
    from scipy.stats import t as tdist
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    G_ = P[P["group"] == "mPFC pooled"]
    Y = {c: G_.loc[G_.cond == c, bcols].values for c in ("correct", "late")}
    COLS = {"correct": ("#2CA02C", "Correct"), "late": ("#DD8452", "Late")}

    p_bin = np.array([pairwise_tukeyhsd(np.r_[Y["correct"][:, j], Y["late"][:, j]],
                                        np.r_[["correct"] * len(Y["correct"]), ["late"] * len(Y["late"])]).pvalues[0]
                      for j in range(len(centers))])
    sig = p_bin < 0.01

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(3.6, 2.8), facecolor="white")
    for c in ("correct", "late"):
        y = Y[c]; mu = y.mean(0); ci = tdist.ppf(0.975, len(y) - 1) * y.std(0, ddof=1) / np.sqrt(len(y))
        col, lab = COLS[c]
        ax.fill_between(centers, mu - ci, mu + ci, color=col, alpha=0.2, lw=0)
        ax.plot(centers, mu, color=col, lw=1.6, label=f"{lab} (n = {len(y):,} trials)")
    ax.axvline(0, color="k", lw=0.9)
    ax.text(0, 1.01, "tone", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    if sig.any():                                                                     # black bar over significant bins
        y0, y1 = ax.get_ylim(); yb = y1 + 0.03 * (y1 - y0)
        idx = np.where(sig)[0]; segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for sg in segs:
            ax.plot([centers[sg[0]] - BIN / 2, centers[sg[-1]] + BIN / 2], [yb, yb], color="k", lw=3, solid_capstyle="butt")
        ax.text(centers[segs[-1][-1]] + BIN, yb, "p < 0.01", ha="left", va="center", fontsize=7.5)
        ax.set_ylim(top=yb + 0.08 * (y1 - y0))
    ax.set_xlim(centers[0] - BIN / 2, centers[-1] + BIN / 2)
    ax.set_xlabel("Time from tone (s)"); ax.set_ylabel("Beta power\n(fraction of 90th percentile)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    _savefig(fig, os.path.join(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report",
                             "Fig 6 beta power correct vs late pooled mPFC.png"))
    plt.show()
    print(f"{len(G_.session.unique())} sessions; significant bins (Tukey p < 0.01): "
          + ", ".join(f"{centers[sg[0]] - BIN / 2:+.3f} to {centers[sg[-1]] + BIN / 2:+.3f} s" for sg in segs) if sig.any() else "none")


@cell(needs={
    'load_burst_tables': ['TABS'],
})
def step0_simple():
    """STEP 0. late − correct beta burst occupancy per region, SIMPLE TEST (session means, ≥5 late trials) (needs Settings, 6a)"""
    global BAND, D0, MIN_CORR, MIN_LATE, R0, REGS, YEL, ax, d, dots, fig, fp, g, half, hi, k, lo, m, n, ok, p, r, rng, rows, s
    global sp, stars, tdist, v
    # ── STEP 0. late − correct beta burst occupancy per region, SIMPLE TEST (session means, ≥5 late trials) (needs Settings, 6a) ──
    globals().pop("C", None)
    from scipy.stats import t as tdist
    BAND = "beta 15-30"; YEL = "#E6B800"
    REGS = ["IL", "PL", "Cg1", "mPFC pooled"]
    MIN_LATE, MIN_CORR = 5, 3
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    D0 = TABS[BAND]["late"]
    D0 = D0[(D0.ref == "LOCAL") & (D0.typ == "long") & D0.cond.isin(["correct", "late"])].dropna(subset=["occ"]).copy()
    D0["late"] = (D0.cond == "late").astype(int); D0["occ_pct"] = 100 * D0.occ

    rows, dots = [], {}
    for g in REGS:
        d = D0[D0.group == g]
        if d.empty: continue
        s = d.groupby(["session", "late"])["occ_pct"].agg(["mean", "size"]).unstack("late")
        ok = (s[("size", 1)] >= MIN_LATE) & (s[("size", 0)] >= MIN_CORR)
        v = (s.loc[ok, ("mean", 1)] - s.loc[ok, ("mean", 0)]).values
        m, p, n = fn.sign_flip_p(v)
        half = tdist.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
        dots[g] = v
        rows.append(dict(region=g, sessions=n, sessions_up=int((v > 0).sum()),
                         occ_correct=s.loc[ok, ("mean", 0)].mean(), occ_late=s.loc[ok, ("mean", 1)].mean(),
                         late_minus_correct=m, ci_low=m - half, ci_high=m + half, p=p, sig=stars(p)))
    R0 = pd.DataFrame(rows)
    print(f"Simple test: late − correct beta occupancy (%-points), session means (≥{MIN_LATE} late, ≥{MIN_CORR} correct), LOCAL, long trials")
    print(R0.round(3).to_string(index=False))
    _table(R0, os.path.join(BETA_DIR, "step0_simple_late_vs_correct_regions.csv"), index=False)

    # ── figure (same style as Step 1) ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(4.6, 3.8), facecolor="white")
    rng = np.random.default_rng(0)
    for k, r in R0.iterrows():
        ax.bar(k, r.late_minus_correct, width=0.6, color=YEL if r.p < 0.05 else "0.75", edgecolor="k", lw=0.6, zorder=1)
        v = dots[r.region]
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color="k", alpha=0.3, lw=0, zorder=2)
        ax.errorbar(k, r.late_minus_correct, yerr=[[r.late_minus_correct - r.ci_low], [r.ci_high - r.late_minus_correct]],
                    color="k", capsize=4, lw=1.1, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    lo = min(min(v.min() for v in dots.values()), R0.ci_low.min()); hi = max(max(v.max() for v in dots.values()), R0.ci_high.max())
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.12 * (hi - lo))
    for k, r in R0.iterrows():
        ax.text(k, hi + 0.04 * (hi - lo), r.sig, ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(R0))); ax.set_xticklabels([g.replace("mPFC pooled", "mPFC\npooled") for g in R0.region])
    ax.set_ylabel("Late − correct\nbeta burst occupancy (%-points)")
    ax.set_title(f"Beta bursts before late responses, session means\n(sessions with ≥{MIN_LATE} late trials, LOCAL, long trials)", fontsize=10, fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fp = os.path.join(BETA_DIR, "step0_simple_late_vs_correct_regions.png")
    _savefig(fig, fp); plt.close(fig)
    display(Image(fp))


@cell(needs={
    'load_burst_tables': ['TABS', 'files'],
})
def step1_model():
    """STEP 1. LATE vs CORRECT BETA BURST OCCUPANCY 0.6 s BEFORE THE TONE PER REGION: MIXED MODEL (rebuilt; makes D, TP, R1 for Step 1 figure, 1-check, 1b, 1c)"""
    global BAND1, D, OLD1, POSCOLS, PT, R1, TP, d, e, g, m, p, rows, se, smf, t0_, t1_, vol, warnings
    # ── STEP 1. LATE vs CORRECT BETA BURST OCCUPANCY 0.6 s BEFORE THE TONE PER REGION: MIXED MODEL (rebuilt; makes D, TP, R1 for Step 1 figure, 1-check, 1b, 1c) ──
    import warnings
    import statsmodels.formula.api as smf
    BAND1 = "beta 15-30"
    # ── trial positions (same as Step 3): trial number, clock time, water drunk ──
    PT = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False).sort_values(["session", "trial_index"])
    PT["position"] = PT.groupby("session")["trial_index"].rank(pct=True)
    t0_ = PT.groupby("session")["press_neural_s"].transform("min"); t1_ = PT.groupby("session")["press_neural_s"].transform("max")
    PT["v_time"] = (PT["press_neural_s"] - t0_) / (t1_ - t0_)
    vol = PT["reward_size"].fillna(0).clip(lower=0)
    PT["v_water"] = (vol.groupby(PT["session"]).cumsum() - vol) / vol.groupby(PT["session"]).transform("sum")
    POSCOLS = ["position", "v_time", "v_water"]
    TP = pd.concat([pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))[["trial_index", "long_prob"]]
                    .assign(session=s, trial=lambda x: x.index) for s in files], ignore_index=True)
    TP = TP.merge(PT[["session", "trial_index", *POSCOLS]], on=["session", "trial_index"], how="left")

    # ── one row per trial and region: occupancy in the 0.6 s before the tone, LOCAL, long correct / late ──
    D = TABS[BAND1]["late"].merge(TP, on=["session", "trial"], how="left")
    D = D[(D["ref"] == "LOCAL") & (D["typ"] == "long") & D["cond"].isin(["correct", "late"])].dropna(subset=["occ", "position"]).copy()
    D["late"] = (D["cond"] == "late").astype(int); D["occ_pct"] = 100 * D["occ"]
    D["block"] = D["long_prob"].round(2).astype(str); D["rat"] = D["rat"].astype(str)

    # ── model per region: occupancy ~ late + position in the session + block + (1 | session) ──
    rows = []
    for g in ["mPFC pooled", "IL", "PL", "Cg1"]:
        d = D[D.group == g]
        if d.empty: continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("occ_pct ~ late + position + C(block)", d, groups=d["session"]).fit(reml=True, method=["lbfgs", "powell", "nm"])
        e, se, p = m.params["late"], m.bse["late"], m.pvalues["late"]
        rows.append(dict(region=g, sessions=d.session.nunique(), n_correct=int((d.late == 0).sum()), n_late=int(d.late.sum()),
                         occ_correct=d.loc[d.late == 0, "occ_pct"].mean(), occ_late=d.loc[d.late == 1, "occ_pct"].mean(),
                         late_minus_correct=e, ci_low=e - 1.96 * se, ci_high=e + 1.96 * se, p=p, converged=m.converged))
    R1 = pd.DataFrame(rows)
    print("late − correct beta occupancy (%-points), 0.6 s before the tone, LOCAL, long trials, model with block (rebuilt):")
    print(R1.round(4).to_string(index=False))
    OLD1 = os.path.join(BETA_DIR, "step1_late_vs_correct_regions.csv")
    if os.path.exists(OLD1):
        print("\nthe table saved by the old Step 1 (should be the same; if not, tell me):")
        print(pd.read_csv(OLD1)[["region", "sessions", "n_correct", "n_late", "late_minus_correct", "ci_low", "ci_high", "p"]].round(4).to_string(index=False))


@cell(needs={
    'step1_model': ['R1'],
    'step0_simple': ['stars'],
})
def step1_regions():
    """FDR over the 4 groups, then figure (paper style, report Fig 8)"""
    global BETA_COL, R1, ax, fig, k, multipletests, r, sp, span
    # ── FDR over the 4 groups, then figure (paper style, report Fig 8) ──
    from statsmodels.stats.multitest import multipletests
    R1["q"] = multipletests(R1["p"], method="fdr_bh")[1]
    R1["sig_q"] = R1["p"].apply(stars)                     # raw p, no FDR
    R1 = R1.set_index("region").loc[["mPFC pooled", "IL", "PL", "Cg1"]].reset_index()
    print("\nwith FDR over the 4 groups:"); print(R1[["region", "sessions", "n_correct", "n_late", "occ_correct", "occ_late",
                                                      "late_minus_correct", "ci_low", "ci_high", "p", "q", "sig_q"]].round(4).to_string(index=False))
    _table(R1, os.path.join(BETA_DIR, "step1_late_vs_correct_regions.csv"), index=False)

    BETA_COL = "#1565C0"
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(3.6, 3.0), facecolor="white")
    span = R1.ci_high.max() - min(0, R1.ci_low.min())
    for k, r in R1.iterrows():
        ax.bar(k, r.late_minus_correct, width=0.6, color=BETA_COL if r.p < 0.05 else "0.8", edgecolor="k", lw=0.4)
        ax.errorbar(k, r.late_minus_correct, yerr=[[r.late_minus_correct - r.ci_low], [r.ci_high - r.late_minus_correct]],
                    color="k", capsize=3, lw=1)
        ax.text(k, max(r.ci_high, 0) + 0.04 * span, r.sig_q, ha="center", va="bottom", fontsize=8.5)     # always above
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(R1.ci_low.min() - 0.1 * span, R1.ci_high.max() + 0.2 * span)
    ax.set_xticks(range(len(R1))); ax.set_xticklabels(["mPFC\npooled", "IL", "PL", "Cg1"])
    ax.set_ylabel("Late − correct\nbeta burst occupancy (%-points)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _savefig(fig, os.path.join(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report",
                             "Fig 8 beta occupancy late vs correct per region.png"))
    plt.show()


@cell(needs={
    'step1_model': ['D'],
    'step0_simple': ['stars'],
})
def step1_check():
    """STEP 1-check. Simple session test vs mixed model (no block), pooled and each region, LOCAL, 0.6 s before the tone"""
    global CHK, S, d, diff, g, m, min_late, mm, n, ok, p, rows, smf, warnings
    # ── STEP 1-check. Simple session test vs mixed model (no block), pooled and each region, LOCAL, 0.6 s before the tone ──
    import warnings
    import statsmodels.formula.api as smf
    rows = []
    for g in ["mPFC pooled", "IL", "PL", "Cg1"]:
        d = D[D.group == g].dropna(subset=["occ_pct", "position"])
        # mixed model WITHOUT block: occupancy ~ late + position + (1 | session)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("occ_pct ~ late + position", d, groups=d["session"]).fit(reml=True, method=["lbfgs", "powell", "nm"])
        rows.append(dict(region=g, test="mixed model (all trials)", sessions=d.session.nunique(), late_trials=int(d.late.sum()),
                         late_minus_correct=m.params["late"], p=m.pvalues["late"], sig=stars(m.pvalues["late"])))
        # simple test: per session mean(late) − mean(correct), sign-flip across sessions, for different minimum numbers of late trials
        S = d.groupby(["session", "late"])["occ_pct"].agg(["mean", "size"]).unstack("late")
        for min_late in (1, 3, 5):
            ok = (S[("size", 1)] >= min_late) & (S[("size", 0)] >= 3)
            diff = (S[("mean", 1)] - S[("mean", 0)])[ok].dropna()
            mm, p, n = fn.sign_flip_p(diff.values)
            rows.append(dict(region=g, test=f"session means, ≥{min_late} late", sessions=n,
                             late_trials=int(S[("size", 1)][ok].sum()), late_minus_correct=mm, p=p, sig=stars(p),
                             sessions_up=f"{int((diff > 0).sum())}/{n}"))
    CHK = pd.DataFrame(rows)
    print("late − correct beta occupancy (%-points), LOCAL, long trials, 0.6 s before the tone (no block term)")
    print(CHK.round(3).to_string(index=False))
    _table(CHK, os.path.join(BETA_DIR, "step1_check_simple_vs_mixed.csv"), index=False)


@cell(needs={
    'step1_model': ['D'],
})
def step1b_il_vs_pl():
    """STEP 1b. IL vs PL IN THE SAME SESSIONS AND TRIALS (beta occupancy before the tone, LOCAL, long trials) (needs Step 1: D)"""
    global E, MIN_CORR, MIN_LATE, R1b_s, R1b_t, RCOL, S, W, a1, a2, ax, col, e, fig, fp, half, k, m, n, ok, p, r, rat, rng
    global rows, s, se, smf, sp, tdist, v, warnings
    # ── STEP 1b. IL vs PL IN THE SAME SESSIONS AND TRIALS (beta occupancy before the tone, LOCAL, long trials) (needs Step 1: D) ──
    import warnings
    import statsmodels.formula.api as smf
    from scipy.stats import t as tdist
    MIN_LATE, MIN_CORR = 5, 3
    RCOL = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}

    # trials that have both IL and PL (same sessions, same trials)
    W = (D[D.group.isin(["IL", "PL"])].pivot_table(index=["session", "trial"], columns="group", values="occ_pct")
         .dropna().reset_index())
    W = W.merge(D[D.group == "IL"][["session", "trial", "late", "position", "rat"]], on=["session", "trial"], how="left")
    W["rat"] = W["rat"].astype(str)
    print(f"sessions with IL and PL: {W.session.nunique()} | trials: {len(W)} (late {int(W.late.sum())})")

    # (1) per session: late − correct in IL and in PL, then IL − PL
    S = W.groupby(["session", "rat", "late"])[["IL", "PL"]].mean().unstack("late")
    n = W.groupby(["session", "late"]).size().unstack(fill_value=0)
    ok = n[(n.get(1, 0) >= MIN_LATE) & (n.get(0, 0) >= MIN_CORR)].index
    S = S[S.index.get_level_values("session").isin(ok)]
    E = pd.DataFrame({"IL": S[("IL", 1)] - S[("IL", 0)], "PL": S[("PL", 1)] - S[("PL", 0)]})
    E["IL_minus_PL"] = E["IL"] - E["PL"]
    rows = []
    for col in ["IL", "PL", "IL_minus_PL"]:
        v = E[col].values; m, p, k = fn.sign_flip_p(v)
        half = tdist.ppf(0.975, k - 1) * v.std(ddof=1) / np.sqrt(k)
        rows.append(dict(measure=col, sessions=k, mean=m, ci_low=m - half, ci_high=m + half, p=p, sessions_up=int((v > 0).sum())))
    R1b_s = pd.DataFrame(rows)
    print(f"\n(1) per session, late − correct (%-points), sessions with ≥{MIN_LATE} late and ≥{MIN_CORR} correct:")
    print(R1b_s.round(3).to_string(index=False))

    # (2) per trial: IL − PL, mixed model
    W["diff"] = W["IL"] - W["PL"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("diff ~ late + position", W, groups=W["session"]).fit(reml=True, method=["lbfgs", "powell", "nm"])
    e, se, p = m.params["late"], m.bse["late"], m.pvalues["late"]
    R1b_t = pd.DataFrame([dict(sessions=W.session.nunique(), n_correct=int((W.late == 0).sum()), n_late=int(W.late.sum()),
                               IL_minus_PL_late_minus_correct=e, ci_low=e - 1.96 * se, ci_high=e + 1.96 * se, p=p, converged=m.converged)])
    print("\n(2) per trial, model diff (IL − PL) ~ late + position + (1 | session):")
    print(R1b_t.round(3).to_string(index=False))
    _table(R1b_s, os.path.join(BETA_DIR, "step1b_IL_vs_PL_sessions.csv"), index=False)
    _table(R1b_t, os.path.join(BETA_DIR, "step1b_IL_vs_PL_trials.csv"), index=False)

    # ── figure ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]), facecolor="white")
    for (s, rat), r in E.iterrows():
        a1.plot([0, 1], [r.PL, r.IL], "-o", color=RCOL.get(str(rat), "0.5"), ms=3, lw=0.9, alpha=0.8)
    for rat, col in RCOL.items():
        a1.plot([], [], "-o", color=col, ms=3, label=f"rat {rat}")
    a1.axhline(0, color="k", lw=0.8); a1.set_xticks([0, 1]); a1.set_xticklabels(["PL", "IL"]); a1.set_xlim(-0.3, 1.3)
    a1.set_ylabel("Late − correct\nbeta occupancy (%-points)"); a1.legend(frameon=False, fontsize=8)
    a1.set_title("A  each session (same trials)", loc="left", fontsize=10, fontweight="bold")
    r = R1b_s.set_index("measure").loc["IL_minus_PL"]
    rng = np.random.default_rng(0)
    a2.scatter(rng.uniform(-0.15, 0.15, len(E)), E["IL_minus_PL"], s=12, color="#1565c0", alpha=0.5, lw=0)
    a2.errorbar(0, r["mean"], yerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]], color="k", marker="o", capsize=5, lw=1.5)
    a2.axhline(0, color="k", lw=0.8); a2.set_xticks([0]); a2.set_xticklabels(["IL − PL"]); a2.set_xlim(-0.6, 0.6)
    a2.set_ylabel("Difference of the late − correct effects\n(%-points)")
    a2.set_title(f"B  {int(r.sessions)} sessions, p = {r.p:.3f}", loc="left", fontsize=10, fontweight="bold")
    for ax in (a1, a2):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fp = os.path.join(BETA_DIR, "step1b_IL_vs_PL_same_sessions.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell(needs={
    'step1_model': ['TP'],
})
def step1c_other_channels():
    """STEP 1c. IL and PL WITH OTHER CHANNELS (whole-probe burst files, LOCAL, beta, 0.6 s before the tone) (needs Step 1: TP)"""
    global ALL_DIR, C1, MIN_CORR, MIN_LATE, NCH, O, R1c, T, VERS, arr, cid, d, f, g, glob, js, keep, m, nch, occ, ok, out, reg
    global rk, rows, s, sm, smf, smn, sn, sp, sv, v, ver, warnings, yy, z, zb
    # ── STEP 1c. IL and PL WITH OTHER CHANNELS (whole-probe burst files, LOCAL, beta, 0.6 s before the tone) (needs Step 1: TP) ──
    import glob, warnings
    import statsmodels.formula.api as smf
    ALL_DIR = os.path.join(DATA_OUT, "Beta and gamma bursts across channels")
    MIN_LATE, MIN_CORR = 5, 3
    VERS = ["representative", "all channels", "dorsal-most", "ventral-most"]

    rows, nch = [], []
    for f in sorted(glob.glob(os.path.join(ALL_DIR, "*.npz"))):
        if f.endswith(".tmp.npz"): continue
        s = os.path.basename(f)[:-4]
        z = np.load(f); zb = np.load(os.path.join(BURST_DIR, s + ".npz"))
        T = pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))
        keep = np.where((T["type"] == "long") & T["label"].isin(["correct", "late"]))[0]
        O = z["beta 15-30|tone|occ"]; reg, yy, cid = z["region"], z["y"], z["channel_id"]
        for g in ["IL", "PL"]:
            js = np.flatnonzero(reg == fn.REGION_FULL[g])
            if not len(js): continue
            occ = np.stack([fn.burst_window(O[j].astype(float), "tone", -0.6, 0.0) for j in js])     # channels × trials
            v = {"all channels": np.nanmean(occ, axis=0),
                 "dorsal-most": occ[np.argmax(yy[js])], "ventral-most": occ[np.argmin(yy[js])]}
            rk = f"LOCAL|{g}|channel"
            if rk in zb.files and str(zb[rk][0]) in cid[js]:
                v["representative"] = occ[list(cid[js]).index(str(zb[rk][0]))]
            nch.append(dict(session=s, group=g, channels=len(js), depth_span_um=float(yy[js].max() - yy[js].min())))
            for ver, arr in v.items():
                rows.append(pd.DataFrame(dict(session=s, group=g, version=ver, trial=keep,
                                              late=(T["label"].values[keep] == "late").astype(int), occ_pct=100 * arr[keep])))
    C1 = pd.concat(rows, ignore_index=True).merge(TP[["session", "trial", "position"]], on=["session", "trial"], how="left")
    C1 = C1.dropna(subset=["occ_pct", "position"])
    NCH = pd.DataFrame(nch)
    print("channels per session in each region (median, min–max):")
    print(NCH.groupby("group")["channels"].describe()[["50%", "min", "max"]])

    out = []
    for g in ["IL", "PL"]:
        for ver in VERS:
            d = C1[(C1.group == g) & (C1.version == ver)]
            if d.empty: continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.mixedlm("occ_pct ~ late + position", d, groups=d["session"]).fit(reml=True, method=["lbfgs", "powell", "nm"])
            sm = d.groupby(["session", "late"])["occ_pct"].agg(["mean", "size"]).unstack("late")
            ok = (sm[("size", 1)] >= MIN_LATE) & (sm[("size", 0)] >= MIN_CORR)
            sv = (sm.loc[ok, ("mean", 1)] - sm.loc[ok, ("mean", 0)]).values
            smn, sp, sn = fn.sign_flip_p(sv)
            out.append(dict(group=g, version=ver, sessions=d.session.nunique(), n_late=int(d.late.sum()),
                            model_late_minus_correct=m.params["late"], model_p=m.pvalues["late"], converged=m.converged,
                            simple_late_minus_correct=smn, simple_p=sp, simple_sessions=sn))
    R1c = pd.DataFrame(out)
    print("\nlate − correct beta occupancy (%-points), 0.6 s before the tone, LOCAL, long trials:")
    print(R1c.round(3).to_string(index=False))
    _table(R1c, os.path.join(BETA_DIR, "step1c_IL_PL_other_channels.csv"), index=False)


@cell()
def step2_burst_example():
    """STEP 2. WHAT A BETA BURST LOOKS LIKE: the clearest burst inside the 600 ms before a tone (illustration)"""
    global BAND_EX, BETA_COL, CAND, Pw, REF_EX, REG_EX, S_EX, T3, WIN_EX, a, a_, aen3, ast3, ax, axs, b0, b1, be3, best, beta3
    global bs3, butter, cax, ch3, clean, e_, env3, f, fftconvolve, fig, freqs, fs3, hilbert, i, i_a, i_t, idx, im, inside, k3
    global lim, m, pw, r, rec3, rest, rows, s_, sd, sosfiltfilt, sp, t3, tone, tr, tt, tw, w, x3, y, ze3
    # ── STEP 2. WHAT A BETA BURST LOOKS LIKE: the clearest burst inside the 600 ms before a tone (illustration) ──
    from scipy.signal import butter, sosfiltfilt, hilbert, fftconvolve
    S_EX, REF_EX, REG_EX, BAND_EX, WIN_EX = TEST, "LOCAL", "IL", "beta 15-30", 0.6
    ze3 = np.load(os.path.join(EXACT_DIR, S_EX + ".npz")); T3 = pd.read_csv(os.path.join(EXACT_DIR, S_EX + "_trials.csv"))
    k3 = f"{REF_EX}|{BAND_EX}|{REG_EX}"
    bs3, be3 = ze3[k3 + "|burst_start_s"], ze3[k3 + "|burst_end_s"]
    ast3, aen3 = ze3[f"{REF_EX}|{REG_EX}|art_start_s"], ze3[f"{REF_EX}|{REG_EX}|art_end_s"]
    ch3 = str(ze3[f"{REF_EX}|{REG_EX}|channel"][0])

    # whole session, this channel
    rec3 = fn.load_session(PROC, S_EX, XB, refs=(REF_EX,))[0][REF_EX]; fs3 = rec3.get_sampling_frequency()
    x3 = rec3.get_traces(channel_ids=[ch3], return_in_uV=True)[:, 0].astype(float)
    t3 = np.arange(len(x3)) / fs3
    beta3 = sosfiltfilt(butter(4, [15, 30], btype="band", fs=fs3, output="sos"), x3)
    env3 = np.abs(hilbert(beta3))                                     # only used to choose the clearest burst

    # candidate bursts: fully inside the 600 ms before the tone of a long correct/late trial without artifacts
    tr = T3[(T3["type"] == "long") & T3["label"].isin(["correct", "late"]) & T3["tone_s"].notna() &
            ((T3["long_prob"] == 0.5) | (T3["long_prob"] >= 0.75))]
    rows = []
    for idx, r in tr.iterrows():
        tone = r["tone_s"]; a = tone - WIN_EX
        if np.any((ast3 < tone) & (aen3 > a)):
            continue
        i_a, i_t = int(a * fs3), int(tone * fs3)
        for s_, e_ in zip(bs3, be3):
            if s_ >= a and e_ <= tone and 0.05 <= e_ - s_ <= 0.30:
                inside = env3[int(s_ * fs3): int(e_ * fs3)].max()
                rest = np.median(np.r_[env3[i_a: int(s_ * fs3)], env3[int(e_ * fs3): i_t]]) if (s_ > a or e_ < tone) else np.nan
                rows.append(dict(trial=idx, label=r["label"], tone=tone, start=s_, end=e_, dur_ms=1000 * (e_ - s_), score=inside / rest))
    CAND = pd.DataFrame(rows).dropna(subset=["score"]).sort_values("score", ascending=False)
    best = CAND.iloc[0]; tone = best["tone"]
    print(f"{S_EX} | {REG_EX} {REF_EX} channel {ch3} | candidate bursts fully inside the 600 ms: {len(CAND)} "
          f"(median {CAND['dur_ms'].median():.0f} ms)")
    print(f"shown: trial {best['trial']} ({best['label']}), burst {best['start'] - tone:+.3f} to {best['end'] - tone:+.3f} s from the tone, "
          f"{best['dur_ms']:.0f} ms, peak amplitude {best['score']:.1f}x the rest of the window")

    # time–frequency power 15–40 Hz (Morlet, 7 cycles), / 90th percentile per frequency over the whole session (artifacts excluded)
    freqs = np.arange(15, 41, 1.0)
    clean = np.ones(len(x3), bool)
    for a_, e_ in zip(ast3, aen3):
        clean[(t3 >= a_) & (t3 < e_)] = False
    m = (t3 >= tone - WIN_EX) & (t3 <= tone); tt = t3[m] - tone
    Pw = np.empty((len(freqs), m.sum()), np.float32)
    for i, f in enumerate(freqs):
        sd = 7 / (2 * np.pi * f); tw = np.arange(-3.5 * sd, 3.5 * sd, 1 / fs3)
        w = np.exp(2j * np.pi * f * tw) * np.exp(-tw ** 2 / (2 * sd ** 2)); w /= np.abs(w).sum()
        pw = np.abs(fftconvolve(x3, w, mode="same")) ** 2
        Pw[i] = pw[m] / np.percentile(pw[clean], 90)

    # figure (paper style, report): raw signal, beta-filtered signal, time–frequency power with the 90th-percentile threshold
    BETA_COL = "#1565C0"                                                    # beta: dark blue (theta #2e7d32, delta #F9A825)
    b0, b1 = best["start"] - tone, best["end"] - tone
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, axs = plt.subplots(3, 1, figsize=(3.4, 4.2), sharex=True, facecolor="white", gridspec_kw=dict(height_ratios=[0.8, 0.8, 1.3]))
    for ax in axs:
        ax.axvspan(b0, b1, color="#C0392B", alpha=0.10, lw=0)                          # burst window, red as in the theta figures
    axs[0].plot(tt, x3[m], color="0.45", lw=0.7); axs[0].set_ylabel("LFP (µV)")
    axs[1].plot(tt, beta3[m], color=BETA_COL, lw=0.9); axs[1].set_ylabel("15–30 Hz (µV)")
    for ax, y in ((axs[0], x3[m]), (axs[1], beta3[m])):
        lim = 1.1 * np.abs(y).max(); ax.set_ylim(-lim, lim); ax.locator_params(axis="y", nbins=3)
    im = axs[2].pcolormesh(tt, freqs, Pw, cmap="viridis", vmin=0, vmax=3, shading="gouraud")                       # marks the beta band on the left edge
    axs[2].set_ylabel("Frequency (Hz)"); axs[2].set_xlabel("Time from tone (s)"); axs[2].set_xlim(-WIN_EX, 0)
    for ax in axs:
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.80, top=0.97, bottom=0.11, hspace=0.12)
    cax = fig.add_axes([0.83, 0.11, 0.03, 0.30]); fig.colorbar(im, cax=cax, label="Power\n(fraction of 90th percentile)")
    _savefig(fig, os.path.join(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report",
                             "Fig 7 example beta burst.png"))
    plt.show()
    print(f"Fig 7: burst {b0:+.3f} to {b1:+.3f} s from the tone, {best['dur_ms']:.0f} ms ({best['label']} trial)")


@cell(needs={
    'load_burst_tables': ['BURST_BANDS_R', 'MIN_TR', 'TABS', 'files'],
})
def step5_success():
    """Step 5. success vs pre-tone beta burst occupancy, IL LOCAL, long trials, all blocks, whole session (needs Settings, 6a)"""
    global BAND, BINCOL, F, L, MIN_BIN_SES, MIN_BIN_TR, OBINS, OK, OUT, PS, PSb, P_ANOVA, P_TREND, RES, ST, TP, TREND, ax, b
    global f_oneway, fig, fp, mb, n_, ptxt, sm, smf, sp
    # ── Step 5. success vs pre-tone beta burst occupancy, IL LOCAL, long trials, all blocks, whole session (needs Settings, 6a) ──
    import statsmodels.formula.api as smf
    from scipy.stats import f_oneway

    BAND = [b for b in BURST_BANDS_R if b.startswith("beta")][0]
    OBINS = ["Quiet\n0 %", "Low\n≤25 %", "Medium\n≤50 %", "High\n>50 %"]
    BINCOL = ["#fde0d9", "#f4a582", "#d6604d", "#b2182b"]
    MIN_BIN_TR, MIN_BIN_SES = 3, 5

    # same trials as Steps 3–4 (all blocks)
    TP = pd.concat([pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))[["long_prob"]]
                    .assign(session=s, trial=lambda d: d.index) for s in files], ignore_index=True)
    L = TABS[BAND]["late"].merge(TP, on=["session", "trial"], how="left")
    L = L[(L.ref == "LOCAL") & (L.group == "IL") & (L.typ == "long") & L.cond.isin(["correct", "late"])].dropna(subset=["occ"]).copy()
    L["success"] = (L.cond == "correct").astype(float)
    L["obin"] = np.select([L.occ <= 0, L.occ <= 0.25, L.occ <= 0.5], [0, 1, 2], default=3)
    print(f"check occ scale: min {L.occ.min():.3f}, max {L.occ.max():.3f} (must be 0–1); "
          f"trials {len(L)} (correct {(L.cond=='correct').sum()}, late {(L.cond=='late').sum()}) — Step 3: 2743 / 490")

    # sessions with at least one correct and one late trial
    n_ = L.groupby("session")["cond"].agg(n_c=lambda x: (x == "correct").sum(), n_l=lambda x: (x == "late").sum())
    L = L[L.session.isin(n_[(n_.n_c >= MIN_TR) & (n_.n_l >= MIN_TR)].index)].copy()

    # per session and group: success as % change from the session mean
    sm = L.groupby("session")["success"].mean()
    PS = L.groupby(["session", "obin"])["success"].agg(["mean", "size"]).reset_index()
    PS = PS[PS["size"] >= MIN_BIN_TR].copy()
    PS["pcm"] = 100 * (PS["mean"] - PS["session"].map(sm)) / PS["session"].map(sm)
    ST = PS.groupby("obin")["pcm"].agg(["mean", "sem", "size"]).reindex(range(4))
    ST["size"] = ST["size"].fillna(0).astype(int)
    ST["trials"] = PS.groupby("obin")["size"].sum().reindex(range(4)).fillna(0).astype(int)
    OK = [b for b in range(4) if ST["size"].iat[b] >= MIN_BIN_SES]

    # tests: ANOVA across groups + straight line through the session values (same units as the bars)
    F, P_ANOVA = f_oneway(*[PS.loc[PS.obin == b, "pcm"].values for b in OK]) if len(OK) >= 2 else (np.nan, np.nan)
    PSb = PS[PS.obin.isin(OK)].copy()
    mb = smf.ols("pcm ~ obin", PSb).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(PSb["session"])[0]})
    TREND, P_TREND = mb.params["obin"], mb.pvalues["obin"]

    OUT = pd.DataFrame(dict(group=[o.replace("\n", " ") for o in OBINS], PCM_mean=ST["mean"].values,
                            PCM_sem=ST["sem"].values, sessions=ST["size"].values, trials=ST["trials"].values))
    RES = pd.DataFrame([dict(sessions=L.session.nunique(), n_correct=int((L.cond == "correct").sum()),
                             n_late=int((L.cond == "late").sum()), points=len(PSb), anova_F=F, anova_p=P_ANOVA,
                             trend_pct_per_group=TREND, trend_p=P_TREND)])
    print("\nsuccess (% change from session mean) per group:"); print(OUT.round(2).to_string(index=False))
    print("\ntests:"); print(RES.round(4).to_string(index=False))
    _table(OUT, os.path.join(BETA_DIR, "step5_success_vs_beta_occupancy.csv"), index=False)
    _table(RES, os.path.join(BETA_DIR, "step5_success_vs_beta_occupancy_tests.csv"), index=False)

    # ── figure ──
    ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for b in OK:
        ax.bar(b, ST["mean"].iat[b], width=0.7, color=BINCOL[b], edgecolor="k", lw=0.6)
        ax.errorbar(b, ST["mean"].iat[b], yerr=ST["sem"].iat[b], color="k", capsize=4, lw=1)
        ax.text(b, 0.02, f"{ST['trials'].iat[b]} trials", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color="0.4")
    ax.text(0.98, 0.97, f"ANOVA {ptxt(P_ANOVA)}\ntrend {TREND:+.1f} % per group, {ptxt(P_TREND)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(OBINS); ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel("Burst occupancy before the tone")
    ax.set_ylabel("Success rate\n(% change from session mean)")
    ax.set_title(f"IL beta bursts and success\n({L.session.nunique()} sessions, 3 rats, LOCAL, long trials)", fontsize=10, fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fp = os.path.join(BETA_DIR, "step5_success_vs_beta_occupancy.png")
    _savefig(fig, fp); plt.close(fig)
    display(Image(fp))


@cell(needs={
    'step5_success': ['OBINS', 'OK', 'P_ANOVA', 'P_TREND', 'ST', 'TREND'],
})
def step5_figure():
    """figure (paper style, report Fig 13): beta blue from light (quiet) to dark (high)"""
    global BLUES, ax, b, fig, ptxt, sp
    # ── figure (paper style, report Fig 13): beta blue from light (quiet) to dark (high) ──
    BLUES = ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"]
    ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(3.6, 3.0), facecolor="white")
    for b in OK:
        ax.bar(b, ST["mean"].iat[b], width=0.65, color=BLUES[b], edgecolor="k", lw=0.4)
        ax.errorbar(b, ST["mean"].iat[b], yerr=ST["sem"].iat[b], color="k", capsize=3, lw=1)
    ax.text(0.98, 0.97, f"ANOVA {ptxt(P_ANOVA)}\ntrend {TREND:+.1f} % per group, {ptxt(P_TREND)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(OBINS); ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel("Beta burst occupancy 0.6 s before the tone")
    ax.set_ylabel("Success rate\n(% change from session mean)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _savefig(fig, os.path.join(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report",
                             "Fig 13 success vs beta burst occupancy.png"))
    plt.show()


@cell()
def step5b_examples():
    """Step 5b. Example trials for the four occupancy groups (quiet, low, medium, high), IL, LOCAL"""
    global BETA_COL, BLUES, C, GCOL, GROUPS, G_, MODE, PW, REF_EX, REG_EX, S_EX, T, WIN_EX, _, a, a_, aen, ast, ax, axs, be
    global beta, bs, butter, ch, clarity, clean, d, e_, env, f, fftconvolve, fig, freqs, fs, g, hilbert, i, i_, j, k, lim, m
    global occ, pick, pw, r, rec, rows, s_, sd, sosfiltfilt, sp, t, tone, tt, tw, w, x, z
    # ── Step 5b. Example trials for the four occupancy groups (quiet, low, medium, high), IL, LOCAL ──
    from scipy.signal import butter, sosfiltfilt, hilbert, fftconvolve
    S_EX, REF_EX, REG_EX, WIN_EX, MODE = TEST, "LOCAL", "IL", 0.6, "typical"     # MODE: "typical" or "clear"
    GROUPS = [("Quiet", 0.0, 0.0), ("Low", 1e-9, 0.25), ("Medium", 0.25 + 1e-9, 0.50), ("High", 0.50 + 1e-9, 1.0)]
    GCOL = ["#fde0d9", "#f4a582", "#d6604d", "#b2182b"]

    z = np.load(os.path.join(EXACT_DIR, S_EX + ".npz")); T = pd.read_csv(os.path.join(EXACT_DIR, S_EX + "_trials.csv"))
    k = f"{REF_EX}|beta 15-30|{REG_EX}"
    bs, be = z[k + "|burst_start_s"], z[k + "|burst_end_s"]
    ast, aen = z[f"{REF_EX}|{REG_EX}|art_start_s"], z[f"{REF_EX}|{REG_EX}|art_end_s"]
    ch = str(z[f"{REF_EX}|{REG_EX}|channel"][0])

    # occupancy in the 0.6 s before the tone (exact times), long correct + late, no artifact in the window
    rows = []
    for i, r in T[(T["type"] == "long") & T["label"].isin(["correct", "late"]) & T["tone_s"].notna()].iterrows():
        tone = r["tone_s"]; a = tone - WIN_EX
        if np.any((ast < tone) & (aen > a)): continue
        occ = np.clip(np.minimum(be, tone) - np.maximum(bs, a), 0, None).sum() / WIN_EX
        rows.append(dict(trial=i, label=r["label"], tone=tone, occ=occ))
    C = pd.DataFrame(rows)
    C["group"] = np.select([C.occ <= 0, C.occ <= 0.25, C.occ <= 0.5], ["Quiet", "Low", "Medium"], default="High")
    print(f"{S_EX} | {REG_EX} {REF_EX} channel {ch} | trials per group:", C.group.value_counts().reindex([g for g, _, _ in GROUPS]).to_dict())

    # LFP, beta-filtered signal and envelope of this channel
    rec = fn.load_session(PROC, S_EX, XB, refs=(REF_EX,))[0][REF_EX]; fs = rec.get_sampling_frequency()
    x = rec.get_traces(channel_ids=[ch], return_in_uV=True)[:, 0].astype(float); t = np.arange(len(x)) / fs
    beta = sosfiltfilt(butter(4, [15, 30], btype="band", fs=fs, output="sos"), x); env = np.abs(hilbert(beta))

    def clarity(r):                                   # strongest burst envelope / median envelope of the window
        a, b = int((r.tone - WIN_EX) * fs), int(r.tone * fs); w = env[a:b]
        return w.max() / np.median(w)
    C["clarity"] = C.apply(clarity, axis=1)

    pick = {}
    for g, _, _ in GROUPS:
        d = C[C.group == g]
        if d.empty: print(f"{g}: no trial in this session"); continue
        if MODE == "clear" and g != "Quiet":
            pick[g] = d.sort_values("clarity", ascending=False).iloc[0]
        else:
            pick[g] = d.iloc[(d.occ - d.occ.median()).abs().argsort().iloc[0]]
        print(f"{g}: {len(d)} trials, occupancy median {100 * d.occ.median():.0f} % | shown trial {pick[g].trial} "
              f"({pick[g].label}, {100 * pick[g].occ:.0f} %)")

    # time–frequency power 15–40 Hz (Morlet 7 cycles) / 90th percentile per frequency over the session (artifacts excluded), as Step 2
    freqs = np.arange(15, 41, 1.0); clean = np.ones(len(x), bool)
    for a_, e_ in zip(ast, aen): clean[(t >= a_) & (t < e_)] = False
    PW = []
    for f in freqs:
        sd = 7 / (2 * np.pi * f); tw = np.arange(-3.5 * sd, 3.5 * sd, 1 / fs)
        w = np.exp(2j * np.pi * f * tw) * np.exp(-tw ** 2 / (2 * sd ** 2)); w /= np.abs(w).sum()
        pw = np.abs(fftconvolve(x, w, mode="same")) ** 2; PW.append(pw / np.percentile(pw[clean], 90))
    PW = np.array(PW, dtype=np.float32)

    # figure (paper style, supplement): one column per group, rows = raw LFP and 15–30 Hz signal; no time–frequency map
    BLUES = ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"]               # as Fig 13 (quiet → high)
    BETA_COL = "#1565C0"
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    G_ = [g for g, _, _ in GROUPS if g in pick]
    fig, axs = plt.subplots(2, len(G_), figsize=(2.3 * len(G_), 3.4), sharex=True, facecolor="white")
    for j, g in enumerate(G_):
        r = pick[g]; m = (t >= r.tone - WIN_EX) & (t <= r.tone); tt = t[m] - r.tone
        for s_, e_ in zip(bs, be):                                       # shade bursts inside the window (red, as Fig 7)
            if e_ > r.tone - WIN_EX and s_ < r.tone:
                for ax in axs[:, j]: ax.axvspan(max(s_, r.tone - WIN_EX) - r.tone, min(e_, r.tone) - r.tone, color="#C0392B", alpha=0.12, lw=0)
        axs[0, j].plot(tt, x[m], color="0.45", lw=0.7)
        axs[1, j].plot(tt, beta[m], color=BETA_COL, lw=0.9)
        axs[0, j].set_title(f"{g} ({100 * r.occ:.0f} %)", fontsize=9, fontweight="bold")
        axs[0, j].add_patch(plt.Rectangle((0, 1.16), 1, 0.05, transform=axs[0, j].transAxes,
                                          color=BLUES[[x_[0] for x_ in GROUPS].index(g)], clip_on=False))
        axs[1, j].set_xlabel("Time from tone (s)")
        for ax in axs[:, j]:
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0, 0].set_ylabel("LFP (µV)"); axs[1, 0].set_ylabel("15–30 Hz (µV)")
    for i_ in range(2):                                                 # same y-scale across groups per row
        lim = max(max(abs(v) for v in a_.get_ylim()) for a_ in axs[i_]); [a_.set_ylim(-lim, lim) for a_ in axs[i_]]
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    _savefig(fig, os.path.join(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\supplementary",
                             "Fig S17 beta occupancy group examples.png"))
    plt.show()


@cell(needs={
    'step5b_examples': ['GROUPS', 'WIN_EX', 'be', 'beta', 'bs', 'pick', 't', 'x'],
    'step5_success': ['OBINS', 'OK', 'P_ANOVA', 'P_TREND', 'ST', 'TREND'],
})
def fig13():
    """Fig 13 (report): A example trials of the four occupancy groups (LFP, 15–30 Hz), B success rate per group"""
    global BETA_COL, BLUES, G_, REPORT, SUPP, ax, axA, axB, b_, e_, fig, g, gA, gs, i_, j, lim, m, ptxt, r, s_, sp, tt
    # ── Fig 13 (report): A example trials of the four occupancy groups (LFP, 15–30 Hz), B success rate per group ──
    #    needs: Step 5 (ST, OK, OBINS, P_ANOVA, TREND, P_TREND) and Step 5b (pick, t, x, beta, bs, be, GROUPS, WIN_EX)
    REPORT = r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report"
    SUPP   = r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\supplementary"

    BLUES = ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"]               # quiet → high
    BETA_COL = "#1565C0"
    ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig = plt.figure(figsize=(11.5, 3.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.25)
    gA = gs[0, 0].subgridspec(2, 4, hspace=0.15, wspace=0.3)
    G_ = [g for g, _, _ in GROUPS if g in pick]
    axA = np.array([[fig.add_subplot(gA[i, j]) for j in range(len(G_))] for i in range(2)])

    # A: example trials
    for j, g in enumerate(G_):
        r = pick[g]; m = (t >= r.tone - WIN_EX) & (t <= r.tone); tt = t[m] - r.tone
        for s_, e_ in zip(bs, be):                                       # bursts in the window (red, as Fig 7)
            if e_ > r.tone - WIN_EX and s_ < r.tone:
                for ax in axA[:, j]: ax.axvspan(max(s_, r.tone - WIN_EX) - r.tone, min(e_, r.tone) - r.tone, color="#C0392B", alpha=0.12, lw=0)
        axA[0, j].plot(tt, x[m], color="0.45", lw=0.7)
        axA[1, j].plot(tt, beta[m], color=BETA_COL, lw=0.9)
        axA[0, j].set_title(f"{g} ({100 * r.occ:.0f} %)", fontsize=8.5, fontweight="bold")
        axA[0, j].add_patch(plt.Rectangle((0, 1.17), 1, 0.05, transform=axA[0, j].transAxes,
                                          color=BLUES[[x_[0] for x_ in GROUPS].index(g)], clip_on=False))
        axA[0, j].tick_params(labelbottom=False)
        axA[1, j].set_xlabel("Time from tone (s)")
        for ax in axA[:, j]:
            ax.set_xlim(-WIN_EX, 0)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axA[0, 0].set_ylabel("LFP (µV)"); axA[1, 0].set_ylabel("15–30 Hz (µV)")
    for i_ in range(2):                                                 # same y-scale across groups per row
        lim = max(max(abs(v) for v in a_.get_ylim()) for a_ in axA[i_]); [a_.set_ylim(-lim, lim) for a_ in axA[i_]]
    axA[0, 0].text(-0.45, 1.3, "A", transform=axA[0, 0].transAxes, fontsize=12, fontweight="bold")

    # B: success rate per group
    axB = fig.add_subplot(gs[0, 1])
    for b_ in OK:
        axB.bar(b_, ST["mean"].iat[b_], width=0.65, color=BLUES[b_], edgecolor="k", lw=0.4)
        axB.errorbar(b_, ST["mean"].iat[b_], yerr=ST["sem"].iat[b_], color="k", capsize=3, lw=1)
    axB.text(0.98, 0.98, f"ANOVA {ptxt(P_ANOVA)}\ntrend {TREND:+.1f} % per group\n{ptxt(P_TREND)}",
             transform=axB.transAxes, ha="right", va="top", fontsize=7.5)
    axB.axhline(0, color="k", lw=0.8)
    axB.set_xticks(range(4)); axB.set_xticklabels(OBINS, fontsize=7.5); axB.set_xlim(-0.6, 3.6)
    axB.set_xlabel("Burst occupancy 0.6 s before the tone"); axB.set_ylabel("Success rate\n(% change from session mean)")
    for sp in ("top", "right"): axB.spines[sp].set_visible(False)
    axB.text(-0.38, 1.06, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")

    _savefig(fig, os.path.join(REPORT, "Fig 13 beta occupancy groups and success.png"))
    plt.show()


@cell()
def step6_rt():
    """STEP 6. LAST BETA BURST BEFORE THE TONE AND RT (exact times, IL LOCAL, long correct + late, no block)"""
    global BETA_COL, BINS, BLAB, CHECKS, Db, F_D2, G, K, K6, L, P, R6, RCOL6, REPORT, RT_DIR, S, T, TB6, WIN6, Xb, a, ax, b
    global b_, b_ax, base, c, d6, dat, e, exact_trials, f, fig, fit6, fits, g, gap, gbar, glob, hi_b, idx, lab, line, lo_b, lp
    global lse, m, npk, ong, p_, p_bg, p_br, p_c, pct_closer, ptxt, r, r_, rats_, rd, res, rho_bg, rho_br, rho_c, rl, rows, rt
    global s, se_, se_b, se_level, smf, sp, spearmanr, v, warnings, x_, yhi, ylo, z
    # ── STEP 6. LAST BETA BURST BEFORE THE TONE AND RT (exact times, IL LOCAL, long correct + late, no block)
    #    Fig 15 (rats, session, number of bursts) and Fig 16 (mixed model with full 95 % CI) → report; per-rat models ──
    import glob, warnings
    import statsmodels.formula.api as smf
    from scipy.stats import spearmanr
    RT_DIR = os.path.join(BETA_DIR, "RT"); os.makedirs(RT_DIR, exist_ok=True)
    REPORT = r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report"
    BETA_COL, WIN6 = "#1565C0", 1.0                                     # beta blue; window before the tone (s)
    RCOL6 = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}       # rats as in Fig 2D
    pct_closer = lambda b: 100 * (10 ** (-1.0 * b) - 1)                 # % RT change when the last burst ends 1 s CLOSER to the tone
    ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    # ── 1. per-trial table from the exact burst times ──
    def exact_trials(bs, be, pk, ast, aen, tones, w):
        n = len(tones); gap = np.full(n, np.nan); ong = np.zeros(n, bool); npk = np.full(n, np.nan)
        for i, T in enumerate(tones):
            if not np.isfinite(T):
                continue
            a = T - w
            j = np.searchsorted(ast, T, "left") - 1                     # artifact overlapping the window -> trial left out
            if j >= 0 and aen[j] > a:
                continue
            k1 = np.searchsorted(bs, T, "left") - 1                     # last burst that started before the tone
            if k1 >= 0 and be[k1] > a:
                ong[i] = be[k1] > T
                gap[i] = 0.0 if ong[i] else T - be[k1]
            npk[i] = np.searchsorted(pk, T, "left") - np.searchsorted(pk, a, "left")
        return gap, ong, npk

    rows, K6 = [], "LOCAL|beta 15-30|IL"
    for f in sorted(glob.glob(os.path.join(EXACT_DIR, "*.npz"))):
        s = os.path.basename(f)[:-4]; z = np.load(f); T = pd.read_csv(f[:-4] + "_trials.csv")
        if K6 + "|burst_start_s" not in z.files:
            print("no IL LOCAL bursts:", s); continue
        rt = (T["release_s"] - T["tone_s"]).values
        idx = np.where(np.isin(T["label"].values, ["correct", "late"]) & (T["type"].values == "long") & np.isfinite(rt))[0]
        gap, ong, npk = exact_trials(z[K6 + "|burst_start_s"], z[K6 + "|burst_end_s"], z[K6 + "|peak_s"],
                                     z["LOCAL|IL|art_start_s"], z["LOCAL|IL|art_end_s"], T["tone_s"].values[idx], WIN6)
        rows.append(pd.DataFrame({"session": s, "rat": str(RAT[s]), "label": T["label"].values[idx], "trial": idx,
                                  "pos": idx / max(len(T) - 1, 1), "rt_s": rt[idx], "gap_s": gap, "n_peaks": npk,
                                  "burst_at_tone": ong.astype(float)}))
    TB6 = pd.concat(rows, ignore_index=True)
    _table(TB6, os.path.join(RT_DIR, "step6_RT_trials.csv"), index=False)

    Db = TB6[TB6.gap_s.notna() & TB6.n_peaks.notna() & (TB6.rt_s > 0)].copy()
    Db["logrt"] = np.log10(Db["rt_s"])
    Db["trial_z"] = Db.groupby("session")["trial"].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0 * x)
    d6 = Db[Db.gap_s > 0].copy()                                         # last burst ended before the tone
    d6["peaks_z"] = (d6["n_peaks"] - d6["n_peaks"].mean()) / d6["n_peaks"].std()
    d6["late"] = (d6["label"] == "late").astype(float); d6["rt_ms"] = 1000 * d6["rt_s"]
    print(f"{TB6.session.nunique()} sessions | long correct+late trials {len(TB6)} | with a burst in the 1 s {len(Db)} | "
          f"still ON at the tone (left out) {int((Db.gap_s == 0).sum())} | kept {len(d6)} ({int(d6.late.sum())} late)")

    # ── 2. rats, position in the session, number of bursts ──
    S = d6.groupby(["session", "rat"]).agg(gap=("gap_s", "median"), rt=("rt_ms", "median"), n=("rt_ms", "size")).reset_index()
    S["rat"] = S["rat"].astype(str); rats_ = ["134", "135", "136"]
    d6["tenth"] = np.clip(np.ceil(d6["pos"] * 10), 1, 10).astype(int)
    P = d6.groupby("tenth").agg(rt=("rt_ms", "median"), gap=("gap_s", "median"), n=("rt_ms", "size"))
    rho_br, p_br = spearmanr(d6["pos"], d6["rt_ms"]); rho_bg, p_bg = spearmanr(d6["pos"], d6["gap_s"])
    d6["k"] = d6["n_peaks"].clip(upper=8).astype(int)
    K = d6.groupby("k")["gap_s"].agg(med="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75), n="size")
    rho_c, p_c = spearmanr(d6["n_peaks"], d6["gap_s"])
    print("\nA. per rat: RT (ms), mean of session medians ± SEM")
    for r in rats_:
        v = S.loc[S.rat == r, "rt"]; print(f"   rat {r}: {v.mean():.0f} ± {v.sem():.0f}  ({len(v)} sessions)")
    print(f"B. position in session vs RT: rho = {rho_br:+.3f}, p = {p_br:.4f} ; vs end of last burst: rho = {rho_bg:+.3f}, p = {p_bg:.4f}")
    print(f"C. number of bursts vs end of last burst: rho = {rho_c:+.3f}, p = {p_c:.4g}")

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.0), facecolor="white", gridspec_kw=dict(width_ratios=[0.7, 1.2, 1], wspace=0.45))
    a = ax[0]                                                         # A: rats differ in RT
    m = [S.loc[S.rat == r, "rt"].mean() for r in rats_]; e = [S.loc[S.rat == r, "rt"].sem() for r in rats_]
    a.bar(range(3), m, yerr=e, width=0.6, color=[RCOL6[r] for r in rats_], edgecolor="k", lw=0.4, capsize=3)
    a.set_xticks(range(3)); a.set_xticklabels(["Rat 1", "Rat 2", "Rat 3"]); a.set_ylabel("Reaction time (ms)")
    b_ax = ax[1]                                                      # B: RT along the session
    b_ax.plot(P.index, P.rt, "o-", color="0.25", lw=1.4, ms=3)
    b_ax.set_xticks(range(1, 11)); b_ax.set_xlabel("Position in the session (tenths)"); b_ax.set_ylabel("Median reaction time (ms)")
    b_ax.text(0.02, 0.97, f"rho = {rho_br:+.2f}, {ptxt(p_br)}", transform=b_ax.transAxes, ha="left", va="top", fontsize=8)
    c = ax[2]                                                         # C: more bursts -> last burst closer to the tone
    c.errorbar(K.index, 1000 * K.med, yerr=[1000 * (K.med - K.q1), 1000 * (K.q3 - K.med)], fmt="o-", color=BETA_COL, capsize=3, lw=1.4, ms=3)
    c.set_xticks(K.index); c.set_xticklabels([str(i) if i < 8 else "8+" for i in K.index])
    c.set_xlabel("Number of beta bursts in the 1 s before the tone"); c.set_ylabel("End of the last burst\nbefore the tone (ms)")
    c.text(0.98, 0.97, f"rho = {rho_c:+.2f}, {ptxt(p_c)}", transform=c.transAxes, ha="right", va="top", fontsize=8)
    for x_, L in zip(ax, "ABC"):
        for sp in ("top", "right"): x_.spines[sp].set_visible(False)
        x_.text(-0.3 if L == "A" else -0.22, 1.06, L, transform=x_.transAxes, fontsize=12, fontweight="bold")
    _savefig(fig, os.path.join(REPORT, "Fig 16 reaction time rats session and bursts.png"))
    plt.show()

    # ── 3. mixed model and checks ──
    F_D2 = "logrt ~ gap_s + rat + trial_z + peaks_z"
    def fit6(f, dat, lab=""):
        for m_ in ("lbfgs", "bfgs", "cg", "nm"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r_ = smf.mixedlm(f, dat, groups=dat["session"]).fit(method=m_)
                if np.isfinite(r_.bse["gap_s"]):
                    if m_ != "lbfgs": print(f"  [{lab}] fitted with method '{m_}'")
                    return r_
            except Exception:
                pass
        print(f"  [{lab}] mixed model failed -> OLS with session-clustered SE")
        return smf.ols(f, dat).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dat["session"])[0]})

    CHECKS = [("all trials", F_D2, d6), ("+ late term", F_D2 + " + late", d6),
              ("correct trials only", F_D2, d6[d6.label == "correct"]), ("late trials only", F_D2, d6[d6.label == "late"])]
    CHECKS += [(f"rat {r} alone", "logrt ~ gap_s + trial_z + peaks_z", d6[d6.rat == r]) for r in rats_]     # per rat (methods)
    rows, fits = [], {}
    for lab, f, dat in CHECKS:
        r_ = fit6(f, dat, lab); fits[lab] = r_
        b_, se_, p_ = r_.params["gap_s"], r_.bse["gap_s"], r_.pvalues["gap_s"]
        rows.append(dict(model=lab, n=len(dat), sessions=dat.session.nunique(), pct=pct_closer(b_),
                         lo=pct_closer(b_ + 1.96 * se_), hi=pct_closer(b_ - 1.96 * se_), p=p_))
    R6 = pd.DataFrame(rows)
    print("\n% change in RT when the last burst ends 1 s closer to the tone:"); print(R6.round(3).to_string(index=False))
    rl, rd = fits["+ late term"], fits["all trials"]
    print(f"late term: {100 * (10 ** rl.params['late'] - 1):+.0f} % RT, p = {rl.pvalues['late']:.2g} | "
          f"number of bursts: {100 * (10 ** rd.params['peaks_z'] - 1):+.1f} % RT per +1 SD, p = {rd.pvalues['peaks_z']:.4f}")
    _table(R6, os.path.join(RT_DIR, "step6_RT.csv"), index=False)

    # ── 4. Fig 16: model line with full 95 % CI + adjusted RT per range (ranges for illustration only) ──
    b, gbar, g = rd.params["gap_s"], d6["gap_s"].mean(), np.linspace(0, 1, 101)
    Xb = np.dot(rd.model.exog, rd.fe_params.values)
    base = d6["logrt"].mean() - b * gbar                               # anchor at the real mean RT (slope and test unchanged)
    res = d6["logrt"].values - Xb
    res = res - pd.Series(res, index=d6.index).groupby(d6["session"]).transform("mean").values   # remove each session's baseline
    d6["lpr"] = base + b * d6["gap_s"].values + res
    BINS, BLAB = [0.0, 0.1, 0.25, 0.5, 1.0001], ["0–0.1", "0.1–0.25", "0.25–0.5", "0.5–1.0"]
    d6["gbin"] = pd.cut(d6["gap_s"], BINS, labels=BLAB)
    G = d6.groupby("gbin", observed=True).agg(x=("gap_s", "mean"), m=("lpr", "mean"), se=("lpr", "sem"), n=("lpr", "size")).reset_index()
    G["ms"], G["lo"], G["hi"] = 1000 * 10 ** G.m, 1000 * 10 ** (G.m - G.se), 1000 * 10 ** (G.m + G.se)
    print("\nadjusted RT per range (illustration only)"); print(G[["gbin", "n", "ms"]].round(0).to_string(index=False))
    _table(G, os.path.join(RT_DIR, "step6_RT_ranges.csv"), index=False)

    se_b = rd.bse["gap_s"]                                                        # SE of the slope (model)
    se_level = d6.groupby("session")["lpr"].mean().sem()                          # SE of the mean (log) RT level across sessions
    lp = base + b * g                                                              # model line (log10 RT)
    lse = np.sqrt(se_level ** 2 + (se_b * (g - gbar)) ** 2)                        # level + slope uncertainty
    line, lo_b, hi_b = 1000 * 10 ** lp, 1000 * 10 ** (lp - 1.96 * lse), 1000 * 10 ** (lp + 1.96 * lse)

    fig, a = plt.subplots(figsize=(3.8, 3.0), facecolor="white")
    a.fill_between(g, lo_b, hi_b, color=BETA_COL, alpha=0.15, lw=0)
    a.plot(g, line, color=BETA_COL, lw=1.8, label="Mixed model")
    a.errorbar(G.x, G.ms, yerr=[G.ms - G.lo, G.hi - G.ms], fmt="o", color="0.25", ms=4, capsize=3, lw=1, zorder=3,
               label="Adjusted RT per range (± SEM)")
    ylo, yhi = min(G.lo.min(), lo_b.min()), max(G.hi.max(), hi_b.max())
    a.set_ylim(ylo - 0.12 * (yhi - ylo), yhi + 0.3 * (yhi - ylo)); a.set_xlim(-0.03, 1.03)
    a.set_xlabel("End of the last beta burst before the tone (s)"); a.set_ylabel("Reaction time (ms, adjusted)")
    a.text(0.97, 0.03, f"{R6.loc[0, 'pct']:+.1f} % per 1 s closer, {ptxt(R6.loc[0, 'p'])}", transform=a.transAxes,
           ha="right", va="bottom", fontsize=8)
    a.legend(frameon=False, fontsize=7.5, loc="upper right")
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    _savefig(fig, os.path.join(REPORT, "Fig 17 last beta burst and RT.png"))
    plt.show()


@cell(needs={
    'load_burst_tables': ['files'],
})
def step7_after_error():
    """STEP 7. BETA 1 s BEFORE / AFTER THE PRESS, AFTER AN ERROR vs AFTER A CORRECT — SIGN TEST ACROSS SESSIONS"""
    global A0P, BAND, BIN, CV, E, G, GCOL, MIN_TR, Op, RES, SES, T, WIN, X, a, a_, b_, binomtest, col, dif, fig, g, hh, hi_, i
    global kc, ke, kk, kp, lab, lo_, m_, n_, nn, ptxt, r, rows, s, se, sel, sig, sp, stars, tc, tp, tw, up, v, w, y0, y1, yy
    global z
    # ── STEP 7. BETA 1 s BEFORE / AFTER THE PRESS, AFTER AN ERROR vs AFTER A CORRECT — SIGN TEST ACROSS SESSIONS
    #    (IL LOCAL, correct long trials) → report Fig 17 ──
    from scipy.stats import binomtest
    BAND = "beta 15-30"; BIN = fn.BURST_BIN; A0P = fn.BURST_A0["press"]; MIN_TR = 3
    GCOL = {"correct": ("#2CA02C", "After a correct trial"), "error": ("#C0392B", "After an error")}
    WIN = {"1 s before the press": (-1.0, 0.0), "1 s after the press": (0.0, 1.0)}
    ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    # ── 1. correct long trials after a correct trial or an error (late or early), 50-ms bins, −1 to +1 s around the press ──
    rows, CV = [], []
    for s in files:
        z = np.load(os.path.join(BURST_DIR, s + ".npz")); T = pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))
        kp = f"LOCAL|{BAND}|IL|press|occ"
        if kp not in z.files or z[kp].shape[0] != len(T): continue
        Op = z[kp]; tp = A0P + BIN * np.arange(Op.shape[1]); sel = (tp >= -1.0 - 1e-9) & (tp + BIN <= 1.0 + 1e-9); tw = tp[sel]
        for i in np.where((T["type"].values == "long") & (T["label"].values == "correct"))[0]:
            if i == 0 or T["trial_index"].iat[i - 1] != T["trial_index"].iat[i] - 1 or T["label"].iat[i - 1] not in ("correct", "late", "early"): continue
            v = Op[i, sel]
            if not np.isfinite(v).all(): continue
            r = dict(session=s, grp="correct" if T["label"].iat[i - 1] == "correct" else "error")
            for w, (a_, b_) in WIN.items():
                r[w] = 100 * v[(tw >= a_ - 1e-9) & (tw + BIN <= b_ + 1e-9)].mean()
            rows.append(r); CV.append(100 * v)
    E = pd.DataFrame(rows); CV = np.vstack(CV); tc = tw + BIN / 2
    kc, ke = (E.grp == "correct").values, (E.grp == "error").values

    # ── 2. sign test across sessions (sessions with ≥ 3 trials of each kind), per window ──
    n_ = E.groupby(["session", "grp"]).size().unstack(fill_value=0)
    SES = n_[(n_["correct"] >= MIN_TR) & (n_["error"] >= MIN_TR)].index
    G = E[E.session.isin(SES)].groupby(["session", "grp"])[list(WIN)].mean().unstack("grp")
    RES = {}
    for w in WIN:
        dif = G[(w, "error")] - G[(w, "correct")]; up = int((dif > 0).sum()); nn = int((dif != 0).sum())
        RES[w] = dict(up=up, n=nn, p=binomtest(up, nn).pvalue, mean_diff=dif.mean(),
                      after_error=G[(w, "error")].mean(), after_correct=G[(w, "correct")].mean())
        print(f"{w}: after an error {RES[w]['after_error']:.1f} %, after a correct {RES[w]['after_correct']:.1f} %; "
              f"higher after an error in {up} of {nn} sessions, sign test {ptxt(RES[w]['p'])} (mean difference {dif.mean():+.2f} %-points)")
    print(f"trials: after correct {kc.sum()}, after an error {ke.sum()} | sessions: {len(SES)}")
    _table(pd.DataFrame(RES).T, os.path.join(BETA_DIR, "step7_final_signtest.csv"))

    # ── 3. figure (paper style, report Fig 17) ──
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, a = plt.subplots(figsize=(4.4, 3.2), facecolor="white")
    for g, kk in (("correct", kc), ("error", ke)):
        X = pd.DataFrame(CV[kk]).T.rolling(7, center=True, min_periods=1).mean().T.values       # 350-ms smoothing (display only)
        m_, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(kk.sum())
        col, lab = GCOL[g]
        a.fill_between(tc, m_ - se, m_ + se, color=col, alpha=0.15, lw=0)
        a.plot(tc, m_, color=col, lw=1.6, label=lab)
    a.axvline(0, color="k", ls="--", lw=0.8)
    a.text(0, 1.0, "press", transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    a.set_xlim(-1, 1)
    y0, y1 = a.get_ylim(); hh = y1 - y0; a.set_ylim(y0, y1 + 0.3 * hh); yy = y1 + 0.08 * hh
    for w, (lo_, hi_) in WIN.items():
        r = RES[w]; sig = r["p"] < 0.05; col = "k" if sig else "0.6"
        a.plot([lo_ + 0.03, hi_ - 0.03], [yy, yy], color=col, lw=2.5, solid_capstyle="butt")
        a.text((lo_ + hi_) / 2, yy + 0.02 * hh, f"{stars(r['p'])} ({r['up']} of {r['n']} sessions)", ha="center", va="bottom",
               fontsize=8, color=col)
    a.set_xlabel("Time from press (s)"); a.set_ylabel("Beta burst occupancy (%)\n(350-ms smoothing)")
    a.legend(frameon=False, fontsize=8, loc="lower left")
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    _savefig(fig, os.path.join(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report",
                             "Fig 18 beta after an error.png"))
    plt.show()


@cell()
def step8_properties():
    """STEP 8. IL BETA BURST PROPERTIES AT THE 98th PERCENTILE, DESCRIPTIVE (as Karvat 2020 Fig. 2): ALL SESSIONS, RATS POOLED (needs cell 0)"""
    global B, BETA_COL, COL, DB, GEN_DIR, IBb, INFO, MEAS, MIN, P, PROP_DIR, SHOW_SHORT, T, ax, axs, be, bins, bs, col, d0, f
    global fig, glob, h, i, i0, lab, letter, ok, rows, s, sp, t0, t1, v, xl, xt, yt, z
    # ── STEP 8. IL BETA BURST PROPERTIES AT THE 98th PERCENTILE, DESCRIPTIVE (as Karvat 2020 Fig. 2): ALL SESSIONS, RATS POOLED (needs cell 0) ──
    import glob
    GEN_DIR  = os.path.join(BETA_DIR, "Burst properties (qualitative)"); os.makedirs(GEN_DIR, exist_ok=True)
    PROP_DIR = os.path.join(DATA_OUT, "Burst properties p98")
    COL = "#2CA02C"                                                      # green bars
    SHOW_SHORT = False                                                   # do not DRAW bursts < 10 ms (duration) and intervals < 0.1 s (interval)                                                  # False = do not DRAW bursts < 10 ms (duration) and intervals < 0.1 s (interval)
    pd.set_option("display.width", 250)

    # ── 1. per session, task period: one row per merged burst = its strongest peak ──
    rows, INFO = [], []
    for f in sorted(glob.glob(os.path.join(PROP_DIR, "*.npz"))):
        s = os.path.basename(f)[:-4]; z = np.load(f)
        t0, t1 = float(z["task_start_s"][0]), float(z["task_end_s"][0])
        P = pd.DataFrame(dict(t=z["peak_s"], freq=z["peak_hz"], span=z["span_hz"] + 1, dur_ms=1000 * z["dur_s"], power=z["power_x_median"]))
        P = P[(P.t >= t0) & (P.t <= t1)]
        bs, be = z["burst_start_s"], z["burst_end_s"]
        i = np.searchsorted(bs, P.t.values, "right") - 1
        ok = (i >= 0) & (P.t.values <= be[np.clip(i, 0, None)])                     # peak lies inside a merged burst
        P = P[ok].assign(b=i[ok]).sort_values("power").groupby("b").last().sort_values("t")
        P["ibi_s"] = np.r_[np.nan, np.diff(P.t.values)]
        rows.append(P.assign(session=s, rat=s.split("_r0000")[1][:3]))
        INFO.append(dict(session=s, rat=s.split("_r0000")[1][:3], minutes=(t1 - t0) / 60, n_bursts=len(P)))
    B = pd.concat(rows, ignore_index=True); INFO = pd.DataFrame(INFO)
    MIN = INFO.minutes.sum()
    print(f"{len(INFO)} sessions, {MIN:.0f} min of task time | {len(B)} bursts | per session median "
          f"{INFO.n_bursts.median():.0f} bursts, {(INFO.n_bursts / INFO.minutes).median():.0f} bursts per minute")

    # first bar edge for duration / interval: the smallest value, or 10 ms / 0.1 s if the short ones are not drawn
    d0 = B.dur_ms.min() if SHOW_SHORT else 10
    i0 = B.ibi_s.min() if SHOW_SHORT else 0.1
    DB, IBb = np.arange(10, 301, 10), np.arange(0.1, 5.01, 0.1)
    # (column, x-label, bin edges, x-limits, x-tick labels, y-tick labels) — ticks are only the numbers shown, not limits
    MEAS = [("freq",   "Peak frequency (Hz)",       np.arange(14.5, 31, 1.0),           (13.5, 30.5), [15, 20, 25, 30],          [0, 2, 4]),
            ("span",   "Frequency span (Hz)",       np.arange(0.5, 21, 1.0),            (0, 20.5),    [0, 5, 10, 15, 20],        [0, 5, 10]),
            ("dur_ms", "Burst duration (ms)",       np.r_[d0, DB[DB > d0]],             (0, 300),     [0, 100, 200, 300],        [0, 5]),
            ("ibi_s",  "Inter-burst interval (s)",  np.r_[i0, IBb[IBb > i0 + 1e-9]],    (0, 5),       [0, 1, 2, 3, 4, 5],        [0, 5]),
            ("power",  "Burst power (FOM)",         np.arange(5, 25.01, 0.5),           (5, 25),      [6, 8, 10, 12, 14, 16, 18, 20, 22, 24], [0, 5, 10])]

    # ── 2. numbers: median and minimum (ALL bursts, also those outside the drawn x-range) ──
    T = pd.DataFrame({"median": B[[m[0] for m in MEAS]].median(), "minimum": B[[m[0] for m in MEAS]].min(),
                      "% outside x-range": [100 * np.mean((B[c].dropna() < b[0]) | (B[c].dropna() > b[-1])) for c, _, b, *_ in MEAS]})
    T.loc["bursts_per_min"] = [len(B) / MIN, np.nan, np.nan]
    T.index = ["peak frequency (Hz)", "span (Hz)", "duration (ms)", "inter-burst interval (s)", "power (FOM)", "bursts per minute"]
    print("\nall bursts of the task period:"); print(T.round(3).to_string())
    _table(T, os.path.join(GEN_DIR, "step8_burst_properties_p98.csv"))

    # ── 3. figure (paper style, supplement Fig S18): beta blue bars, one row of five panels ──
    BETA_COL = "#1565C0"
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, axs = plt.subplots(1, len(MEAS), figsize=(12.5, 2.4), facecolor="white", gridspec_kw={"wspace": 0.45})
    for ax, (col, lab, bins, xl, xt, yt), letter in zip(axs, MEAS, "ABCDE"):
        v = B[col].dropna()
        h = np.histogram(v, bins)[0] / MIN
        ax.bar(bins[:-1], h, width=np.diff(bins), align="edge", color=BETA_COL, edgecolor="white", lw=0.6)
        ax.axvline(v.median(), color="k", ls="--", lw=1)
        ax.set_xlim(*xl); ax.set_xticks(xt[::2] if len(xt) > 6 else xt)
        ax.set_ylim(0, h.max() * 1.1); ax.set_yticks(yt)
        ax.set_xlabel(lab); ax.set_ylabel("Bursts per minute" if letter == "A" else "")
        ax.text(-0.25, 1.05, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _savefig(fig, os.path.join(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\supplementary",
                             "Fig S18 beta burst properties.png"))
    plt.show()


@cell()
def step9_single_bursts():
    """STEP 9. SINGLE IL BETA BURSTS AND THEIR AVERAGE IN 4 TYPICAL SESSIONS (as Karvat 2020 Fig. 3a), 98th PERCENTILE, LOCAL (needs cell 0; data from 06 cell 5e)"""
    global A, AVG_DIR, E, M, MIN_SEL, ODD, OFF, OUT_DIR, R, SAVE, SB_M, SB_TR, SCOL, SHOW_MS, a, ax_, axs, b, c, col, fig, fp
    global load, mhi, mlo, nice, pad, q, r, sp, y, y0
    # ── STEP 9. SINGLE IL BETA BURSTS AND THEIR AVERAGE IN 4 TYPICAL SESSIONS (as Karvat 2020 Fig. 3a), 98th PERCENTILE, LOCAL (needs cell 0; data from 06 cell 5e) ──
    AVG_DIR = os.path.join(DATA_OUT, "Burst averages p98 LOCAL")
    OUT_DIR = os.path.join(BETA_DIR, "Burst properties (qualitative)"); os.makedirs(OUT_DIR, exist_ok=True)
    SAVE = True
    SHOW_MS, MIN_SEL = 150, 150
    ODD = ["20260519_r0000136_wt_2031", "20260510_r0000135_wt_1625", "20260515_r0000135_wt_1953", "20260515_r0000135_wt_1957"]
    SCOL = ["#90CAF9", "#42A5F5", "#1E88E5", "#0D47A1"]                  # blue gradient (beta colour), light → dark (Session 1 → 4)
    pd.set_option("display.width", 250)

    # ── 1. the rule: one typical session per rat + the next closest overall ──
    M = pd.read_csv(os.path.join(AVG_DIR, "burst_average_metrics.csv"), dtype={"rat": str})
    E = M[(M.n_sel >= MIN_SEL) & ~M.session.isin(ODD)].copy()
    for c in ("sel_depth_uV", "sel_clarity"):
        E[c + "_z"] = E.groupby("rat")[c].transform(lambda v: (v - v.median()) / v.std(ddof=1))
    E["dist"] = np.hypot(E.sel_depth_uV_z, E.sel_clarity_z); E = E.sort_values("dist")
    A = [E[E.rat == r].session.iloc[0] for r in sorted(E.rat.unique())]
    A = sorted(A + [E[~E.session.isin(A)].session.iloc[0]])                    # 4 sessions, date order
    print(f"eligible {len(E)} of {len(M)} sessions | chosen (Session 1–4):")
    print(E.set_index("session").loc[A].reset_index()[["session", "n_sel", "sel_depth_uV", "sel_clarity", "sel_cycles", "dist"]]
          .assign(label=[f"Session {i + 1}" for i in range(4)]).round(2).to_string(index=False))
    print(f"all 60 sessions: mean of selected bursts {M.sel_cycles.median():.2f} cycles (range {M.sel_cycles.min():.2f}–{M.sel_cycles.max():.2f})")

    # ── 2. load the saved traces / means ──
    def load(s):
        z = np.load(os.path.join(AVG_DIR, s + ".npz")); t = 1000 * z["t_s"]; k = np.abs(t) <= SHOW_MS
        return dict(t=t[k], tr=z["traces_sel"][:, k], on=z["on_sel"][:, k],
                    m=z["mean_sel"][k] - np.median(z["mean_sel"][np.abs(t) > 200]), n=int(z["n_sel"][0]))
    R = [load(s) for s in A]
    def nice(v):
        e = 10 ** np.floor(np.log10(v)); return min((1, 2, 5, 10), key=lambda m: abs(m * e - v)) * e
    OFF = 1.1 * np.median([np.ptp(r["tr"], axis=1).mean() for r in R])
    SB_TR, SB_M = nice(OFF * 0.6), nice(0.5 * max(np.ptp(r["m"]) for r in R))
    mlo = min(r["m"].min() for r in R); mhi = max(r["m"].max() for r in R); pad = 0.15 * (mhi - mlo)

    # ── 3. figure ──
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, axs = plt.subplots(2, 4, figsize=(13.2, 9), facecolor="white", gridspec_kw=dict(height_ratios=[3, 1.2]), constrained_layout=True)
    for c, (r, col) in enumerate(zip(R, SCOL)):
        a, b = axs[0, c], axs[1, c]
        for q in range(len(r["tr"])):
            y = r["tr"][q] - q * OFF
            a.plot(r["t"], y, color="k", lw=1.0); a.plot(r["t"], np.where(r["on"][q], y, np.nan), color=col, lw=1.8)
        a.axvline(0, color="k", ls="--", lw=1.4); a.set_title(f"Session {c + 1}", color=col, fontweight="bold")
        b.plot(r["t"], r["m"], color=col, lw=2.2); b.axvline(0, color="k", ls="--", lw=1.4); b.set_ylim(mlo - pad, mhi + pad)
        b.text(0.02, 0.97, f"n = {r['n']}", transform=b.transAxes, va="top", fontsize=10)
        for ax_ in (a, b):
            ax_.set_xlim(-SHOW_MS, SHOW_MS); ax_.set_xticks([-140, -70, 0, 70, 140]); ax_.set_yticks([])
            for sp in ("top", "right", "left"): ax_.spines[sp].set_visible(False)
        b.set_xlabel("Time (ms)")
        if c == 0: a.set_ylabel("Raw LFP"); b.set_ylabel("Mean LFP")
        if c == 3:
            y0 = -(len(r["tr"]) - 0.5) * OFF
            a.plot([SHOW_MS + 8] * 2, [y0, y0 + SB_TR], color="k", lw=2.5, clip_on=False)
            a.text(SHOW_MS + 14, y0 + SB_TR / 2, f"{SB_TR:.0f} µV", va="center", fontsize=11)
            b.plot([SHOW_MS + 8] * 2, [mlo, mlo + SB_M], color="k", lw=2.5, clip_on=False)
            b.text(SHOW_MS + 14, mlo + SB_M / 2, f"{SB_M:.0f} µV", va="center", fontsize=11)
    fig.suptitle("Single IL beta bursts and their average (98th percentile, ≥ 70 ms, 20–25 Hz, LOCAL)", fontsize=15, fontweight="bold")
    fp = os.path.join(OUT_DIR, "step9_single_bursts_4_sessions_p98.png")
    if SAVE: _savefig(fig, fp)
    plt.show()


@cell()
def step10_shape():
    """STEP 10. SHAPE AND TIME-FREQUENCY OF THE AVERAGE IL BETA BURST IN ALL 60 SESSIONS, 98th PERCENTILE, LOCAL (needs cell 0; data from 06 cell 5e)"""
    global AVG_DIR, FREQS, N, Nk, OUT_DIR, P, RAT, RATC, RNAME, S, SAVE, SHOW_MS, T, TF, TFR_MS, a, above, b, c, col, f, fband
    global fig, fmax, fs, glob, hi, im, k, kt, lo, low, m, med, r, rhythm, row, s, sim, sp, ss, t, tk, z
    # ── STEP 10. SHAPE AND TIME-FREQUENCY OF THE AVERAGE IL BETA BURST IN ALL 60 SESSIONS, 98th PERCENTILE, LOCAL (needs cell 0; data from 06 cell 5e) ──
    import glob
    import scipy.signal as ss
    AVG_DIR = os.path.join(DATA_OUT, "Burst averages p98 LOCAL")
    OUT_DIR = os.path.join(BETA_DIR, "Burst properties (qualitative)"); os.makedirs(OUT_DIR, exist_ok=True)
    SAVE, SHOW_MS, TFR_MS = True, 150, 250
    FREQS = np.arange(10, 41)
    RATC = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}

    # ── 1. normalized means and their time-frequency maps ──
    S, N, RAT, TF = [], [], [], []
    for f in sorted(glob.glob(os.path.join(AVG_DIR, "*.npz"))):
        z = np.load(f); s = os.path.basename(f)[:-4]; fs = float(z["fs"][0])
        t = 1000 * z["t_s"]; m = z["mean_sel"] - np.median(z["mean_sel"][np.abs(t) > 200]); c = len(m) // 2
        S.append(s); RAT.append(s.split("_r0000")[1][:3]); N.append(m / abs(m[c]))
        P = np.array([np.abs(ss.fftconvolve(m, fn.morlet(f_, 7, fs), mode="same")) ** 2 for f_ in FREQS])
        TF.append(P / P.max())                                           # each session normalized to its own maximum
    N, TF = np.array(N), np.mean(TF, 0)
    k = np.abs(t) <= SHOW_MS; tk = t[k]; Nk = N[:, k]; med = np.median(Nk, 0)

    # ── 2. shape similarity (leave one out), rhythm, time-frequency summary ──
    sim = np.array([np.corrcoef(Nk[i], np.median(np.delete(Nk, i, 0), 0))[0, 1] for i in range(len(Nk))])
    def rhythm(m):
        b = (tk >= -40) & (tk < 0); a = (tk > 0) & (tk <= 40)
        return 1000 / (tk[a][np.argmax(m[a])] - tk[b][np.argmax(m[b])])
    T = pd.DataFrame(dict(session=S, rat=RAT, similarity=sim, rhythm_hz=[rhythm(m) for m in Nk]))
    print(f"{len(T)} sessions | shape similarity with the other sessions: median r = {T.similarity.median():.2f} "
          f"(range {T.similarity.min():.2f}–{T.similarity.max():.2f}); r >= 0.9 in {(T.similarity >= 0.9).sum()} of {len(T)}")
    print(f"rhythm (peak before -> peak after the trough): median {T.rhythm_hz.median():.1f} Hz (range {T.rhythm_hz.min():.1f}–{T.rhythm_hz.max():.1f})")
    print("per rat (median):"); print(T.groupby("rat")[["similarity", "rhythm_hz"]].median().round(2).to_string())
    low = T[T.similarity < 0.9]
    if len(low): print("sessions with r < 0.9:"); print(low.round(2).to_string(index=False))
    c = len(t) // 2; fmax = FREQS[np.argmax(TF[:, c])]
    row = TF[np.argmax(TF[:, c])]; above = row >= 0.5 * row.max(); lo = hi = c
    while lo > 0 and above[lo - 1]: lo -= 1
    while hi < len(row) - 1 and above[hi + 1]: hi += 1
    fband = FREQS[TF[:, c] >= 0.5 * TF[:, c].max()]
    print(f"time-frequency (mean of 60 sessions): maximum at {fmax} Hz; above half maximum {fband.min()}–{fband.max()} Hz "
          f"and {t[lo]:.0f} to {t[hi]:.0f} ms")

    # ── 3. figure (paper style, supplement Fig S19): A shape per session (rats pink) + median, B time–frequency ──
    RNAME = {"134": "Rat 1", "135": "Rat 2", "136": "Rat 3"}
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), facecolor="white", gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.35))
    for m, r in zip(Nk, RAT): a.plot(tk, m, color=RATC[r], lw=0.7, alpha=0.7)
    a.plot(tk, med, color="k", lw=1.8, label="Median of sessions")
    for r, col in RATC.items(): a.plot([], [], color=col, lw=1.5, label=RNAME[r])
    a.axvline(0, color="k", ls="--", lw=0.8); a.axhline(0, color="0.6", lw=0.6)
    a.set_xlim(-SHOW_MS, SHOW_MS); a.set_xticks([-140, -70, 0, 70, 140])
    a.set_xlabel("Time from trough (ms)"); a.set_ylabel("LFP (normalized to the trough)")
    a.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, handlelength=1.5)
    a.text(-0.2, 1.05, "A", transform=a.transAxes, fontsize=11, fontweight="bold")
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    kt = np.abs(t) <= TFR_MS
    im = b.pcolormesh(t[kt], FREQS, TF[:, kt], cmap="viridis", shading="auto", vmin=0, vmax=1)
    b.axvline(0, color="w", ls="--", lw=0.8); b.set_xticks([-200, -100, 0, 100, 200])
    b.set_xlabel("Time from trough (ms)"); b.set_ylabel("Frequency (Hz)")
    b.text(-0.2, 1.05, "B", transform=b.transAxes, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=b, label="Power (normalized)")
    _savefig(fig, os.path.join(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\supplementary",
                             "Fig S19 average beta burst shape.png"))
    plt.show()


@cell()
def step0_across_session():
    """STEP 0. mPFC BETA BURSTS ACROSS THE SESSION — POOLED, IL, PL, Cg1: OCCUPANCY AND RATE, ALL TIME, NO TRIAL CATEGORIES (exact times, LOCAL) (needs Settings, RAT)"""
    global DALL, DP, DR, GEN_DIR, NB, OUT, RCOL, REGS, RES, ROWS, SUB, T, _, a, aen, ast, ax, axs, b, be, binomtest, bs, clean
    global col, d, d0, down, edges, f, fig, fp, g, glob, k, k_, letter, m, name, nn, out, overlap, p, per_rat, pk, ptxt, r, r_
    global reg, rho, rows, s, sf, sig, sp, spearmanr, t0, t1, title, x, ylab, z
    # ── STEP 0. mPFC BETA BURSTS ACROSS THE SESSION — POOLED, IL, PL, Cg1: OCCUPANCY AND RATE, ALL TIME, NO TRIAL CATEGORIES (exact times, LOCAL) (needs Settings, RAT) ──
    import glob
    from scipy.stats import spearmanr, binomtest
    GEN_DIR = os.path.join(BETA_DIR, "How beta behaves in general"); os.makedirs(GEN_DIR, exist_ok=True)
    REGS, NB = ["IL", "PL", "Cg1"], 20
    ROWS = [("pooled", "mPFC pooled (mean of IL, PL, Cg1)"), ("IL", "IL (infralimbic)"), ("PL", "PL (prelimbic)"), ("Cg1", "Cg1 (cingulate)")]
    RCOL = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}
    ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    def overlap(s_, e_, a, b): return np.clip(np.minimum(e_, b) - np.maximum(s_, a), 0, None).sum()

    # ── 1. occupancy and rate in 20 parts of each session, per region ──
    rows = []
    for f in sorted(glob.glob(os.path.join(EXACT_DIR, "*.npz"))):
        s = os.path.basename(f)[:-4]; z = np.load(f); T = pd.read_csv(f[:-4] + "_trials.csv")
        t0, t1 = np.nanmin(T["press_s"]), np.nanmax(T["release_s"]); edges = np.linspace(t0, t1, NB + 1)
        for g in REGS:
            k_ = f"LOCAL|beta 15-30|{g}"
            if k_ + "|burst_start_s" not in z.files: continue
            bs, be, pk = z[k_ + "|burst_start_s"], z[k_ + "|burst_end_s"], z[k_ + "|peak_s"]
            ast, aen = z[f"LOCAL|{g}|art_start_s"], z[f"LOCAL|{g}|art_end_s"]
            for k in range(NB):
                a, b = edges[k], edges[k + 1]; clean = (b - a) - overlap(ast, aen, a, b)
                if clean < 0.5 * (b - a): continue
                rows.append(dict(session=s, rat=str(RAT[s]), region=g, part=k, pos=(k + 0.5) / NB, minutes=(t1 - t0) / 60,
                                 occ=100 * overlap(bs, be, a, b) / clean, rate=((pk >= a) & (pk < b)).sum() / clean))
    DR = pd.DataFrame(rows)
    DP = DR.groupby(["session", "rat", "part", "pos"])[["occ", "rate"]].mean().reset_index().assign(region="pooled")
    DALL = pd.concat([DP, DR], ignore_index=True)
    print("sessions per region:", DALL.groupby("region").session.nunique().to_dict(),
          f"| session length: median {DR.groupby('session').minutes.first().median():.0f} min")

    # ── 2. test: per-session trend, sign test across sessions ──
    RES, out = {}, []
    for reg, _ in ROWS:
        d0 = DALL[DALL.region == reg]
        for col, name in (("occ", "occupancy"), ("rate", "rate")):
            rho = d0.groupby(["session", "rat"]).apply(lambda d: spearmanr(d.pos, d[col])[0] if len(d) >= 5 else np.nan).dropna()
            down, nn = int((rho < 0).sum()), int((rho != 0).sum()); p = binomtest(down, nn).pvalue
            per_rat = {r: f"{(v < 0).sum()}/{len(v)}" for r, v in rho.groupby(level="rat")}
            RES[(reg, col)] = dict(down=down, n=nn, p=p, per_rat=per_rat)
            out.append(dict(region=reg, measure=name, sessions_decreasing=f"{down} of {nn}", median_rho=rho.median(), sign_test_p=p,
                            **{f"rat {r}": v for r, v in per_rat.items()}))
    OUT = pd.DataFrame(out); print(); print(OUT.round(3).to_string(index=False))
    _table(OUT, os.path.join(GEN_DIR, "step0_beta_across_session_tests.csv"), index=False)
    DALL.to_csv(os.path.join(GEN_DIR, "step0_beta_across_session_data.csv"), index=False)

    # ── 3. figure: one row per region (pooled first), each row its own title ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(11, 17), facecolor="white")
    SUB = fig.subfigures(len(ROWS), 1, hspace=0.06)
    for sf, (reg, title) in zip(SUB, ROWS):
        sf.set_facecolor("white")
        d0 = DALL[DALL.region == reg]
        axs = sf.subplots(1, 2)
        for ax, (col, ylab, letter) in zip(axs, (("occ", "Beta burst occupancy (%)", "occupancy"), ("rate", "Beta burst rate (bursts/s)", "rate"))):
            for r, d in d0.groupby("rat"):
                m = d.groupby("part")[col].mean()
                ax.plot((m.index + 0.5) / NB, m.values, color=RCOL[r], lw=1.3, label=f"rat {r} ({d.session.nunique()} sessions)")
            g = d0.groupby(["session", "part"])[col].mean().groupby("part").agg(["mean", "sem"]); x = (g.index + 0.5) / NB
            ax.fill_between(x, g["mean"] - g["sem"], g["mean"] + g["sem"], color="0.3", alpha=0.15, lw=0)
            ax.plot(x, g["mean"], color="k", lw=2.2, label=f"all sessions ({d0.session.nunique()}, ± SEM)")
            r_ = RES[(reg, col)]; sig = r_["p"] < 0.05
            ax.text(0.98, 0.97, f"decreasing in {r_['down']} of {r_['n']} sessions\nsign test {ptxt(r_['p'])}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5, color="k" if sig else "0.5", fontweight="bold" if sig else "normal")
            ax.set_xlim(0, 1); ax.set_xlabel("Position in the session (start → end)"); ax.set_ylabel(ylab)
            ax.set_title(letter, loc="left", fontsize=9.5, color="0.3")
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
        sf.suptitle(f"{title} — beta bursts across the session (LOCAL, all time, no trial categories)", fontsize=11, fontweight="bold")
    fp = os.path.join(GEN_DIR, "step0_beta_across_session_regions.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def step0_figure():
    """Beta bursts across the session: figure pooled mPFC (paper style) + numbers for pooled, IL, PL, Cg1 (FDR over the 4 groups)"""
    global BETA_COL, DALL, GEN_DIR, NB, OUT, R, ax, binomtest, col, d0, down, fig, fmt_q, fname, folder, g, multipletests
    global name, nn, r, reg, rho, rows, sp, spearmanr, x, ylab
    # ── Beta bursts across the session: figure pooled mPFC (paper style) + numbers for pooled, IL, PL, Cg1 (FDR over the 4 groups) ──
    from scipy.stats import spearmanr, binomtest
    from statsmodels.stats.multitest import multipletests
    BETA_COL, NB = "#1565C0", 20
    GEN_DIR = os.path.join(BETA_DIR, "How beta behaves in general")
    DALL = globals().get("DALL")
    if DALL is None:
        DALL = pd.read_csv(os.path.join(GEN_DIR, "step0_beta_across_session_data.csv"), dtype={"rat": str})

    # numbers: per-session Spearman (position vs measure), sign test across sessions; start/end = first/last 10 % of the session
    rows = []
    for reg in ["pooled", "IL", "PL", "Cg1"]:
        d0 = DALL[DALL.region == reg]
        for col, name in (("occ", "occupancy (%)"), ("rate", "rate (bursts/s)")):
            rho = d0.groupby("session").apply(lambda d: spearmanr(d.pos, d[col])[0] if len(d) >= 5 else np.nan).dropna()
            down, nn = int((rho < 0).sum()), int((rho != 0).sum())
            g = d0.groupby(["session", "part"])[col].mean().groupby("part").mean()
            rows.append(dict(region=reg, measure=name, sessions=nn, decreasing=down, median_rho=rho.median(),
                             start=g.loc[g.index < 2].mean(), end=g.loc[g.index >= NB - 2].mean(), p=binomtest(down, nn).pvalue))
    R = pd.DataFrame(rows)
    R["q"] = R.groupby("measure")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    print("Beta bursts across the session (LOCAL; sign test across sessions; q = FDR over pooled, IL, PL, Cg1 within each measure)")
    print(R.round({"median_rho": 3, "start": 3, "end": 3, "p": 4, "q": 4}).to_string(index=False))

    # figures: pooled mPFC — occupancy (report, Fig 8) and rate (supplement, Fig S15), one panel each
    fmt_q = lambda q: "q < 0.001" if q < 0.001 else f"q = {q:.3f}"
    d0 = DALL[DALL.region == "pooled"]
    OUT = {"occ":  ("Beta burst occupancy (%)",   "occupancy (%)",   r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report",
                    "Fig 9 beta burst occupancy across the session.png"),
           "rate": ("Beta burst rate (bursts/s)", "rate (bursts/s)", r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\supplementary",
                    "Fig S15 beta burst rate across the session.png")}
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    for col, (ylab, name, folder, fname) in OUT.items():
        fig, ax = plt.subplots(figsize=(3.2, 2.6), facecolor="white")
        g = d0.groupby(["session", "part"])[col].mean().groupby("part").agg(["mean", "sem"]); x = (g.index + 0.5) / NB
        ax.fill_between(x, g["mean"] - g["sem"], g["mean"] + g["sem"], color=BETA_COL, alpha=0.2, lw=0)
        ax.plot(x, g["mean"], color=BETA_COL, lw=1.6)
        r = R[(R.region == "pooled") & (R.measure == name)].iloc[0]
        ax.text(0.98, 0.97, f"lower in {r.decreasing} of {r.sessions} sessions\n{fmt_q(r.q)}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5)
        ax.set_xlim(0, 1); ax.set_xlabel("Position in the session (start → end)"); ax.set_ylabel(ylab)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        os.makedirs(folder, exist_ok=True)
        _savefig(fig, os.path.join(folder, fname))
        plt.show()
    print(f"{d0.session.nunique()} sessions (pooled mPFC), {NB} parts per session")


@cell(needs={
    'load_burst_tables': ['BURST_BANDS_R', 'TABS', 'files'],
})
def step3_along_session():
    """STEP 3. LATE vs CORRECT ALONG THE SESSION (beta, LOCAL, all blocks, NO block term): occupancy and rate; start effect + three methods to split"""
    global ALPHA, COLC, CT, CUTS, C_, DEFS, DT, D_RATE, KEYC, LT, PHS, POSCOLS, PT, Patch, RCOL, RCOL4, T, TP, WIN3, YEL, dd
    global finish, fit, folder, g, i, keep, line_model, multipletests, norm, occ_table, pcol, phase_fit, r_rate, rr_, s, sb
    global smf, stars, t0_, t1_, tb_, three_methods, vol, warnings, z
    # ── STEP 3. LATE vs CORRECT ALONG THE SESSION (beta, LOCAL, all blocks, NO block term): occupancy and rate; start effect + three methods to split ──
    globals().pop("C", None)
    import warnings
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    from matplotlib.patches import Patch
    from statsmodels.stats.multitest import multipletests
    COLC = {"correct": "#2CA02C", "late": "#DD8452"}
    RCOL = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}; RCOL4 = {"all": "0.25", **RCOL}   # rats: light, medium, dark pink
    YEL = "#E6B800"
    PHS, CUTS, WIN3 = ["start", "middle", "end"], (0.25, 0.85), (-0.6, 0.0)
    DEFS = {"trial number": "position", "clock time": "v_time", "water drunk": "v_water"}
    ALPHA = {"trial number": 0.95, "clock time": 0.6, "water drunk": 0.3}
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    # ── trial positions: trial number, clock time, water drunk ──
    PT = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False).sort_values(["session", "trial_index"])
    PT["position"] = PT.groupby("session")["trial_index"].rank(pct=True)
    t0_ = PT.groupby("session")["press_neural_s"].transform("min"); t1_ = PT.groupby("session")["press_neural_s"].transform("max")
    PT["v_time"] = (PT["press_neural_s"] - t0_) / (t1_ - t0_)
    vol = PT["reward_size"].fillna(0).clip(lower=0)
    PT["v_water"] = (vol.groupby(PT["session"]).cumsum() - vol) / vol.groupby(PT["session"]).transform("sum")
    POSCOLS = ["position", "v_time", "v_water"]

    def finish(D):
        D = D.dropna(subset=POSCOLS).copy()
        D["late"] = (D["cond"] == "late").astype(int); D["block"] = D["long_prob"].round(2).astype(str); D["rat"] = D["rat"].astype(str)
        D["pbin"] = pd.cut(D["position"], np.linspace(0, 1, 11), labels=False, include_lowest=True)
        D["phase"] = pd.cut(D["position"], [-0.001, *CUTS, 1.001], labels=PHS).astype(str)
        return D

    # ── occupancy table (from TABS, all regions and bands) ──
    TP = pd.concat([pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))[["trial_index", "long_prob"]]
                    .assign(session=s, trial=lambda x: x.index) for s in files], ignore_index=True)
    TP = TP.merge(PT[["session", "trial_index", *POSCOLS]], on=["session", "trial_index"], how="left")
    def occ_table(band):
        D = TABS[band]["late"].merge(TP, on=["session", "trial"], how="left")
        D = D[(D["ref"] == "LOCAL") & (D["typ"] == "long") & D["cond"].isin(["correct", "late"])].dropna(subset=["occ"]).copy()
        D["occ_pct"] = 100 * D["occ"]
        return finish(D)

    # ── rate table (IL beta LOCAL, from the 50-ms counts) ──
    KEYC = "LOCAL|beta 15-30|IL|tone|cnt"; rr_ = []
    for s in files:
        z = np.load(os.path.join(BURST_DIR, s + ".npz")); T = pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))
        if KEYC not in z.files or z[KEYC].shape[0] != len(T):
            print("rate: skipped", s); continue
        tb_ = fn.BURST_A0["tone"] + fn.BURST_BIN * np.arange(z[KEYC].shape[1])
        sb = (tb_ >= WIN3[0] - 1e-9) & (tb_ + fn.BURST_BIN <= WIN3[1] + 1e-9)
        keep = ((T["type"] == "long") & T["label"].isin(["correct", "late"])).values
        C_ = z[KEYC][:, sb]
        for i in np.where(keep)[0]:
            if np.isfinite(C_[i]).all():
                rr_.append(dict(session=s, rat=RAT[s], trial_index=T["trial_index"].values[i], cond=T["label"].values[i],
                                long_prob=T["long_prob"].values[i], rate=C_[i].sum() / (WIN3[1] - WIN3[0])))
    D_RATE = finish(pd.DataFrame(rr_).merge(PT[["session", "trial_index", *POSCOLS]], on=["session", "trial_index"], how="left"))

    def fit(f, dd):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return smf.mixedlm(f, dd, groups=dd["session"]).fit(reml=True, method=["lbfgs", "powell", "nm"])

    def line_model(dd, y):
        m = fit(f"{y} ~ late * position", dd); P, V = m.params, m.cov_params()                       # no block term
        comb = lambda k: (sum(P[x] for x in k), 2 * norm.sf(abs(sum(P[x] for x in k) / np.sqrt(sum(V.loc[a, b] for a in k for b in k)))))
        ls_, pls = comb(["position", "late:position"]); le_, ple = comb(["late", "late:position"])
        return pd.Series(dict(sessions=dd.session.nunique(), n_correct=int((dd.late == 0).sum()), n_late=int(dd.late.sum()),
                              correct_change=P["position"], p_correct_change=m.pvalues["position"], late_change=ls_, p_late_change=pls,
                              late_x_position=P["late:position"], p_interaction=m.pvalues["late:position"],
                              late_minus_correct_start=P["late"], p_start=m.pvalues["late"], late_minus_correct_end=le_, p_end=ple))

    def phase_fit(dd, var, y):
        dd = dd.assign(phase=pd.cut(dd[var], [-0.001, *CUTS, 1.001], labels=PHS).astype(str))
        m = fit(f"{y} ~ late * C(phase, Treatment('start'))", dd); P, V = m.params, m.cov_params(); out = {}   # no block term
        for ph in PHS:
            k = ["late"] + ([f"late:C(phase, Treatment('start'))[T.{ph}]"] if ph != "start" else [])
            est = sum(P[x] for x in k); se = np.sqrt(sum(V.loc[a, b] for a in k for b in k))
            out[ph] = dict(est=est, se=se, p=2 * norm.sf(abs(est / se)),
                           n_late=int(((dd.phase == ph) & (dd.late == 1)).sum()), n_correct=int(((dd.phase == ph) & (dd.late == 0)).sum()))
        return out

    def three_methods(dd, y):
        rows = []
        for rat, x in [("all", dd)] + list(dd.groupby("rat")):
            for dname, var in DEFS.items():
                for ph, rr in phase_fit(x, var, y).items():
                    rows.append(dict(rat=rat, definition=dname, phase=ph, late_minus_correct=rr["est"], se=rr["se"],
                                     ci_low=rr["est"] - 1.96 * rr["se"], ci_high=rr["est"] + 1.96 * rr["se"], p=rr["p"],
                                     n_correct=rr["n_correct"], n_late=rr["n_late"]))
        CTm = pd.DataFrame(rows); CTm["sig"] = CTm["p"].apply(stars)
        return CTm

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    # ── occupancy: straight-line table for all regions and bands, FDR over the 4 groups ──
    DT = {band: occ_table(band) for band in BURST_BANDS_R}
    LT = pd.DataFrame([dict(band=band, group=g, **line_model(dd, "occ_pct")) for band in BURST_BANDS_R for g, dd in DT[band].groupby("group")])
    LT["group"] = pd.Categorical(LT["group"], ["IL", "PL", "Cg1", "mPFC pooled"]); LT = LT.sort_values(["band", "group"])
    for pcol in ["p_interaction", "p_start", "p_end"]:
        LT["q" + pcol[1:]] = LT.groupby("band")[pcol].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    for folder in (TABLES, BETA_DIR): _table(LT, os.path.join(folder, "step3_beta_occupancy_straight_line_all_regions_noblock.csv"), index=False)
    print("===== OCCUPANCY beta 15-30, LOCAL, no block: change start→end (%-points), late − correct at start / end; q = FDR over 4 groups =====")
    print(LT[LT.band == "beta 15-30"].drop(columns="band").to_string(index=False))

    # ── occupancy per phase, three methods to split, pooled mPFC and IL ──
    CT = {}
    for g in ["mPFC pooled", "IL"]:
        dd = DT["beta 15-30"]; dd = dd[dd.group == g].copy()
        CT[g] = three_methods(dd, "occ_pct")
        for folder in (TABLES, BETA_DIR): _table(CT[g], os.path.join(folder, f"step3_beta_occupancy_three_methods_{g.replace(' ', '_')}_noblock.csv"), index=False)
        print(f"\nOCCUPANCY {g} beta, late − correct (%-points) per phase, three methods to split the session (no block):")
        print(CT[g].drop(columns="se").to_string(index=False))

    # ── rate: IL beta ──
    r_rate = line_model(D_RATE, "rate")
    print("\nRATE IL beta straight-line model (bursts/s, no block):"); print(r_rate.to_string())
    pd.reset_option("display.float_format")


@cell(needs={
    'step3_along_session': ['DT', 'LT'],
})
def fig10():
    """Fig 10. Beta burst occupancy 0.6 s before the tone along the session: correct vs late long trials, 10 equal parts"""
    global COLC, CUTS, PHS, REGION, a1, base, c, col, cut, dd, fig, hi, lab, lo, ph, r, sp, x, xc, xx
    # ── Fig 10. Beta burst occupancy 0.6 s before the tone along the session: correct vs late long trials, 10 equal parts
    #    (mean ± SEM over sessions), straight lines from the mixed model, start / middle / end as in Fig 9 ──
    REGION = "IL"                                                    # or "mPFC pooled"
    COLC = {"correct": ("#2CA02C", "Correct"), "late": ("#DD8452", "Late")}
    CUTS, PHS = (0.25, 0.85), ["start", "middle", "end"]
    dd = DT["beta 15-30"]; dd = dd[dd.group == REGION].copy()
    r = LT[(LT.band == "beta 15-30") & (LT.group == REGION)].iloc[0]

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, a1 = plt.subplots(figsize=(5.0, 3.6), facecolor="white")
    for c, (col, lab) in COLC.items():
        x = dd[dd.cond == c].groupby(["session", "pbin"])["occ_pct"].mean().groupby("pbin").agg(["mean", "sem"]); xc = (x.index + 0.5) / 10
        a1.fill_between(xc, x["mean"] - x["sem"], x["mean"] + x["sem"], color=col, alpha=0.18, lw=0)
        a1.plot(xc, x["mean"], "o-", color=col, lw=1.6, ms=3, label=lab)
    xx = np.linspace(0, 1, 50)
    base = dd.loc[dd.late == 0, "occ_pct"].mean() - r.correct_change * dd.loc[dd.late == 0, "position"].mean()
    a1.plot(xx, base + r.correct_change * xx, "--", color=COLC["correct"][0], lw=1)
    a1.plot(xx, base + r.late_minus_correct_start + r.late_change * xx, "--", color=COLC["late"][0], lw=1)
    for cut in CUTS:
        a1.axvline(cut, color="k", ls="--", lw=0.8)
    for (lo, hi), ph in zip([(0, CUTS[0]), CUTS, (CUTS[1], 1)], PHS):
        a1.text((lo + hi) / 2, 1.02, ph.capitalize(), transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    a1.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(np.mean(CUTS), 1.0), ncol=2)   # under "Middle"
    a1.set_xlim(0, 1); a1.set_xlabel("Position in the session (first → last trial)")
    a1.set_ylabel("Beta burst occupancy (%)\n0.6 s before the tone")
    for sp in ("top", "right"): a1.spines[sp].set_visible(False)
    _savefig(fig, os.path.join(r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report",
                             "Fig 11 beta occupancy late vs correct along the session.png"))
    plt.show()
    print(f"{REGION}: {dd.session.nunique()} sessions | late − correct at start {r.late_minus_correct_start:+.2f} (q = {r.q_start:.4f}), "
          f"late × position {r.late_x_position:+.2f} (q = {r.q_interaction:.4f}), at end {r.late_minus_correct_end:+.2f} (q = {r.q_end:.3f})")


@cell(needs={
    'step3_along_session': ['DT', 'POSCOLS', 'PT', 'WIN3', 'finish', 'line_model', 'three_methods'],
    'fig10': ['PHS'],
    'load_burst_tables': ['files'],
})
def step3_numbers():
    """ALL NUMBERS for late vs correct beta bursts 0.6 s before the tone (LOCAL, no block), IL / PL / Cg1 / pooled:"""
    global Ct, D, D_OCC4, D_RATE4, GROUPS4, LEV, LTR, MEAS, PHASE, R_, SIMPLE, T, allr, d, diff, fdr4, g, i, k_, keep, m_, mm
    global mname, multipletests, n, name, p, part, pcol, rows, rr_, s, sb, tab, tb_, v, x, y, z
    # ── ALL NUMBERS for late vs correct beta bursts 0.6 s before the tone (LOCAL, no block), IL / PL / Cg1 / pooled:
    #    1 simple test (session means, sign-flip) whole session + per part; 2 mixed model per part × 3 splits × all rats + each rat;
    #    3 the same for burst rate; FDR (Benjamini–Hochberg) over the 4 groups within each test; 4 occupancy levels per part ──
    from statsmodels.stats.multitest import multipletests
    GROUPS4 = ["IL", "PL", "Cg1", "mPFC pooled"]
    fdr4 = lambda s: multipletests(s, method="fdr_bh")[1]

    # rate table for all regions (+ pooled = mean of the regions the session has), from the 50-ms counts
    rr_ = []
    for s in files:
        z = np.load(os.path.join(BURST_DIR, s + ".npz")); T = pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))
        keep = np.where(((T["type"] == "long") & T["label"].isin(["correct", "late"])).values)[0]
        R_ = {}
        for g in ["IL", "PL", "Cg1"]:
            k_ = f"LOCAL|beta 15-30|{g}|tone|cnt"
            if k_ not in z.files or z[k_].shape[0] != len(T): continue
            tb_ = fn.BURST_A0["tone"] + fn.BURST_BIN * np.arange(z[k_].shape[1])
            sb = (tb_ >= WIN3[0] - 1e-9) & (tb_ + fn.BURST_BIN <= WIN3[1] + 1e-9)
            R_[g] = np.where(np.isfinite(z[k_][:, sb]).all(1), z[k_][:, sb].sum(1) / (WIN3[1] - WIN3[0]), np.nan)
        if not R_: continue
        R_["mPFC pooled"] = np.nanmean(np.vstack(list(R_.values())), axis=0)
        for g, v in R_.items():
            for i in keep:
                if np.isfinite(v[i]):
                    rr_.append(dict(session=s, rat=RAT[s], group=g, trial_index=T["trial_index"].values[i], cond=T["label"].values[i],
                                    long_prob=T["long_prob"].values[i], rate=v[i]))
    D_RATE4 = finish(pd.DataFrame(rr_).merge(PT[["session", "trial_index", *POSCOLS]], on=["session", "trial_index"], how="left"))
    D_OCC4 = DT["beta 15-30"]
    MEAS = {"occupancy (%-points)": (D_OCC4, "occ_pct"), "rate (bursts/s)": (D_RATE4, "rate")}

    # 1) simple test: per session mean late − mean correct (sessions with both), sign-flip; whole session and per part (trial number)
    rows = []
    for mname, (D, y) in MEAS.items():
        for g in GROUPS4:
            d = D[D.group == g]
            for part in ["whole", *PHS]:
                x = d if part == "whole" else d[d.phase == part]
                m_ = x.groupby(["session", "late"])[y].mean().unstack()
                diff = (m_[1] - m_[0]).dropna(); mm, p, n = fn.sign_flip_p(diff.values)
                rows.append(dict(measure=mname, group=g, part=part, sessions=n, late_higher=int((diff > 0).sum()),
                                 late_minus_correct=mm, p=p))
    SIMPLE = pd.DataFrame(rows)
    SIMPLE["q"] = SIMPLE.groupby(["measure", "part"])["p"].transform(fdr4)

    # 2) + 3) mixed model per part, three splits, all rats + each rat, occupancy and rate, all 4 groups
    rows = []
    for mname, (D, y) in MEAS.items():
        for g in GROUPS4:
            Ct = three_methods(D[D.group == g].copy(), y)
            rows.append(Ct.assign(measure=mname, group=g))
    PHASE = pd.concat(rows, ignore_index=True)
    PHASE["q"] = np.nan
    allr = PHASE.rat == "all"
    PHASE.loc[allr, "q"] = PHASE[allr].groupby(["measure", "definition", "phase"])["p"].transform(fdr4)   # FDR over 4 groups (all rats)

    # rate line model for all 4 groups (+ FDR)
    LTR = pd.DataFrame([dict(group=g, **line_model(D_RATE4[D_RATE4.group == g], "rate")) for g in GROUPS4])
    for pcol in ["p_interaction", "p_start", "p_end"]:
        LTR["q" + pcol[1:]] = fdr4(LTR[pcol])

    # 4) occupancy levels: mean over sessions of the session means, correct and late, per part (trial number)
    LEV = (D_OCC4.groupby(["group", "phase", "cond", "session"])["occ_pct"].mean().groupby(["group", "phase", "cond"]).mean()
           .unstack("cond").reindex(pd.MultiIndex.from_product([GROUPS4, PHS])))

    pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("1) SIMPLE TEST (session means, sign-flip; q = FDR over 4 groups within measure × part)"); print(SIMPLE.to_string(index=False))
    print("\n2) MIXED MODEL PER PART, ALL RATS (q = FDR over 4 groups within measure × split × part)")
    print(PHASE[allr].drop(columns=["rat", "se"]).to_string(index=False))
    print("\n   EACH RAT (raw p)"); print(PHASE[~allr].drop(columns=["se", "q"]).to_string(index=False))
    print("\n3) RATE, line model (late × position), 4 groups, FDR"); print(LTR.to_string(index=False))
    print("\n4) OCCUPANCY LEVELS (%; mean over sessions), correct and late, per part (trial number)"); print(LEV.to_string())
    for name, tab in (("simple", SIMPLE), ("phase_models", PHASE), ("rate_line", LTR)):
        _table(tab, os.path.join(TABLES, f"beta_late_vs_correct_{name}_noblock.csv"), index=False)
    _table(LEV, os.path.join(TABLES, "beta_late_vs_correct_levels_noblock.csv"))
    pd.reset_option("display.float_format")


@cell(needs={
    'step3_along_session': ['CT', 'DT'],
})
def fig11():
    """Fig 11 (all rats, report) and Fig S16 (each rat, supplement): late − correct beta occupancy 0.6 s before the tone"""
    global ALPHA3, COL, CTg, DEFS3, NAME, NS, PHS, Patch, REGION, REPORT, SUPP, ax, axs, dd, fig, panel, rat, stars, y0, y1
    # ── Fig 11 (all rats, report) and Fig S16 (each rat, supplement): late − correct beta occupancy 0.6 s before the tone
    #    per part of the session, three ways to split the session (mixed model, no block, 95 % CI) ──
    REGION = "IL"                                                    # or "mPFC pooled"
    PHS = ["start", "middle", "end"]
    DEFS3 = ["trial number", "clock time", "water drunk"]
    ALPHA3 = {"trial number": 1.0, "clock time": 0.6, "water drunk": 0.3}
    COL = {"all": "#1565C0", "134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}       # beta blue; rats as in Fig 2D
    NAME = {"all": "All rats", "134": "Rat 1", "135": "Rat 2", "136": "Rat 3"}
    REPORT = r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report"
    SUPP   = r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\supplementary"
    from matplotlib.patches import Patch
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    CTg = CT[REGION]
    dd = DT["beta 15-30"]; dd = dd[dd.group == REGION]
    NS = {"all": dd.session.nunique(), **dd.groupby("rat")["session"].nunique().to_dict()}

    def panel(ax, rat):
        X = CTg[CTg.rat == rat]; span = X.ci_high.max() - min(0, X.ci_low.min())
        for k, ph in enumerate(PHS):
            for j, dname in enumerate(DEFS3):
                rr = X[(X.definition == dname) & (X.phase == ph)].iloc[0]; xk = k + (j - 1) * 0.27
                ax.bar(xk, rr.late_minus_correct, width=0.25, color=COL[rat], alpha=ALPHA3[dname], edgecolor="k", lw=0.4)
                ax.errorbar(xk, rr.late_minus_correct, yerr=[[rr.late_minus_correct - rr.ci_low], [rr.ci_high - rr.late_minus_correct]],
                            color="k", capsize=2, lw=0.8)
                ax.text(xk, max(rr.ci_high, 0) + 0.03 * span, stars(rr.p), ha="center", va="bottom", fontsize=6.5)   # always above
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels([ph.capitalize() for ph in PHS])
        ax.set_title(f"{NAME[rat]} ({NS[rat]} sessions)", fontsize=9, fontweight="bold")
        ax.legend(handles=[Patch(facecolor=COL[rat], alpha=ALPHA3[d], edgecolor="k", lw=0.4, label=d.capitalize()) for d in DEFS3],
                  frameon=False, fontsize=7, loc="upper right")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})

    # Fig 11: all rats (report)
    fig, ax = plt.subplots(figsize=(3.8, 3.0), facecolor="white")
    panel(ax, "all"); ax.set_ylabel("Late − correct occupancy (%-points)")
    y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + 0.12 * (y1 - y0))
    _savefig(fig, os.path.join(REPORT, "Fig 12 beta occupancy late vs correct three ways to split.png"))
    plt.show()

    # Fig S16: each rat (supplement), same y-axis
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True, facecolor="white")
    for ax, rat in zip(axs, ["134", "135", "136"]):
        panel(ax, rat)
    axs[0].set_ylabel("Late − correct occupancy (%-points)")
    y0, y1 = axs[0].get_ylim(); axs[0].set_ylim(y0, y1 + 0.12 * (y1 - y0))
    fig.tight_layout()
    _savefig(fig, os.path.join(SUPP, "Fig S16 beta occupancy late vs correct three ways to split per rat.png"))
    plt.show()
    print(f"{REGION}: all rats and each rat, trial number / clock time / water drunk; stars = raw p of the mixed model per part")


@cell(needs={
    'fig10': ['CUTS'],
    'step3_numbers': ['D_OCC4', 'D_RATE4', 'GROUPS4'],
    'fig11': ['PHS'],
    'step3_along_session': ['fit'],
})
def step3_missing_numbers():
    """MISSING NUMBERS: 1 rate whole session (mixed model, 4 groups); 2 + 3 simple test per part for all three splits, all rats + each rat"""
    global D, MEAS, SIMPLE3, SPLITS, WHOLE, d, d0, d1, d2, diff, e, g, m, m_, mm, mname, n, p, ph, rat, rows, se, sname, var
    global y
    # ── MISSING NUMBERS: 1 rate whole session (mixed model, 4 groups); 2 + 3 simple test per part for all three splits, all rats + each rat ──
    MEAS = {"occupancy (%-points)": (D_OCC4, "occ_pct"), "rate (bursts/s)": (D_RATE4, "rate")}
    SPLITS = {"trial number": "position", "clock time": "v_time", "water drunk": "v_water"}

    # 1) whole session, mixed model y ~ late + position + (1 | session), 4 groups (raw p)
    rows = []
    for mname, (D, y) in MEAS.items():
        for g in GROUPS4:
            d = D[D.group == g].dropna(subset=[y, "position"])
            m = fit(f"{y} ~ late + position", d)
            e, se = m.params["late"], m.bse["late"]
            rows.append(dict(measure=mname, group=g, sessions=d.session.nunique(), correct=d.loc[d.late == 0, y].mean(),
                             late=d.loc[d.late == 1, y].mean(), late_minus_correct=e, ci_low=e - 1.96 * se, ci_high=e + 1.96 * se,
                             p=m.pvalues["late"]))
    WHOLE = pd.DataFrame(rows)

    # 2 + 3) simple test per part: per session mean late − mean correct, sign-flip; three splits; all rats and each rat
    rows = []
    for mname, (D, y) in MEAS.items():
        for g in GROUPS4:
            d0 = D[D.group == g]
            for sname, var in SPLITS.items():
                d1 = d0.assign(part=pd.cut(d0[var], [-0.001, *CUTS, 1.001], labels=PHS).astype(str))
                for rat, d2 in [("all", d1)] + list(d1.groupby("rat")):
                    for ph in PHS:
                        m_ = d2[d2.part == ph].groupby(["session", "late"])[y].mean().unstack()
                        if m_.shape[1] < 2: continue
                        diff = (m_[1] - m_[0]).dropna()
                        if len(diff) < 2: continue
                        mm, p, n = fn.sign_flip_p(diff.values)
                        rows.append(dict(measure=mname, group=g, split=sname, rat=rat, part=ph, sessions=n,
                                         late_higher=int((diff > 0).sum()), late_minus_correct=mm, p=p))
    SIMPLE3 = pd.DataFrame(rows)

    pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("1) WHOLE SESSION, mixed model (late + position), raw p"); print(WHOLE.to_string(index=False))
    print("\n2) SIMPLE TEST PER PART, ALL RATS, three splits (raw p)")
    print(SIMPLE3[SIMPLE3.rat == "all"].drop(columns="rat").to_string(index=False))
    print("\n3) SIMPLE TEST PER PART, EACH RAT, IL only (raw p)")
    print(SIMPLE3[(SIMPLE3.rat != "all") & (SIMPLE3.group == "IL")].to_string(index=False))
    _table(WHOLE, os.path.join(TABLES, "beta_late_vs_correct_whole_session_noblock.csv"), index=False)
    _table(SIMPLE3, os.path.join(TABLES, "beta_late_vs_correct_simple_three_splits.csv"), index=False)
    pd.reset_option("display.float_format")


@cell(needs={
    'load_burst_tables': ['TABS', 'files'],
})
def step4_where_in_trial():
    """STEP 4. WHERE IN THE TRIAL IS THE START EFFECT? occupancy and rate, 3 windows (IL beta, LOCAL, long trials, no block)"""
    global A0P, A0T, B6, BAND, BIN, COLC, CUTS, CV, CV_O, CV_R, Cp, Ct, OUT, Op, Ot, PHS, PT, P_, R, T, TONE, T_, W, WCOL
    global WINS, WLAB, _, diff, figure, folder, how, i, keep, kp, kt, m_, mm, n_, name, norm, p_, phase_fit, r, row, rows, s
    global sm, smf, stars, tone, tp, tt_, unit, w, warnings, wmean, ylab, z
    # ── STEP 4. WHERE IN THE TRIAL IS THE START EFFECT? occupancy and rate, 3 windows (IL beta, LOCAL, long trials, no block) ──
    globals().pop("C", None)
    import warnings
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    BAND = "beta 15-30"; BIN = fn.BURST_BIN; A0P, A0T = fn.BURST_A0["press"], fn.BURST_A0["tone"]
    COLC = {"correct": ("#2CA02C", "Correct"), "late": ("#DD8452", "Late")}
    PHS, CUTS = ["start", "middle", "end"], (0.25, 0.85)
    WINS = ["before press", "whole hold", "0.6 s before tone"]
    WLAB = {"before press": "Before press\n(−1.5–0 s)", "whole hold": "Whole hold\n(press → tone)", "0.6 s before tone": "0.6 s\nbefore tone"}
    WCOL = {"before press": "#BBDEFB", "whole hold": "#64B5F6", "0.6 s before tone": "#1565C0"}       # beta blue, light → dark
    OUT = {"occ":  (r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\report", "Fig 14 beta occupancy where in the trial.png"),
           "rate": (r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data\amir_data\supplementary", "Fig S17 beta rate where in the trial.png")}
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    PT = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False).sort_values(["session", "trial_index"])
    PT["position"] = PT.groupby("session")["trial_index"].rank(pct=True)
    B6 = TABS[BAND]["late"]
    B6 = B6[(B6.ref == "LOCAL") & (B6.group == "IL") & (B6.typ == "long") & B6.cond.isin(["correct", "late"])][["session", "rat", "trial", "cond"]]

    def wmean(v, tstart, a, b, how):                                 # mean occupancy (%) or rate (bursts/s) of the 50-ms bins inside [a, b]
        sel = (tstart >= a - 1e-9) & (tstart + BIN <= b + 1e-9)
        if not sel.any() or b <= a or not np.isfinite(v[sel]).all():
            return np.nan
        return 100 * v[sel].mean() if how == "occ" else v[sel].sum() / (sel.sum() * BIN)

    rows, CV_O, CV_R = [], [], []
    for s in files:
        z = np.load(os.path.join(BURST_DIR, s + ".npz")); T = pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))
        kp, kt = f"LOCAL|{BAND}|IL|press", f"LOCAL|{BAND}|IL|tone"
        if kp + "|occ" not in z.files or z[kp + "|occ"].shape[0] != len(T):
            print("skipped", s); continue
        Op, Cp, Ot, Ct = z[kp + "|occ"], z[kp + "|cnt"], z[kt + "|occ"], z[kt + "|cnt"]
        tp = A0P + BIN * np.arange(Op.shape[1]); tt_ = A0T + BIN * np.arange(Ot.shape[1])
        for _, r in B6[B6.session == s].iterrows():
            i = int(r.trial); tone = T["tone_s"].iat[i] - T["press_s"].iat[i]
            row = dict(session=s, trial=i, rat=str(r.rat), cond=r.cond, trial_index=T["trial_index"].iat[i], tone_after_press=tone)
            for how, P_, T_ in (("occ", Op, Ot), ("rate", Cp, Ct)):
                row[f"{how}|before press"] = wmean(P_[i], tp, -1.5, 0.0, how)
                row[f"{how}|whole hold"] = wmean(P_[i], tp, 0.0, min(tone, tp[-1] + BIN), how)
                row[f"{how}|0.6 s before tone"] = wmean(T_[i], tt_, -0.6, 0.0, how)
            rows.append(row); CV_O.append(100 * Op[i]); CV_R.append(Cp[i] / BIN)
    W = pd.DataFrame(rows); CV_O, CV_R = np.vstack(CV_O), np.vstack(CV_R)
    W = W.merge(PT[["session", "trial_index", "position"]], on=["session", "trial_index"], how="left")
    keep = W["position"].notna().values
    W, CV_O, CV_R = W[keep].reset_index(drop=True), CV_O[keep], CV_R[keep]
    W["late"] = (W.cond == "late").astype(int)
    W["phase"] = pd.cut(W["position"], [-0.001, *CUTS, 1.001], labels=PHS).astype(str)     # trial-number split
    TONE = W.tone_after_press.median()
    print(f"{W.session.nunique()} sessions, {len(W)} trials ({W.late.sum()} late); median tone {TONE:.2f} s after the press")

    def phase_fit(d, col):                                            # mixed model, no block term
        d = d.dropna(subset=[col]).assign(y=lambda x: x[col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("y ~ late * C(phase, Treatment('start'))", d, groups=d["session"]).fit(reml=True, method=["lbfgs", "powell", "nm"])
        P, V = m.params, m.cov_params(); out = {}
        for ph in PHS:
            k = ["late"] + ([f"late:C(phase, Treatment('start'))[T.{ph}]"] if ph != "start" else [])
            e = sum(P[x] for x in k); se = np.sqrt(sum(V.loc[a, b] for a in k for b in k)); out[ph] = (e, se, 2 * norm.sf(abs(e / se)))
        return out

    def figure(R, CV, ylab, unit, how, smooth=1):
        tc = A0P + BIN * np.arange(CV.shape[1]) + BIN / 2
        st = (W.phase == "start").values
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.4), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.35), facecolor="white")
        for c, (col, lab) in COLC.items():
            k = st & (W.cond == c).values
            S = pd.DataFrame(CV[k]).assign(session=W.session.values[k]).groupby("session").mean()
            if smooth > 1: S = S.T.rolling(smooth, center=True, min_periods=1).mean().T
            m, se = S.mean().values, S.sem().values
            a1.fill_between(tc, m - se, m + se, color=col, alpha=0.18, lw=0); a1.plot(tc, m, color=col, lw=1.6, label=f"{lab} (start of session)")
        a1.set_xlim(-1.5, TONE)
        for x_, lab in ((0, "press"), (TONE - 0.6, "−0.6 s"), (TONE, "tone")):
            a1.axvline(x_, color="k", ls="--", lw=0.8)
            a1.text(x_, 1.0, lab, transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        y0, y1 = a1.get_ylim(); h = y1 - y0; a1.set_ylim(y0 - 0.22 * h, y1)                  # room for the window bars at the bottom
        for k, (w, (a, b)) in enumerate({"before press": (-1.5, 0.0), "whole hold": (0.0, TONE), "0.6 s before tone": (TONE - 0.6, TONE)}.items()):
            y = y0 - 0.05 * h - k * 0.06 * h; c = WCOL[w]
            a1.plot([a, b], [y, y], color=c, lw=2.4, solid_capstyle="butt")
            a1.plot([a, a], [y - 0.02 * h, y + 0.02 * h], color=c, lw=2.4); a1.plot([b, b], [y - 0.02 * h, y + 0.02 * h], color=c, lw=2.4)
        a1.set_xlabel("Time from press (s)"); a1.set_ylabel(ylab)
        a1.legend(frameon=False, fontsize=7.5, loc="upper left")
        a1.text(-0.17, 1.06, "A", transform=a1.transAxes, fontsize=12, fontweight="bold")
        r0 = R[R.phase == "start"].set_index("window").loc[WINS]
        span = r0.ci_high.max() - min(0, r0.ci_low.min())
        for k, w in enumerate(WINS):
            rr = r0.loc[w]
            a2.bar(k, rr.late_minus_correct, width=0.6, color=WCOL[w], edgecolor="k", lw=0.4)
            a2.errorbar(k, rr.late_minus_correct, yerr=[[rr.late_minus_correct - rr.ci_low], [rr.ci_high - rr.late_minus_correct]],
                        color="k", capsize=3, lw=1)
            a2.text(k, max(rr.ci_high, 0) + 0.04 * span, rr.sig, ha="center", va="bottom", fontsize=8.5)   # always above
        a2.set_ylim(top=r0.ci_high.max() + 0.2 * span)
        a2.axhline(0, color="k", lw=0.8)
        a2.set_xticks(range(len(WINS))); a2.set_xticklabels([WLAB[w] for w in WINS], fontsize=8)
        a2.set_ylabel(f"Late − correct at the\nstart of the session ({unit})")
        a2.text(-0.32, 1.06, "B", transform=a2.transAxes, fontsize=12, fontweight="bold")
        for ax in (a1, a2):
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        _savefig(fig, os.path.join(*OUT[how]))
        plt.show()

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    for how, CV, ylab, unit, sm in (("occ", CV_O, "Beta burst occupancy (%)\n(350-ms smoothing)", "%-points", 7),
                                    ("rate", CV_R, "Beta burst rate (bursts/s)\n(150-ms smoothing)", "bursts/s", 3)):
        name = "occupancy" if how == "occ" else "rate"
        # 1) simple test (start of the session, trial-number split): per session mean late − mean correct, sign-flip
        print(f"\n===== {name} ({unit}) — 1) SIMPLE TEST at the start of the session =====")
        for w in WINS:
            m_ = W[W.phase == "start"].groupby(["session", "late"])[f"{how}|{w}"].mean().unstack()
            diff = (m_[1] - m_[0]).dropna(); mm, p_, n_ = fn.sign_flip_p(diff.values)
            print(f"   {w:18s} {mm:+.3f}, late higher in {int((diff > 0).sum())}/{n_} sessions, p = {p_:.4f}")
        # 2) mixed model per part
        R = pd.DataFrame([dict(window=w, phase=ph, late_minus_correct=e, ci_low=e - 1.96 * se, ci_high=e + 1.96 * se, p=p)
                          for w in WINS for ph, (e, se, p) in phase_fit(W, f"{how}|{w}").items()])
        R["sig"] = R["p"].apply(stars)
        for folder in (TABLES, BETA_DIR): _table(R, os.path.join(folder, f"step4_beta_{name}_where_in_trial_noblock.csv"), index=False)
        print(f"===== {name} ({unit}) — 2) MIXED MODEL, late − correct per part and window =====")
        print(R.to_string(index=False))
        figure(R, CV, ylab, unit, how, sm)
    pd.reset_option("display.float_format")


@cell()
def gamma_step1():
    """GAMMA STEP 1. BETA vs GAMMA BEFORE LATE RESPONSES AT THE SESSION START: A BURSTS, B BAND POWER (IL, LOCAL, long trials, all blocks) (needs cell 0)"""
    global A0P, A0T, BANDS, BIN, BPB, BPW, CUTS, D, GAMMA_DIR, PHS, PROP_DIR, PT, RA, RB, WINS, WLAB, a, b, bp, burst_effect
    global ch, e, f, fig, files, fp, glob, i, ids, j, model, norm, out, p, pc, prep, profile, r, rows, s, se, smf, stars, tr
    global w, warnings, wm
    # ── GAMMA STEP 1. BETA vs GAMMA BEFORE LATE RESPONSES AT THE SESSION START: A BURSTS, B BAND POWER (IL, LOCAL, long trials, all blocks) (needs cell 0) ──
    globals().pop("C", None)
    import glob, warnings
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    GAMMA_DIR = os.path.join(AMIR, "Gamma bursts"); os.makedirs(GAMMA_DIR, exist_ok=True)
    PROP_DIR  = os.path.join(DATA_OUT, "Burst properties p98")         # only to get the IL channel used for the bursts
    PHS, CUTS = ["start", "middle", "end"], (0.25, 0.85)
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    PT = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False).sort_values(["session", "trial_index"])
    PT["position"] = PT.groupby("session")["trial_index"].rank(pct=True)

    def model(dd):                                                       # late − correct at the session start (estimate, SE, p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("y ~ late * C(phase, Treatment('start')) + C(block)", dd, groups=dd["session"]).fit(reml=True, method=["lbfgs", "powell", "nm"])
        e, se = m.params["late"], m.bse["late"]; return e, se, 2 * norm.sf(abs(e / se))

    def prep(W):
        W = W.merge(PT[["session", "trial_index", "position"]], on=["session", "trial_index"], how="left").dropna(subset=["position"])
        W["late"] = (W.cond == "late").astype(int); W["block"] = W.long_prob.round(2).astype(str)
        W["phase"] = pd.cut(W.position, [-0.001, *CUTS, 1.001], labels=PHS).astype(str)
        return W

    # ── A. BURSTS (50-ms bins), relative to each band's correct level at the start ──
    BANDS = {"beta 15-30": ("Beta (15–30 Hz)", "#1565C0"), "gamma 30-100": ("Gamma (30–100 Hz)", "#6a1b9a")}
    BIN, A0P, A0T = fn.BURST_BIN, fn.BURST_A0["press"], fn.BURST_A0["tone"]
    WINS = ["before press", "whole hold", "0.6 s before tone"]
    WLAB = {"before press": "Before press\n(−1.5–0 s)", "whole hold": "Whole hold\n(press → tone)", "0.6 s before tone": "0.6 s\nbefore tone"}
    files = [s for s in final["session"] if os.path.exists(os.path.join(BURST_DIR, s + ".npz"))]

    def wm(v, t_, a, b):                                                 # mean occupancy (%) of the 50-ms bins inside [a, b]
        sel = (t_ >= a - 1e-9) & (t_ + BIN <= b + 1e-9)
        return 100 * v[sel].mean() if sel.any() and b > a and np.isfinite(v[sel]).all() else np.nan

    def burst_effect(band):
        rows = []
        for s in files:
            z = np.load(os.path.join(BURST_DIR, s + ".npz")); T = pd.read_csv(os.path.join(BURST_DIR, s + "_trials.csv"))
            kp, kt = f"LOCAL|{band}|IL|press|occ", f"LOCAL|{band}|IL|tone|occ"
            if kp not in z.files or z[kp].shape[0] != len(T): continue
            Op, Ot = z[kp], z[kt]; tp = A0P + BIN * np.arange(Op.shape[1]); tt_ = A0T + BIN * np.arange(Ot.shape[1])
            for i in np.where((T["type"].values == "long") & T["label"].isin(["correct", "late"]).values)[0]:
                tone = T["tone_s"].iat[i] - T["press_s"].iat[i]
                rows.append(dict(session=s, trial_index=T["trial_index"].iat[i], cond=T["label"].iat[i], long_prob=T["long_prob"].iat[i],
                                 **{"before press": wm(Op[i], tp, -1.5, 0), "whole hold": wm(Op[i], tp, 0, min(tone, tp[-1] + BIN)),
                                    "0.6 s before tone": wm(Ot[i], tt_, -0.6, 0)}))
        W = prep(pd.DataFrame(rows)); out = []
        for w in WINS:
            dd = W.dropna(subset=[w]).rename(columns={w: "y"}); e, se, p = model(dd); st = dd[dd.phase == "start"]
            lvl = st.loc[st.late == 0, "y"].mean()                         # the band's own correct level at the session start
            out.append(dict(band=band, window=w, sessions=dd.session.nunique(), n_correct_start=int((st.late == 0).sum()), n_late_start=int(st.late.sum()),
                            occ_correct_start=lvl, occ_late_start=st.loc[st.late == 1, "y"].mean(),
                            late_minus_correct=e, ci_low=e - 1.96 * se, ci_high=e + 1.96 * se,
                            value=100 * e / lvl, v_low=100 * (e - 1.96 * se) / lvl, v_high=100 * (e + 1.96 * se) / lvl, p=p, sig=stars(p)))
        return pd.DataFrame(out)

    RA = pd.concat([burst_effect(b) for b in BANDS], ignore_index=True)
    pd.set_option("display.width", 250)
    print("A  BURSTS: late − correct occupancy at the START of the session (%-points; value = % of the band's correct level), IL, LOCAL:")
    print(RA.round(3).to_string(index=False))
    print("   (check: beta %-points = beta Step 4, start: before press +3.32, whole hold +4.72, 0.6 s before tone +7.75)")

    # ── B. BAND POWER (band-power files, LOCAL, same IL channel), % change ──
    BPB = {"beta 13-29": ("Beta (13–29 Hz)", "#1565C0"), "lowG 30-60": ("Low gamma (30–60 Hz)", "#6a1b9a"),
           "highG 61-100": ("High gamma (61–100 Hz)", "#C39BD3")}
    BPW = {"pre": "Before press\n(−0.5 to −0.2 s)", "hold": "Hold\n(0.2–0.85 s)", "pre_tone": "0.3 s\nbefore tone"}
    rows = []
    for f in sorted(glob.glob(os.path.join(PROP_DIR, "*.npz"))):
        s = os.path.basename(f)[:-4]; ch = str(np.load(f)["channel"][0])
        try:
            bp = load_bp(s, "LOCAL"); ids = [str(c) for c in bp["meta"]["channel_ids"]]; j = ids.index(ch)
        except (FileNotFoundError, ValueError) as e:
            print(f"   {s}: skipped ({e!r})"); continue
        tr = bp["trials"]
        for i in np.where((tr["type"].values == "long") & tr["label"].isin(["correct", "late"]).values)[0]:
            r = dict(session=s, trial_index=tr["trial_index"].iat[i], cond=tr["label"].iat[i], long_prob=tr["long_prob"].iat[i])
            for b in BPB:
                for w in BPW: r[f"{b}|{w}"] = bp["logpow"][b][w][i, j]
            rows.append(r)
    D = prep(pd.DataFrame(rows)); pc = lambda v: 100 * (10 ** v - 1); out = []
    for b in BPB:
        for w in BPW:
            e, se, p = model(D.dropna(subset=[f"{b}|{w}"]).rename(columns={f"{b}|{w}": "y"}))
            out.append(dict(band=b, window=w, sessions=D.session.nunique(), value=pc(e), v_low=pc(e - 1.96 * se), v_high=pc(e + 1.96 * se), p=p, sig=stars(p)))
    RB = pd.DataFrame(out)
    print(f"\nB  BAND POWER: late − correct power at the START of the session (% change), IL, LOCAL "
          f"({D.session.nunique()} sessions; start: {int(((D.phase == 'start') & (D.late == 0)).sum())} correct, {int(((D.phase == 'start') & (D.late == 1)).sum())} late):")
    print(RB.round(3).to_string(index=False))
    _table(RA, os.path.join(GAMMA_DIR, "gamma_step1_A_bursts.csv"), index=False)
    _table(RB, os.path.join(GAMMA_DIR, "gamma_step1_B_band_power.csv"), index=False)

    # ── figure: A bursts, B band power — one line per band over the three windows ──
    def profile(ax, R, bands, wins, wlab, ylab, letter, title):
        nb = len(bands); step = 0.18 if nb == 2 else 0.16                 # horizontal distance between the bands
        for j, (b, (lab, col)) in enumerate(bands.items()):
            d = R[R.band == b].set_index("window").loc[list(wins)]; x = np.arange(len(wins)) + (j - (nb - 1) / 2) * step
            ax.errorbar(x, d["value"], yerr=[d["value"] - d["v_low"], d["v_high"] - d["value"]], color=col, lw=2.4, capsize=4, label=lab, zorder=2)
            for xi, (_, r) in zip(x, d.iterrows()):
                sig = r.p < 0.05
                ax.plot(xi, r["value"], "o", ms=9, color=col, mfc=col if sig else "white", mew=2, zorder=3)
                ax.text(xi, r["v_high"], r.sig, ha="center", va="bottom", fontsize=9.5, color=col,
                        fontweight="bold" if sig else "normal")                 # stars just above the error bar
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(wins))); ax.set_xticklabels([wlab[w] for w in wins]); ax.set_xlim(-0.5, len(wins) - 0.5)
        y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + 0.08 * (y1 - y0))      # room for the stars at the top
        ax.set_ylabel(ylab); ax.legend(frameon=False, fontsize=9.5, loc="upper left")
        ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9, "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[2.3, 1]), facecolor="white", constrained_layout=True)
    profile(a, RA, BANDS, WINS, WLAB, "Late − correct burst occupancy\n(% of the band's correct level)", "A", "Bursts")
    profile(b, RB, BPB, ["pre_tone"], BPW, "Late − correct power (% change)", "B", "Band power, just before the tone") 
    fig.suptitle("Before late responses at the session start: only beta is raised just before the tone (IL, LOCAL, long trials)",
                 fontsize=13.5, fontweight="bold")
    fp = os.path.join(GAMMA_DIR, "gamma_step1_beta_vs_gamma_start.png")
    _savefig(fig, fp); plt.close(fig)
    print("\nsaved:", fp, "| exists:", os.path.exists(fp)); display(Image(fp))
