"""
fx_res_pilot.py — the pilot steps of the phase coupling (07 section 2, first part): choosing the channel pair and the reference
on the pilot session, rat 3 with the pilot channels, the IL-ahead example, and the all-pairs test.
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fx_results import cell

SHORT = {"Infralimbic area": "IL", "Prelimbic area": "PL", "Cingulate area 1": "Cg1"}


def setup(B):
    """channel regions of every session (used by the reference scan)."""
    CHREG = pd.read_csv(os.path.join(B["TABLES"], "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    return dict(CHREG=CHREG)
SETUP_NAMES = ("CHREG",)


@cell()
def pilot_channels_probe():
    """1o. THE 20 PILOT CHANNELS ON THE PROBE (black), 3 PAIRS TESTED (one colour per pair), REGION BORDERS (2 lines) (needs cell 0)"""
    global CHREG, Line2D, ORDER, P, PAIRS, PCOL, PLV_DIR, SES, a, ax, b, borders, c1, c2, fig, fp, h, hi, hi_s, lo, lo_s, meta
    global name, pdf, per, pname, sel, sp, y_il_pl, y_pl_cg, yb, ymax, ymin
    # ── 1o. THE 20 PILOT CHANNELS ON THE PROBE (black), 3 PAIRS TESTED (one colour per pair), REGION BORDERS (2 lines) (needs cell 0) ──
    from matplotlib.lines import Line2D
    PLV_DIR = os.path.join(AMIR, "Theta", "Phase connectivity"); os.makedirs(PLV_DIR, exist_ok=True)
    SES = TEST
    PAIRS = {"IL–Cg1": ("imec0.ap#AP49", "imec0.ap#AP205"), "IL–PL": ("imec0.ap#AP49", "imec0.ap#AP241"),
             "PL–Cg1": ("imec0.ap#AP257", "imec0.ap#AP156")}
    PCOL = {"IL–Cg1": "#81C784", "IL–PL": "#43A047", "PL–Cg1": "#1B5E20"}            # light, mid, dark green
    ORDER = [("Infralimbic area", "IL"), ("Prelimbic area", "PL"), ("Cingulate area 1", "Cg1")]   # ventral -> dorsal
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)

    meta = load_bp(SES, "CMR")["meta"]
    pdf = pd.DataFrame({"channel_id": [str(c) for c in meta["channel_ids"]], "x": meta["x"], "y": meta["y"]})
    pdf["shank"] = (pdf["x"] // 250).astype(int)
    pdf["region"] = pdf.channel_id.map(CHREG[CHREG.session == SES].set_index("channel_id")["region"])
    sel = fn.select_channels(pdf, n_total=20).copy(); sel["channel_id"] = sel["channel_id"].astype(str)
    P = pdf.set_index("channel_id")

    # region borders: per shank halfway between the regions, then the median over shanks -> one line per border
    borders = {}
    for (lo, lo_s), (hi, hi_s) in zip(ORDER[:-1], ORDER[1:]):
        per = {sh: (d[d.region == lo].y.max() + d[d.region == hi].y.min()) / 2
               for sh, d in pdf.groupby("shank") if (d.region == lo).any() and (d.region == hi).any()}
        borders[f"{lo_s}/{hi_s}"] = np.median(list(per.values()))
        print(f"{lo_s}/{hi_s} border per shank (µm): {{{', '.join(f'shank {k}: {v:.0f}' for k, v in per.items())}}} -> line at {borders[f'{lo_s}/{hi_s}']:.0f} µm")

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(6.5, 9), facecolor="white")
    ax.scatter(pdf.x, pdf.y, s=6, color="0.85", zorder=1)                              # all channels
    for pname, (c1, c2) in PAIRS.items():                                              # pairs tested, one colour each
        ax.plot(P.loc[[c1, c2], "x"], P.loc[[c1, c2], "y"], color=PCOL[pname], lw=2, ls="--", zorder=2)
    ax.scatter(sel.x, sel.y, s=55, color="k", marker="s", zorder=3)                    # 20 pilot channels (black)

    # 2 horizontal lines + region names
    y_il_pl, y_pl_cg = borders["IL/PL"], borders["PL/Cg1"]
    for yb in (y_il_pl, y_pl_cg):
        ax.axhline(yb, color="k", lw=1.2, zorder=0)
    ymin, ymax = pdf.y.min() - 100, pdf.y.max() + 100
    for name, (a, b) in (("IL", (ymin, y_il_pl)), ("PL", (y_il_pl, y_pl_cg)), ("Cg1", (y_pl_cg, ymax))):
        ax.text(-140, (a + b) / 2, name, fontsize=14, fontweight="bold", ha="center", va="center")
    ax.set_xlim(-200, 800); ax.set_ylim(ymin, ymax)

    h = [Line2D([], [], marker="s", ls="", color="0.85", ms=4, label="all channels"),
         Line2D([], [], marker="s", ls="", color="k", ms=7, label="20 pilot channels")]
    h += [Line2D([], [], color=col, ls="--", lw=2, label=f"pair {p}") for p, col in PCOL.items()]
    ax.legend(handles=h, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm, higher = more dorsal)"); ax.set_xticks([0, 250, 500, 750])
    ax.set_title(f"Theta coupling pilot: 20 channels and the 3 pairs tested\n({SES})", fontsize=12, fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fp = os.path.join(PLV_DIR, "pilot_20_channels_pairs_probe.png")
    _savefig(fig, fp); plt.show()


@cell()
def pilot_coupling_cmr():
    """1p. THETA PLV, PLI, PPC, DEBIASED wPLI² (MNE), 3 PAIRS, PILOT SESSION, CORRECT LONG, HOLD, CMR — one figure (needs cell 0)"""
    global A, B, CHS, E, FR, HOLD, MEAS, NCYC, NSH, PAIRS, PCOL, PC_DIR, REF, RNG, ROWS, SES, STEP_S, T, Tk, W, WV, X, _, a0
    global a1, ax, axs, c1, c2, ci, cohy, cons, fig, fp, fs, full, i, idx, lab, m, mne_connectivity, objs, ok, own, pname
    global ptxt, r_, real, rec, recN, row, sh, sp, spectral_connectivity_epochs, ss, t, top, v
    # ── 1p. THETA PLV, PLI, PPC, DEBIASED wPLI² (MNE), 3 PAIRS, PILOT SESSION, CORRECT LONG, HOLD, CMR — one figure (needs cell 0) ──
    import scipy.signal as ss
    import mne_connectivity
    from mne_connectivity import spectral_connectivity_epochs
    PCOL = {"IL–Cg1": "#81C784", "IL–PL": "#43A047", "PL–Cg1": "#1B5E20"}            # light, mid, dark green
    PC_DIR = os.path.join(AMIR, "Theta", "Phase connectivity"); os.makedirs(PC_DIR, exist_ok=True)
    SES, REF = TEST, "CMR"
    PAIRS = {"IL–Cg1": ("imec0.ap#AP49", "imec0.ap#AP205"), "IL–PL": ("imec0.ap#AP49", "imec0.ap#AP241"),
             "PL–Cg1": ("imec0.ap#AP257", "imec0.ap#AP156")}
    FR, NCYC = np.array(fn.MAIN_BANDS["theta 4-7"]["freqs"]), fn.MAIN_BANDS["theta 4-7"]["n_cycles"]
    HOLD, STEP_S, NSH, RNG = (0.20, 0.85), 0.010, 500, np.random.default_rng(0)
    MEAS = [("plv", "plv", "PLV"), ("pli", "pli", "PLI"), ("ppc", "ppc", "PPC"), ("dwpli", "wpli2_debiased", "Debiased wPLI²")]
    CHS = sorted({c for p in PAIRS.values() for c in p})

    T = load_bp(SES, "LOCAL")["trials"]; T = T[(T["type"] == "long") & (T["label"] == "correct")].reset_index(drop=True)
    rec = fn.load_session(PROC, SES, XB, refs=(REF,))[0][REF]
    full = [str(c) for c in rec.get_channel_ids()]
    objs = [rec.get_channel_ids()[full.index(c)] for c in CHS]
    recN = rec.select_channels(objs) if hasattr(rec, "select_channels") else rec.channel_slice(channel_ids=objs)
    a0, a1 = fn.MAIN_EPOCHS["Press"]
    X, ok = fn.read_epochs(recN, T["press_s"].values, a0, a1); fs = recN.get_sampling_frequency(); Tk = T[ok].reset_index(drop=True)
    assert ((Tk["release_s"] - Tk["press_s"]).values >= HOLD[1]).all(), "some trials released before the end of the hold window"
    t = np.arange(X.shape[-1]) / fs + a0
    idx = np.where((t >= HOLD[0]) & (t <= HOLD[1]))[0][::int(round(STEP_S * fs))]
    ci = {c: i for i, c in enumerate(CHS)}

    def own(A, B):
        """validated own formulas (identical to MNE within 0.0008) — used for the shuffles only"""
        S = A * np.conj(B); E = S / np.abs(S).clip(1e-30); n = A.shape[0]
        im = S.imag; num = im.sum(0) ** 2 - (im ** 2).sum(0); den = np.abs(im).sum(0) ** 2 - (im ** 2).sum(0)
        return dict(plv=np.abs(E.mean(0)).mean(0).mean(), pli=np.abs(np.sign(E.imag).mean(0)).mean(0).mean(),
                    ppc=((np.abs(E.sum(0)) ** 2 - n) / (n * (n - 1))).mean(),
                    dwpli=np.nanmean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)))

    W = np.stack([ss.fftconvolve(X.astype(np.float64), fn.morlet(f, NCYC, fs)[None, None, :], mode="same", axes=-1) for f in FR], axis=2)[..., idx]
    ROWS = []
    for pname, (c1, c2) in PAIRS.items():
        # real values: MNE
        cons = spectral_connectivity_epochs(X[:, [ci[c1], ci[c2]]].astype(np.float64), method=[m[1] for m in MEAS],
                                            indices=(np.array([0]), np.array([1])), sfreq=fs, mode="cwt_morlet", cwt_freqs=FR,
                                            cwt_n_cycles=NCYC, fmin=FR[0], fmax=FR[-1], faverage=False, verbose=False)
        real = {m[0]: np.nanmean(np.squeeze(con.get_data())[:, idx].mean(0)) for m, con in zip(MEAS, cons)}
        cohy = spectral_connectivity_epochs(X[:, [ci[c1], ci[c2]]].astype(np.float64), method="cohy", indices=(np.array([0]), np.array([1])),
                                            sfreq=fs, mode="cwt_morlet", cwt_freqs=FR, cwt_n_cycles=NCYC, fmin=FR[0], fmax=FR[-1],
                                            faverage=False, verbose=False)
        A, B = W[:, ci[c1]], W[:, ci[c2]]
        E = A * np.conj(B); E = E / np.abs(E).clip(1e-30)
        row = dict(pair=pname, trials=len(Tk), mean_dphi_deg=np.degrees(np.angle(E.mean())))
        # chance: own validated code, channel 2 from another trial
        sh = [own(A, B[RNG.permutation(len(A))]) for _ in range(NSH)]
        for m, _, _ in MEAS:
            v = np.array([d[m] for d in sh])
            row.update({m: real[m], f"chance95_{m}": np.percentile(v, 95), f"p_{m}": (1 + (v >= real[m]).sum()) / (1 + NSH)})
        ROWS.append(row)
    del W, X
    WV = pd.DataFrame(ROWS); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
    print(f"{SES}: {len(Tk)} correct long trials, CMR, hold {HOLD[0]}–{HOLD[1]} s | values: MNE {mne_connectivity.__version__} "
          f"(dwpli = wpli2_debiased) | chance: {NSH} shuffles | mean Δφ > 0 = first region leads")
    print(WV.round(3).to_string(index=False))
    _table(WV, os.path.join(PC_DIR, "pilot_3pairs_coupling_hold_CMR.csv"), index=False)

    # ── one figure: 2 x 2 panels; one bar per pair (pair colours as in 1o), black line = 95 % chance, raw p above ──
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), facecolor="white", constrained_layout=True)
    ptxt = lambda p: "p < 0.002" if p < 0.002 else f"p = {p:.3f}"
    for ax, (m, _, lab) in zip(axs.flat, MEAS):
        top = max(WV[m].max(), WV[f"chance95_{m}"].max())
        for i, r_ in WV.iterrows():
            ax.bar(i, r_[m], width=0.55, color=PCOL[r_["pair"]])
            ax.plot([i - 0.275, i + 0.275], [r_[f"chance95_{m}"]] * 2, color="k", lw=2, label="chance (95 % of shuffles)" if i == 0 else None)
            ax.text(i, max(r_[m], r_[f"chance95_{m}"], 0) + 0.02 * top, ptxt(r_[f"p_{m}"]), ha="center", va="bottom", fontsize=10)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(range(len(WV))); ax.set_xticklabels(WV.pair, fontsize=13)
        ax.set_ylabel(f"Theta {lab} (hold)", fontsize=14); ax.set_ylim(min(0, WV[m].min() * 1.2), top * 1.25)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0, 0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle(f"Theta phase coupling during the hold, CMR, {len(Tk)} correct long trials ({SES})", fontsize=15, fontweight="bold")
    fp = os.path.join(PC_DIR, "pilot_3pairs_coupling_hold_CMR.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def theta_left_local():
    """THETA LEFT IN LOCAL (% of CMR), hold, correct long: one point per session, regions in colour (representative channels) (needs cell 0)"""
    global BAND, L_, PC_DIR, R, RC, REFS, REGS, Tr, ax, c, cl, d, fb, fig, fp, g, i, ids, r, rng, rows, s, sp, v, zb
    # ── THETA LEFT IN LOCAL (% of CMR), hold, correct long: one point per session, regions in colour (representative channels) (needs cell 0) ──
    PC_DIR = os.path.join(AMIR, "Theta", "Phase connectivity"); os.makedirs(PC_DIR, exist_ok=True)
    REFS, REGS, BAND = ("CMR", "LOCAL"), ("Cg1", "PL", "IL"), "theta 4-7"
    RC = {"Cg1": "#9467bd", "PL": "#1f77b4", "IL": "#ff7f0e"}

    rows = []
    for s in final["session"]:
        fb = os.path.join(BURST_DIR, s + ".npz")
        if not (os.path.exists(fb) and all(os.path.exists(os.path.join(BP_DIRS[r], s + ".pkl")) for r in REFS)): continue
        zb = np.load(fb); R = {r: load_bp(s, r) for r in REFS}
        for g in REGS:
            if f"CMR|{g}|channel" not in zb.files: continue
            c = str(zb[f"CMR|{g}|channel"][0]); v = {}
            for r in REFS:
                ids = [str(x) for x in R[r]["meta"]["channel_ids"]]
                Tr = R[r]["trials"]; cl = ((Tr["label"] == "correct") & (Tr["type"] == "long")).values
                if c not in ids or cl.sum() < 10: break
                v[r] = np.nanmean(R[r]["logpow"][BAND]["hold"][cl, ids.index(c)])        # mean log10 power over trials
            if len(v) == 2:
                rows.append(dict(session=s, rat=str(RAT[s]), region=g, LOCAL_pct_of_CMR=100 * 10 ** (v["LOCAL"] - v["CMR"])))
        del R
    L_ = pd.DataFrame(rows)
    print("theta (4–7 Hz) in LOCAL, % of CMR, hold, correct long, representative channels:")
    for g in REGS:
        d = L_[L_.region == g].LOCAL_pct_of_CMR
        print(f"  {g:4s} ({len(d)} sessions): median {d.median():.1f} %  [25th–75th {d.quantile(.25):.1f}–{d.quantile(.75):.1f} %]")
    _table(L_, os.path.join(PC_DIR, "theta_left_LOCAL_vs_CMR_all_sessions.csv"), index=False)

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="white")
    rng = np.random.default_rng(0)
    for i, g in enumerate(REGS):
        d = L_[L_.region == g].LOCAL_pct_of_CMR
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=28, color=RC[g], alpha=0.8, zorder=2)
        ax.plot([i - 0.25, i + 0.25], [d.median()] * 2, color="k", lw=2.5, zorder=3)
        ax.text(i + 0.3, d.median(), f"{d.median():.0f} %", va="center", fontsize=11)
    ax.axhline(100, color="0.5", lw=0.9, ls="--")
    ax.set_yscale("log"); ax.set_xticks(range(len(REGS))); ax.set_xticklabels(REGS, fontsize=14)
    ax.set_ylabel("Theta in LOCAL (% of CMR), hold", fontsize=13)
    ax.set_title("Theta left after LOCAL referencing\n(one point = one session; black = median; dashed = same as CMR)", fontsize=12, fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fp = os.path.join(PC_DIR, "theta_left_LOCAL_vs_CMR_all_sessions.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def reference_scan():
    """MOST LOCAL REFERENCE THAT STILL HAS THETA: same-region rings, theta peak by FOOOF (specparam), pilot, correct long, hold (needs cell 0 and 1q-A)"""
    global DIST, FOOOF, FOOOF_KW, FQ, HOLD, IDS, INNER, MAIN, MIN_CH, NC, OUTER, P, P0, PCOL, REG, SC, SEEDS, SES, STEP_S, T
    global W_, X, a0, a1, c, cell, cf, fs, h, hi, i, idx, lo, loc_, m1, m2, name, ok, ok_all, ok_main, rec0, regmap_, rows
    global s_set, spec, ss, t, th0, theta_peak, x0
    # ── MOST LOCAL REFERENCE THAT STILL HAS THETA: same-region rings, theta peak by FOOOF (specparam), pilot, correct long, hold (needs cell 0 and 1q-A) ──
    import scipy.signal as ss
    from fooof import FOOOF
    SEEDS = {"IL AP49": "imec0.ap#AP49", "PL AP241": "imec0.ap#AP241", "Cg1 AP205": "imec0.ap#AP205", "PL AP257": "imec0.ap#AP257", "Cg1 AP156": "imec0.ap#AP156"}
    MAIN = ("IL AP49", "PL AP241")
    INNER, OUTER, MIN_CH = (30, 50, 100, 150, 200, 300, 500), (150, 200, 250, 300, 400, 500, 750, 1000), 5
    FQ, NC, HOLD, STEP_S = np.arange(2, 31), 5, (0.20, 0.85), 0.010
    FOOOF_KW = dict(peak_width_limits=(1, 8), max_n_peaks=4, aperiodic_mode="fixed", verbose=False)
    PCOL = {"IL–Cg1": "#81C784", "IL–PL": "#43A047", "PL–Cg1": "#1B5E20"}            # light, mid, dark green

    # ── pilot recording, channel ids, regions and distances (these lines were in the old cell 1q-A, which is gone) ──
    SES = TEST
    rec0 = fn.load_session(PROC, SES, XB, refs=("original",))[0]["original"]
    IDS = [str(c) for c in rec0.get_channel_ids()]; loc_ = rec0.get_channel_locations()
    DIST = np.hypot(loc_[:, None, 0] - loc_[None, :, 0], loc_[:, None, 1] - loc_[None, :, 1])     # µm between every two channels
    regmap_ = CHREG[CHREG.session == SES].set_index("channel_id")["region"]
    REG = np.array([SHORT.get(regmap_.get(k, ""), "other") for k in IDS])

    T = load_bp(SES, "LOCAL")["trials"]; T = T[(T["type"] == "long") & (T["label"] == "correct")].reset_index(drop=True)
    a0, a1 = fn.MAIN_EPOCHS["Press"]
    X, ok = fn.read_epochs(rec0, T["press_s"].values, a0, a1); fs = rec0.get_sampling_frequency()      # all channels, original
    t = np.arange(X.shape[-1]) / fs + a0; idx = np.where((t >= HOLD[0]) & (t <= HOLD[1]))[0][::int(round(STEP_S * fs))]
    spec = lambda x: np.array([(np.abs(ss.fftconvolve(x, fn.morlet(f, NC, fs)[None, :], mode="same", axes=-1)[:, idx]) ** 2).mean() for f in FQ])

    def theta_peak(P):
        """FOOOF fit 2–30 Hz -> (peak centre Hz, height above 1/f in log10) of the strongest peak with centre 4–8 Hz, or (nan, 0)"""
        fm = FOOOF(**FOOOF_KW); fm.fit(FQ.astype(float), P, [2, 30])
        pk = [p for p in fm.peak_params_ if 4 <= p[0] <= 8]
        return (max(pk, key=lambda p: p[1])[0], max(pk, key=lambda p: p[1])[1]) if pk else (np.nan, 0.0)

    rows = []
    for name, c in SEEDS.items():
        i = IDS.index(c); x0 = X[:, i].astype(np.float64); P0 = spec(x0); th0 = P0[(FQ >= 4) & (FQ <= 7)].mean()
        cf, h = theta_peak(P0)
        rows.append(dict(seed=name, ring="original", inner=np.nan, outer=np.nan, n=0, theta_left_pct=100.0, peak=bool(h > 0), peak_hz=cf, peak_height=h))
        for hi in OUTER:
            for lo in INNER:
                if lo >= hi: continue
                s_set = np.where((REG == REG[i]) & (DIST[i] > lo) & (DIST[i] <= hi))[0]
                if len(s_set) < MIN_CH: continue
                P = spec(x0 - np.median(X[:, s_set], axis=1)); cf, h = theta_peak(P)
                rows.append(dict(seed=name, ring=f"{lo}–{hi}", inner=lo, outer=hi, n=len(s_set),
                                 theta_left_pct=100 * P[(FQ >= 4) & (FQ <= 7)].mean() / th0, peak=bool(h > 0), peak_hz=cf, peak_height=h))
        print(f"{name} done", flush=True)
    del X
    SC = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    _table(SC, os.path.join(AMIR, "Theta", "Phase connectivity", "reference_scan_fooof_pilot.csv"), index=False)

    print(f"\n{SES}: {int(ok.sum())} correct long trials, hold | theta peak = FOOOF peak with centre 4–8 Hz (height = log10 above 1/f)")
    print("original:"); print(SC[SC.ring == "original"][["seed", "peak", "peak_hz", "peak_height"]].round(2).to_string(index=False))
    W_ = SC[SC.ring != "original"].copy()
    cell = W_.apply(lambda r: f"{r.theta_left_pct:.0f}% {'P' if r.peak else '-'}{'' if not r.peak else f'{r.peak_height:.2f}'}", axis=1)
    print("\nper ring (sorted most local first): theta left % | P = theta peak (height) / - = no peak")
    print(W_.assign(v=cell).pivot_table(index=["outer", "inner"], columns="seed", values="v", aggfunc="first").to_string())

    # most local ring with a theta peak at the main pair / at all seeds (rings available for all of them)
    ok_main = W_[W_.seed.isin(MAIN)].groupby(["outer", "inner"]).agg(n_seeds=("peak", "size"), all_peak=("peak", "all")).reset_index()
    ok_all  = W_.groupby(["outer", "inner"]).agg(n_seeds=("peak", "size"), all_peak=("peak", "all")).reset_index()
    m1 = ok_main[(ok_main.n_seeds == len(MAIN)) & ok_main.all_peak].sort_values(["outer", "inner"])
    m2 = ok_all[(ok_all.n_seeds == len(SEEDS)) & ok_all.all_peak].sort_values(["outer", "inner"])
    print(f"\nmost local ring with a theta peak at {MAIN}: " + (f"{int(m1.iloc[0].inner)}–{int(m1.iloc[0].outer)} µm" if len(m1) else "none"))
    print("most local ring with a theta peak at all 5 seeds: " + (f"{int(m2.iloc[0].inner)}–{int(m2.iloc[0].outer)} µm" if len(m2) else "none"))


@cell()
def pilot_coupling_sr():
    """1q. THETA PLV, PLI, PPC, DEBIASED wPLI² (MNE), 3 PAIRS, PILOT, CORRECT LONG, HOLD — SAME-REGION 30–500 µm REFERENCE (needs cell 0)"""
    global A, ALPHA, B, CHREG, DIST, E, FR, HOLD, IDS, LOC, MEAS, NCYC, NSH, PAIRS, PCOL, PC_DIR, REG, RING, RNG, ROWS, SES
    global SHORT, STEP_S, T, Tk, WV, X, _, a0, a1, ax, axs, c1, c2, cons, derived, fig, fp, fs, i, idx, lab, m
    global mne_connectivity, ok, own, pname, ptxt, r_, real, rec0, ring_set, row, s1, s2, sh, sp, spectral_connectivity_epochs
    global ss, surv, t, top, v, wav, x1, x2, y0
    # ── 1q. THETA PLV, PLI, PPC, DEBIASED wPLI² (MNE), 3 PAIRS, PILOT, CORRECT LONG, HOLD — SAME-REGION 30–500 µm REFERENCE (needs cell 0) ──
    import scipy.signal as ss
    import mne_connectivity
    from mne_connectivity import spectral_connectivity_epochs
    PC_DIR = os.path.join(AMIR, "Theta", "Phase connectivity"); os.makedirs(PC_DIR, exist_ok=True)
    SES, RING = TEST, (30, 500)
    PAIRS = {"IL–Cg1": ("imec0.ap#AP49", "imec0.ap#AP205"), "IL–PL": ("imec0.ap#AP49", "imec0.ap#AP241"),
             "PL–Cg1": ("imec0.ap#AP257", "imec0.ap#AP156")}
    PCOL = {"IL–Cg1": "#81C784", "IL–PL": "#43A047", "PL–Cg1": "#1B5E20"}            # light, mid, dark green
    FR, NCYC = np.array(fn.MAIN_BANDS["theta 4-7"]["freqs"]), fn.MAIN_BANDS["theta 4-7"]["n_cycles"]
    HOLD, STEP_S, NSH, RNG = (0.20, 0.85), 0.010, 500, np.random.default_rng(0)
    MEAS = [("plv", "plv", "PLV"), ("pli", "pli", "PLI"), ("ppc", "ppc", "PPC"), ("dwpli", "wpli2_debiased", "Debiased wPLI²")]
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    SHORT = {"Infralimbic area": "IL", "Prelimbic area": "PL", "Cingulate area 1": "Cg1"}

    # ── recording, channel regions and distances ──
    rec0 = fn.load_session(PROC, SES, XB, refs=("original",))[0]["original"]
    IDS = [str(c) for c in rec0.get_channel_ids()]; LOC = rec0.get_channel_locations()
    REG = np.array([SHORT.get(r, "other") for r in pd.Series(IDS).map(CHREG[CHREG.session == SES].set_index("channel_id")["region"]).fillna("")])
    DIST = np.hypot(LOC[:, None, 0] - LOC[None, :, 0], LOC[:, None, 1] - LOC[None, :, 1])
    ring_set = lambda i: np.where((REG == REG[i]) & (DIST[i] > RING[0]) & (DIST[i] <= RING[1]))[0]

    # ── reference check: channels subtracted per side, shared channels (must be 0) ──
    print(f"{SES}: same-region ring {RING[0]}–{RING[1]} µm")
    for pname, (c1, c2) in PAIRS.items():
        s1, s2 = ring_set(IDS.index(c1)), ring_set(IDS.index(c2))
        print(f"  {pname}: {c1.split('#')[-1]} ({REG[IDS.index(c1)]}) minus {len(s1)} ch | {c2.split('#')[-1]} ({REG[IDS.index(c2)]}) minus {len(s2)} ch"
              f" | shared {len(np.intersect1d(s1, s2))}")

    # ── epochs (all channels, original), derived signals ──
    T = load_bp(SES, "LOCAL")["trials"]; T = T[(T["type"] == "long") & (T["label"] == "correct")].reset_index(drop=True)
    a0, a1 = fn.MAIN_EPOCHS["Press"]
    X, ok = fn.read_epochs(rec0, T["press_s"].values, a0, a1); fs = rec0.get_sampling_frequency(); Tk = T[ok].reset_index(drop=True)
    assert ((Tk["release_s"] - Tk["press_s"]).values >= HOLD[1]).all(), "some trials released before the end of the hold window"
    t = np.arange(X.shape[-1]) / fs + a0
    idx = np.where((t >= HOLD[0]) & (t <= HOLD[1]))[0][::int(round(STEP_S * fs))]
    derived = lambda c: (X[:, IDS.index(c)] - np.median(X[:, ring_set(IDS.index(c))], axis=1)).astype(np.float64)
    wav = lambda x: np.stack([ss.fftconvolve(x, fn.morlet(f, NCYC, fs)[None, :], mode="same", axes=-1) for f in FR], axis=1)[..., idx]

    def own(A, B):
        """validated own formulas (identical to MNE within 0.0008) — used for the shuffles only"""
        S = A * np.conj(B); E = S / np.abs(S).clip(1e-30); n = A.shape[0]
        im = S.imag; num = im.sum(0) ** 2 - (im ** 2).sum(0); den = np.abs(im).sum(0) ** 2 - (im ** 2).sum(0)
        return dict(plv=np.abs(E.mean(0)).mean(0).mean(), pli=np.abs(np.sign(E.imag).mean(0)).mean(0).mean(),
                    ppc=((np.abs(E.sum(0)) ** 2 - n) / (n * (n - 1))).mean(),
                    dwpli=np.nanmean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)))

    ROWS = []
    for pname, (c1, c2) in PAIRS.items():
        x1, x2 = derived(c1), derived(c2)
        cons = spectral_connectivity_epochs(np.stack([x1, x2], axis=1), method=[m[1] for m in MEAS], indices=(np.array([0]), np.array([1])),
                                            sfreq=fs, mode="cwt_morlet", cwt_freqs=FR, cwt_n_cycles=NCYC, fmin=FR[0], fmax=FR[-1],
                                            faverage=False, verbose=False)
        real = {m[0]: np.nanmean(np.squeeze(con.get_data())[:, idx].mean(0)) for m, con in zip(MEAS, cons)}
        A, B = wav(x1), wav(x2)
        E = A * np.conj(B); E = E / np.abs(E).clip(1e-30)
        row = dict(pair=pname, trials=len(Tk), mean_dphi_deg=np.degrees(np.angle(E.mean())))
        sh = [own(A, B[RNG.permutation(len(A))]) for _ in range(NSH)]
        for m, _, _ in MEAS:
            v = np.array([d[m] for d in sh])
            row.update({m: real[m], f"chance95_{m}": np.percentile(v, 95), f"p_{m}": (1 + (v >= real[m]).sum()) / (1 + NSH)})
        ROWS.append(row)
    del X
    WV = pd.DataFrame(ROWS); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
    print(f"\n{len(Tk)} correct long trials, hold {HOLD[0]}–{HOLD[1]} s | MNE {mne_connectivity.__version__} | {NSH} shuffles | Δφ > 0 = first region leads")
    print(WV.round(3).to_string(index=False))
    _table(WV, os.path.join(PC_DIR, "pilot_3pairs_coupling_hold_sameregion_30-500.csv"), index=False)

    # ── one figure: 4 panels; one bar per pair (pair colours), black line = 95 % chance; ★ = pair above chance in all four measures ──
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), facecolor="white", constrained_layout=True)
    ptxt = lambda p: "p < 0.002" if p < 0.002 else f"p = {p:.3f}"
    ALPHA = 0.05
    surv = {r_["pair"]: all(r_[f"p_{m}"] < ALPHA for m, _, _ in MEAS) for _, r_ in WV.iterrows()}
    for ax, (m, _, lab) in zip(axs, MEAS):
        top = max(WV[m].max(), WV[f"chance95_{m}"].max())
        for i, r_ in WV.iterrows():
            ax.bar(i, r_[m], width=0.55, color=PCOL[r_["pair"]])
            ax.plot([i - 0.275, i + 0.275], [r_[f"chance95_{m}"]] * 2, color="k", lw=2, label="chance (95 % of shuffles)" if i == 0 else None)
            y0 = max(r_[m], r_[f"chance95_{m}"], 0) + 0.02 * top
            ax.text(i, y0, ptxt(r_[f"p_{m}"]), ha="center", va="bottom", fontsize=10)
            if surv[r_["pair"]]: ax.text(i, y0 + 0.09 * top, "★", ha="center", va="bottom", fontsize=16)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(range(len(WV))); ax.set_xticklabels(WV.pair, fontsize=13)
        ax.set_ylabel(f"Theta {lab} (hold)", fontsize=14); ax.set_ylim(min(0, WV[m].min() * 1.2), top * 1.4)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle(f"Theta coupling in the hold, same-region 30–500 µm reference ({len(Tk)} correct long trials)\n★ = above chance in all four measures",
                 fontsize=14, fontweight="bold")
    fp = os.path.join(PC_DIR, "pilot_3pairs_coupling_hold_sameregion_30-500.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))
    print("pairs above chance in all four measures:", [p for p, v in surv.items() if v] or "none")


@cell()
def rat136_sessions():
    """STEP A-136 (compute): IL–PL theta coupling with the pilot channels AP49 – AP241, rat 136, all sessions with them, SR30-500, correct long, hold (needs cell 0)"""
    global A, B, C, CHREG, COL, D, E, FR, HOLD, MEAS, MIN_RING, MIN_TR, NCYC, NSH, PCH, PC_DIR, PXY, RING, RNG, SESS, SHORT
    global STEP_S, T, X, _, a0, a1, ax, axs, ch, cons, d, fig, fp, fs, ids, idx, lab, loc, m, mean_dphi, n_, n_ahead, ok, own
    global p, ptxt, real, rec, reg, ring_set, rng, row, rows, rs, s, sh, sp, spectral_connectivity_epochs, ss, t, t_, time, v
    global wav, x1, x2
    # ── STEP A-136 (compute): IL–PL theta coupling with the pilot channels AP49 – AP241, rat 136, all sessions with them, SR30-500, correct long, hold (needs cell 0) ──
    import time, scipy.signal as ss
    from mne_connectivity import spectral_connectivity_epochs
    PC_DIR = os.path.join(AMIR, "Theta", "Phase connectivity", "Rat 136 pilot channels"); os.makedirs(PC_DIR, exist_ok=True)
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    SHORT = {"Infralimbic area": "IL", "Prelimbic area": "PL", "Cingulate area 1": "Cg1"}
    PCH = {"IL": "imec0.ap#AP49", "PL": "imec0.ap#AP241"}; PXY = {"IL": (282, 0), "PL": (532, 2160)}
    RING = (30, 500)
    FR, NCYC = np.array(fn.MAIN_BANDS["theta 4-7"]["freqs"]), fn.MAIN_BANDS["theta 4-7"]["n_cycles"]
    HOLD, STEP_S, NSH, RNG = (0.20, 0.85), 0.010, 200, np.random.default_rng(0)
    MIN_TR, MIN_RING = 10, 5
    MEAS = [("plv", "plv"), ("pli", "pli"), ("ppc", "ppc"), ("dwpli", "wpli2_debiased")]

    def own(A, B):
        """validated own formulas (identical to MNE within 0.0008) — shuffles only"""
        S = A * np.conj(B); E = S / np.abs(S).clip(1e-30); n = A.shape[0]
        im = S.imag; num = im.sum(0) ** 2 - (im ** 2).sum(0); den = np.abs(im).sum(0) ** 2 - (im ** 2).sum(0)
        return dict(plv=np.abs(E.mean(0)).mean(0).mean(), pli=np.abs(np.sign(E.imag).mean(0)).mean(0).mean(),
                    ppc=((np.abs(E.sum(0)) ** 2 - n) / (n * (n - 1))).mean(),
                    dwpli=np.nanmean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)))

    C = CHREG[CHREG.session.isin(final["session"]) & (CHREG.rat.astype(str) == "136")]
    SESS = sorted(s for s, d in C.groupby("session")
                  if all(((d.channel_id == PCH[g]) & (np.hypot(d.x - PXY[g][0], d.y - PXY[g][1]) < 1)).any() for g in PCH))
    print(f"rat 136 sessions with both pilot channels: {len(SESS)}", flush=True)
    rows, t_ = [], time.time()
    for n_, s in enumerate(SESS, 1):
        T = fn.session_trials(K, s)
        T = T[(T["type"] == "long") & (T["label"] == "correct") & ((T["release_s"] - T["press_s"]) >= HOLD[1])].reset_index(drop=True)
        if len(T) < MIN_TR: print(f"[{n_}] {s}: {len(T)} trials, skipped"); continue
        rec = fn.load_session(PROC, s, XB, refs=("original",))[0]["original"]; fs = rec.get_sampling_frequency()
        ids = [str(c) for c in rec.get_channel_ids()]; loc = rec.get_channel_locations()
        reg = np.array([SHORT.get(r, "other") for r in pd.Series(ids).map(CHREG[CHREG.session == s].set_index("channel_id")["region"]).fillna("")])
        def ring_set(i): dd = np.hypot(loc[:, 0] - loc[i, 0], loc[:, 1] - loc[i, 1]); return np.where((reg == reg[i]) & (dd > RING[0]) & (dd <= RING[1]))[0]
        rs = {g: ring_set(ids.index(c)) for g, c in PCH.items()}
        if min(len(v) for v in rs.values()) < MIN_RING: print(f"[{n_}] {s}: ring too small, skipped"); continue
        a0, a1 = fn.MAIN_EPOCHS["Press"]
        X, ok = fn.read_epochs(rec, T["press_s"].values, a0, a1); t = np.arange(X.shape[-1]) / fs + a0
        idx = np.where((t >= HOLD[0]) & (t <= HOLD[1]))[0][::int(round(STEP_S * fs))]
        x1, x2 = [(X[:, ids.index(PCH[g])] - np.median(X[:, rs[g]], axis=1)).astype(np.float64) for g in ("IL", "PL")]; del X
        cons = spectral_connectivity_epochs(np.stack([x1, x2], axis=1), method=[m[1] for m in MEAS], indices=(np.array([0]), np.array([1])),
                                            sfreq=fs, mode="cwt_morlet", cwt_freqs=FR, cwt_n_cycles=NCYC, fmin=FR[0], fmax=FR[-1],
                                            faverage=False, verbose=False)
        real = {m[0]: np.nanmean(np.squeeze(con.get_data())[:, idx].mean(0)) for m, con in zip(MEAS, cons)}
        wav = lambda x: np.stack([ss.fftconvolve(x, fn.morlet(f_, NCYC, fs)[None, :], mode="same", axes=-1) for f_ in FR], axis=1)[..., idx]
        A, B = wav(x1), wav(x2); E = A * np.conj(B); E = E / np.abs(E).clip(1e-30)
        sh = [own(A, B[RNG.permutation(len(A))]) for _ in range(NSH)]
        row = dict(session=s, pilot=(s == TEST), ref="SR30-500", pair="IL–PL", trials=len(T), dphi_deg=np.degrees(np.angle(E.mean())))
        for m, _ in MEAS:
            v = np.array([d_[m] for d_ in sh]); row.update({m: real[m], f"chance_{m}": v.mean(), f"p_{m}": (1 + (v >= real[m]).sum()) / (1 + NSH)})
        rows.append(row)
        print(f"[{n_}/{len(SESS)}] {s}{' (PILOT)' if s == TEST else ''}: {len(T)} trials, Δφ {row['dphi_deg']:+.0f}° | {(time.time() - t_) / 60:.1f} min", flush=True)
        del rec
    D = pd.DataFrame(rows); fp = os.path.join(PC_DIR, "rat136_pilot_channels_sessions.csv"); D.to_csv(fp, index=False)
    # ── STEP A-136 (IL–PL): 1q FIGURE, SESSIONS AVERAGED — pilot channels AP49 – AP241, rat 136, 10 sessions without the pilot, SR30-500 (needs cell 0) ──
    PC_DIR = os.path.join(AMIR, "Theta", "Phase connectivity", "Rat 136 pilot channels")
    D = pd.read_csv(os.path.join(PC_DIR, "rat136_pilot_channels_sessions.csv"))
    d = D[(D.pair == "IL–PL") & (D.ref == "SR30-500")].reset_index(drop=True)
    MEAS = [("plv", "PLV"), ("pli", "PLI")]
    COL = "#43A047"; ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    rng = np.random.default_rng(0)
    mean_dphi = np.degrees(np.angle(np.exp(1j * np.radians(d.dphi_deg)).mean())); n_ahead = int((np.sin(np.radians(d.dphi_deg)) > 0).sum())

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), facecolor="white", constrained_layout=True)
    rows = []
    for ax, (m, lab) in zip(axs, MEAS):
        real, ch = d[m].values, d[f"chance_{m}"].values
        _, p, _ = fn.sign_flip_p(real - ch, n_perm=10000)
        rows.append(dict(measure=lab, sessions=len(d), mean=real.mean(), sem=real.std(ddof=1) / np.sqrt(len(d)), chance=ch.mean(),
                         mean_minus_chance=(real - ch).mean(), p=p))
        ax.bar(0, real.mean(), width=0.55, color=COL, zorder=1)
        ax.errorbar(0, real.mean(), yerr=real.std(ddof=1) / np.sqrt(len(d)), color="k", capsize=6, lw=1.5, zorder=3)
        ax.scatter(rng.uniform(-0.15, 0.15, len(d)), real, s=22, color="0.45", zorder=2)
        ax.plot([-0.3, 0.3], [ch.mean()] * 2, color="k", lw=2.5, zorder=4, label="chance (mean of shuffles)")

        ax.axhline(0, color="0.5", lw=0.8); ax.set_xlim(-0.6, 0.6); ax.set_xticks([0]); ax.set_xticklabels(["IL–PL"], fontsize=13)
        ax.set_ylabel(f"Theta {lab} (hold)", fontsize=14); ax.set_ylim(min(0, real.min() * 1.3), max(real.max(), ch.mean()) * 1.3)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle(f"IL–PL theta coupling, rat 136 (pilot channels), {len(d)} sessions\nIL ahead in {n_ahead}/{len(d)} sessions, mean Δφ {mean_dphi:+.0f}°",
                 fontsize=13, fontweight="bold")
    fp = os.path.join(PC_DIR, "rat136_pilot_channels_IL_PL_averaged.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


@cell()
def il_ahead_example():
    """IL AHEAD OF PL: example trial (typical, by rule) + Δφ per session (polar) — rat 136, pilot channels AP49 – AP241, SR30-500, correct long (needs cell 0)"""
    global A, B, CHREG, COL, COL_IL, COL_PL, D, E, FR, F_LAG, HOLD, IL_C, NCYC, PC_DIR, PL_C, R, RING, SHORT, STEP_S, T, TONE
    global WIN, X, a0, a1, a_, amp, ax, axp, d, der, dphi_tr, fIL, fPL, fig, fp, fs, gs, ids, idx, k, lag_ms, lb, loc
    global mean_ang, mw, nIL, nPL, near, ok, p, pl, rec, reg, ring_set, s_ex, show, sos, sp, ss, t, th, wav, xIL, xPL, x_
    # ── IL AHEAD OF PL: example trial (typical, by rule) + Δφ per session (polar) — rat 136, pilot channels AP49 – AP241, SR30-500, correct long (needs cell 0) ──
    import scipy.signal as ss
    PC_DIR = os.path.join(AMIR, "Theta", "Phase connectivity", "Rat 136 pilot channels")
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    SHORT = {"Infralimbic area": "IL", "Prelimbic area": "PL", "Cingulate area 1": "Cg1"}
    IL_C, PL_C, RING = "imec0.ap#AP49", "imec0.ap#AP241", (30, 500)
    FR, NCYC = np.array(fn.MAIN_BANDS["theta 4-7"]["freqs"]), fn.MAIN_BANDS["theta 4-7"]["n_cycles"]
    HOLD, STEP_S, TONE, F_LAG, WIN = (0.20, 0.85), 0.010, 1.05, 6.0, 20
    COL_IL, COL_PL, COL = "#ff7f0e", "#1f77b4", "#1E88E5"

    # ── sessions: mean Δφ, session closest to it ──
    D = pd.read_csv(os.path.join(PC_DIR, "rat136_pilot_channels_sessions.csv"))
    d = D[(D.pair == "IL–PL") & (D.ref == "SR30-500")].sort_values("session").reset_index(drop=True); d["pilot"] = d["pilot"].astype(bool)
    th = np.radians(d.dphi_deg.values); mean_ang = np.angle(np.exp(1j * th).mean()); R = np.abs(np.exp(1j * th).mean())
    s_ex = d.session[np.argmin(np.abs(np.angle(np.exp(1j * (th - mean_ang)))))]
    print(f"mean Δφ over {len(d)} sessions {np.degrees(mean_ang):+.0f}° | example session (closest to the mean): {s_ex}", flush=True)

    # ── example session: SR30-500 signals, per-trial Δφ in the hold, trial within ±WIN° of the mean with the strongest theta ──
    T = fn.session_trials(K, s_ex)
    T = T[(T["type"] == "long") & (T["label"] == "correct") & ((T["release_s"] - T["press_s"]) >= HOLD[1])].reset_index(drop=True)
    rec = fn.load_session(PROC, s_ex, XB, refs=("original",))[0]["original"]; fs = rec.get_sampling_frequency()
    ids = [str(c) for c in rec.get_channel_ids()]; loc = rec.get_channel_locations()
    reg = np.array([SHORT.get(r, "other") for r in pd.Series(ids).map(CHREG[CHREG.session == s_ex].set_index("channel_id")["region"]).fillna("")])
    def ring_set(i): dd = np.hypot(loc[:, 0] - loc[i, 0], loc[:, 1] - loc[i, 1]); return np.where((reg == reg[i]) & (dd > RING[0]) & (dd <= RING[1]))[0]
    a0, a1 = fn.MAIN_EPOCHS["Press"]
    X, ok = fn.read_epochs(rec, T["press_s"].values, a0, a1); t = np.arange(X.shape[-1]) / fs + a0
    der = lambda c: (X[:, ids.index(c)] - np.median(X[:, ring_set(ids.index(c))], axis=1)).astype(np.float64)
    xIL, xPL = der(IL_C), der(PL_C); del X
    idx = np.where((t >= HOLD[0]) & (t <= HOLD[1]))[0][::int(round(STEP_S * fs))]
    wav = lambda x: np.stack([ss.fftconvolve(x, fn.morlet(f, NCYC, fs)[None, :], mode="same", axes=-1) for f in FR], axis=1)[..., idx]
    A, B = wav(xIL), wav(xPL); E = A * np.conj(B); E = E / np.abs(E).clip(1e-30)
    dphi_tr = np.angle(E.mean(axis=(1, 2)))
    amp = 0.5 * (np.abs(A).mean(axis=(1, 2)) + np.abs(B).mean(axis=(1, 2)))
    near = np.where(np.abs(np.angle(np.exp(1j * (dphi_tr - mean_ang)))) <= np.radians(WIN))[0]
    k = near[np.argmax(amp[near])] if len(near) else int(np.argmax(amp))
    sos = ss.butter(4, [4, 7], btype="bandpass", fs=fs, output="sos")                  # zero-phase theta filter (display only)
    fIL, fPL = ss.sosfiltfilt(sos, xIL[k]), ss.sosfiltfilt(sos, xPL[k])
    show = (t >= -0.5) & (t <= 1.2); nIL, nPL = fIL / np.abs(fIL[show]).max(), fPL / np.abs(fPL[show]).max()
    print(f"example trial {k}: Δφ {np.degrees(dphi_tr[k]):+.0f}° ({len(near)} trials within ±{WIN}° of the mean)")

    # ── figure ──
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(17, 5.3), facecolor="white"); gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1], wspace=0.2)
    ax = fig.add_subplot(gs[0])
    ax.axvspan(*HOLD, color="0.93", zorder=0)
    ax.plot(t[show], nIL[show] + 1.2, color=COL_IL, lw=2.2, label="IL (AP49)"); ax.plot(t[show], nPL[show] - 1.2, color=COL_PL, lw=2.2, label="PL (AP241)")
    mw = (t >= HOLD[0]) & (t <= HOLD[1])
    for p in t[mw][ss.find_peaks(nIL[mw], distance=int(0.1 * fs))[0]]: ax.axvline(p, ymin=0.5, ymax=0.95, color=COL_IL, ls="--", lw=1.2)
    for p in t[mw][ss.find_peaks(nPL[mw], distance=int(0.1 * fs))[0]]: ax.axvline(p, ymin=0.05, ymax=0.5, color=COL_PL, ls="--", lw=1.2)
    for x_, lb in ((0, "press"), (TONE, "tone")): ax.axvline(x_, color="k", lw=1); ax.text(x_, 2.45, lb, ha="center", fontsize=10)
    ax.set_yticks([]); ax.set_ylim(-2.5, 2.6); ax.set_xlabel("Time from press (s)")
    ax.set_title(f"A. Typical trial ({s_ex[4:6]}/{s_ex[6:8]}, trial {k}, Δφ {np.degrees(dphi_tr[k]):+.0f}°): theta 4–7 Hz, dashed = peaks in the hold (grey)",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    axp = fig.add_subplot(gs[1], projection="polar")
    axp.scatter(th, np.ones(len(th)), s=60, color=COL, zorder=3)
    for a_, pl in zip(th, d.pilot):
        if pl: axp.text(a_, 1.22, "pilot", ha="center", va="center", fontsize=9)
    axp.annotate("", xy=(mean_ang, R), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5))
    axp.set_ylim(0, 1.3); axp.set_yticks([]); axp.set_xticks(np.radians([0, 90, 180, 270]))
    axp.set_xticklabels(["0°", "+90°\nIL ahead", "180°", "−90°\nPL ahead"], fontsize=10)
    lag_ms = np.degrees(mean_ang) / 360 * 1000 / F_LAG
    axp.set_title(f"B. Δφ per session (dot = session)\nmean {np.degrees(mean_ang):+.0f}° (≈ {lag_ms:+.0f} ms at {F_LAG:.0f} Hz), "
                  f"IL ahead in {int((np.sin(th) > 0).sum())}/{len(th)}", fontsize=11, fontweight="bold", pad=18)
    fig.suptitle("IL theta leads PL — rat 136, pilot channels, same-region reference 30–500 µm, correct long trials", fontsize=13, fontweight="bold", y=1.05)
    fp = os.path.join(PC_DIR, "rat136_IL_PL_lead_example_sessions.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def all_pairs():
    """06: ALL-PAIRS IL–PL THETA PLV and PLI, CORRECT LONG, ALL SESSIONS — SR30-500, hold, every 4th channel, pairs ≥ 1000 µm, vs trial-shuffle chance (needs 06 cell 0)"""
    global CH, CHREG, D, FILE, FR, M, MIN_PAIR_UM, MIN_RING, MIN_TR, NCYC, NSH, OUT, PD, RING, RNG, SESS, SHANK_W, SHORT
    global STEP_S, STRIDE, S_, T, UW, W, WIN, X, _, a, a0, a1, ax, axs, b, ch, chv, d, done, fig, fp, fs, g, i, ids, idx, lab
    global loc, m, n_, name, ok, out, p, p_, ptxt, pv_all, r, rats, rec, reg, ri, ring_set, rng, row, rv, s, sp_, ss, t, t_
    global time, top, u, v, x, x_
    # ── 06: ALL-PAIRS IL–PL THETA PLV and PLI, CORRECT LONG, ALL SESSIONS — SR30-500, hold, every 4th channel, pairs ≥ 1000 µm, vs trial-shuffle chance (needs 06 cell 0) ──
    import time, scipy.signal as ss
    OUT = os.path.join(AMIR, "Theta", "Phase connectivity", "All sessions all pairs"); os.makedirs(OUT, exist_ok=True)
    FILE = os.path.join(OUT, "all_pairs_correct_long_sessions.csv")
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    SHORT = {"Infralimbic area": "IL", "Prelimbic area": "PL", "Cingulate area 1": "Cg1"}
    RING, MIN_RING, STRIDE, MIN_PAIR_UM, SHANK_W = (30, 500), 5, 4, 1000, 250
    WIN, STEP_S, NSH, MIN_TR, RNG = (0.20, 0.85), 0.010, 50, 10, np.random.default_rng(0)
    FR, NCYC = np.array(fn.MAIN_BANDS["theta 4-7"]["freqs"]), fn.MAIN_BANDS["theta 4-7"]["n_cycles"]
    globals().pop("C", None)

    def pv_all(a, b, M):
        """mean PLV and PLI over all IL × PL pairs in mask M; a: (trials, nIL, F, T), b: (trials, nPL, F, T), unit phases"""
        plv = pli = 0.0; cnt = 0
        for i in range(a.shape[1]):
            m = M[i]
            if not m.any(): continue
            E = a[:, i:i + 1] * np.conj(b[:, m])
            plv += np.abs(E.mean(0)).mean(axis=(1, 2)).sum(); pli += np.abs(np.sign(E.imag).mean(0)).mean(axis=(1, 2)).sum(); cnt += int(m.sum())
        return plv / cnt, pli / cnt

    done = pd.read_csv(FILE) if os.path.exists(FILE) else pd.DataFrame()
    SESS = sorted(final["session"]); t_ = time.time()
    for n_, s in enumerate(SESS, 1):
        if len(done) and s in set(done.session): continue
        T = fn.session_trials(K, s); T = T[(T["type"] == "long") & (T["label"] == "correct") & ((T["release_s"] - T["press_s"]) >= WIN[1])].reset_index(drop=True)
        if len(T) < MIN_TR: print(f"[{n_}/{len(SESS)}] {s}: {len(T)} trials, skipped"); continue
        rec = fn.load_session(PROC, s, XB, refs=("original",))[0]["original"]; fs = rec.get_sampling_frequency()
        ids = [str(c) for c in rec.get_channel_ids()]; loc = rec.get_channel_locations()
        reg = np.array([SHORT.get(r, "other") for r in pd.Series(ids).map(CHREG[CHREG.session == s].set_index("channel_id")["region"]).fillna("")])
        def ring_set(i): dd = np.hypot(loc[:, 0] - loc[i, 0], loc[:, 1] - loc[i, 1]); return np.where((reg == reg[i]) & (dd > RING[0]) & (dd <= RING[1]))[0]
        CH = {}
        for g in ("IL", "PL"):
            u = [i for i in np.where(reg == g)[0] if len(ring_set(i)) >= MIN_RING]
            CH[g] = sorted(u, key=lambda i: (loc[i, 0] // SHANK_W, loc[i, 1], loc[i, 0]))[::STRIDE]
        if not CH["IL"] or not CH["PL"]: print(f"[{n_}/{len(SESS)}] {s}: no usable IL or PL channel, skipped"); continue
        PD = np.hypot(loc[CH["IL"], 0][:, None] - loc[CH["PL"], 0][None, :], loc[CH["IL"], 1][:, None] - loc[CH["PL"], 1][None, :])
        M = PD >= MIN_PAIR_UM
        if not M.any(): print(f"[{n_}/{len(SESS)}] {s}: no pair ≥ {MIN_PAIR_UM} µm, skipped"); continue
        print(f"[{n_}/{len(SESS)}] {s}: {len(T)} trials, {len(CH['IL'])} IL × {len(CH['PL'])} PL, {int(M.sum())} pairs — computing...", flush=True)
        a0, a1 = fn.MAIN_EPOCHS["Press"]
        X, ok = fn.read_epochs(rec, T["press_s"].values, a0, a1); del rec
        t = np.arange(X.shape[-1]) / fs + a0; idx = np.where((t >= WIN[0]) & (t <= WIN[1]))[0][::int(round(STEP_S * fs))]
        UW = {}
        for g in ("IL", "PL"):
            u = []
            for i in CH[g]:
                x = (X[:, i] - np.median(X[:, ring_set(i)], axis=1)).astype(np.float64)                     # SR30-500 for this channel
                W = np.stack([ss.fftconvolve(x, fn.morlet(f_, NCYC, fs)[None, :], mode="same", axes=-1) for f_ in FR], axis=1)[..., idx]
                u.append((W / np.abs(W).clip(1e-30)).astype(np.complex64))
            UW[g] = np.stack(u, axis=1)
        del X
        a, b = UW["IL"], UW["PL"]; del UW
        rv, ri = pv_all(a, b, M)
        ch = np.array([pv_all(a, b[RNG.permutation(len(a))], M) for _ in range(NSH)])
        row = pd.DataFrame([dict(session=s, rat=s.split("_")[1][-3:], trials=len(a), n_IL=len(CH["IL"]), n_PL=len(CH["PL"]), pairs=int(M.sum()),
                                 mean_pair_dist_um=PD[M].mean(), plv=rv, chance_plv=ch[:, 0].mean(), p_plv=(1 + (ch[:, 0] >= rv).sum()) / (1 + NSH),
                                 pli=ri, chance_pli=ch[:, 1].mean(), p_pli=(1 + (ch[:, 1] >= ri).sum()) / (1 + NSH))])
        done = pd.concat([done, row], ignore_index=True); done.to_csv(FILE, index=False)                   # saved after every session
        print(f"      PLV {rv:.3f} (chance {ch[:, 0].mean():.3f}) | PLI {ri:.3f} (chance {ch[:, 1].mean():.3f}) | {(time.time() - t_) / 60:.1f} min", flush=True)
        del a, b

    # ── across sessions: all, and per rat; sign-flip on (real − chance), raw p ──
    D = pd.read_csv(FILE, dtype={"rat": str}); out = []
    for name, d in [("all", D)] + [(f"rat {r}", D[D.rat == r]) for r in sorted(D.rat.unique())]:
        for m in ("plv", "pli"):
            v = (d[m] - d[f"chance_{m}"]).values; _, p, _ = fn.sign_flip_p(v, n_perm=10000)
            out.append(dict(set=name, measure=m.upper(), sessions=len(v), mean=d[m].mean(), chance=d[f"chance_{m}"].mean(), mean_minus_chance=v.mean(),
                            sessions_above=f"{int((v > 0).sum())}/{len(v)}", p=p))
    S_ = pd.DataFrame(out); pd.set_option("display.width", 250)
    print(f"\nIL–PL theta, ALL PAIRS (every {STRIDE}th channel, pairs ≥ {MIN_PAIR_UM} µm), SR30-500, correct long, hold {WIN[0]}–{WIN[1]} s; "
          f"sign-flip across sessions on (real − chance), raw p")
    print(S_.round(4).to_string(index=False))
    _table(S_, os.path.join(OUT, "all_pairs_correct_long_summary.csv"), index=False)

    # ── figure (averaged style): per rat, bar = mean, SEM, grey dots = sessions, black line = chance, p (vs chance) above ──
    ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    rats = sorted(D.rat.unique()); rng = np.random.default_rng(0)
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), facecolor="white", constrained_layout=True)
    for ax, (m, lab) in zip(axs, (("plv", "PLV"), ("pli", "PLI"))):
        top = D[m].max()
        for x_, r in enumerate(rats):
            d = D[D.rat == r]; v, chv = d[m].values, d[f"chance_{m}"].values
            ax.bar(x_, v.mean(), width=0.55, color="#1E88E5", zorder=1)
            ax.errorbar(x_, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0, color="k", capsize=6, lw=1.5, zorder=3)
            ax.scatter(x_ + rng.uniform(-0.15, 0.15, len(v)), v, s=18, color="0.45", zorder=2)
            ax.plot([x_ - 0.3, x_ + 0.3], [chv.mean()] * 2, color="k", lw=2.5, zorder=4, label="chance (mean of shuffles)" if x_ == 0 else None)
            p_ = S_[(S_.set == f"rat {r}") & (S_.measure == m.upper())].p.values[0]
            ax.text(x_, top * 1.05, f"{ptxt(p_)}\n{len(v)} sessions", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(rats))); ax.set_xticklabels([f"rat {r}" for r in rats]); ax.set_ylim(0, top * 1.3)
        ax.set_ylabel(f"Theta {lab} (hold), mean over pairs", fontsize=13)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    axs[0].legend(frameon=False, fontsize=10, loc="lower left")
    fig.suptitle(f"IL–PL theta coupling, all channel pairs, correct long — {len(D)} sessions, 3 rats", fontsize=13, fontweight="bold")
    fp = os.path.join(OUT, "all_pairs_correct_long_per_rat.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))
