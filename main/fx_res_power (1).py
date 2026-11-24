"""
fx_res_power.py — band power during the hold: theta, delta (total vs induced = ERP removed), high gamma and multi-unit firing.
Cells of 07 section 1, the theta-along-the-session cells of section 4, and the multi-unit cache step run from 06 (mua_curves).
Every cell uses only the shared names of fx_results.setup (paths, load_bp, done_sessions, RAT, colours, ...).
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fx_results import cell


@cell()
def mua_curves():
    """HIGH GAMMA AND MULTI-UNIT FIRING OVER TIME IN ONE PLOT (correct long, all sessions, CMR) — desktop (needs cell 0)"""
    global BIN_S, C, CHREG, CUR_DIR, D, DEAD_MS, DESK, EDG, GCOL, HGc, K_THR, MCOL, MIN_RATE, MPFC, MUc, N, N_TR, PRE, Path
    global RAW, SHOW, T, TT, X, XB_, _, a0, a1, a_, al, allid, ap_bin, ax, axs, c, cl, col, cols, crossings, dead, e, fc, fig
    global fp, fs, glob, h, hold, i, k, keep_ids, l, lab, lr, m, n_ok, n_tot, noise, ok, on, pcf, r, rate, raw, rec, reg, rng
    global s, se, sel, sh, si, sp, sp_, starts, t, t_start, thr, time, tr, w, warnings, x, xc, xl, z, z_
    if not RUN:
        print("RUN = False in the settings: nothing is computed here (the cell only computes missing sessions when RUN = True)")
        return
    # ── HIGH GAMMA AND MULTI-UNIT FIRING OVER TIME IN ONE PLOT (correct long, all sessions, CMR) — desktop (needs cell 0) ──
    #    High gamma: as before — saved 06 curves = log10(power(t) / own pre-press), each trial only while on the lever; median over ALL channels.
    #    Spikes (MUA): raw ap.bin, same processing as the MUA cell (300–6000 Hz, phase_shift, CMR, crossings of −4.5 × noise, 1-ms dead time),
    #    ≤ N_TR correct long trials per session (same random trials as the MUA cell, seed 0). Spike counts in 50-ms bins, press-aligned (only while on
    #    the lever) and release-aligned (only from the press on). Per channel: mean rate over trials / its own mean rate before the press (−0.5…−0.2 s);
    #    channels of IL, PL, Cg1 with ≥ 1 spike/s before the press; session curve = median over channels of log10(ratio).
    #    Both lines = mean ± SEM over sessions, in % vs before the press. Slow (raw 30 kHz) -> one cached file per session in tables\mua_curves\.
    import glob, time, warnings
    from pathlib import Path
    import spikeinterface.full as si
    DESK = next((p for p in (os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"), os.path.join(os.path.expanduser("~"), "Desktop"))
                 if os.path.isdir(p)), os.path.expanduser("~"))
    RAW = os.path.join(DATA_ROOT, "0_raw")
    K_THR, N_TR, DEAD_MS, MIN_RATE, BIN_S = 4.5, 60, 1.0, 1.0, 0.05
    SHOW = {"Press": (-0.7, 1.2), "Release": (-0.7, 0.7)}
    CUR_DIR = os.path.join(TABLES, "mua_curves"); os.makedirs(CUR_DIR, exist_ok=True)
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    MPFC = {"Infralimbic area", "Prelimbic area", "Cingulate area 1"}
    XB_ = pd.read_csv(os.path.join(TABLES, "extra_bad_channels.csv"))
    GCOL, MCOL = "#6a1b9a", "#455A64"
    pcf = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    EDG = {al: np.arange(lo, hi + 1e-9, BIN_S) for al, (lo, hi) in SHOW.items()}

    def crossings(x, thr, dead):
        """onsets (samples) of downward crossings of -thr per channel; x (samples, channels)"""
        below = x < -thr[None, :]; on = below[1:] & ~below[:-1]; out = []
        for j in range(x.shape[1]):
            t = np.flatnonzero(on[:, j]) + 1
            if len(t) > 1: t = t[np.r_[True, np.diff(t) > dead]]
            out.append(t)
        return out

    # ── 1) MUA curves per session (cached) ──
    for i, s in enumerate(done_sessions("CMR")):
        fc = os.path.join(CUR_DIR, s + ".npz")
        if os.path.exists(fc): continue
        t_start = time.time()
        T = load_bp(s, "CMR")["trials"]
        cl = T[(T["label"] == "correct") & (T["type"] == "long")]
        if len(cl) < 10: continue
        cl = cl.sample(min(N_TR, len(cl)), random_state=0)
        keep_ids = [str(c) for c in fn.load_session(PROC, s, XB_, refs=("original",))[0]["original"].get_channel_ids()]
        reg = CHREG[CHREG["session"] == s].set_index("channel_id")["region"]
        keep_ids = [c for c in keep_ids if reg.get(c) in MPFC]                                      # mPFC channels only
        if not keep_ids: continue
        ap_bin = glob.glob(os.path.join(RAW, s, "**", "*.ap.bin"), recursive=True)[0]
        raw = si.read_spikeglx(Path(ap_bin).parent, stream_name="imec0.ap")
        rec = si.common_reference(si.phase_shift(si.bandpass_filter(raw, freq_min=300, freq_max=6000, dtype="float32")),
                                  reference="global", operator="median")                          # CMR over ALL channels (as the MUA cell)
        allid = [str(c) for c in rec.get_channel_ids()]; cols = [allid.index(c) for c in keep_ids if c in allid]
        fs = rec.get_sampling_frequency(); n_tot = rec.get_num_samples(); sel = [rec.get_channel_ids()[k] for k in cols]
        rng = np.random.default_rng(0); starts = rng.integers(int(10 * fs), n_tot - int(2 * fs), 20)
        noise = np.median([np.median(np.abs(rec.get_traces(start_frame=a, end_frame=a + int(fs), channel_ids=sel, return_in_uV=True)), axis=0) / 0.6745
                           for a in starts], axis=0)
        thr, dead = K_THR * noise, int(DEAD_MS * fs / 1000)
        C = {al: np.zeros((len(sel), len(EDG[al]) - 1)) for al in SHOW}; N = {al: np.zeros(len(EDG[al]) - 1) for al in SHOW}
        PRE = np.zeros(len(sel)); n_ok = 0
        for _, tr in cl.iterrows():
            hold = tr["release_s"] - tr["press_s"]
            a0 = int((tr["press_s"] - 0.7) * fs); a1 = int((tr["release_s"] + 0.7) * fs)
            if a0 < 0 or a1 > n_tot: continue
            x = rec.get_traces(start_frame=a0, end_frame=a1, channel_ids=sel, return_in_uV=True)
            sp = [t / fs - 0.7 for t in crossings(x, thr, dead)]                                  # spike times from the press (s)
            for al in SHOW:
                sh = 0.0 if al == "Press" else hold; e = EDG[al]
                on = (e[1:] <= hold) if al == "Press" else (e[:-1] >= -hold)                     # press: until release | release: from press
                for k, t in enumerate(sp):
                    C[al][k] += np.histogram(t - sh, e)[0] * on
                N[al] += on
            PRE += np.array([((t >= -0.5) & (t < -0.2)).sum() for t in sp]); n_ok += 1
        np.savez(fc, channel_id=np.array(keep_ids)[:len(sel)], pre_rate=PRE / (0.3 * max(n_ok, 1)), n_trials=n_ok,
                 **{f"{al}_counts": C[al] for al in SHOW}, **{f"{al}_ntr": N[al] for al in SHOW})
        print(f"  {i + 1} {s}: {n_ok} trials, {len(sel)} mPFC channels, {time.time() - t_start:.0f} s", flush=True)

    # ── 2) session curves: high gamma (saved curves) and MUA (cache) ──
    HGc = {al: [] for al in SHOW}; MUc = {al: [] for al in SHOW}; TT = {}
    for s in done_sessions("CMR"):
        r = load_bp(s, "CMR")
        for al in SHOW:
            TT.setdefault(al, r["times"][al]); c = fn.session_curve(r, al, "highG 61-100", "correct long")
            if c is not None: HGc[al].append(np.interp(EDG[al][:-1] + BIN_S / 2, TT[al], c))       # on the same 50-ms bins
        fc = os.path.join(CUR_DIR, s + ".npz")
        if not os.path.exists(fc): continue
        z = np.load(fc); ok = z["pre_rate"] >= MIN_RATE
        if ok.sum() == 0: continue
        for al in SHOW:
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = z[f"{al}_counts"][ok] / (z[f"{al}_ntr"][None, :] * BIN_S)                   # spikes/s per channel and bin
                lr = np.log10((rate + 0.1) / (z["pre_rate"][ok][:, None] + 0.1))
            lr[:, z[f"{al}_ntr"] < 3] = np.nan                                                    # bins with < 3 trials on the lever
            MUc[al].append(np.nanmedian(lr, axis=0))

    # ── 3) figure ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), facecolor="white", sharey=True, gridspec_kw={"width_ratios": [1.9, 1.4], "wspace": 0.08})
    for ax, al, xl in zip(axs, SHOW, ["Time from press (s)", "Time from release (s)"]):
        xc = EDG[al][:-1] + BIN_S / 2
        for D, col, lab in ((HGc, GCOL, f"high gamma ({len(HGc[al])} sessions)"), (MUc, MCOL, f"multi-unit firing ({len(MUc[al])} sessions)")):
            X = np.array(D[al])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(X, 0); se = np.nanstd(X, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(X), 0))
            ax.fill_between(xc, pcf(m - se), pcf(m + se), color=col, alpha=0.2, lw=0); ax.plot(xc, pcf(m), color=col, lw=2, label=lab)
        ax.axhline(0, color="0.5", lw=0.8, ls="--"); ax.axvline(0, color="k", lw=0.9)
        for w in {"Press": ["pre", "hold"], "Release": ["post"]}[al]:
            a_, z_ = fn.MAIN_WINDOWS[w][1]
            ax.axvspan(a_, z_, color="#F6DADD", zorder=0, lw=0)
            ax.text((a_ + z_) / 2, 1.0, w, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c62828")
        ax.set_xlim(*SHOW[al]); ax.set_xlabel(xl)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    axs[0].set_ylabel("Change vs before the press (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fp = os.path.join(DESK, "high_gamma_and_spikes_over_time.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def m1_all_sessions():
    """1b. M1, ALL FINISHED SESSIONS (sign-flip across sessions, raw p + FDR)"""
    global M1, M1_stat, MIN_CL, REF, b, d, idx, ii, m, mean, meas, multipletests, n, n_cl, ok, p, rat, rows, s, stat, v
    # ── 1b. M1, ALL FINISHED SESSIONS (sign-flip across sessions, raw p + FDR) ──
    from statsmodels.stats.multitest import multipletests
    REF = "CMR"
    MIN_CL = globals().get("MIN_CL", 10)                  # from 1a; 10 if 1a was not run

    rows = []
    for s in done_sessions(REF):
        m, n_cl = fn.m1_session(load_bp(s, REF), BANDS, MIN_CL)
        if m is None:
            continue
        for b in BANDS:
            rows.append({"session": s, "rat": RAT[s], "band": b, "n_cl": n_cl, **m[b]})
    M1 = pd.DataFrame(rows)
    M1.to_csv(os.path.join(TABLES, f"M1_session_values_{REF}.csv"), index=False)
    print(f"sessions with >= {MIN_CL} correct long trials: {M1['session'].nunique()} of {len(done_sessions(REF))} finished")

    stat = []
    for meas in ["hold-pre", "hold-post"]:
        for b in BANDS:
            for rat in ["all"] + sorted(M1["rat"].unique()):
                v = M1[(M1["band"] == b) & ((M1["rat"] == rat) if rat != "all" else True)][meas]
                mean, p, n = fn.sign_flip_p(v.values)
                stat.append({"measure": meas, "band": b, "rat": rat, "n_sessions": n, "sessions_up": int((v > 0).sum()),
                             "mean_%": round(100 * (10 ** mean - 1), 1) if n else np.nan, "p": round(p, 4) if n >= 2 else np.nan})
    M1_stat = pd.DataFrame(stat)

    # FDR (Benjamini–Hochberg) per rat: 6 bands x 2 comparisons = 12 tests
    M1_stat["q"] = np.nan
    for rat, idx in M1_stat.groupby("rat").groups.items():
        ok = M1_stat.loc[idx, "p"].notna()
        ii = M1_stat.loc[idx][ok].index
        M1_stat.loc[ii, "q"] = multipletests(M1_stat.loc[ii, "p"], method="fdr_bh")[1].round(4)
    _table(M1_stat, os.path.join(TABLES, f"M1_stats_{REF}.csv"), index=False)

    for meas, d in M1_stat.groupby("measure", sort=False):
        print(f"\n===== M1 {meas} ({REF}) =====")
        print(d.pivot(index="band", columns="rat", values=["mean_%", "sessions_up", "n_sessions", "p", "q"]).loc[BANDS].to_string())


@cell()
def pilot_time_frequency():
    """FIG 2C. Pilot time–frequency map (1–12 Hz): hold vs the mean of pre and post (combined, for illustration),"""
    global BAND, DEC, GAP, KEEP, LP, LR, L_SHOW, ML, MR, PP, PR, PROC, REF, R_SHOW, Rectangle, SES, T, WIN_COL, XB, XP, XR, a
    global a0, a1, ax, b, b0, b1, b_both, b_post, b_pre, band_of, cl, f, fb, fig, freqs, fs, full, hold, im, kw, mL, mR_, meta
    global name, objs, off, okP, okR, onP, onR, pdf, press, r, rec, recN, release, report_dir, sel_ids, sep, sp, spec, ss, tP
    global tR, theta_dir, tl, tr, w, w_hold, w_post, w_pre, win, wv, x0, z
    # ── FIG 2C. Pilot time–frequency map (1–12 Hz): hold vs the mean of pre and post (combined, for illustration),
    #    20 selected channels, own pre/post per trial, trials only while on the lever, mean over trials, median over channels ──
    import scipy.signal as ss
    from matplotlib.patches import Rectangle
    PROC = os.path.join(DATA_ROOT, "2_processed")
    XB   = pd.read_csv(os.path.join(TABLES, "extra_bad_channels.csv"))
    SES, REF, DEC, BAND = TEST, "CMR", 10, "theta 4-7"
    KEEP = ["delta 1-3", "theta 4-7", "alpha 8-12"]
    WIN_COL = "#C0392B"

    # 1) same session, trials and 20 channels as Fig. 2B
    r = load_bp(SES, REF); T, meta = r["trials"], r["meta"]
    cl = ((T["label"] == "correct") & (T["type"] == "long")).values
    press, release, hold = T.loc[cl, "press_s"].values, T.loc[cl, "release_s"].values, T.loc[cl, "hold_s"].values
    pdf = pd.DataFrame({"channel_id": meta["channel_ids"], "x": meta["x"], "y": meta["y"]}); pdf["shank"] = (pdf["x"] // 250).astype(int)
    sel_ids = [str(c) for c in fn.select_channels(pdf, n_total=20)["channel_id"]]

    rec = fn.load_session(PROC, SES, XB, refs=(REF,))[0][REF]
    full = [str(c) for c in rec.get_channel_ids()]
    objs = [rec.get_channel_ids()[full.index(c)] for c in sel_ids]
    recN = rec.select_channels(objs) if hasattr(rec, "select_channels") else rec.channel_slice(channel_ids=objs)
    fs = recN.get_sampling_frequency()
    (a0, a1), (b0, b1) = fn.MAIN_EPOCHS["Press"], fn.MAIN_EPOCHS["Release"]
    XP, okP = fn.read_epochs(recN, press, a0, a1)
    XR, okR = fn.read_epochs(recN, release, b0, b1)
    assert okP.all() and okR.all(), "trial lost at a recording edge: should not happen (saved trials are in both)"
    tP = (np.arange(XP.shape[-1]) / fs + a0)[::DEC]; tR = (np.arange(XR.shape[-1]) / fs + b0)[::DEC]
    onP = tP[None, :] <= hold[:, None]; onR = tR[None, :] >= -hold[:, None]        # only while on the lever
    win = lambda tt, name: (tt >= fn.MAIN_WINDOWS[name][1][0] - 1e-9) & (tt <= fn.MAIN_WINDOWS[name][1][1] + 1e-9)

    # 2) Morlet power, same frequencies and cycles as 06
    freqs, band_of, PP, PR = [], [], [], []
    for b in KEEP:
        spec = fn.MAIN_BANDS[b]
        for f in spec["freqs"]:
            w = fn.morlet(f, spec["n_cycles"], fs)[None, None, :]
            PP.append((np.abs(ss.fftconvolve(XP.astype(np.float64), w, mode="same", axes=-1)) ** 2)[..., ::DEC])
            PR.append((np.abs(ss.fftconvolve(XR.astype(np.float64), w, mode="same", axes=-1)) ** 2)[..., ::DEC])
            freqs.append(f); band_of.append(b)
    PP, PR = np.stack(PP, axis=2), np.stack(PR, axis=2)                             # (trials, channels, freqs, times)
    freqs, band_of = np.array(freqs), np.array(band_of)
    LP, LR = np.log10(PP), np.log10(PR)

    # 3) map: reference = mean of the trial's own pre and post levels (per frequency); mean over trials, median over channels
    b_pre  = LP[..., win(tP, "pre")].mean(axis=-1, keepdims=True)
    b_post = LR[..., win(tR, "post")].mean(axis=-1, keepdims=True)
    b_both = (b_pre + b_post) / 2
    with np.errstate(invalid="ignore"):
        ML = 100 * (10 ** np.nanmedian(np.nanmean(np.where(onP[:, None, None, :], LP - b_both, np.nan), axis=0), axis=0) - 1)
        MR = 100 * (10 ** np.nanmedian(np.nanmean(np.where(onR[:, None, None, :], LR - b_both, np.nan), axis=0), axis=0) - 1)

    # 4) the two separate differences, as in the analysis (window values: log10 of mean band power; per trial; mean over trials, median over channels)
    fb = band_of == BAND
    wv = lambda P, tt, name: np.log10(P[:, :, fb][..., win(tt, name)].mean(axis=(2, 3)))    # (trials, channels)
    w_pre, w_hold, w_post = wv(PP, tP, "pre"), wv(PP, tP, "hold"), wv(PR, tR, "post")
    sep = lambda d: 100 * (10 ** np.median(d.mean(axis=0)) - 1)

    # 5) figure: press | release on one axis
    L_SHOW, R_SHOW, GAP = (-0.7, 1.0), (-0.3, 0.6), 0.15
    mL = (tP >= L_SHOW[0]) & (tP <= L_SHOW[1]); mR_ = (tR >= R_SHOW[0]) & (tR <= R_SHOW[1])
    off = L_SHOW[1] + GAP - R_SHOW[0]
    kw = dict(cmap="viridis", vmin=-50, vmax=50, shading="nearest")

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="white")
    ax.pcolormesh(tP[mL], freqs, ML[:, mL], **kw)
    im = ax.pcolormesh(tR[mR_] + off, freqs, MR[:, mR_], **kw)
    ax.axvspan(L_SHOW[1], L_SHOW[1] + GAP, color="white", zorder=3)
    for x0, name in [(0, "pre"), (0, "hold"), (off, "post")]:
        a, z = fn.MAIN_WINDOWS[name][1]
        ax.add_patch(Rectangle((a + x0, 3.5), z - a, 4.0, fill=False, ec=WIN_COL, lw=2, zorder=5))   # theta rows
        ax.text((a + z) / 2 + x0, 12.6, name, ha="center", va="bottom", fontsize=10, fontweight="bold", color=WIN_COL)
    ax.axvline(0, color="k", ls="--", lw=1); ax.axvline(off, color="k", ls="--", lw=1)
    tl, tr = [-0.5, 0, 0.5, 1.0], [0, 0.5]
    ax.set_xticks(tl + [v + off for v in tr]); ax.set_xticklabels([f"{v:g}" for v in tl + tr])
    ax.text(0, -0.13, "press", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(off, -0.13, "release", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (s)", labelpad=16); ax.set_xlim(L_SHOW[0], R_SHOW[1] + off); ax.set_ylim(0.5, 12.5)
    ax.set_ylabel("Frequency (Hz)"); ax.set_yticks(range(1, 13))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Power change from the mean\nof pre and post (%)")
    ax.text(-0.09, 1.06, "C", transform=ax.transAxes, fontsize=14, fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    theta_dir  = os.path.join(AMIR, "Theta", "theta computation example"); os.makedirs(theta_dir, exist_ok=True)
    report_dir = os.path.join(AMIR, "report"); os.makedirs(report_dir, exist_ok=True)
    _savefig(fig, os.path.join(theta_dir, "pilot_comparison.png"))
    _savefig(fig, os.path.join(report_dir, "Fig 2C pilot time-frequency.png"))
    plt.show()
    print(f"pilot {SES}, {REF}: {cl.sum()} correct long trials, {len(sel_ids)} channels, {freqs.min():g}–{freqs.max():g} Hz")
    print(f"theta, computed separately (as in the analysis): hold vs pre {sep(w_hold - w_pre):+.1f} %, hold vs post {sep(w_hold - w_post):+.1f} %")


@cell()
def pilot_significance():
    """1d. PILOT: SIGNIFICANCE PER CHANNEL (sign-flip across trials) + FDR count + probe plots"""
    global NAMES, NPERM_1D, R, REF_1D, Rectangle, SES_1D, T, _, ax, axes, b, blue, cb, ci, cl, counts, counts_fdr, fig, i
    global ids_all, im, is_sel, is_sig, j, jt, k, kind, lp, m, meta, multipletests, n_tr, p, pct, pdf, pv, q, r, rest, s
    global same_dir, sel, sel_ids, shanks, spec, spec_fdr, sup_dir, theta_dir, ttl, xs, ys
    # ── 1d. PILOT: SIGNIFICANCE PER CHANNEL (sign-flip across trials) + FDR count + probe plots ──
    from matplotlib.patches import Rectangle
    from statsmodels.stats.multitest import multipletests
    SES_1D, REF_1D, NPERM_1D = TEST, "CMR", 5000
    NAMES = {"delta 1-3": "Delta", "theta 4-7": "Theta", "alpha 8-12": "Alpha", "beta 13-29": "Beta",
             "lowG 30-60": "Low γ", "highG 61-100": "High γ"}
    theta_dir = os.path.join(AMIR, "Theta", "20 selected channels_pilot"); os.makedirs(theta_dir, exist_ok=True)

    r = load_bp(SES_1D, REF_1D)
    T, lp, meta = r["trials"], r["logpow"], r["meta"]
    pdf = pd.DataFrame({"channel_id": meta["channel_ids"], "x": meta["x"], "y": meta["y"]})
    pdf["shank"] = (pdf["x"] // 250).astype(int)
    sel = fn.select_channels(pdf, n_total=20).sort_values(["shank", "y"], ascending=[True, False])
    ci = [list(map(str, meta["channel_ids"])).index(str(c)) for c in sel["channel_id"]]
    cl = ((T["label"] == "correct") & (T["type"] == "long")).values
    n_tr = int(cl.sum())

    # 1) per channel and band: sign-flip across trials
    pct = {k: np.full((len(ci), len(BANDS)), np.nan) for k in ("pre", "post")}
    pv  = {k: np.full((len(ci), len(BANDS)), np.nan) for k in ("pre", "post")}
    for j, b in enumerate(BANDS):
        for k in ("pre", "post"):
            R = lp[b]["hold"][cl][:, ci] - lp[b][k][cl][:, ci]                  # (trials, channels) log10 ratios
            for i in range(len(ci)):
                m, p, _ = fn.sign_flip_p(R[:, i], n_perm=NPERM_1D)
                pct[k][i, j], pv[k][i, j] = 100 * (10 ** m - 1), p
    same_dir = np.sign(pct["pre"]) == np.sign(pct["post"])
    spec = (pv["pre"] < 0.05) & (pv["post"] < 0.05) & same_dir
    counts = ", ".join(f"{NAMES[b]} {int(spec[:, j].sum())}/{len(ci)}" for j, b in enumerate(BANDS))

    # FDR per band over 20 channels x 2 comparisons
    spec_fdr = np.zeros_like(spec)
    for j in range(len(BANDS)):
        q = multipletests(np.concatenate([pv["pre"][:, j], pv["post"][:, j]]), method="fdr_bh")[1]
        spec_fdr[:, j] = (q[:len(ci)] < 0.05) & (q[len(ci):] < 0.05) & same_dir[:, j]
    counts_fdr = ", ".join(f"{NAMES[b]} {int(spec_fdr[:, j].sum())}/{len(ci)}" for j, b in enumerate(BANDS))

    # 2) significance figure (raw p, as on the pilot slide)
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 7), sharey=True, facecolor="white")
    jt = BANDS.index("theta 4-7")
    for ax, k, ttl in zip(axes, ("pre", "post"), ("Hold vs before press", "Hold vs after release")):
        im = ax.imshow(pct[k], cmap="viridis", vmin=-50, vmax=50, aspect="auto")
        ax.add_patch(Rectangle((jt - 0.5, -0.5), 1, len(ci), fill=False, ec="white", lw=2.5, zorder=5))   # theta frame
        for i in range(len(ci)):
            for j in range(len(BANDS)):
                if spec[i, j]:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=13, fontweight="bold")
                elif pv[k][i, j] < 0.05:
                    ax.text(j, i, "·", ha="center", va="center", fontsize=13)
        ax.set_xticks(range(len(BANDS))); ax.set_xticklabels([NAMES[b] for b in BANDS], fontsize=8)
        ax.set_title(ttl, fontsize=10, pad=26)
    axes[0].set_yticks([]); axes[0].set_ylabel("Channels")
    fig.suptitle(f"Correct long trials, hold window 0.20–0.85 s (n = {n_tr} trials, {REF_1D})", fontsize=10.5, y=0.99)
    fig.text(0.5, 0.01, "* p < 0.05 vs before press and vs after release, same direction   ·  p < 0.05 in this panel only\n"
             f"Sign-flip test across trials, uncorrected. Channels with * (of {len(ci)}): " + counts,
             ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(bottom=0.13, top=0.86, left=0.12, right=0.88, wspace=0.06)
    cb = fig.colorbar(im, cax=fig.add_axes([0.91, 0.35, 0.02, 0.3])); cb.set_label("Power change (%)")
    _savefig(fig, os.path.join(theta_dir, f"pilot_significance_{REF_1D}.png"))
    plt.show()
    print("raw:", counts)
    print("FDR:", counts_fdr)

    # 3) probe plots: (a) selected channels (blue) | (b) theta-significant (red) + rest of the selected (blue)
    xs, ys = np.asarray(meta["x"], float), np.asarray(meta["y"], float)
    ids_all = np.array([str(c) for c in meta["channel_ids"]])
    sel_ids = np.array([str(c) for c in sel["channel_id"]])                 # same order as the rows of spec
    is_sel = np.isin(ids_all, sel_ids)
    is_sig = np.isin(ids_all, sel_ids[spec[:, jt]])
    shanks = np.unique(np.round(xs / 250).astype(int))
    for kind in ("selected", "significant"):
        fig, ax = plt.subplots(figsize=(4, 7), facecolor="white")
        rest = ~is_sel
        ax.scatter(xs[rest], ys[rest], s=6, marker="s", color="0.8", lw=0, label=f"recorded, not selected ({rest.sum()})")
        blue = is_sel if kind == "selected" else (is_sel & ~is_sig)
        ax.scatter(xs[blue], ys[blue], s=40, marker="s", color="#000000FF", edgecolor="k", lw=0.4, zorder=3,
                   label=f"selected ({blue.sum()})" if kind == "selected" else f"selected, n.s. ({blue.sum()})")
        if kind == "significant":
            ax.scatter(xs[is_sig], ys[is_sig], s=40, marker="s", color="red", edgecolor="k", lw=0.4, zorder=4,
                       label=f"theta significant ({is_sig.sum()})")
        ax.set_xticks(shanks * 250); ax.set_xticklabels([f"shank {s + 1}" for s in shanks], fontsize=8)
        ax.set_xlim(xs.min() - 120, xs.max() + 120); ax.set_ylabel("Distance from probe tip (µm)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, frameon=False, fontsize=8)
        ttl = "Selected channels" if kind == "selected" else f"Theta during the hold: {is_sig.sum()}/{len(sel_ids)} channels"
        ax.set_title(f"{ttl}\n{SES_1D}", fontsize=10, fontweight="bold")
        _savefig(fig, os.path.join(theta_dir, f"pilot_{kind}_channels_{REF_1D}.png"))
        if kind == "selected":
            sup_dir = os.path.join(AMIR, "supplementary"); os.makedirs(sup_dir, exist_ok=True)
            _savefig(fig, os.path.join(sup_dir, "Fig S6 pilot selected channels.png"))
        plt.show()


@cell()
def pilot_theta_over_time():
    """FIG 2B. Pilot: raw theta power over time (mean over trials and the 20 channels), correct long, CMR"""
    global BAND, DEC, PROC, REF_R, SES_R, T, THETA_COL, WIN_COL, XB, Z, _, a, avg, ax, axes, cl, fig, fs, full, m_, meta, name
    global objs, pdf, r, rawP, rawR, rec, recN, recs, report_dir, rng_, sel_ids, sp, ss, tP, tR, theta_dir, theta_raw, tt, win
    global xl, z
    # ── FIG 2B. Pilot: raw theta power over time (mean over trials and the 20 channels), correct long, CMR ──
    import scipy.signal as ss

    PROC = os.path.join(DATA_ROOT, "2_processed")
    XB   = pd.read_csv(os.path.join(TABLES, "extra_bad_channels.csv"))
    SES_R, REF_R, BAND, DEC = TEST, "CMR", "theta 4-7", 10
    THETA_COL, WIN_COL = "#2e7d32", "#C0392B"

    r  = load_bp(SES_R, REF_R)
    T, meta = r["trials"], r["meta"]
    cl = ((T["label"] == "correct") & (T["type"] == "long")).values
    pdf = pd.DataFrame({"channel_id": meta["channel_ids"], "x": meta["x"], "y": meta["y"]})
    pdf["shank"] = (pdf["x"] // 250).astype(int)
    sel_ids = [str(c) for c in fn.select_channels(pdf, n_total=20)["channel_id"]]

    recs, _ = fn.load_session(PROC, SES_R, XB, refs=(REF_R,))
    rec = recs[REF_R]
    full = [str(c) for c in rec.get_channel_ids()]
    objs = [rec.get_channel_ids()[full.index(c)] for c in sel_ids]
    recN = rec.select_channels(objs) if hasattr(rec, "select_channels") else rec.channel_slice(channel_ids=objs)
    fs = recN.get_sampling_frequency()

    def theta_raw(event_col, align):
        """log10 of the mean theta power (4–7 Hz) over time, per trial and channel"""
        a0, a1 = fn.MAIN_EPOCHS[align]
        X, ok = fn.read_epochs(recN, T.loc[cl, event_col].values, a0, a1)
        X = X.astype(np.float64)
        spec = fn.MAIN_BANDS[BAND]
        Pw = np.mean([np.abs(ss.fftconvolve(X, fn.morlet(f, spec["n_cycles"], fs)[None, None, :], mode="same", axes=-1)) ** 2
                      for f in spec["freqs"]], axis=0)[..., ::DEC]
        tt = (np.arange(X.shape[-1]) / fs + a0)[::DEC]
        return tt, np.log10(Pw)                                      # (trials, channels, times)

    tP, rawP = theta_raw("press_s", "Press")
    tR, rawR = theta_raw("release_s", "Release")
    avg = lambda Z: Z.mean(axis=0).mean(axis=0)                     # mean over trials, then channels

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, facecolor="white",
                             gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.08})
    for ax, tt, Z, rng_, win, xl in ((axes[0], tP, rawP, (-0.7, 1.0), ["pre", "hold"], "Time from press (s)"),
                                     (axes[1], tR, rawR, (-0.3, 0.7), ["post"], "Time from release (s)")):
        m_ = (tt >= rng_[0]) & (tt <= rng_[1])
        ax.plot(tt[m_], avg(Z)[m_], color=THETA_COL, lw=2.2)
        for name in win:
            a, z = fn.MAIN_WINDOWS[name][1]
            ax.axvspan(a, z, color=WIN_COL, alpha=0.12, zorder=0, lw=0)
            ax.text((a + z) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=WIN_COL)
        ax.axvline(0, color="k", ls="--", lw=1); ax.set_xlabel(xl)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Theta power (log10, a.u.)")
    axes[0].text(-0.14, 1.08, "B", transform=axes[0].transAxes, fontsize=14, fontweight="bold")

    theta_dir  = os.path.join(AMIR, "Theta", "20 selected channels_pilot"); os.makedirs(theta_dir, exist_ok=True)
    report_dir = os.path.join(AMIR, "report"); os.makedirs(report_dir, exist_ok=True)
    _savefig(fig, os.path.join(theta_dir, "pilot theta across time.png"))
    _savefig(fig, os.path.join(report_dir, "Fig 2B pilot theta time course.png"))
    plt.show()
    print(f"pilot {SES_R}, {REF_R}: {cl.sum()} correct long trials, {len(sel_ids)} channels")


@cell()
def theta_all_sessions():
    """1e. THETA ACROSS ALL SESSIONS (paper figure: Fig. 2D CMR; LOCAL kept in the theta folder) + table of all bands"""
    global BAND, BANDS_, M1t, MIN_CL, MV, PAN, Q, RAT_NAME, RCOL, REF, REPORT_NAME, SUP_NAME, _, ax, b, band_rows, col, fig
    global hi, i, k, lo, m, mean_pct, multipletests, n, n_ses, n_tr, p, pc, q, qs, qs_, rat, report_dir, rng, se, ses_dir
    global session_values, sp, stats, summary_rows, sup_dir, tcrit, tests, up, v, vr, x
    # ── 1e. THETA ACROSS ALL SESSIONS (paper figure: Fig. 2D CMR; LOCAL kept in the theta folder) + table of all bands ──
    from statsmodels.stats.multitest import multipletests
    from scipy import stats
    BAND, MIN_CL = "theta 4-7", globals().get("MIN_CL", 10)
    BANDS_ = globals().get("BANDS", list(fn.MAIN_BANDS))
    RCOL = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}                  # rat colours used in the project
    RAT_NAME = {"134": "Rat 1", "135": "Rat 2", "136": "Rat 3"}
    PAN  = [("hold-pre", "Hold vs\nbefore press"), ("hold-post", "Hold vs\nafter release")]
    pc   = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    ses_dir = os.path.join(AMIR, "Theta", "hold vs pre and post all sessions"); os.makedirs(ses_dir, exist_ok=True)
    report_dir = os.path.join(AMIR, "report"); os.makedirs(report_dir, exist_ok=True)
    REPORT_NAME = {"CMR": "Fig 2D all sessions theta CMR.png"}
    SUP_NAME    = {"LOCAL": "Fig S7 theta all sessions LOCAL.png"}
    sup_dir = os.path.join(AMIR, "supplementary"); os.makedirs(sup_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif",
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8})

    def session_values(ref):
        """M1 session values (correct long, all bands); read from tables or computed with fn.m1_session (as 1b)"""
        f = os.path.join(TABLES, f"M1_session_values_{ref}.csv")
        if os.path.exists(f):
            return pd.read_csv(f, dtype={"rat": str})
        rows = []
        for s in done_sessions(ref):
            m, n_cl = fn.m1_session(load_bp(s, ref), BANDS_, MIN_CL)
            if m is not None:
                rows += [{"session": s, "rat": RAT[s], "band": b, "n_cl": n_cl, **m[b]} for b in BANDS_]
        M = pd.DataFrame(rows); M.to_csv(f, index=False)
        return M

    summary_rows, band_rows = [], []
    for REF in ["CMR", "LOCAL"]:
        MV = session_values(REF)
        # FDR over the 12 tests (6 bands x 2 comparisons, all rats)
        tests = [(col, b, *fn.sign_flip_p(MV.loc[MV["band"] == b, col].values)) for col, _ in PAN for b in BANDS_]
        qs = multipletests([t[3] for t in tests], method="fdr_bh")[1]
        Q = {(t[0], t[1]): q for t, q in zip(tests, qs)}
        for (col, b, m, p, n), q in zip(tests, qs):
            v = MV.loc[MV["band"] == b, col]
            band_rows.append({"ref": REF, "band": b, "measure": col, "n_sessions": n, "sessions_up": int((v > 0).sum()),
                              "mean_%": round(float(pc(m)), 1), "p": p, "q": q})

        M1t = MV[MV["band"] == BAND].copy()
        n_ses, n_tr = M1t["session"].nunique(), int(M1t["n_cl"].sum())
        fig, ax = plt.subplots(figsize=(3.35, 3.1), facecolor="white")               # ~8.5 x 8 cm, single column
        rng = np.random.default_rng(0)
        for i, (col, _) in enumerate(PAN):
            v = pc(M1t[col])
            for k, rat in enumerate(sorted(RCOL)):
                vr = v[M1t["rat"] == rat]
                ax.scatter(i + (k - 1) * 0.11 + rng.uniform(-0.025, 0.025, len(vr)), vr, s=14, color=RCOL[rat],
                           edgecolor="0.35", linewidth=0.35, zorder=2,
                           label=f"{RAT_NAME[rat]} (n = {len(vr)})" if i == 0 else None)
            m, p, n = fn.sign_flip_p(M1t[col].values)                 # test on log10 values
            x = M1t[col].values                                        # session values, log10 ratios
            se, tcrit = x.std(ddof=1) / np.sqrt(len(x)), stats.t.ppf(0.975, len(x) - 1)
            mean_pct = pc(x.mean())
            lo, hi = pc(x.mean() - tcrit * se), pc(x.mean() + tcrit * se)   # 95 % CI on log values, back to %
            q = Q[(col, BAND)]
            ax.errorbar(i + 0.28, mean_pct, yerr=[[mean_pct - lo], [hi - mean_pct]], fmt="o", color="k",
                        ms=5, capsize=3, lw=1.2, zorder=4)
            qs_ = "q < 0.001" if q < 0.001 else f"q = {q:.3f}"
            ax.text(i, 1.02, f"{mean_pct:+.0f} %\n{qs_}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8)
            up = int((M1t[col] > 0).sum())
            summary_rows.append({"ref": REF, "measure": col, "n_sessions": n, "sessions_up": up,
                                 "mean_%": round(float(mean_pct), 1), "ci95_%": f"{lo:.1f} to {hi:.1f}", "p": p, "q": q})
        ax.axhline(0, color="0.4", lw=0.8, ls="--", zorder=1)
        ax.set_xticks(range(len(PAN))); ax.set_xticklabels([t for _, t in PAN]); ax.set_xlim(-0.5, len(PAN) - 0.5)
        ax.set_ylabel("Theta power change (%)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2)
        if REF == "CMR":
            ax.text(-0.28, 1.15, "D", transform=ax.transAxes, fontsize=12, fontweight="bold")
        _savefig(fig, os.path.join(ses_dir, f"all_sessions_theta_{REF}.png"))
        if REF in SUP_NAME:
            _savefig(fig, os.path.join(sup_dir, SUP_NAME[REF]))
        if REF in REPORT_NAME:
            _savefig(fig, os.path.join(report_dir, REPORT_NAME[REF]))
        plt.show()
        print(f"{REF}: {n_ses} sessions, {n_tr} correct long trials | sessions per rat:",
              {RAT_NAME[k]: v for k, v in M1t.groupby("rat")["session"].nunique().items()})

    print("\ntheta, correct long — CMR vs LOCAL (mean and 95 % CI over sessions, computed on log values)")
    print(pd.DataFrame(summary_rows).round({"p": 4, "q": 4}).to_string(index=False))
    print("\nall bands (FDR over 12 tests per reference)")
    print(pd.DataFrame(band_rows).round({"p": 4, "q": 4}).to_string(index=False))


@cell()
def theta_each_condition_numbers():
    """1e-b. Fig. 2D measures for EACH condition alone (numbers only): hold vs pre and hold vs post, CMR and LOCAL"""
    global BAND, D, HOLD_FULL, L, MIN_N, R, REF, T, g, hold, k, lab, lng, m, meas, mk, multipletests, n, p, pc, q, r, rows, s
    global se, stats, t_, tc, tests, v, vals
    # ── 1e-b. Fig. 2D measures for EACH condition alone (numbers only): hold vs pre and hold vs post, CMR and LOCAL ──
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    BAND, HOLD_FULL = "theta 4-7", fn.MAIN_WINDOWS["hold"][1][1]                   # 0.85 s
    MIN_N = {"correct": 10, "errors": 5, "late": 5, "early": 5}                      # correct as Fig. 2D
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    rows = []
    for REF in ["CMR", "LOCAL"]:
        vals = []
        for s in done_sessions(REF):
            r = load_bp(s, REF); T = r["trials"]; L = r["logpow"][BAND]
            lng, lab, hold = (T["type"] == "long").values, T["label"].values, T["hold_s"].values
            mk = {"correct": lng & (lab == "correct"), "late": lng & (lab == "late"),
                  "early": lng & (lab == "early") & (hold >= HOLD_FULL)}
            mk["errors"] = mk["late"] | mk["early"]
            for g, k in mk.items():
                if k.sum() >= MIN_N[g]:
                    vals.append({"group": g, "n": int(k.sum()),
                                 "hold-pre":  float(np.median((L["hold"][k] - L["pre"][k]).mean(0))),
                                 "hold-post": float(np.median((L["hold"][k] - L["post"][k]).mean(0)))})
        D = pd.DataFrame(vals); tests = []
        for meas in ["hold-pre", "hold-post"]:
            for g in ["correct", "errors", "late", "early"]:
                v = D.loc[D["group"] == g, meas].values; m, p, n = fn.sign_flip_p(v)
                se, tc = v.std(ddof=1) / np.sqrt(len(v)), stats.t.ppf(0.975, len(v) - 1)
                tests.append({"ref": REF, "measure": meas, "group": g, "n_sessions": n,
                              "n_trials": int(D.loc[D["group"] == g, "n"].sum()), "sessions_up": int((v > 0).sum()),
                              "mean_%": round(float(pc(v.mean())), 1),
                              "ci95_%": f"{pc(v.mean() - tc * se):.1f} to {pc(v.mean() + tc * se):.1f}", "p": p})
        for t_, q in zip(tests, multipletests([x["p"] for x in tests], method="fdr_bh")[1]):
            t_["q"] = q
        rows += tests
    R = pd.DataFrame(rows)
    print("each condition alone (as Fig. 2D); sign-flip across sessions; q = FDR over the 8 tests per reference; early held >= 0.85 s")
    print(R.round({"p": 4, "q": 4}).to_string(index=False))
    _table(R, os.path.join(TABLES, "theta_hold_vs_pre_post_all_conditions.csv"), index=False)


@cell()
def theta_time_course_long():
    """1g. THETA TIME COURSE, all long trials (all sessions, all rats): correct, late, early long — no tests"""
    global ALIGN, BAND, C, CONDS, MIN_FRAC, MIN_ON, MIN_SES_FRAC, MIN_TRIALS, REF_TC, SC, TONE_LONG, TT, _, al, c, cm, cur, h
    global hold, hp, i, info, k, long_dir, m, make_fig, n, n_s, n_t, ok, pc, post, pre_ch, r, raw, s, style, summary, tt
    global win_mask
    # ── 1g. THETA TIME COURSE, all long trials (all sessions, all rats): correct, late, early long — no tests ──
    REF_TC, BAND = "CMR", "theta 4-7"                 # "LOCAL" for the LOCAL version
    MIN_TRIALS = {"correct long": 10, "late long": 5, "early long (hold>0.3)": 5}   # per session
    MIN_ON, MIN_FRAC, MIN_SES_FRAC = 3, 0.5, 0.5
    CONDS = {"correct long": "#2CA02C", "late long": "#DD8452", "early long (hold>0.3)": "#E6B800"}
    ALIGN = [("Press", "Time from press (s)"), ("Release", "Time from release (s)")]
    TONE_LONG = 1.05
    pc = globals().get("pc", lambda x: 100 * (10 ** np.asarray(x, float) - 1))
    long_dir = os.path.join(AMIR, "Theta", "all long trials"); os.makedirs(long_dir, exist_ok=True)

    def win_mask(t, name):
        a, z = fn.MAIN_WINDOWS[name][1]
        return (t >= a - 1e-9) & (t <= z + 1e-9)

    # 1) per session and condition: change curve (vs own pre-press, as stored), raw curve, hold and post levels
    SC = {c: [] for c in CONDS}; TT = {}
    for s in done_sessions(REF_TC):
        r = load_bp(s, REF_TC); cm = fn.main_conditions(r["trials"])
        for al, _ in ALIGN:
            TT.setdefault(al, r["times"][al])
        for c in CONDS:
            k = cm[c].values; n = int(k.sum())
            if n < MIN_TRIALS[c]:
                continue
            cur, raw = {}, {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pre_ch = np.nanmean(r["logpow"][BAND]["pre"][k], axis=0)            # (channels,) pre-press level of this condition
                for al, _ in ALIGN:
                    C = r["curves"][al][BAND][c]                                     # (channels, times), vs own pre-press
                    ok = r["counts"][al][c] >= max(MIN_ON, MIN_FRAC * n)             # >= 50 % of the trials on the lever
                    cur[al] = np.where(ok, np.nanmedian(C, axis=0), np.nan)
                    raw[al] = np.where(ok, np.nanmedian(C + pre_ch[:, None], axis=0), np.nan)
            if any(len(cur[al]) != len(TT[al]) for al, _ in ALIGN):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                hold = np.nanmean(cur["Press"][win_mask(TT["Press"], "hold")])       # hold level (on the lever), vs pre
                post = np.nanmean(cur["Release"][win_mask(TT["Release"], "post")])   # post level, vs pre
            SC[c].append({"session": s, "rat": RAT[s], "n": n, "cur": cur, "raw": raw, "hold": hold, "post": post})

    def summary(c, al, mode):
        """mean ± SEM across sessions. mode 'raw': log10 power; 'pre': % vs pre-press; 'hold': release part % vs hold level"""
        rows = [x for x in SC[c] if mode != "post" or np.isfinite(x["post"])]
        if mode == "raw":
            X = np.array([x["raw"][al] for x in rows])
        else:
            X = np.array([x["cur"][al] - (x["post"] if (mode == "post" and al == "Release") else 0) for x in rows])
        n_s = np.sum(np.isfinite(X), 0); ok = n_s >= max(3, MIN_SES_FRAC * len(X))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m, se = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1) / np.sqrt(n_s)
        f = (lambda v: v) if mode == "raw" else pc
        return TT[al][ok], f(m)[ok], f(m - se)[ok], f(m + se)[ok]

    info = lambda c: (len(SC[c]), sum(x["n"] for x in SC[c]))
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9})

    def style(ax, al, xlabel, wins, zero=True):
        if zero:
            ax.axhline(0, color="0.3", lw=0.8)
        ax.axvline(0, color="k", ls="--", lw=1)
        for name in wins:
            a, z = fn.MAIN_WINDOWS[name][1]
            ax.axvspan(a, z, color="0.85", alpha=0.6, zorder=0, lw=0)
            ax.text((a + z) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, fontweight="bold")
        if al == "Press":
            ax.axvline(TONE_LONG, color="0.4", ls=":", lw=1.2)
            ax.text(TONE_LONG, 0.02, " tone", transform=ax.get_xaxis_transform(), fontsize=8, color="0.4")
        ax.set_xlabel(xlabel)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def make_fig(mode, title, ylabels, fname):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), facecolor="white", gridspec_kw={"width_ratios": [2.7, 2.1], "wspace": 0.25})
        for ax, (al, xl), yl in zip(axes, ALIGN, ylabels):
            for c in CONDS:
                if len(SC[c]) < 3:
                    continue
                tt, m, lo, hi = summary(c, al, mode); n_s, n_t = info(c)
                ax.fill_between(tt, lo, hi, color=CONDS[c], alpha=0.2, lw=0)
                ax.plot(tt, m, color=CONDS[c], lw=2.2, label=f"{c} ({n_s} sessions, {n_t:,} trials)")
            style(ax, al, xl, ["pre", "hold"] if al == "Press" else ["post"], zero=(mode != "raw"))
            ax.set_ylabel(yl)
        axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"{title}\nlong trials by outcome ({REF_TC}, mean ± SEM across sessions)", fontsize=11, fontweight="bold", y=1.12)
        _savefig(fig, os.path.join(long_dir, fname))
        plt.show()

    # Figure 1: theta power across time, no baseline
    make_fig("raw", "Theta power (4–7 Hz) across time, no baseline",
             ["Theta power, log10 (a.u.)", "Theta power, log10 (a.u.)"], f"theta_raw_timecourse_long_{REF_TC}.png")
    # Figure 2: hold and after the release, both vs before the press
    make_fig("pre", "Theta power: hold and after the release compared to before the press",
             ["Theta power change (%)\nvs own pre-press", "Theta power change (%)\nvs own pre-press"],
             f"theta_hold_post_vs_pre_long_{REF_TC}.png")
    # Figure 3: hold vs before the press (left), hold vs after the release (right; post level = 0)
    make_fig("post", "Theta power: hold vs before the press (left) and hold vs after the release (right)",
             ["Theta power change (%)\nvs own pre-press", "Theta power change (%)\nvs own post-release level"],
             f"theta_hold_vs_pre_and_post_long_{REF_TC}.png")

    # numbers (no tests): peak, hold vs pre, hold vs post — mean across sessions
    for c in CONDS:
        if len(SC[c]) < 3:
            continue
        tt, m, _, _ = summary(c, "Press", "pre"); i = np.nanargmax(m)
        h = np.array([x["hold"] for x in SC[c]]); hp = np.array([x["hold"] - x["post"] for x in SC[c]])
        n_s, n_t = info(c)
        print(f"{c:24s} {n_s:2d} sessions, {n_t:5,} trials | peak {m[i]:+.0f} % at {tt[i]:.2f} s | "
              f"hold vs pre {pc(np.nanmean(h)):+.1f} % | hold vs post {pc(np.nanmean(hp)):+.1f} %")


@cell()
def theta_all_long_pooled():
    """1h. THETA DURING THE HOLD, ALL LONG TRIALS POOLED (correct + late + early held >= 0.85 s), CMR and LOCAL"""
    global BAND, HOLD_FULL, MIN_TR, PAN, PL, QH, RCOL, REF_P, T, _, ax, col, fig, i, k, lab, lng, long_dir, lp, m, mean_pct
    global mk, multipletests, n, n_ses, n_tr, p, pc, ps, r, rat, rng, rows, s, sem_pct, sp, summary_rows, tests, up, v, vr
    # ── 1h. THETA DURING THE HOLD, ALL LONG TRIALS POOLED (correct + late + early held >= 0.85 s), CMR and LOCAL ──
    from statsmodels.stats.multitest import multipletests
    BAND, MIN_TR = "theta 4-7", 10
    HOLD_FULL = fn.MAIN_WINDOWS["hold"][1][1]               # 0.85 s
    RCOL = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}
    PAN  = [("hold-pre", "Hold vs\nbefore the press"), ("hold-post", "Hold vs\nafter the release")]
    pc   = globals().get("pc", lambda x: 100 * (10 ** np.asarray(x, float) - 1))
    long_dir = os.path.join(AMIR, "Theta", "all long trials"); os.makedirs(long_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9, "font.family": "sans-serif"})

    summary_rows = []
    for REF_P in ["CMR", "LOCAL"]:
        rows = []
        for s in done_sessions(REF_P):
            r = load_bp(s, REF_P)
            T, lp = r["trials"], r["logpow"][BAND]
            lab, lng = T["label"].values, (T["type"] == "long").values
            mk = lng & (np.isin(lab, ["correct", "late"]) | ((lab == "early") & (T["hold_s"].values >= HOLD_FULL)))
            if mk.sum() < MIN_TR:
                continue
            rows.append({"session": s, "rat": RAT[s], "n": int(mk.sum()),
                         "n_correct": int((mk & (lab == "correct")).sum()), "n_late": int((mk & (lab == "late")).sum()),
                         "n_early": int((mk & (lab == "early")).sum()),
                         "hold-pre":  float(np.median((lp["hold"][mk] - lp["pre"][mk]).mean(0))),
                         "hold-post": float(np.median((lp["hold"][mk] - lp["post"][mk]).mean(0)))})
        PL = pd.DataFrame(rows)
        _table(PL, os.path.join(TABLES, f"theta_all_long_pooled_{REF_P}.csv"), index=False)
        n_ses, n_tr = PL["session"].nunique(), int(PL["n"].sum())
        tests = {col: fn.sign_flip_p(PL[col].values) for col, _ in PAN}
        QH = dict(zip(tests, multipletests([t[1] for t in tests.values()], method="fdr_bh")[1]))   # FDR over the 2 tests

        fig, ax = plt.subplots(figsize=(4.6, 4.6), facecolor="white")
        rng = np.random.default_rng(0)
        for i, (col, _) in enumerate(PAN):
            v = pc(PL[col])
            for k, rat in enumerate(sorted(RCOL)):
                vr = v[PL["rat"] == rat]
                ax.scatter(i + (k - 1) * 0.12 + rng.uniform(-0.03, 0.03, len(vr)), vr, s=22, color=RCOL[rat],
                           alpha=0.75, edgecolor="white", linewidth=0.4, zorder=2,
                           label=f"Rat {rat} (n = {len(vr)})" if i == 0 else None)
            m, p, n = tests[col]
            mean_pct, sem_pct = pc(m), v.std(ddof=1) / np.sqrt(len(v))
            ax.errorbar(i + 0.30, mean_pct, yerr=sem_pct, fmt="o", color="k", ms=7, capsize=4, lw=1.6, zorder=4)
            ps = "q < 0.001 (FDR)" if QH[col] < 0.001 else f"q = {QH[col]:.3f} (FDR)"
            up = int((PL[col] > 0).sum())
            ax.text(i, 1.01, f"{mean_pct:+.0f} %\n{up}/{n} sessions up\n{ps}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
            summary_rows.append({"ref": REF_P, "measure": col, "n_sessions": n, "sessions_up": up,
                                 "mean_%": round(float(mean_pct), 1), "p": p, "q": QH[col]})
        ax.axhline(0, color="0.3", lw=0.9, ls="--", zorder=1)
        ax.set_xticks(range(len(PAN))); ax.set_xticklabels([t for _, t in PAN]); ax.set_xlim(-0.5, len(PAN) - 0.5)
        ax.set_ylabel("Theta power change (%)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_title(f"Theta (4–7 Hz) during the hold, all long trials\n{n_ses} sessions, {n_tr:,} trials ({REF_P})",
                     fontsize=11, fontweight="bold", pad=58)
        _savefig(fig, os.path.join(long_dir, f"theta_all_long_pooled_{REF_P}.png"))
        plt.show()
        print(f"{REF_P}: {n_ses} sessions, {n_tr} trials: correct {PL['n_correct'].sum()}, late {PL['n_late'].sum()}, "
              f"early (held >= {HOLD_FULL} s) {PL['n_early'].sum()} | per rat:", PL.groupby("rat")["session"].nunique().to_dict())

    print("\ntheta, all long trials pooled — CMR vs LOCAL")
    print(pd.DataFrame(summary_rows).round({"p": 4, "q": 4}).to_string(index=False))


@cell()
def theta_per_region():
    """1i. THETA DURING THE HOLD PER REGION (PL, IL, Cg1), CMR and LOCAL (personal note)"""
    global BAND, CHREG, GCOL, L, MEAS, MIN_CH, MIN_CL, PAIRS, REGS, RS, RV, W, _, a, ax, axes, b, ch, cl, d, diffs, fam, fam_d
    global fam_r, fdr, fig, k, m, masks, meas, mr, multipletests, n, nr, p, pc, pr, ps, q, qq, r, rat, ref, reg, reg_dir, rng
    global row, rows, s, sp, stats, sub, t, v, vr, w0, x
    # ── 1i. THETA DURING THE HOLD PER REGION (PL, IL, Cg1), CMR and LOCAL (personal note) ──
    from statsmodels.stats.multitest import multipletests
    fdr = lambda p: multipletests(np.asarray(p, float), alpha=0.05, method="fdr_bh")[1]
    pc  = globals().get("pc", lambda x: 100 * (10 ** np.asarray(x, float) - 1))
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    BAND, MIN_CL, MIN_CH = "theta 4-7", 10, 10
    REGS  = list(fn.REGIONS)                                     # PL, IL, Cg1
    PAIRS = [("IL", "PL"), ("IL", "Cg1"), ("PL", "Cg1")]
    MEAS  = [("hold-pre", "pre"), ("hold-post", "post")]
    GCOL  = {"PL": "#1f77b4", "IL": "#ff7f0e", "Cg1": "#9467bd"}
    reg_dir = os.path.join(AMIR, "Theta", "across regions"); os.makedirs(reg_dir, exist_ok=True)

    # 1) one value per session x region x comparison x reference
    rows = []
    for ref in ["CMR", "LOCAL"]:
        for s in done_sessions(ref):
            r = load_bp(s, ref)
            cl = ((r["trials"]["label"] == "correct") & (r["trials"]["type"] == "long")).values
            if cl.sum() < MIN_CL:
                continue
            masks, _ = fn.region_masks(r, CHREG, s)
            L = r["logpow"][BAND]
            for reg, m in masks.items():
                if m.sum() < MIN_CH:
                    continue
                for meas, w0 in MEAS:
                    ch = (L["hold"][cl][:, m] - L[w0][cl][:, m]).mean(axis=0)
                    rows.append({"ref": ref, "session": s, "rat": RAT[s], "region": reg, "n_ch": int(m.sum()),
                                 "measure": meas, "value": float(np.median(ch))})
    RV = pd.DataFrame(rows)
    _table(RV, os.path.join(TABLES, "theta_regions_values.csv"), index=False)

    # 2) statistics: each region vs 0, paired differences, per rat; FDR per reference and family
    stats = []
    for ref in ["CMR", "LOCAL"]:
        d = RV[RV["ref"] == ref]
        fam_r, fam_d = [], []
        for meas, _ in MEAS:
            W = d[d["measure"] == meas].pivot(index="session", columns="region", values="value")
            for reg in REGS:
                v = W[reg].dropna(); m, p, n = fn.sign_flip_p(v.values)
                fam_r.append({"ref": ref, "measure": meas, "test": reg, "n_sessions": n, "up": int((v > 0).sum()), "mean_%": pc(m), "p": p})
                for rat in sorted(d["rat"].unique()):
                    vr = v[[RAT[s] == rat for s in v.index]]
                    if len(vr) >= 2:
                        mr, pr, nr = fn.sign_flip_p(vr.values)
                        stats.append({"ref": ref, "measure": meas, "test": reg, "rat": rat, "n_sessions": nr,
                                      "up": int((vr > 0).sum()), "mean_%": pc(mr), "p": pr, "p_fdr": np.nan})
            for a, b in PAIRS:
                v = (W[a] - W[b]).dropna(); m, p, n = fn.sign_flip_p(v.values)
                fam_d.append({"ref": ref, "measure": meas, "test": f"{a} − {b}", "n_sessions": n, "up": int((v > 0).sum()), "mean_%": pc(m), "p": p})
        for fam in (fam_r, fam_d):
            q = fdr([x["p"] for x in fam])
            for x, qq in zip(fam, q):
                stats.append({**x, "rat": "all", "p_fdr": qq})
    RS = pd.DataFrame(stats)
    _table(RS, os.path.join(TABLES, "theta_regions_stats.csv"), index=False)

    for ref in ["CMR", "LOCAL"]:
        print(f"\n===== theta, {ref}: regions and differences (all rats; p_fdr per family of 6) =====")
        t = RS[(RS["ref"] == ref) & (RS["rat"] == "all")][["measure", "test", "n_sessions", "up", "mean_%", "p", "p_fdr"]]
        print(t.round({"mean_%": 1, "p": 4, "p_fdr": 4}).to_string(index=False))
        print("--- per rat (hold-pre, mean %) ---")
        t = RS[(RS["ref"] == ref) & (RS["rat"] != "all") & (RS["measure"] == "hold-pre")]
        print(t.pivot(index="test", columns="rat", values="mean_%").round(1).astype(str).add(" %").to_string())
    print("\nsessions with >= 10 channels per region:\n", RV[(RV["ref"] == "LOCAL") & (RV["measure"] == "hold-pre")]
          .groupby(["rat", "region"])["session"].nunique().unstack().to_string())

    # 3) figure per reference: one dot = one session, per region, two comparisons
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9})
    for ref in ["CMR", "LOCAL"]:
        d = RV[RV["ref"] == ref]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.4), sharey=True, facecolor="white"); rng = np.random.default_rng(0)
        fig.subplots_adjust(top=0.70, bottom=0.16, wspace=0.08)
        for ax, (meas, _) in zip(axes, MEAS):
            for k, reg in enumerate(REGS):
                v = pc(d[(d["measure"] == meas) & (d["region"] == reg)]["value"])
                ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=18, color=GCOL[reg], alpha=0.7, lw=0)
                row = RS[(RS["ref"] == ref) & (RS["rat"] == "all") & (RS["measure"] == meas) & (RS["test"] == reg)].iloc[0]
                ax.errorbar(k + 0.28, row["mean_%"], yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", color="k", ms=6, capsize=3, lw=1.4)
                ps = "q < 0.001" if row["p_fdr"] < 0.001 else f"q = {row['p_fdr']:.3f}"
                ax.text(k, 1.01, f"{row['mean_%']:+.0f} %\n{row['up']}/{row['n_sessions']}\n{ps}", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8.5)
            ax.axhline(0, color="0.3", lw=0.8, ls="--")
            ax.set_xticks(range(len(REGS))); ax.set_xticklabels(REGS); ax.set_xlim(-0.5, len(REGS) - 0.4)
            ax.set_title("Hold vs before the press" if meas == "hold-pre" else "Hold vs after the release", fontsize=10, pad=50)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel("Theta power change (%), one dot = one session")
        diffs = RS[(RS["ref"] == ref) & (RS["rat"] == "all") & (RS["measure"] == "hold-pre") & RS["test"].str.contains("−")]
        sub = "  |  ".join(f"{x['test']} {x['mean_%']:+.1f} % (q = {x['p_fdr']:.3f})" for _, x in diffs.iterrows())
        fig.suptitle(f"Theta (4–7 Hz) during the hold per region ({ref})", fontsize=11, fontweight="bold", y=0.98)
        fig.text(0.5, 0.02, f"Region differences, hold vs pre: {sub}\nq = FDR-corrected p (6 tests per family)",
                 ha="center", va="bottom", fontsize=8.5)
        _savefig(fig, os.path.join(reg_dir, f"theta_regions_{ref}.png"))
        plt.show()


@cell()
def theta_findings_per_region():
    """1r. ALL THETA FINDINGS PER REGION (IL, PL, Cg1): does any finding differ between regions? CMR and LOCAL"""
    global BAND, CHREG, FULL2SHORT, H, HG, HOLD_FULL, L, Lg, MEAS, MIN_CH, MIN_CL, MIN_TR, RCOLR, REF, REGS, RPAIRS, S, T, T5
    global TS_DIR, V, W, WG_WIN, _, a, add, alone, ax, axes, b, bot, ci, cl, ct, d, err_h, err_p, f, fdr, fig, g, go, hg, hi
    global hit, hold, i, j, k, kind, lab, lab_, late, lng, lo, m, masks, meas, med, mm, mu, multipletests, n_, ok, p, pairs
    global pc, pos, pre, q, r, ref, reg, reg_dir, regch, row, rows, s, sp_, span, spearmanr, stats, step, t, th, thp, tone
    global top, ttl, v, val, wait, wwin, x, x0, x1, y, yb, z
    # ── 1r. ALL THETA FINDINGS PER REGION (IL, PL, Cg1): does any finding differ between regions? CMR and LOCAL ──
    from scipy import stats
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    BAND, HG = "theta 4-7", "highG 61-100"
    MIN_CL, MIN_TR, MIN_CH = 10, 5, 10
    HOLD_FULL = fn.MAIN_WINDOWS["hold"][1][1]                                       # 0.85 s
    WG_WIN = (0.50, 1.00)                                                            # wait vs go window (after short tone, before long tone)
    REGS = ["IL", "PL", "Cg1"]; RPAIRS = [("IL", "PL"), ("IL", "Cg1"), ("PL", "Cg1")]
    RCOLR = {"PL": "#1f77b4", "IL": "#ff7f0e", "Cg1": "#9467bd"}
    FULL2SHORT = {v: k for k, v in fn.REGIONS.items()}
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    TS_DIR = os.path.join(DATA_OUT, "Theta signal all channels")                    # 06 cell 5i
    MEAS = [("hold_pre",  "1. Hold vs before press\n(correct)", "pct"),
            ("hold_post", "1. Hold vs after release\n(correct)", "pct"),
            ("wait_go",   "2. Wait − go\n(0.50–1.00 s)", "pct"),
            ("out_hold",  "3. Correct − errors\n(hold vs pre)", "pct"),
            ("out_post",  "3. Correct − errors\n(after release vs pre)", "pct"),
            ("mov_icpt",  "4. Theta drop at zero\nhigh-gamma change", "pct"),
            ("rt_rho",    "5. Rho: pre-tone theta\nvs reaction time", "rho")]
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    fdr = lambda p: multipletests(np.asarray(p, float), method="fdr_bh")[1]
    lab_ = lambda q: "q < 0.001" if q < 0.001 else "q < 0.01" if q < 0.01 else "q < 0.05" if q < 0.05 else "n.s."
    reg_dir = os.path.join(AMIR, "Theta", "across regions"); os.makedirs(reg_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})

    rows = []
    for REF in ["CMR", "LOCAL"]:
        # ── measures 1, 3, 4, 5 from the band-power files ──
        for s in done_sessions(REF):
            r = load_bp(s, REF); T = r["trials"]; L, H = r["logpow"][BAND], r["logpow"][HG]
            masks, _ = fn.region_masks(r, CHREG, s)
            lng, lab, hold = (T["type"] == "long").values, T["label"].values, T["hold_s"].values
            cl, late = lng & (lab == "correct"), lng & (lab == "late")
            err_h = late | (lng & (lab == "early") & (hold >= HOLD_FULL))
            err_p = late | (lng & (lab == "early") & (hold > 0.3))
            tone = T["tone_delay_s"].values.astype(float)
            for reg, m in masks.items():
                if m.sum() < MIN_CH:
                    continue
                med = lambda X, k: float(np.median(X[k][:, m].mean(0)))              # per channel mean over trials, median over region
                add = lambda meas, v: rows.append({"ref": REF, "session": s, "rat": RAT[s], "region": reg, "measure": meas, "value": v})
                if cl.sum() >= MIN_CL:
                    add("hold_pre",  med(L["hold"] - L["pre"], cl))
                    add("hold_post", med(L["hold"] - L["post"], cl))
                    th = np.median((L["post"] - L["hold"])[cl][:, m], axis=1)          # per trial, median over region channels
                    hg = np.median((H["post"] - H["hold"])[cl][:, m], axis=1)
                    ok = np.isfinite(th) & np.isfinite(hg)
                    if ok.sum() >= MIN_CL:
                        add("mov_icpt", float(np.polyfit(hg[ok], th[ok], 1)[1]))     # intercept = theta drop at zero high-gamma change
                    ct = cl & np.isfinite(tone)
                    if ct.sum() >= MIN_CL:
                        thp = np.median((L["pre_tone"] - L["pre"])[ct][:, m], axis=1)
                        add("rt_rho", float(spearmanr(thp, (hold - tone)[ct], nan_policy="omit")[0]))
                if cl.sum() >= MIN_TR and err_h.sum() >= MIN_TR:
                    add("out_hold", med(L["hold"] - L["pre"], cl) - med(L["hold"] - L["pre"], err_h))
                if cl.sum() >= MIN_TR and err_p.sum() >= MIN_TR:
                    add("out_post", med(L["post"] - L["pre"], cl) - med(L["post"] - L["pre"], err_p))

        # ── measure 2 (wait − go) from the 5i theta signal (go trials continue after their release) ──
        for s in done_sessions(REF):
            f = os.path.join(TS_DIR, REF, s + ".npz")
            if not os.path.exists(f):
                continue
            T5 = pd.read_csv(os.path.join(TS_DIR, REF, s + "_trials.csv"))
            wait = ((T5["type"] == "long") & (T5["label"] == "correct")).values
            go   = ((T5["type"] == "short") & (T5["label"] == "correct")).values
            if wait.sum() < MIN_TR or go.sum() < MIN_TR:
                continue
            z = np.load(f, allow_pickle=True); t = z["t"]
            regch = np.array([FULL2SHORT.get(str(x), str(x)) for x in z["region"]])
            pre = (t >= fn.MAIN_WINDOWS["pre"][1][0] - 1e-9) & (t <= fn.MAIN_WINDOWS["pre"][1][1] + 1e-9)
            wwin = (t >= WG_WIN[0] - 1e-9) & (t <= WG_WIN[1] + 1e-9)
            val = {}
            for g, k in (("wait", wait), ("go", go)):
                Lg = np.log10(np.abs(z["coef"][k]).astype(np.float64) ** 2)          # (trials, channels, times)
                val[g] = (Lg[..., wwin].mean(-1) - Lg[..., pre].mean(-1)).mean(0)   # per channel: mean over trials, vs own pre-press
            for reg in REGS:
                m = regch == reg
                if m.sum() >= MIN_CH:
                    rows.append({"ref": REF, "session": s, "rat": RAT[s], "region": reg, "measure": "wait_go",
                                 "value": float(np.median(val["wait"][m]) - np.median(val["go"][m]))})

    V = pd.DataFrame(rows)
    _table(V, os.path.join(TABLES, "theta_findings_per_region.csv"), index=False)

    # ── statistics: each region vs 0; region pairs (paired within session) ──
    S = []
    for (ref, meas), d in V.groupby(["ref", "measure"]):
        W = d.pivot_table(index="session", columns="region", values="value")
        alone = []
        for reg in REGS:
            v = W[reg].dropna().values if reg in W else np.array([])
            if len(v) >= 5:
                mm, p, n_ = fn.sign_flip_p(v)
                alone.append({"ref": ref, "measure": meas, "test": reg, "n_sessions": n_, "mean": mm, "p": p})
        for row, q in zip(alone, fdr([x["p"] for x in alone])):
            row["q"] = q
        pairs = []
        for a, b in RPAIRS:
            if a in W and b in W:
                x = (W[a] - W[b]).dropna()
                if len(x) >= 5:
                    mm, p, n_ = fn.sign_flip_p(x.values)
                    pairs.append({"ref": ref, "measure": meas, "test": f"{a} − {b}", "n_sessions": n_, "mean": mm, "p": p})
        for row, q in zip(pairs, fdr([x["p"] for x in pairs])):
            row["q"] = q
        S += alone + pairs
    S = pd.DataFrame(S)
    kind = dict((m, k) for m, _, k in MEAS)
    S["value"] = [round(float(pc(m)), 1) if kind[ms] == "pct" else round(float(m), 3) for m, ms in zip(S["mean"], S["measure"])]
    S = S.drop(columns="mean")
    _table(S, os.path.join(TABLES, "theta_findings_per_region_stats.csv"), index=False)

    # ── figure per reference: one panel per measure, regions with 95 % CI, brackets for region pairs ──
    for REF in ["CMR", "LOCAL"]:
        fig, axes = plt.subplots(1, len(MEAS), figsize=(13, 3.3), facecolor="white")
        for ax, (meas, ttl, k) in zip(axes, MEAS):
            d = V[(V["ref"] == REF) & (V["measure"] == meas)]; top, bot = 0.0, 0.0
            f = pc if k == "pct" else (lambda x: x)
            for i, reg in enumerate(REGS):
                v = d.loc[d["region"] == reg, "value"].dropna().values
                if len(v) < 3:
                    continue
                mu, ci = v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                y, lo, hi = f(mu), f(mu - ci), f(mu + ci)
                ax.errorbar(i, y, yerr=[[y - lo], [hi - y]], fmt="o", color=RCOLR[reg], ms=5, capsize=3, lw=1.2)
                top, bot = max(top, hi), min(bot, lo)
            span = (top - bot) or 1; step = 0.14 * span; pos = {r_: i for i, r_ in enumerate(REGS)}
            for j, (a, b) in enumerate(RPAIRS):
                hit = S[(S["ref"] == REF) & (S["measure"] == meas) & (S["test"] == f"{a} − {b}")]
                if hit.empty:
                    continue
                yb = top + (0.12 + j) * step; x0, x1 = pos[a], pos[b]
                ax.plot([x0, x0, x1, x1], [yb - 0.25 * step, yb, yb, yb - 0.25 * step], color="k", lw=0.8)
                ax.text((x0 + x1) / 2, yb + 0.05 * step, lab_(hit["q"].iloc[0]), ha="center", va="bottom", fontsize=6.5)
            ax.set_ylim(bot - 0.1 * span, top + (0.12 + len(RPAIRS) + 0.4) * step)
            ax.axhline(0, color="0.5", lw=0.8, ls="--")
            ax.set_xticks(range(len(REGS))); ax.set_xticklabels(REGS); ax.set_xlim(-0.5, len(REGS) - 0.5)
            ax.set_title(ttl, fontsize=7.5); ax.set_ylabel("%" if k == "pct" else "rho")
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
        fig.suptitle(f"Theta findings per region ({REF}; mean with 95 % CI across sessions; brackets: paired region differences, FDR over 3 pairs)",
                     fontsize=9, y=1.04)
        fig.tight_layout()
        _savefig(fig, os.path.join(reg_dir, f"theta_findings_per_region_{REF}.png"))
        plt.show()

    pd.set_option("display.width", 200)
    print("value: mean across sessions (% for log measures, rho for measure 5); p: sign-flip; q: FDR over the 3 regions / over the 3 pairs")
    for ref in ["CMR", "LOCAL"]:
        print(f"\n── {ref} ──")
        print(S[S["ref"] == ref].drop(columns="ref").round({"p": 4, "q": 4}).to_string(index=False))


@cell()
def theta_drop_before_lick():
    """Fig. S11. Theta drops after the release before the first lick (correct long, CMR)"""
    global BAND, LCOL, LK, MIN_CL, REF_L, X, a, ax, ax2, bins, c, cur, fig, fl, h, h1, h2, l1, l2, lick_dir, m, pc, r, s, se
    global show, sp_, sup_dir, tR, z
    # ── Fig. S11. Theta drops after the release before the first lick (correct long, CMR) ──
    REF_L, BAND, MIN_CL = "CMR", "theta 4-7", 10
    LCOL = {"correct": "#2CA02C", "late": "#DD8452", "early": "#E6B800"}
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    sup_dir = os.path.join(AMIR, "supplementary"); os.makedirs(sup_dir, exist_ok=True)
    lick_dir = os.path.join(AMIR, "Theta", "check licks"); os.makedirs(lick_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})

    # theta after the release: saved curves (vs own pre-press, only while on the lever before the release), median over channels
    cur, tR = [], None
    for s in done_sessions(REF_L):
        r = load_bp(s, REF_L)
        c = fn.session_curve(r, "Release", BAND, "correct long", min_trials=MIN_CL)
        if c is not None:
            cur.append(c); tR = r["times"]["Release"]
    X = np.array(cur)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m, se = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(X), 0))
    show = tR >= -0.5

    # first licks
    LK = pd.read_csv(os.path.join(TABLES, "licks_after_release_long.csv"), dtype={"rat": str})
    bins = np.arange(0, 2.0001, 0.05)

    fig, ax = plt.subplots(figsize=(4.6, 3.0), facecolor="white")
    a, z = fn.MAIN_WINDOWS["post"][1]
    ax.axvspan(a, z, color="#C0392B", alpha=0.10, lw=0)
    ax.text((a + z) / 2, 1.01, "post", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#C0392B")
    ax.fill_between(tR[show], pc(m - se)[show], pc(m + se)[show], color="0.2", alpha=0.2, lw=0)
    ax.plot(tR[show], pc(m)[show], color="0.1", lw=1.6, label=f"Theta, correct long ({len(X)} sessions)")
    ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="k", lw=0.9)
    ax.set_xlabel("Time from release (s)"); ax.set_ylabel("Theta power change\nfrom pre-press (%)")
    ax.set_xlim(-0.5, 2.0)

    ax2 = ax.twinx()
    for c in ["correct", "late", "early"]:
        fl = LK.loc[LK["label"] == c, "first_lick_s"].dropna().values
        h = 100 * np.histogram(fl, bins=bins)[0] / max(len(fl), 1)                  # % of trials with a lick, per 50 ms
        if c == "correct":
            ax2.bar(bins[:-1], h, width=0.05, align="edge", color=LCOL[c], alpha=0.35, lw=0,
                    label=f"First lick, correct (median {np.median(fl):.2f} s)")
        else:
            ax2.step(bins[:-1], h, where="post", color=LCOL[c], lw=1.2,
                     label=f"First lick, {c} (median {np.median(fl):.2f} s)")
    ax2.set_ylabel("First licks (% of trials per 50 ms)")
    for sp_ in ("top",):
        ax.spines[sp_].set_visible(False); ax2.spines[sp_].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper right")
    _savefig(fig, os.path.join(lick_dir, "theta_drop_before_first_lick_CMR.png"))
    _savefig(fig, os.path.join(sup_dir, "Fig S10 theta drop before first lick CMR.png"))
    plt.show()
    print(f"{len(X)} sessions; saved in: {sup_dir}")


@cell()
def first_lick_vs_window():
    """1j-b. First lick after the release vs the theta window after the release (0.2–0.5 s), long trials"""
    global LK, a, c, d, fl, lick, rows, z
    # ── 1j-b. First lick after the release vs the theta window after the release (0.2–0.5 s), long trials ──
    LK = pd.read_csv(os.path.join(TABLES, "licks_after_release_long.csv"), dtype={"rat": str})
    a, z = fn.MAIN_WINDOWS["post"][1]                                               # 0.2–0.5 s after the release
    rows = []
    for c in ["correct", "late", "early"]:
        d = LK[LK["label"] == c]; fl = d["first_lick_s"]
        lick = fl.dropna()
        rows.append({"outcome": c, "trials": len(d), "lick_%": round(100 * fl.notna().mean(), 1),
                     "median_s": round(lick.median(), 2),
                     "IQR_s": f"{lick.quantile(0.25):.2f}–{lick.quantile(0.75):.2f}",
                     "before_window_%": round(100 * (lick < a).mean(), 1),            # first lick < 0.2 s
                     "inside_window_%": round(100 * ((lick >= a) & (lick <= z)).mean(), 1),
                     "after_window_%": round(100 * (lick > z).mean(), 1)})           # first lick > 0.5 s
    print("first lick after the release (s), long trials; % columns = share of trials that licked")
    print(pd.DataFrame(rows).to_string(index=False))


@cell()
def theta_drop_vs_movement():
    """1k. THETA DROP vs MOVEMENT PROXY (high gamma), trial by trial within sessions, CMR and LOCAL"""
    global COLS, MIN_TR, MVt, Q, RCOL, REF_N, ST, T, TESTS, all_stats, ax, axA, axB, cl, col, fig, fmt_q, hg, i, icpt, is_pct
    global k, lab, lp, m, mov_dir, multipletests, n, name, ok, p, pc, q1, q2, r, rat, rho, rng, rows, s, slope, sp, spearmanr
    global st, th, v, vr
    # ── 1k. THETA DROP vs MOVEMENT PROXY (high gamma), trial by trial within sessions, CMR and LOCAL ──
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    MIN_TR = 10
    pc   = globals().get("pc", lambda x: 100 * (10 ** np.asarray(x, float) - 1))
    RCOL = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}
    mov_dir = os.path.join(AMIR, "Theta", "movement check"); os.makedirs(mov_dir, exist_ok=True)
    fmt_q = lambda x: "q < 0.001" if x < 0.001 else f"q = {x:.3f}"
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9})

    TESTS = [("theta_drop", "theta drop after the release (all trials)", True),
             ("hg_change", "high-gamma change after the release", True),
             ("rho", "Spearman rho (theta drop vs high-gamma change)", False),
             ("intercept", "theta drop at zero high-gamma change (intercept)", True),
             ("drop_low_hg", "theta drop, lowest third of high-gamma change", True),
             ("drop_high_hg", "theta drop, highest third of high-gamma change", True),
             ("high_minus_low", "highest − lowest third", True)]

    all_stats = []
    for REF_N in ["CMR", "LOCAL"]:
        # 1) per session: trial-by-trial theta drop vs high-gamma change (correct long, median over channels per trial)
        rows = []
        for s in done_sessions(REF_N):
            r = load_bp(s, REF_N); T, lp = r["trials"], r["logpow"]
            cl = ((T["label"] == "correct") & (T["type"] == "long")).values
            if cl.sum() < MIN_TR:
                continue
            th = np.median(lp["theta 4-7"]["post"][cl] - lp["theta 4-7"]["hold"][cl], axis=1)          # per trial
            hg = np.median(lp["highG 61-100"]["post"][cl] - lp["highG 61-100"]["hold"][cl], axis=1)
            ok = np.isfinite(th) & np.isfinite(hg); th, hg = th[ok], hg[ok]
            rho = spearmanr(th, hg)[0]
            slope, icpt = np.polyfit(hg, th, 1)                                                         # theta = icpt + slope * hg
            q1, q2 = np.quantile(hg, [1 / 3, 2 / 3])
            rows.append({"session": s, "rat": RAT[s], "n": len(th), "rho": rho, "slope": slope, "intercept": icpt,
                         "theta_drop": th.mean(), "hg_change": hg.mean(),
                         "drop_low_hg": th[hg <= q1].mean(), "drop_high_hg": th[hg >= q2].mean()})
        MVt = pd.DataFrame(rows)
        MVt["high_minus_low"] = MVt["drop_high_hg"] - MVt["drop_low_hg"]
        MVt.to_csv(os.path.join(TABLES, f"theta_drop_vs_movement_{REF_N}.csv"), index=False)

        # 2) across sessions: sign-flip, FDR over the 7 tests
        st = []
        for col, name, is_pct in TESTS:
            v = MVt[col].dropna(); m, p, n = fn.sign_flip_p(v.values)
            st.append({"ref": REF_N, "test": name, "col": col, "n_sessions": n, "below_0": int((v < 0).sum()),
                       "mean": round(float(pc(m)), 1) if is_pct else round(m, 3), "unit": "%" if is_pct else "rho", "p": p})
        ST = pd.DataFrame(st); ST["q"] = multipletests(ST["p"], method="fdr_bh")[1]
        all_stats.append(ST)
        Q = dict(zip(ST["col"], ST["q"]))
        print(f"\n===== correct long, {REF_N}, {len(MVt)} sessions (FDR over {len(ST)} tests) =====")
        print(ST.drop(columns="col").round({"p": 4, "q": 4}).to_string(index=False))
        print("per rat (intercept):", {rat: f"{pc(d['intercept'].mean()):+.1f} % ({int((d['intercept'] < 0).sum())}/{len(d)} < 0)"
                                       for rat, d in MVt.groupby("rat")})

        # 3) figure: A theta drop overall / at zero movement / by movement third; B rho per session
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 5.2), facecolor="white", gridspec_kw={"width_ratios": [2.4, 1]})
        fig.subplots_adjust(top=0.72, bottom=0.16, wspace=0.35)
        rng = np.random.default_rng(0)
        COLS = [("theta_drop", "all trials"), ("intercept", "at zero\nhigh-gamma change"),
                ("drop_low_hg", "lowest third\nhigh-gamma change"), ("drop_high_hg", "highest third\nhigh-gamma change")]
        for i, (col, lab) in enumerate(COLS):
            v = pc(MVt[col])
            for k, rat in enumerate(sorted(RCOL)):
                vr = v[MVt["rat"] == rat]
                axA.scatter(i + (k - 1) * 0.10 + rng.uniform(-0.03, 0.03, len(vr)), vr, s=16, color=RCOL[rat], alpha=0.75,
                            lw=0, label=f"Rat {rat}" if i == 0 else None)
            m = fn.sign_flip_p(MVt[col].values)[0]
            axA.errorbar(i + 0.28, pc(m), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", color="k", ms=6, capsize=3, lw=1.4, zorder=4)
            axA.text(i, 1.01, f"{pc(m):+.0f} %\n{int((MVt[col] < 0).sum())}/{len(MVt)} down\n{fmt_q(Q[col])}",
                     transform=axA.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)
        axA.axhline(0, color="0.3", lw=0.8, ls="--")
        axA.set_xticks(range(len(COLS))); axA.set_xticklabels([l for _, l in COLS], fontsize=8.5); axA.set_xlim(-0.5, len(COLS) - 0.4)
        axA.set_ylabel("Theta after vs during the hold (%)\none dot = one session")
        axA.set_title("A  Theta drop with and without movement", fontsize=10.5, fontweight="bold", pad=50, loc="left")
        axA.legend(frameon=False, fontsize=8, loc="lower left")

        v = MVt["rho"]
        axB.scatter(rng.uniform(-0.12, 0.12, len(v)), v, s=16, color=[RCOL[x] for x in MVt["rat"]], alpha=0.75, lw=0)
        axB.errorbar(0.28, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", color="k", ms=6, capsize=3, lw=1.4, zorder=4)
        axB.text(0, 1.01, f"rho {v.mean():+.2f}\n{int((v < 0).sum())}/{len(v)} < 0\n{fmt_q(Q['rho'])}",
                 transform=axB.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)
        axB.axhline(0, color="0.3", lw=0.8, ls="--")
        axB.set_xticks([0]); axB.set_xticklabels(["theta drop vs\nhigh-gamma change"], fontsize=8.5); axB.set_xlim(-0.5, 0.6)
        axB.set_ylabel("Spearman rho (trials within a session)")
        axB.set_title("B  Correlation", fontsize=10.5, fontweight="bold", pad=50, loc="left")
        for ax in (axA, axB):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle(f"Is the theta drop after the release explained by movement? ({REF_N}, correct long, {len(MVt)} sessions)",
                     fontsize=11, fontweight="bold", y=0.98)
        fig.text(0.5, 0.01, "Movement proxy: high gamma (61–100 Hz) after vs during the hold, per trial. "
                 f"Highest − lowest third: {pc(MVt['high_minus_low'].mean()):+.1f} % ({fmt_q(Q['high_minus_low'])}). "
                 "Sign-flip across sessions, FDR over 7 tests.", ha="center", va="bottom", fontsize=8.5)
        _savefig(fig, os.path.join(mov_dir, f"theta_drop_vs_movement_{REF_N}.png"))
        plt.show()

    pd.concat(all_stats).drop(columns="col").to_csv(os.path.join(mov_dir, "theta_drop_vs_movement_stats_CMR_LOCAL.csv"), index=False)


@cell()
def theta_drop_movement_figure():
    """Fig. S12. Theta drop after the release with and without movement (high gamma as proxy), correct long, CMR"""
    global COLS, MV, RAT_NAME, RCOL, REF_M, ax, col, fig, hi, i, k, lab, lo, mu, pc, rat, rng, se, sp_, stats, sup_dir, tc, v
    global vr
    # ── Fig. S12. Theta drop after the release with and without movement (high gamma as proxy), correct long, CMR ──
    from scipy import stats
    REF_M = "CMR"
    MV = pd.read_csv(os.path.join(TABLES, f"theta_drop_vs_movement_{REF_M}.csv"), dtype={"rat": str})   # saved by cell 1k
    COLS = [("theta_drop", "All trials"), ("intercept", "No high-gamma\nchange"),
            ("drop_low_hg", "Lowest third\nhigh gamma"), ("drop_high_hg", "Highest third\nhigh gamma")]
    RCOL = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}
    RAT_NAME = {"134": "Rat 1", "135": "Rat 2", "136": "Rat 3"}
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    sup_dir = os.path.join(AMIR, "supplementary"); os.makedirs(sup_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})

    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor="white")
    rng = np.random.default_rng(0)
    for i, (col, lab) in enumerate(COLS):
        v = MV[col].values
        for k, rat in enumerate(sorted(RCOL)):
            vr = pc(v[MV["rat"].values == rat])
            ax.scatter(i + (k - 1) * 0.09 + rng.uniform(-0.02, 0.02, len(vr)), vr, s=10, color=RCOL[rat],
                       edgecolor="0.35", lw=0.3, zorder=2, label=RAT_NAME[rat] if i == 0 else None)
        se, tc = v.std(ddof=1) / np.sqrt(len(v)), stats.t.ppf(0.975, len(v) - 1)
        mu, lo, hi = pc(v.mean()), pc(v.mean() - tc * se), pc(v.mean() + tc * se)
        ax.errorbar(i + 0.25, mu, yerr=[[mu - lo], [hi - mu]], fmt="o", color="k", ms=4.5, capsize=3, lw=1.2, zorder=4)
        ax.text(i, 1.01, f"{mu:+.0f} %", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([l for _, l in COLS], fontsize=7.5); ax.set_xlim(-0.5, len(COLS) - 0.4)
    ax.set_ylabel("Theta power after vs\nduring the hold (%)")
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1)
    _savefig(fig, os.path.join(sup_dir, "Fig S11 theta drop vs movement CMR.png"))
    plt.show()
    print(f"{len(MV)} sessions; saved in: {sup_dir}")


@cell()
def theta_outcome_rt():
    """1l. THETA DOES NOT PREDICT THE OUTCOME OR THE REACTION TIME, CMR and LOCAL"""
    global BAND, CND, DQ, HOLD_FULL, L, LABS, MIN_RT, MIN_TR, Q, RCOL, REF_Q, RQ, T, W, ax, axA, axB, c, cl, col, fdr, fig
    global fmt_q, hold, k, lab, lng, m, masks, mk, mr, multipletests, n, name, p, pc, r, rhos, rng, rows, rt, rt_dir, s, sp_
    global spearmanr, sub, summary_rows, tests, th, v
    # ── 1l. THETA DOES NOT PREDICT THE OUTCOME OR THE REACTION TIME, CMR and LOCAL ──
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    fdr = lambda p: multipletests(np.asarray(p, float), alpha=0.05, method="fdr_bh")[1]
    pc  = globals().get("pc", lambda x: 100 * (10 ** np.asarray(x, float) - 1))
    BAND, MIN_TR, MIN_RT = "theta 4-7", 5, 10
    HOLD_FULL = fn.MAIN_WINDOWS["hold"][1][1]                  # 0.85 s
    CND  = {"correct": "#2CA02C", "late": "#DD8452", "early": "#E6B800"}
    LABS = {"correct": "correct", "late": "late", "early": f"early\n(held ≥ {HOLD_FULL} s)"}
    RCOL = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}
    rt_dir = os.path.join(AMIR, "Theta", "reaction time"); os.makedirs(rt_dir, exist_ok=True)
    fmt_q = lambda x: "q < 0.001" if x < 0.001 else f"q = {x:.3f}"
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9, "font.family": "sans-serif"})

    summary_rows = []
    for REF_Q in ["CMR", "LOCAL"]:
        rows, rhos = [], []
        for s in done_sessions(REF_Q):
            r = load_bp(s, REF_Q); T, L = r["trials"], r["logpow"][BAND]
            lng, lab, hold = (T["type"] == "long").values, T["label"].values, T["hold_s"].values
            masks = {"correct": lng & (lab == "correct"), "late": lng & (lab == "late"),
                     "early": lng & (lab == "early") & (hold >= HOLD_FULL)}
            for c, mk in masks.items():
                if mk.sum() >= MIN_TR:
                    rows.append({"session": s, "rat": RAT[s], "cond": c, "rise": float(np.median((L["hold"][mk] - L["pre"][mk]).mean(0)))})
            cl = masks["correct"] & np.isfinite(T["tone_delay_s"].values)
            if cl.sum() >= MIN_RT:
                th = np.median(L["pre_tone"][cl] - L["pre"][cl], axis=1)                # pre-tone theta per trial
                rt = (T["hold_s"] - T["tone_delay_s"]).values[cl]                       # reaction time = release − tone
                rhos.append({"session": s, "rat": RAT[s], "rho": spearmanr(th, rt, nan_policy="omit")[0]})
        DQ, RQ = pd.DataFrame(rows), pd.DataFrame(rhos)
        _table(DQ, os.path.join(TABLES, f"theta_rise_by_outcome_{REF_Q}.csv"), index=False)
        _table(RQ, os.path.join(TABLES, f"theta_pretone_rt_rho_{REF_Q}.csv"), index=False)
        W = DQ.pivot(index="session", columns="cond", values="rise")

        # tests: late − correct, early − correct (paired), rho vs 0; FDR over the 3
        tests = {f"{c} − correct": fn.sign_flip_p((W[c] - W["correct"]).dropna().values) for c in ["late", "early"]}
        tests["rho (pre-tone theta vs RT)"] = fn.sign_flip_p(RQ["rho"].dropna().values)
        Q = dict(zip(tests, fdr([t[1] for t in tests.values()])))
        for name, (m, p, n) in tests.items():
            summary_rows.append({"ref": REF_Q, "test": name, "n_sessions": n,
                                 "mean": round(float(m), 3) if name.startswith("rho") else round(float(pc(m)), 1),
                                 "unit": "rho" if name.startswith("rho") else "%", "p": p, "q": Q[name]})
        print(f"{REF_Q}  rise during the hold:", {c: f"{pc(W[c].mean()):+.1f} % (n={W[c].notna().sum()})" for c in CND})

        # figure
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 5.2), facecolor="white", gridspec_kw={"width_ratios": [1.4, 1]})
        fig.subplots_adjust(top=0.72, bottom=0.12, wspace=0.4)
        rng = np.random.default_rng(0)
        for k, (c, col) in enumerate(CND.items()):
            v = pc(W[c].dropna())
            axA.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=18, color=col, alpha=0.75, lw=0)
            axA.errorbar(k + 0.28, pc(W[c].dropna().mean()), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", color="k", ms=6, capsize=3, lw=1.4)
            sub = f"{pc(W[c].mean()):+.0f} %\nn = {len(v)}" + ("" if c == "correct" else f"\nvs correct {fmt_q(Q[f'{c} − correct'])}")
            axA.text(k, 1.01, sub, transform=axA.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)
        axA.axhline(0, color="0.3", lw=0.8, ls="--")
        axA.set_xticks(range(3)); axA.set_xticklabels([LABS[c] for c in CND], fontsize=9); axA.set_xlim(-0.5, 2.6)
        axA.set_ylabel("Theta during the hold vs before the press (%)\none dot = one session")
        axA.set_title("A  Same rise before every outcome", fontsize=10.5, fontweight="bold", loc="left", pad=50)

        v = RQ["rho"].dropna(); mr = tests["rho (pre-tone theta vs RT)"][0]
        axB.scatter(rng.uniform(-0.1, 0.1, len(v)), v, s=18, color=[RCOL[x] for x in RQ.loc[v.index, "rat"]], alpha=0.75, lw=0)
        axB.errorbar(0.25, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", color="k", ms=6, capsize=3, lw=1.4)
        axB.text(0, 1.01, f"mean rho {mr:+.2f}\n{int((v > 0).sum())}/{len(v)} positive\n{fmt_q(Q['rho (pre-tone theta vs RT)'])}",
                 transform=axB.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)
        axB.axhline(0, color="0.3", lw=0.8, ls="--"); axB.set_xlim(-0.5, 0.6); axB.set_xticks([])
        axB.set_ylabel("Spearman rho (pre-tone theta vs RT)\none dot = one session")
        axB.set_title("B  No link to reaction time", fontsize=10.5, fontweight="bold", loc="left", pad=50)
        for ax in (axA, axB):
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
        fig.suptitle(f"Theta does not predict the outcome or the reaction time ({REF_Q})", fontsize=11, fontweight="bold", y=0.98)
        fig.text(0.5, 0.01, "Sign-flip across sessions; FDR over 3 tests (late − correct, early − correct, rho). "
                 "B: correct long trials; pre-tone = 0.3 s before the tone; RT = release − tone.", ha="center", va="bottom", fontsize=8)
        _savefig(fig, os.path.join(rt_dir, f"theta_outcome_rt_{REF_Q}.png"))
        plt.show()

    print("\nCMR vs LOCAL (FDR over 3 tests per reference)")
    print(pd.DataFrame(summary_rows).round({"p": 4, "q": 4}).to_string(index=False))


@cell()
def theta_wait_vs_go():
    """1m. THETA WAIT vs GO, paper figure (Fig. 3B: CMR; LOCAL kept for the supplement), from the saved maps"""
    global C_GO, C_WAIT, FREQS, MG, MW, REF_G, REPORT_NAME, ax, col, early, f, fb, fig, g_in, idx, lab, m, multipletests, p1
    global p2, pc, q1, q2, qq, report_dir, se, segs, segs_txt, sg, sig, sp_, sup_dir, t1, t2, thg, thw, tt, ttest_1samp, used
    global v, w_in, wg_dir, x, y0, y1, yb, z
    # ── 1m. THETA WAIT vs GO, paper figure (Fig. 3B: CMR; LOCAL kept for the supplement), from the saved maps ──
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    C_WAIT, C_GO = "#2CA02C", "#C62828"                 # wait = correct long (green), go = correct short (red), as in Fig. 3A
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    wg_dir = os.path.join(AMIR, "Theta", "wait vs go"); os.makedirs(wg_dir, exist_ok=True)
    report_dir = os.path.join(AMIR, "report"); os.makedirs(report_dir, exist_ok=True)
    REPORT_NAME = {"CMR": "Fig 3B wait vs go theta CMR.png"}                       # LOCAL: supplementary folder later
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif",
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8})

    for REF_G in ["CMR", "LOCAL"]:
        f = os.path.join(TABLES, f"wait_vs_go_maps_{REF_G}.npz")
        if not os.path.exists(f):
            print(f"{REF_G}: {f} not found -> run the wait vs go maps cell with REF = '{REF_G}' first"); continue
        z = np.load(f, allow_pickle=True)
        MW, MG, FREQS, tt, used = z["wait"], z["go"], z["freqs"], z["times"], z["sessions"]

        fb = (FREQS >= 4) & (FREQS <= 7)
        thw, thg = MW[:, fb].mean(1), MG[:, fb].mean(1)                              # (sessions, times), log10 vs pre-press
        t1, p1 = ttest_1samp(thw, 0, axis=0);       q1 = multipletests(p1, method="fdr_bh")[1]
        t2, p2 = ttest_1samp(thw - thg, 0, axis=0); q2 = multipletests(p2, method="fdr_bh")[1]
        sig = (q1 < 0.05) & (t1 > 0) & (q2 < 0.05) & (t2 > 0)                        # wait > baseline AND wait > go
        qq  = np.maximum(q1, q2)                                                     # a point needs both tests -> larger q

        fig, ax = plt.subplots(figsize=(4.5, 3.2), facecolor="white")                # ~11.5 x 8 cm
        for v, col, lab in ((thw, C_WAIT, "Wait (correct long)"), (thg, C_GO, "Go (correct short)")):
            m, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))
            ax.fill_between(tt, pc(m - se), pc(m + se), color=col, alpha=0.25, lw=0)
            ax.plot(tt, pc(m), color=col, lw=1.6, label=lab)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="k", lw=0.9)
        ax.text(0, 1.01, "press", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        for x, col, lab in ((0.35, C_GO, "short tone"), (1.05, C_WAIT, "long tone")):
            ax.axvline(x, color=col, ls="--", lw=1)
            ax.text(x, 1.01, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=col)
        if sig.any():                                                                # significance bar (contiguous segments)
            y0, y1 = ax.get_ylim(); yb = y1 + 0.03 * (y1 - y0)
            idx = np.where(sig)[0]; segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for sg in segs:
                ax.plot([tt[sg[0]], tt[sg[-1]]], [yb, yb], color="k", lw=3, solid_capstyle="butt")
            ax.text(tt[segs[0][0]], yb + 0.02 * (y1 - y0), "q < 0.05", ha="left", va="bottom", fontsize=7.5)
            ax.set_ylim(top=yb + 0.14 * (y1 - y0))
        ax.set_xlim(tt[0], tt[-1])
        ax.set_xlabel("Time from press (s)")
        ax.set_ylabel("Theta power change from pre-press (%)")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left", handlelength=1.6)
        if REF_G == "CMR":
            ax.text(-0.17, 1.10, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
        _savefig(fig, os.path.join(wg_dir, f"wait_vs_go_paper_{REF_G}.png"))
        if REF_G == "LOCAL":
            sup_dir = os.path.join(AMIR, "supplementary"); os.makedirs(sup_dir, exist_ok=True)
            _savefig(fig, os.path.join(sup_dir, "Fig S8 theta wait vs go LOCAL.png"))
        if REF_G in REPORT_NAME:
            _savefig(fig, os.path.join(report_dir, REPORT_NAME[REF_G]))
        plt.show()

        # numbers for the results text
        print(f"\n{REF_G}: {len(used)} sessions")
        if sig.any():
            segs_txt = ", ".join(f"{tt[sg[0]]:.2f}–{tt[sg[-1]]:.2f} s" for sg in segs)
            w_in, g_in = thw[:, sig].mean(1), thg[:, sig].mean(1)                    # per session, mean inside the significant points
            print(f"  significant (wait > pre AND wait > go): {segs_txt}; q inside: {qq[sig].min():.1e} to {qq[sig].max():.3f}")
            print(f"  inside: wait {pc(w_in.mean()):+.1f} %, go {pc(g_in.mean()):+.1f} % vs pre-press; "
                  f"wait > go in {int((w_in > g_in).sum())}/{len(w_in)} sessions")
        else:
            print("  no significant time point")
        early = (tt >= 0.2) & (tt <= 0.35)                                           # both trial types still identical
        print(f"  0.20–0.35 s (same in both): wait {pc(thw[:, early].mean()):+.1f} %, go {pc(thg[:, early].mean()):+.1f} %")


@cell()
def theta_course_correct_vs_errors():
    """1n. THETA TIME COURSE: correct vs early and correct vs late long (black bar = significant time points), CMR and LOCAL"""
    global BAND, C, D, G2, GRP, LABELS, MIN_TR, OTHER, P, Q, REF_T, SHOW, TT, X, _, a, al, allq, ax, axes, ce_dir, cm, col
    global cur, fig, g, i0, idx, k, keep, lab, m, multipletests, n, n_s, name, ok, pc, r, s, se, segs, sg, sig, sp_
    global ttest_1samp, used, xl, yb, yhi, ylo, z
    # ── 1n. THETA TIME COURSE: correct vs early and correct vs late long (black bar = significant time points), CMR and LOCAL ──
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    BAND, MIN_TR = "theta 4-7", 5
    GRP = {"correct": ("#2CA02C", "Correct long", "correct long"),
           "early":   ("#E6B800", "Early long", "early long (hold>0.3)"),
           "late":    ("#DD8452", "Late long", "late long")}
    SHOW = {"Press": (-0.7, 1.2), "Release": (-1.0, 0.7)}
    LABELS = {"Press": ["pre", "hold"], "Release": ["post"]}
    pc = globals().get("pc", lambda x: 100 * (10 ** np.asarray(x, float) - 1))
    ce_dir = os.path.join(AMIR, "Theta", "correct vs error"); os.makedirs(ce_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1, "font.family": "sans-serif"})

    for REF_T in ["CMR", "LOCAL"]:
        for OTHER in ["early", "late"]:
            G2 = ["correct", OTHER]
            # 1) paired session curves (sessions with both groups)
            C = {al: {g: [] for g in G2} for al in SHOW}; TT = {}; used = []
            for s in done_sessions(REF_T):
                r = load_bp(s, REF_T); cm = fn.main_conditions(r["trials"])
                n = {g: int(cm[GRP[g][2]].sum()) for g in G2}
                if min(n.values()) < MIN_TR:
                    continue
                used.append(s)
                for al in SHOW:
                    TT.setdefault(al, r["times"][al])
                    for g in G2:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            cur = np.nanmedian(r["curves"][al][BAND][GRP[g][2]], axis=0)
                        C[al][g].append(np.where(r["counts"][al][GRP[g][2]] >= max(3, 0.5 * n[g]), cur, np.nan))
            C = {al: {g: np.array(v) for g, v in d.items()} for al, d in C.items()}

            # 2) paired t-test (other − correct) at every time point, FDR over all shown time points of both panels
            P, keep = {}, {}
            for al in SHOW:
                D = C[al][OTHER] - C[al]["correct"]
                keep[al] = (np.sum(np.isfinite(D), 0) >= max(5, 0.5 * len(used))) & (TT[al] >= SHOW[al][0]) & (TT[al] <= SHOW[al][1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    P[al] = ttest_1samp(D, 0, axis=0, nan_policy="omit")[1]
            allq = multipletests(np.concatenate([P[al][keep[al]] for al in SHOW]), method="fdr_bh")[1]
            Q, i0 = {}, 0
            for al in SHOW:
                Q[al] = np.full(len(TT[al]), np.nan); k = keep[al].sum(); Q[al][keep[al]] = allq[i0:i0 + k]; i0 += k

            # 3) figure
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, facecolor="white", gridspec_kw={"width_ratios": [1.9, 1.7], "wspace": 0.08})
            for ax, al, xl in zip(axes, SHOW, ["Time from press (s)", "Time from release (s)"]):
                for g in G2:
                    col, lab, _ = GRP[g]
                    X = C[al][g]; n_s = np.sum(np.isfinite(X), 0)
                    ok = (n_s >= max(3, 0.5 * len(X))) & (TT[al] >= SHOW[al][0]) & (TT[al] <= SHOW[al][1])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        m, se = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1) / np.sqrt(n_s)
                    ax.fill_between(TT[al][ok], pc(m - se)[ok], pc(m + se)[ok], color=col, alpha=0.2, lw=0)
                    ax.plot(TT[al][ok], pc(m)[ok], color=col, lw=2.4, label=lab)
                for name in LABELS[al]:                                       # window labels only, no shading
                    a, z = fn.MAIN_WINDOWS[name][1]
                    ax.text((a + z) / 2, 1.01, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=11, fontweight="bold")
                ax.axhline(0, color="0.5", lw=0.9); ax.axvline(0, color="k", lw=1.2)
                if al == "Press":
                    ax.axvline(1.05, color="0.4", ls=":", lw=1.2); ax.text(1.05, 0.02, " long tone", transform=ax.get_xaxis_transform(), fontsize=9, color="0.4")
                ax.set_xlim(*SHOW[al]); ax.set_xlabel(xl)
                for sp_ in ("top", "right"):
                    ax.spines[sp_].set_visible(False)
            ylo, yhi = axes[0].get_ylim(); yb = yhi + 0.04 * (yhi - ylo)
            for ax, al in zip(axes, SHOW):                                    # black bar where other ≠ correct (q < 0.05)
                sig = Q[al] < 0.05
                if sig.any():
                    idx = np.where(sig)[0]; segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                    for sg in segs:
                        ax.plot([TT[al][sg[0]], TT[al][sg[-1]]], [yb, yb], color="k", lw=4, solid_capstyle="butt")
                    print(f"{REF_T} | {OTHER} vs correct, {al}-aligned: q < 0.05 at " +
                          ", ".join(f"{TT[al][sg[0]]:.2f}–{TT[al][sg[-1]]:.2f} s" for sg in segs))
                else:
                    print(f"{REF_T} | {OTHER} vs correct, {al}-aligned: no significant time point")
            axes[0].set_ylim(ylo, yb + 0.06 * (yhi - ylo))
            axes[0].set_ylabel("Theta (4–7 Hz) power change (%)\nvs own pre-press")
            axes[1].legend(frameon=False, fontsize=10, loc="lower left")
            fig.suptitle(f"Theta in correct vs {OTHER} long trials ({len(used)} sessions, {REF_T}; mean ± SEM; black bar: FDR q < 0.05)",
                         fontsize=12.5, fontweight="bold", y=1.04)
            _savefig(fig, os.path.join(ce_dir, f"theta_course_correct_vs_{OTHER}_{REF_T}.png"))
            plt.show()


@cell()
def theta_correct_vs_errors_vs_pre():
    """1n. THETA: correct vs errors (late + early), everything vs before the press; CMR and LOCAL"""
    global BAND, COL, CURVE_COND, EARLY_POST, GRP, HOLD_FULL, L, MEAS, MIN_TR, NAME, R, REF_T, REPORT_NAME, SHOW, T, TESTS_FU
    global TESTS_MAIN, TT, V, VAL, X, XPOS, _, a_, al, all_rows, alone, ax, axA, bot, c, ce_dir, check_rows, checks, ci, cm
    global cur, curves, fdr, fig, fu, g, gs, gsA, hi, j, k, lab_, lbl, lo, m, main, masks_for, mcol, mh, mp, mu, multipletests
    global n3, n_, n_s, name, ok, p, paired, pc, q, r, report_dir, row, rows_, s, se, sp_, span, stats, step, sup_dir, top
    global ttl, v, x0, x1, xl, y, yb, z_
    # ── 1n. THETA: correct vs errors (late + early), everything vs before the press; CMR and LOCAL ──
    #    hold − pre: early held >= 0.85 s (hold window fully on the lever); post − pre: early held > 0.3 s
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    BAND, MIN_TR, HOLD_FULL, EARLY_POST = "theta 4-7", 5, fn.MAIN_WINDOWS["hold"][1][1], 0.3
    GRP  = ["correct", "errors", "late", "early"]
    COL  = {"correct": "#2CA02C", "errors": "#C0392B", "late": "#DD8452", "early": "#E6B800"}
    NAME = {"correct": "Correct", "errors": "Errors", "late": "Late", "early": "Early"}
    MEAS = [("hold", "Hold vs before the press\n(early held ≥ 0.85 s)"), ("post", "After release vs before the press\n(early held > 0.3 s)")]
    TESTS_MAIN = [("correct", "errors")]
    TESTS_FU   = [("correct", "late"), ("correct", "early")]
    SHOW = {"Press": (-0.7, 1.2), "Release": (-1.0, 0.7)}
    CURVE_COND = {"correct": "correct long", "late": "late long", "early": "early long (hold>0.3)"}
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    fdr = lambda p: multipletests(np.asarray(p, float), method="fdr_bh")[1]
    lab_ = lambda q: "q < 0.001" if q < 0.001 else "q < 0.01" if q < 0.01 else "q < 0.05" if q < 0.05 else "n.s."
    ce_dir = os.path.join(AMIR, "Theta", "correct vs error"); os.makedirs(ce_dir, exist_ok=True)
    report_dir = os.path.join(AMIR, "report"); os.makedirs(report_dir, exist_ok=True)
    REPORT_NAME = {"CMR": "Fig 4 theta correct vs errors CMR.png"}                 # LOCAL: supplementary folder later
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif",
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8})

    def masks_for(T, early_min, strict):
        lng, lab, hold = (T["type"] == "long").values, T["label"].values, T["hold_s"].values
        e = lng & (lab == "early") & ((hold >= early_min) if strict else (hold > early_min))
        m = {"correct": lng & (lab == "correct"), "late": lng & (lab == "late"), "early": e}
        m["errors"] = m["late"] | m["early"]
        return m

    all_rows, check_rows = [], []
    for REF_T in ["CMR", "LOCAL"]:
        curves = {al: {g: [] for g in CURVE_COND} for al in SHOW}; TT = {}; VAL = []
        for s in done_sessions(REF_T):
            r = load_bp(s, REF_T); T = r["trials"]; L = r["logpow"][BAND]; cm = fn.main_conditions(T)
            # A) saved time courses (illustration): sessions with >= MIN_TR in all three
            n3 = {g: int(cm[c].sum()) for g, c in CURVE_COND.items()}
            if min(n3.values()) >= MIN_TR:
                for al in SHOW:
                    TT.setdefault(al, r["times"][al])
                    for g, c in CURVE_COND.items():
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            cur = np.nanmedian(r["curves"][al][BAND][c], axis=0)
                        curves[al][g].append(np.where(r["counts"][al][c] >= max(3, 0.5 * n3[g]), cur, np.nan))
            # B) one value per session and group, as fn.m1_session (per channel mean over trials, median over channels)
            mh = masks_for(T, HOLD_FULL, strict=True)                                # for hold − pre
            mp = masks_for(T, EARLY_POST, strict=False)                              # for post − pre
            for g in GRP:
                row = {"session": s, "rat": RAT[s], "group": g}
                if mh[g].sum() >= MIN_TR:
                    row["hold"] = float(np.median((L["hold"][mh[g]] - L["pre"][mh[g]]).mean(0))); row["n_hold"] = int(mh[g].sum())
                    row["post_strict"] = float(np.median((L["post"][mh[g]] - L["pre"][mh[g]]).mean(0)))   # check
                if mp[g].sum() >= MIN_TR:
                    row["post"] = float(np.median((L["post"][mp[g]] - L["pre"][mp[g]]).mean(0))); row["n_post"] = int(mp[g].sum())
                    row["pre_raw"] = float(np.median(L["pre"][mp[g]].mean(0)))                              # check
                VAL.append(row)
        V = pd.DataFrame(VAL)
        curves = {al: {g: np.array(v) for g, v in d.items()} for al, d in curves.items()}

        # each group alone vs 0
        alone = []
        for mcol, _ in MEAS:
            for g in GRP:
                v = V.loc[V["group"] == g, mcol].dropna().values; m, p, n_ = fn.sign_flip_p(v)
                alone.append({"ref": REF_T, "measure": f"{mcol} − pre", "group": g, "n_sessions": n_,
                              "sessions_up": int((v > 0).sum()), "mean_%": round(float(pc(m)), 1), "p": p})
        for row, q in zip(alone, fdr([x["p"] for x in alone])):
            row["q"] = q

        # comparisons: paired within session; FDR separately for the main (2) and follow-up (4) tests
        def paired(mcol, a, b):
            piv = V.pivot(index="session", columns="group", values=mcol)
            d = (piv[a] - piv[b]).dropna(); m, p, n_ = fn.sign_flip_p(d.values)
            return {"ref": REF_T, "measure": f"{mcol} − pre" if mcol in ("hold", "post", "post_strict") else mcol,
                    "pair": f"{a} − {b}", "a": a, "b": b, "mcol": mcol, "n_sessions": n_,
                    "mean_a_%": round(float(pc(piv.loc[d.index, a].mean())), 1),
                    "mean_b_%": round(float(pc(piv.loc[d.index, b].mean())), 1),
                    "diff_%": round(float(pc(m)), 1), "sessions_a_higher": int((d > 0).sum()), "p": p}
        main = [paired(mcol, a, b) for mcol, _ in MEAS for a, b in TESTS_MAIN]
        fu   = [paired(mcol, a, b) for mcol, _ in MEAS for a, b in TESTS_FU]
        for rows_ in (main, fu):
            for row, q in zip(rows_, fdr([x["p"] for x in rows_])):
                row["q"] = q
        for row in main: row["family"] = "main"
        for row in fu:   row["family"] = "follow-up"
        checks = [paired(c, a, b) for c in ("post_strict", "pre_raw") for a, b in TESTS_MAIN + TESTS_FU]
        for row in checks: row["family"] = "check (raw p)"
        all_rows += main + fu; check_rows += checks

        # figure
        fig = plt.figure(figsize=(7.2, 5.8), facecolor="white")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.6, wspace=0.3)
        gsA = gs[0, :].subgridspec(1, 2, width_ratios=[1.9, 1.7], wspace=0.08)
        axA = [fig.add_subplot(gsA[0, 0])]; axA.append(fig.add_subplot(gsA[0, 1], sharey=axA[0]))
        for ax, al, xl in zip(axA, SHOW, ["Time from press (s)", "Time from release (s)"]):
            for g in CURVE_COND:
                X = curves[al][g]; n_s = np.sum(np.isfinite(X), 0)
                ok = (n_s >= max(3, 0.5 * len(X))) & (TT[al] >= SHOW[al][0]) & (TT[al] <= SHOW[al][1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    m, se = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1) / np.sqrt(n_s)
                lbl = NAME[g] + " long"
                ax.fill_between(TT[al][ok], pc(m - se)[ok], pc(m + se)[ok], color=COL[g], alpha=0.2, lw=0)
                ax.plot(TT[al][ok], pc(m)[ok], color=COL[g], lw=1.5, label=lbl)
            for name in (["pre", "hold"] if al == "Press" else ["post"]):
                a_, z_ = fn.MAIN_WINDOWS[name][1]
                ax.axvspan(a_, z_, color="#C0392B", alpha=0.10, lw=0)
                ax.text((a_ + z_) / 2, 1.01, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#C0392B")
            ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="k", lw=0.9)
            if al == "Press":
                ax.axvline(1.05, color="0.4", ls=":", lw=1)
            ax.set_xlim(*SHOW[al]); ax.set_xlabel(xl)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
        plt.setp(axA[1].get_yticklabels(), visible=False)
        axA[0].set_ylabel("Theta power change\nfrom pre-press (%)")
        axA[1].legend(frameon=False, fontsize=7, loc="lower left")
        axA[0].text(-0.2, 1.12, "A", transform=axA[0].transAxes, fontsize=12, fontweight="bold")

        XPOS = {g: i for i, g in enumerate(GRP)}
        for k, (mcol, ttl) in enumerate(MEAS):
            ax = fig.add_subplot(gs[1, k]); top, bot = 0.0, 0.0
            for g in GRP:
                v = V.loc[V["group"] == g, mcol].dropna().values
                mu, ci = v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                y, lo, hi = pc(mu), pc(mu - ci), pc(mu + ci)
                ax.errorbar(XPOS[g], y, yerr=[[y - lo], [hi - y]], fmt="o", color=COL[g], ms=5, capsize=3, lw=1.2)
                top, bot = max(top, hi), min(bot, lo)
            span = top - bot; step = 0.13 * span
            for j, row in enumerate([x for x in main + fu if x["mcol"] == mcol]):    # brackets: main first, then follow-up
                yb = top + (0.12 + j) * step; x0, x1 = XPOS[row["a"]], XPOS[row["b"]]
                ax.plot([x0, x0, x1, x1], [yb - 0.25 * step, yb, yb, yb - 0.25 * step], color="k", lw=0.8)
                ax.text((x0 + x1) / 2, yb + 0.05 * step, lab_(row["q"]), ha="center", va="bottom", fontsize=7)
            ax.set_ylim(bot - 0.1 * span, top + (0.12 + 3 + 0.4) * step)
            ax.axhline(0, color="0.5", lw=0.8, ls="--")
            ax.set_xticks(range(len(GRP))); ax.set_xticklabels([NAME[g] for g in GRP]); ax.set_xlim(-0.5, len(GRP) - 0.5)
            ax.set_title(ttl, fontsize=8); ax.set_ylabel("Theta power change (%)" if k == 0 else "")
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            if k == 0:
                ax.text(-0.25, 1.12, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
        _savefig(fig, os.path.join(ce_dir, f"theta_correct_vs_errors_{REF_T}.png"))
        if REF_T == "LOCAL":
            sup_dir = os.path.join(AMIR, "supplementary"); os.makedirs(sup_dir, exist_ok=True)
            _savefig(fig, os.path.join(sup_dir, "Fig S9 theta correct vs errors LOCAL.png"))
        if REF_T in REPORT_NAME:
            _savefig(fig, os.path.join(report_dir, REPORT_NAME[REF_T]))
        plt.show()
        print(f"\n{REF_T}: each group alone vs 0 (sign-flip; FDR over the 8 tests)")
        print(pd.DataFrame(alone).round({"p": 4, "q": 4}).to_string(index=False))

    R = pd.DataFrame(all_rows).drop(columns=["a", "b", "mcol"])
    print("\ncomparisons (paired within session, sign-flip; q: FDR within 'main' (2 tests) and within 'follow-up' (4 tests) per reference)")
    print(R.round({"p": 4, "q": 4}).to_string(index=False))
    print("\nchecks (raw p): post − pre with early held >= 0.85 s; raw pre level")
    print(pd.DataFrame(check_rows).drop(columns=["a", "b", "mcol"]).round({"p": 4}).to_string(index=False))
    _table(R, os.path.join(TABLES, "theta_correct_vs_errors.csv"), index=False)


@cell()
def theta_after_release_regions():
    """1q. THETA AFTER THE RELEASE: CORRECT vs EARLY, PER REGION, IL vs PL"""
    global BAND_E, C, CHREG, L, MEAS_E, MIN_CH, MIN_CL, MIN_EL, OE, VE, Wd, _, cl, d_ch, el, m, masks, meas, mm, n, out, p, pc
    global r, ref, reg, rows, s, test, v, w0, warnings
    # ── 1q. THETA AFTER THE RELEASE: CORRECT vs EARLY, PER REGION, IL vs PL ──
    import warnings
    pc = globals().get("pc", lambda x: 100 * (10 ** np.asarray(x, float) - 1))
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    BAND_E, MIN_CL, MIN_EL, MIN_CH = "theta 4-7", 10, 5, 10
    MEAS_E = {"post − pre": "pre", "post − pre_release": "pre_release"}

    rows = []
    for ref in ["CMR", "LOCAL"]:
        for s in done_sessions(ref):
            r = load_bp(s, ref)
            C = fn.main_conditions(r["trials"])
            cl, el = C["correct long"].values, C["early long (hold>0.3)"].values
            if cl.sum() < MIN_CL or el.sum() < MIN_EL:
                continue
            masks, _ = fn.region_masks(r, CHREG, s)
            L = r["logpow"][BAND_E]
            for meas, w0 in MEAS_E.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    d_ch = np.nanmean(L["post"][cl] - L[w0][cl], axis=0) - np.nanmean(L["post"][el] - L[w0][el], axis=0)
                for reg in ("IL", "PL"):
                    m = masks[reg]
                    if m.sum() >= MIN_CH:
                        rows.append({"ref": ref, "session": s, "rat": RAT[s], "measure": meas, "region": reg,
                                     "value": float(np.nanmedian(d_ch[m]))})
    VE = pd.DataFrame(rows)

    out = []
    for ref in ["CMR", "LOCAL"]:
        for meas in MEAS_E:
            Wd = VE[(VE.ref == ref) & (VE.measure == meas)].pivot(index="session", columns="region", values="value")
            for test, v in (("IL", Wd["IL"]), ("PL", Wd["PL"]), ("IL − PL", Wd["IL"] - Wd["PL"])):
                v = v.dropna(); mm, p, n = fn.sign_flip_p(v.values)
                out.append({"ref": ref, "measure": meas, "test": test, "n_sessions": n, "up": int((v > 0).sum()),
                            "correct_minus_early_%": pc(mm), "p": p})
    OE = pd.DataFrame(out)
    _table(OE, os.path.join(TABLES, "theta_post_release_correct_vs_early_regions.csv"), index=False)
    for ref in ["CMR", "LOCAL"]:
        print(f"\n===== theta after the release, correct − early long (%), {ref} (raw p) =====")
        print(OE[OE.ref == ref].drop(columns="ref").round({"correct_minus_early_%": 1, "p": 4}).to_string(index=False))


@cell()
def delta_theta_quick_look():
    """1r. QUICK LOOK: DELTA vs THETA TIME COURSE AROUND THE PRESS AND THE RELEASE (correct long)"""
    global BCOL, X, al, ax, b, c, col, curves, fig, fp, k, m, n, ok, out_dir, pc, r, ref, s, se, sp, times
    # ── 1r. QUICK LOOK: DELTA vs THETA TIME COURSE AROUND THE PRESS AND THE RELEASE (correct long) ──
    pc = globals().get("pc", lambda x: 100 * (10 ** np.asarray(x, float) - 1))
    BCOL = {"delta 1-3": "#FFA000", "theta 4-7": "#2e7d32"}
    out_dir = os.path.join(AMIR, "Theta", "delta check"); os.makedirs(out_dir, exist_ok=True)
    for ref in ["CMR", "LOCAL"]:
        curves, times = {b: {"Press": [], "Release": []} for b in BCOL}, {}
        for s in done_sessions(ref):
            r = load_bp(s, ref)
            for al in ("Press", "Release"):
                times[al] = r["times"][al]
                for b in BCOL:
                    c = fn.session_curve(r, al, b, "correct long", min_trials=10)
                    if c is not None:
                        curves[b][al].append(c)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor="white", gridspec_kw=dict(width_ratios=[2.7, 2.1]))
        for k, al in enumerate(("Press", "Release")):
            for b, col in BCOL.items():
                X = np.vstack(curves[b][al]); n = np.sum(np.isfinite(X), 0)
                m = np.nanmean(X, 0); se = np.nanstd(X, 0, ddof=1) / np.sqrt(n)
                ok = n >= 0.5 * len(X)
                ax[k].plot(times[al][ok], pc(m[ok]), color=col, lw=1.8, label=f"{b.split()[0]} ({len(X)} sessions)")
                ax[k].fill_between(times[al][ok], pc((m - se)[ok]), pc((m + se)[ok]), color=col, alpha=0.2, lw=0)
            ax[k].axvline(0, color="k", lw=0.8); ax[k].axhline(0, color="0.6", lw=0.6)
            if al == "Press": ax[k].axvline(1.05, color="k", lw=0.8, ls="--")
            ax[k].set_xlabel(f"Time from {'press' if al == 'Press' else 'release'} (s)")
            for sp in ("top", "right"): ax[k].spines[sp].set_visible(False)
        ax[0].set_ylabel("Power change vs before the press (%)"); ax[0].legend(frameon=False, fontsize=9)
        fig.suptitle(f"Delta vs theta, correct long trials ({ref}); dashed = tone", fontsize=10.5, fontweight="bold")
        fig.tight_layout(); fp = os.path.join(out_dir, f"delta_vs_theta_timecourse_{ref}.png")
        _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def total_vs_induced():
    """1s. TOTAL vs INDUCED POWER (mean ERP removed), delta and theta, correct long, LOCAL — windows + Fig. S14"""
    global BANDS_D, CUR, DEC, EP, HOLDW, N_CH, ONLY_TEST, P, PEAK, POSTW, PRE, PT, REF_D, SH, SHARE, SHOW, THETA_COL, TT, WIN
    global WV, X, Xk, Y, _, a_, al, ax, axes, b, band, band_pow, base, col, fig, fs, g, hold_v, ids, k, keep, lab, ls, lw, m
    global meas, n, name, ok, out_dir, p, pc, pm, post_v, pre_v, rec, recs, res, rm, s, se, sel, sessions, sfft, share, sm, sp
    global sub, sup_dir, t, tr, win, x, xl, z_
    # ── 1s. TOTAL vs INDUCED POWER (mean ERP removed), delta and theta, correct long, LOCAL — windows + Fig. S14 ──
    import scipy.fft as sfft
    REF_D, N_CH, DEC, ONLY_TEST = "LOCAL", 20, 10, False          # ONLY_TEST = True -> run on TEST first (fast check)
    BANDS_D = {b: fn.MAIN_BANDS[b] for b in ["delta 1-3", "theta 4-7"]}
    EP   = {"Press": (-2.5, 3.5), "Release": (-3.0, 2.5)}          # same epochs as 06
    SHOW = {"Press": (-0.7, 1.05), "Release": (-1.0, 0.6)}
    PRE, HOLDW, POSTW, PEAK = fn.MAIN_WINDOWS["pre"][1], fn.MAIN_WINDOWS["hold"][1], fn.MAIN_WINDOWS["post"][1], (-0.2, 0.1)
    THETA_COL = "#2e7d32"
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    PT = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False)
    PT = PT[PT["keep"]].copy()                                                       # kept trials only, as in 06
    out_dir = os.path.join(AMIR, "Theta", "delta check"); os.makedirs(out_dir, exist_ok=True)
    sup_dir = os.path.join(AMIR, "supplementary"); os.makedirs(sup_dir, exist_ok=True)

    def band_pow(X, fs, band):
        """(trials, ch, n) -> mean Morlet power over the band's frequencies (trials, ch, n // DEC), as in 06."""
        n = X.shape[-1]; nfft = int(2 ** np.ceil(np.log2(n + 4 * fs)))
        Xf = sfft.fft(X, nfft, axis=-1, workers=-1)
        return np.mean([fn._power(Xf, nfft, n, f, band["n_cycles"], fs, DEC) for f in band["freqs"]], axis=0)

    win = lambda tt, w: (tt >= w[0] - 1e-9) & (tt <= w[1] + 1e-9)
    sessions = [TEST] if ONLY_TEST else list(final["session"])
    CUR = {(b, k, al): [] for b in BANDS_D for k in ("total", "induced") for al in EP}
    SHARE, WIN, TT = [], [], {}
    for s in sessions:
        tr = fn.session_trials(PT, s); tr = tr[(tr.label == "correct") & (tr.type == "long")].reset_index(drop=True)
        if len(tr) < 10:
            continue
        recs, _ = fn.load_session(PROC, s, extra_bad=XB, refs=(REF_D,)); rec = recs[REF_D]
        sel = set(fn.select_channels(fn.probe_table(rec), N_CH)["channel_id"])
        ids = [c for c in rec.get_channel_ids() if str(c) in sel]
        sub = rec.select_channels(ids) if hasattr(rec, "select_channels") else rec.channel_slice(ids)
        fs = sub.get_sampling_frequency()
        X, ok = {}, {}
        for al, col in (("Press", "press_s"), ("Release", "release_s")):
            X[al], ok[al] = fn.read_epochs(sub, tr[col].values, *EP[al])
        keep = ok["Press"] & ok["Release"]                              # same trials in both alignments
        X = {al: X[al][keep[ok[al]]] for al in X}
        t = {al: EP[al][0] + np.arange(0, X[al].shape[-1], DEC) / fs for al in X}; TT = t
        for b, band in BANDS_D.items():
            P = {}
            for k in ("total", "induced"):
                for al in X:
                    Xk = X[al] if k == "total" else X[al] - X[al].mean(axis=0, keepdims=True)   # remove the mean ERP
                    P[(k, al)] = band_pow(Xk, fs, band)
                pm = win(t["Press"], PRE)
                base = P[(k, "Press")][..., pm].mean(axis=-1, keepdims=True)                     # own pre-press power
                for al in X:                                                                     # time courses
                    sm = (t[al] >= SHOW[al][0]) & (t[al] <= SHOW[al][1])
                    CUR[(b, k, al)].append(np.median(np.log10(P[(k, al)][..., sm] / base).mean(axis=0), axis=0))
                lw = lambda al, m: np.log10(P[(k, al)][..., m].mean(axis=-1))                    # log10 of mean power in window, as 06
                pre_v, hold_v, post_v = lw("Press", pm), lw("Press", win(t["Press"], HOLDW)), lw("Release", win(t["Release"], POSTW))
                WIN.append({"session": s, "band": b, "kind": k,
                            "hold-pre":  float(np.median((hold_v - pre_v).mean(axis=0))),
                            "hold-post": float(np.median((hold_v - post_v).mean(axis=0)))})
            rm = win(t["Release"], PEAK)
            share = 1 - np.median(P[("induced", "Release")][..., rm].mean(axis=(0, 2)) / P[("total", "Release")][..., rm].mean(axis=(0, 2)))
            SHARE.append({"session": s, "band": b, "evoked_share_%": 100 * share, "n_trials": int(keep.sum())})
        print(f"{s}: {int(keep.sum())} trials, {len(ids)} channels", flush=True)

    # numbers: windows (total vs induced) and evoked share around the release
    WV = pd.DataFrame(WIN); res = []
    for (b, k), g in WV.groupby(["band", "kind"], sort=False):
        for meas in ["hold-pre", "hold-post"]:
            m, p, n = fn.sign_flip_p(g[meas].values)
            res.append({"band": b, "kind": k, "measure": meas, "n_sessions": n, "sessions_up": int((g[meas] > 0).sum()),
                        "mean_%": round(float(pc(m)), 1), "p": round(p, 4)})
    SH = pd.DataFrame(SHARE)
    print(f"\nTotal vs induced power (mean ERP removed), correct long, {REF_D}, {N_CH} channels/session")
    print(pd.DataFrame(res).to_string(index=False))
    print(f"\nEvoked share of power around the release ({PEAK[0]}..{PEAK[1]} s): mean over sessions (min–max)")
    print(SH.groupby("band")["evoked_share_%"].agg(["mean", "min", "max"]).round(1).to_string())
    _table(WV, os.path.join(out_dir, f"total_vs_induced_windows_{REF_D}.csv"), index=False)
    _table(SH, os.path.join(out_dir, f"evoked_share_{REF_D}.csv"), index=False)

    # Fig. S14: theta, total (solid) vs induced (dashed), press | release
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True, facecolor="white", gridspec_kw={"width_ratios": [1.75, 1.6], "wspace": 0.08})
    b = "theta 4-7"
    for ax, al, xl in zip(axes, EP, ["Time from press (s)", "Time from release (s)"]):
        sm = (TT[al] >= SHOW[al][0]) & (TT[al] <= SHOW[al][1]); x = TT[al][sm]
        for name in (["pre", "hold"] if al == "Press" else ["post"]):
            a_, z_ = fn.MAIN_WINDOWS[name][1]
            ax.axvspan(a_, z_, color="#C0392B", alpha=0.10, lw=0)
            ax.text((a_ + z_) / 2, 1.01, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#C0392B")
        for k, ls, lab in (("total", "-", "Total power"), ("induced", "--", "Induced power (ERP removed)")):
            Y = np.vstack(CUR[(b, k, al)]); m, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))
            ax.fill_between(x, pc(m - se), pc(m + se), color=THETA_COL, alpha=0.15, lw=0)
            ax.plot(x, pc(m), ls, color=THETA_COL, lw=1.5, label=f"{lab}")
        ax.axvline(0, color="k", lw=0.9); ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlim(*SHOW[al]); ax.set_xlabel(xl)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Theta power change\nfrom pre-press (%)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    _savefig(fig, os.path.join(out_dir, f"theta_total_vs_induced_{REF_D}.png"))
    _savefig(fig, os.path.join(sup_dir, f"Fig S14 theta total vs induced {REF_D}.png"))
    plt.show()
    print(f"{len(CUR[(b, 'total', 'Press')])} sessions; saved in: {sup_dir}")


@cell(needs={
    'total_vs_induced': ['CUR', 'REF_D', 'SHOW', 'TT'],
})
def total_vs_induced_figure():
    """1s-plot. THETA + DELTA: total vs induced power, one figure for the report (uses CUR from cell 1s; no recompute)"""
    global BAND, HOLD_W, Line2D, PEAK, PRE_W, Y, _, a_, al, ax, axes, b, col, d, fig, hand, k, lab, ls, m, miss, n, name, need
    global off, out, p, pc, pw, ref_al, ref_w, rep_dir, se, sp, wmean, wname, x, xl, xp, xs, z_
    # ── 1s-plot. THETA + DELTA: total vs induced power, one figure for the report (uses CUR from cell 1s; no recompute) ──
    from matplotlib.lines import Line2D
    need = [(b, k, al) for b in ("theta 4-7", "delta 1-3") for k in ("total", "induced") for al in ("Press", "Release")]
    miss = [x for x in need if x not in CUR]
    assert not miss, f"not in CUR (kernel restarted? re-run 1s): {miss}"
    PEAK = (-0.2, 0.1)                                                                   # release peak window
    BAND = {"theta 4-7": ("#2e7d32", "Theta"), "delta 1-3": ("#FFA000", "Delta")}
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    rep_dir = os.path.join(AMIR, "report"); os.makedirs(rep_dir, exist_ok=True)
    xs = lambda al: TT[al][(TT[al] >= SHOW[al][0]) & (TT[al] <= SHOW[al][1])]          # times of the CUR curves

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True, facecolor="white",
                             gridspec_kw={"width_ratios": [1.75, 1.6], "wspace": 0.08})
    for ax, al, xl in zip(axes, ("Press", "Release"), ("Time from press (s)", "Time from release (s)")):
        x = xs(al)
        for wname in (["pre", "hold"] if al == "Press" else ["post"]):
            a_, z_ = fn.MAIN_WINDOWS[wname][1]
            ax.axvspan(a_, z_, color="#C0392B", alpha=0.08, lw=0)
            ax.text((a_ + z_) / 2, 1.01, wname, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#C0392B")
        if al == "Release":
            ax.axvspan(*PEAK, color="0.85", lw=0, zorder=0)
        for b, (col, name) in BAND.items():
            for k, ls in (("total", "-"), ("induced", "--")):
                xp = xs("Press"); pw = (xp >= fn.MAIN_WINDOWS["pre"][1][0]) & (xp <= fn.MAIN_WINDOWS["pre"][1][1])
                off = np.vstack(CUR[(b, k, "Press")])[:, pw].mean(1, keepdims=True)              # each session's pre-window level
                Y = np.vstack(CUR[(b, k, al)]) - off; m, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))
                ax.fill_between(x, pc(m - se), pc(m + se), color=col, alpha=0.15, lw=0)
                ax.plot(x, pc(m), ls, color=col, lw=1.5)
        ax.axvline(0, color="k", lw=0.9); ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlim(*SHOW[al]); ax.set_xlabel(xl)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Power change from pre-press (%)")
    hand = [Line2D([], [], color=c, lw=1.8, label=n) for c, n in BAND.values()] + \
           [Line2D([], [], color="0.3", lw=1.5, ls="-", label="Total"),
            Line2D([], [], color="0.3", lw=1.5, ls="--", label="Induced (ERP removed)")]
    axes[1].legend(handles=hand, frameon=False, fontsize=7, loc="lower left")
    _savefig(fig, os.path.join(rep_dir, f"Fig 5 theta and delta total vs induced {REF_D}.png"))
    plt.show()

    # numbers: release peak vs pre-press and vs hold, total vs induced (from the session curves)
    def wmean(b, k, al, w):
        x = xs(al); m = (x >= w[0]) & (x <= w[1]); assert m.any(), f"window {w} outside curve ({al})"
        return np.vstack(CUR[(b, k, al)])[:, m].mean(1)
    PRE_W, HOLD_W = fn.MAIN_WINDOWS["pre"][1], fn.MAIN_WINDOWS["hold"][1]
    print(f"{REF_D}, {len(CUR[('theta 4-7', 'total', 'Press')])} sessions; release peak {PEAK[0]}..{PEAK[1]} s")
    for b, (_, name) in BAND.items():
        for lab, ref_al, ref_w in (("release-pre", "Press", PRE_W), ("release-hold", "Press", HOLD_W)):
            d = {k: wmean(b, k, "Release", PEAK) - wmean(b, k, ref_al, ref_w) for k in ("total", "induced")}
            out = []
            for k in ("total", "induced"):
                m, p, n = fn.sign_flip_p(d[k]); out.append(f"{k} {pc(m):+.1f} % (p = {p:.4f}, up in {int((d[k] > 0).sum())}/{n})")
            m, p, n = fn.sign_flip_p(d["total"] - d["induced"])
            print(f"  {name:5s} {lab:12s}: " + "; ".join(out) + f"; total − induced {pc(m):+.1f} % (p = {p:.4f})")


@cell()
def delta_go_vs_wait():
    """1m-delta. DELTA (1–3 Hz) GO vs WAIT from the saved maps (as Fig. 3, mirrored: go > pre AND go > wait), CMR and LOCAL"""
    global CUE_WIN, C_GO, C_WAIT, FREQS, MG, MW, REF_G, ax, col, cw, dg, dw, fb, fig, g_in, idx, lab, m, multipletests, n, p
    global p1, p2, pc, q1, q2, qq, se, segs, sg, sig, sp_, sup_dir, t1, t2, tt, ttest_1samp, used, v, w_in, wg_dir, x, y0, y1
    global yb, z
    # ── 1m-delta. DELTA (1–3 Hz) GO vs WAIT from the saved maps (as Fig. 3, mirrored: go > pre AND go > wait), CMR and LOCAL ──
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests
    C_WAIT, C_GO = "#2CA02C", "#C62828"                                               # as Fig. 3
    CUE_WIN = (0.35 + 0.30, 0.35 + 0.50)                                             # 300–500 ms after the short tone (Fakhraei 2021)
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    wg_dir = os.path.join(AMIR, "Theta", "wait vs go"); os.makedirs(wg_dir, exist_ok=True)
    sup_dir = os.path.join(AMIR, "supplementary"); os.makedirs(sup_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    for REF_G in ["CMR", "LOCAL"]:
        z = np.load(os.path.join(TABLES, f"wait_vs_go_maps_{REF_G}.npz"), allow_pickle=True)
        MW, MG, FREQS, tt, used = z["wait"], z["go"], z["freqs"], z["times"], z["sessions"]
        fb = (FREQS >= 1) & (FREQS <= 3)
        dw, dg = MW[:, fb].mean(1), MG[:, fb].mean(1)                                # (sessions, times), log10 vs pre-press
        t1, p1 = ttest_1samp(dg, 0, axis=0);       q1 = multipletests(p1, method="fdr_bh")[1]
        t2, p2 = ttest_1samp(dg - dw, 0, axis=0);  q2 = multipletests(p2, method="fdr_bh")[1]
        sig = (q1 < 0.05) & (t1 > 0) & (q2 < 0.05) & (t2 > 0)                        # go > pre AND go > wait
        qq = np.maximum(q1, q2)

        fig, ax = plt.subplots(figsize=(4.5, 3.2), facecolor="white")
        for v, col, lab in ((dw, C_WAIT, "Wait (correct long)"), (dg, C_GO, "Go (correct short)")):
            m, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))
            ax.fill_between(tt, pc(m - se), pc(m + se), color=col, alpha=0.25, lw=0)
            ax.plot(tt, pc(m), color=col, lw=1.6, label=lab)
        ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="k", lw=0.9)
        ax.axvspan(*CUE_WIN, color="0.9", lw=0, zorder=0)
        for x, col, lab in ((0.35, C_GO, "short tone"), (1.05, C_WAIT, "long tone")):
            ax.axvline(x, color=col, ls="--", lw=1)
            ax.text(x, 1.01, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=col)
        if sig.any():
            y0, y1 = ax.get_ylim(); yb = y1 + 0.03 * (y1 - y0)
            idx = np.where(sig)[0]; segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for sg in segs:
                ax.plot([tt[sg[0]], tt[sg[-1]]], [yb, yb], color="k", lw=3, solid_capstyle="butt")
            ax.text(tt[segs[0][0]], yb + 0.02 * (y1 - y0), "q < 0.05", ha="left", va="bottom", fontsize=7.5)
            ax.set_ylim(top=yb + 0.14 * (y1 - y0))
        ax.set_xlim(tt[0], tt[-1]); ax.set_xlabel("Time from press (s)")
        ax.set_ylabel("Delta power change from pre-press (%)")
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=1.6)
        _savefig(fig, os.path.join(wg_dir, f"delta_wait_vs_go_{REF_G}.png"))
        if REF_G == "LOCAL":
            _savefig(fig, os.path.join(sup_dir, "Fig S13 delta wait vs go LOCAL.png"))
        plt.show()

        # numbers
        cw = (tt >= CUE_WIN[0]) & (tt <= CUE_WIN[1])
        g_in, w_in = dg[:, cw].mean(1), dw[:, cw].mean(1)
        m, p, n = fn.sign_flip_p(g_in - w_in)
        print(f"\n{REF_G}: {len(used)} sessions")
        print(f"  300–500 ms after the short tone ({CUE_WIN[0]:.2f}–{CUE_WIN[1]:.2f} s): go {pc(g_in.mean()):+.1f} %, wait {pc(w_in.mean()):+.1f} % "
              f"vs pre-press; go − wait {pc(m):+.1f} %, p = {p:.4f}, go higher in {int((g_in > w_in).sum())}/{n} sessions")
        if sig.any():
            print("  time points go > pre AND go > wait (q < 0.05): " + ", ".join(f"{tt[sg[0]]:.2f}–{tt[sg[-1]]:.2f} s" for sg in segs)
                  + f"; q inside {qq[sig].min():.1e} to {qq[sig].max():.3f}")
        else:
            print("  no time point with go > pre AND go > wait")


@cell()
def mua_hold_vs_pre_post():
    """MULTI-UNIT FIRING (MUA) DURING THE HOLD vs BEFORE THE PRESS and vs AFTER THE RELEASE — does it drop like high gamma? (correct long) — desktop (needs cell 0)"""
    global CHREG, D, DEAD_MS, DESK, HG, K_THR, L, M, MCOL, MIN_RATE, MUA_DIR, N_TR, Path, RAT_S, RAW, RCOL, RM, SHORT, SV, T
    global WINS, XB_, _, a, a0, a1, a2, ap_bin, ax, c, cl, col, crossings, d, dead, eps, fc, fig, fmt_q, fp, fs, g, glob
    global gname, hi, i, ids, k, keep_ids, lab, lo, m, multipletests, n, n_tot, noise, out, p, p_rho, pcf, q, r_, rate, raw
    global rec, reg, rho, rng, rows, s, sel, si, sp, sp_, spearmanr, starts, t_start, thr, time, tr, v, w, warnings, x, z
    # ── MULTI-UNIT FIRING (MUA) DURING THE HOLD vs BEFORE THE PRESS and vs AFTER THE RELEASE — does it drop like high gamma? (correct long) — desktop (needs cell 0) ──
    #    No spike sorting: threshold crossings on the raw 30 kHz AP band (as is common for MUA).
    #    Per session: raw ap.bin (si.read_spikeglx, as 02) -> same channels as the 1000 Hz files (bad channels out) -> bandpass 300–6000 Hz -> phase_shift
    #    -> CMR (global median; fine for spikes, which are local). Noise per channel = MAD / 0.6745 from 20 random 1-s pieces; spike = downward crossing
    #    of −K_THR × noise, 1-ms dead time. Windows as the band power: pre −0.5…−0.2 s and hold 0.2–0.85 s (press), post 0.2–0.5 s (release).
    #    Up to N_TR correct long trials per session (random, seed 0). Rate per channel = mean over trials (spikes/s); channels with < 1 spike/s before
    #    the press are left out (no units). Session value per region = median over its channels of log10(hold / pre) and log10(hold / post).
    #    Test: sign-flip across sessions (fn.sign_flip_p); q = FDR-BH over the 8 tests (4 groups × 2 comparisons).
    #    Link to high gamma: per session, MUA hold vs pre against high gamma hold vs pre (CMR, M1 table of 1e), Spearman across sessions.
    #    Slow (reads raw 30 kHz data around every trial) -> one cached file per session in tables\mua_hold\.
    import glob, time, warnings
    from pathlib import Path
    import spikeinterface.full as si
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    DESK = next((p for p in (os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"), os.path.join(os.path.expanduser("~"), "Desktop"))
                 if os.path.isdir(p)), os.path.expanduser("~"))
    RAW = os.path.join(DATA_ROOT, "0_raw")
    K_THR, N_TR, DEAD_MS, MIN_RATE = 4.5, 60, 1.0, 1.0
    WINS = {"pre": ("press_s", -0.5, -0.2), "hold": ("press_s", 0.20, 0.85), "post": ("release_s", 0.20, 0.50)}
    MUA_DIR = os.path.join(TABLES, "mua_hold"); os.makedirs(MUA_DIR, exist_ok=True)
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    SHORT = {"Infralimbic area": "IL", "Prelimbic area": "PL", "Cingulate area 1": "Cg1"}
    RAT_S = dict(zip(final["session"], final["rat"].astype(str)))
    XB_ = pd.read_csv(os.path.join(TABLES, "extra_bad_channels.csv"))
    pcf = lambda x: 100 * (10 ** np.asarray(x, float) - 1)

    def crossings(x, thr, dead):
        """onsets (samples) of downward crossings of -thr per channel; x (samples, channels)"""
        below = x < -thr[None, :]
        on = below[1:] & ~below[:-1]
        out = []
        for j in range(x.shape[1]):
            t = np.flatnonzero(on[:, j]) + 1
            if len(t) > 1:                                                      # dead time
                keep = np.r_[True, np.diff(t) > dead]; t = t[keep]
            out.append(t)
        return out

    # ── 1) MUA per channel, window and trial (cached per session) ──
    for i, s in enumerate(done_sessions("CMR")):
        fc = os.path.join(MUA_DIR, s + ".csv")
        if os.path.exists(fc): continue
        t_start = time.time()
        T = load_bp(s, "CMR")["trials"]
        cl = T[(T["label"] == "correct") & (T["type"] == "long")]
        if len(cl) < 10: continue
        cl = cl.sample(min(N_TR, len(cl)), random_state=0)
        keep_ids = [str(c) for c in fn.load_session(PROC, s, XB_, refs=("original",))[0]["original"].get_channel_ids()]   # same channels as the LFP files
        ap_bin = glob.glob(os.path.join(RAW, s, "**", "*.ap.bin"), recursive=True)[0]
        raw = si.read_spikeglx(Path(ap_bin).parent, stream_name="imec0.ap")
        sel = [c for c in raw.get_channel_ids() if str(c) in set(keep_ids)]
        raw = raw.select_channels(sel) if hasattr(raw, "select_channels") else raw.channel_slice(channel_ids=sel)
        rec = si.common_reference(si.phase_shift(si.bandpass_filter(raw, freq_min=300, freq_max=6000, dtype="float32")),
                                  reference="global", operator="median")
        fs = rec.get_sampling_frequency(); n_tot = rec.get_num_samples(); ids = [str(c) for c in rec.get_channel_ids()]
        rng = np.random.default_rng(0); starts = rng.integers(int(10 * fs), n_tot - int(2 * fs), 20)
        noise = np.median([np.median(np.abs(rec.get_traces(start_frame=a, end_frame=a + int(fs), return_in_uV=True)), axis=0) / 0.6745
                           for a in starts], axis=0)
        thr, dead = K_THR * noise, int(DEAD_MS * fs / 1000)
        rows = []
        for _, tr in cl.iterrows():
            a0 = int((tr["press_s"] - 0.6) * fs); a1 = int((tr["release_s"] + 0.6) * fs)
            if a0 < 0 or a1 > n_tot: continue
            x = rec.get_traces(start_frame=a0, end_frame=a1, return_in_uV=True)
            sp = crossings(x, thr, dead)
            for w, (col, a, z) in WINS.items():
                lo, hi = int((tr[col] + a) * fs) - a0, int((tr[col] + z) * fs) - a0
                rate = np.array([((t >= lo) & (t < hi)).sum() for t in sp]) / (z - a)
                rows += [dict(session=s, trial_index=tr["trial_index"], window=w, channel_id=c, rate=float(v)) for c, v in zip(ids, rate)]
        D = pd.DataFrame(rows).groupby(["session", "window", "channel_id"])["rate"].mean().unstack("window").reset_index()
        reg = CHREG[CHREG["session"] == s].set_index("channel_id")["region"]
        D["region"] = D["channel_id"].map(lambda c: SHORT.get(reg.get(c), "other")); D["n_trials"] = len(cl)
        D.to_csv(fc, index=False)
        print(f"  {i + 1} {s}: {len(cl)} trials, {len(ids)} channels, {time.time() - t_start:.0f} s", flush=True)

    # ── 2) session values (median over channels with units) ──
    M = pd.concat([pd.read_csv(f, dtype={"channel_id": str}) for f in sorted(glob.glob(os.path.join(MUA_DIR, "*.csv")))], ignore_index=True)
    M = M[(M["pre"] >= MIN_RATE) & (M["region"] != "other")].copy()
    eps = 0.1                                                                   # spikes/s, avoids log(0)
    M["hold-pre"] = np.log10((M["hold"] + eps) / (M["pre"] + eps)); M["hold-post"] = np.log10((M["hold"] + eps) / (M["post"] + eps))
    SV = pd.concat([M.groupby("session")[["hold-pre", "hold-post"]].median().assign(group="mPFC (IL+PL+Cg1)"),
                    *[M[M.region == g].groupby("session")[["hold-pre", "hold-post"]].median().assign(group=g) for g in ("IL", "PL", "Cg1")]]).reset_index()
    SV["rat"] = SV["session"].map(RAT_S)
    out = []
    for gname, d in SV.groupby("group", sort=False):
        for c in ("hold-pre", "hold-post"):
            m, p, n = fn.sign_flip_p(d[c].values)
            out.append(dict(group=gname, comparison=c, n_sessions=n, sessions_lower_in_hold=int((d[c] < 0).sum()), mean_pct=round(float(pcf(m)), 1), p=p))
    RM = pd.DataFrame(out); RM["q"] = multipletests(RM["p"], method="fdr_bh")[1]
    _table(RM, os.path.join(DESK, "mua_hold_vs_pre_post.csv"), index=False)

    # ── 3) link to high gamma (CMR, M1 session values of 1e: median over all channels) ──
    HG = pd.read_csv(os.path.join(TABLES, "M1_session_values_CMR.csv"))
    HG = HG[HG["band"] == "highG 61-100"].set_index("session")["hold-pre"]
    L = SV[SV.group == "mPFC (IL+PL+Cg1)"].set_index("session")[["hold-pre"]].join(HG.rename("hg"), how="inner").dropna()
    rho, p_rho = spearmanr(L["hold-pre"], L["hg"])

    # ── 4) figure: A bars (mPFC MUA hold vs pre / vs post), B MUA vs high gamma per session ──
    fmt_q = lambda q: "q < 0.001" if q < 0.001 else f"q = {q:.3f}"
    MCOL, RCOL = "#455A64", {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4), facecolor="white", gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.4})
    d = SV[SV.group == "mPFC (IL+PL+Cg1)"]
    for k, (c, lab) in enumerate((("hold-pre", "Hold vs\nbefore press"), ("hold-post", "Hold vs\nafter release"))):
        v = pcf(d[c]); m, _, _ = fn.sign_flip_p(d[c].values); q = RM[(RM.group == "mPFC (IL+PL+Cg1)") & (RM.comparison == c)].q.iloc[0]
        a1.bar(k, pcf(m), width=0.6, color=MCOL, alpha=0.85, edgecolor="k", lw=0.8, zorder=2)
        a1.errorbar(k, pcf(m), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="none", color="k", capsize=5, lw=1.3, zorder=3)
        a1.text(k, 1.01, f"{pcf(m):+.1f} %\n{fmt_q(q)}", transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9.5)
    a1.axhline(0, color="k", lw=0.9); a1.set_xticks([0, 1]); a1.set_xticklabels(["Hold vs\nbefore press", "Hold vs\nafter release"]); a1.set_xlim(-0.6, 1.6)
    a1.set_ylabel("Multi-unit firing change (%)")
    for r_, g in L.join(SV[SV.group == "mPFC (IL+PL+Cg1)"].set_index("session")["rat"]).groupby("rat"):
        a2.scatter(pcf(g["hg"]), pcf(g["hold-pre"]), s=24, color=RCOL.get(r_, "0.5"), edgecolor="white", lw=0.4, label=f"rat {r_} ({len(g)})")
    a2.axhline(0, color="0.6", lw=0.8, ls="--"); a2.axvline(0, color="0.6", lw=0.8, ls="--")
    a2.set_xlabel("High gamma, hold vs before press (%)"); a2.set_ylabel("Multi-unit firing, hold vs before press (%)")
    a2.set_title(f"Spearman ρ = {rho:.2f}, p = {p_rho:.3f} ({len(L)} sessions)", fontsize=9.5)
    a2.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in (a1, a2):
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    fp = os.path.join(DESK, "mua_hold_vs_pre_post.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))

    print(f"MUA, correct long (≤ {N_TR} per session), threshold −{K_THR} × noise, channels with ≥ {MIN_RATE} spikes/s before the press")
    print("channels per session (median):", M.groupby(["session", "region"]).size().groupby("region").median().to_dict())
    print(RM.round({"p": 4, "q": 4}).to_string(index=False))
    print(f"MUA vs high gamma (hold vs pre, per session): Spearman rho = {rho:.2f}, p = {p_rho:.4f}, {len(L)} sessions | saved: {fp}")


@cell(needs={
    'mua_hold_vs_pre_post': ['DEAD_MS', 'DESK', 'K_THR', 'MUA_DIR', 'N_TR', 'RAT_S', 'RAW', 'RM', 'SV', 'XB_', 'crossings', 'pcf'],
})
def mua_gamma_figure():
    """FIGURE: SPIKING AND HIGH GAMMA DROP DURING THE HOLD — A) one example trial, B) the same channel over all its trials, C) bars of all sessions"""
    global BIN_S, EX_CH, EX_REG, EX_SES, HGC, MCOL, Mall, POST_S, PRE_S, Path, T, WIN_C, _, a, a0, a1, a2, aligned, amp
    global ap_bin, ax, b0, b1, b3, c, c1, c_, cand, cl, d, dead, ex, fig, fmt_q, fp, fs, fsl, glob, gs, gsb, h1, h2, hi, hi_
    global hm, hold_s, ids, j, k, keep_ids, l1, l2, lfp, lj, lo, lo_, m, mean_ratio, mp, n_tot, nm, noise, ok_s, pre_m, q
    global r_hold, r_pre, raw, rec, ref, rm, rng, rse, sel, shade, si, sm, sp_, spk, ss, starts, t_ap, th, thr, tr, trials, tw
    global twins, v, wav, wins, x, xc, xl, y, z
    # ── FIGURE: SPIKING AND HIGH GAMMA DROP DURING THE HOLD — A) one example trial, B) the same channel over all its trials, C) bars of all sessions ──
    #    Needs cell 0 and the MUA cell (cache tables\mua_hold\ and its settings K_THR, N_TR, DEAD_MS, MIN_RATE, WINS, crossings, RAW, XB_, pcf, SV, RM).
    #    Example: session EX_SES (default: the session whose mPFC MUA drop is closest to the mean of all sessions, ≥ 40 trials), channel EX_CH
    #    (default: an mPFC channel of that session with ≥ 10 spikes/s before the press and a clear drop = lowest hold/pre ratio) — an EXAMPLE, chosen.
    #    Trial in A = the trial of that channel whose hold/pre ratio is closest to the channel's mean ratio (a typical trial of that channel).
    #    AP band = same processing as the MUA cell (300–6000 Hz, phase_shift, CMR, threshold −K_THR × noise). High gamma = Morlet 65–96 Hz (10 cycles)
    #    on the 1000 Hz CMR signal of the same channel, amplitude smoothed over 50 ms, relative to the trial's own pre-press mean.
    import glob
    from pathlib import Path
    import scipy.signal as ss
    import spikeinterface.full as si
    EX_SES, EX_CH, BIN_S = None, None, 0.05
    HGC, MCOL, WIN_C = "#6a1b9a", "#455A64", "#F6DADD"

    # ── 1) pick the example session and channel from the cache ──
    Mall = pd.concat([pd.read_csv(f, dtype={"channel_id": str}) for f in sorted(glob.glob(os.path.join(MUA_DIR, "*.csv")))], ignore_index=True)
    mp = SV[SV.group == "mPFC (IL+PL+Cg1)"].set_index("session")["hold-pre"]
    if EX_SES is None:
        ok_s = Mall.groupby("session")["n_trials"].first(); ok_s = ok_s[ok_s >= 40].index
        EX_SES = (mp.loc[mp.index.intersection(ok_s)] - mp.mean()).abs().idxmin()
    if EX_CH is None:
        c_ = Mall[(Mall.session == EX_SES) & (Mall.region != "other") & (Mall["pre"] >= 10)].copy()
        EX_CH = c_.loc[(c_["hold"] / c_["pre"]).idxmin(), "channel_id"]
    EX_REG = Mall[(Mall.session == EX_SES) & (Mall.channel_id == EX_CH)]["region"].iloc[0]
    print(f"example: {EX_SES} (rat {RAT_S[EX_SES]}), channel {EX_CH} ({EX_REG})")

    # ── 2) AP band of that session (same as the MUA cell) and high gamma of the same channel ──
    T = load_bp(EX_SES, "CMR")["trials"]
    cl = T[(T["label"] == "correct") & (T["type"] == "long")]; cl = cl.sample(min(N_TR, len(cl)), random_state=0)
    ap_bin = glob.glob(os.path.join(RAW, EX_SES, "**", "*.ap.bin"), recursive=True)[0]
    keep_ids = [str(c) for c in fn.load_session(PROC, EX_SES, XB_, refs=("original",))[0]["original"].get_channel_ids()]
    raw = si.read_spikeglx(Path(ap_bin).parent, stream_name="imec0.ap")
    sel = [c for c in raw.get_channel_ids() if str(c) in set(keep_ids)]
    raw = raw.select_channels(sel) if hasattr(raw, "select_channels") else raw.channel_slice(channel_ids=sel)
    rec = si.common_reference(si.phase_shift(si.bandpass_filter(raw, freq_min=300, freq_max=6000, dtype="float32")), reference="global", operator="median")
    fs = rec.get_sampling_frequency(); ids = [str(c) for c in rec.get_channel_ids()]; j = ids.index(EX_CH); n_tot = rec.get_num_samples()
    rng = np.random.default_rng(0); starts = rng.integers(int(10 * fs), n_tot - int(2 * fs), 20)
    noise = np.median([np.median(np.abs(rec.get_traces(start_frame=a, end_frame=a + int(fs), return_in_uV=True)[:, j])) / 0.6745 for a in starts])
    thr, dead = K_THR * noise, int(DEAD_MS * fs / 1000)
    lfp = fn.load_session(PROC, EX_SES, XB_, refs=("CMR",))[0]["CMR"]; fsl = lfp.get_sampling_frequency()
    lj = [str(c) for c in lfp.get_channel_ids()].index(EX_CH)
    wav = [fn.morlet(f, 10, fsl) for f in fn.MAIN_BANDS["highG 61-100"]["freqs"]]
    sm = int(0.05 * fsl)

    PRE_S, POST_S = 0.6, 0.7
    trials = []
    for _, tr in cl.iterrows():
        a0 = int((tr["press_s"] - PRE_S) * fs); a1 = int((tr["release_s"] + POST_S) * fs)
        if a0 < 0 or a1 > n_tot: continue
        x = rec.get_traces(start_frame=a0, end_frame=a1, return_in_uV=True)[:, j]
        spk = crossings(x[:, None], np.array([thr]), dead)[0] / fs - PRE_S                       # spike times from the press (s)
        b0, b1 = int((tr["press_s"] - PRE_S - 1) * fsl), int((tr["release_s"] + POST_S + 1) * fsl)   # 1 s padding for the wavelets
        y = lfp.get_traces(start_frame=b0, end_frame=b1, channel_ids=[lfp.get_channel_ids()[lj]], return_in_uV=True)[:, 0].astype(float)
        amp = np.mean([np.abs(ss.fftconvolve(y, w, mode="same")) for w in wav], axis=0)
        amp = np.convolve(amp, np.ones(sm) / sm, mode="same")[int(fsl):-int(fsl)]
        th = np.arange(len(amp)) / fsl - PRE_S
        pre_m = amp[(th >= -0.5) & (th < -0.2)].mean()
        hold_s = tr["release_s"] - tr["press_s"]
        r_pre = ((spk >= -0.5) & (spk < -0.2)).sum() / 0.3; r_hold = ((spk >= 0.2) & (spk < 0.85)).sum() / 0.65
        trials.append(dict(x=x, spk=spk, th=th, hg=100 * (amp / pre_m - 1), hold=hold_s, ratio=(r_hold + 0.1) / (r_pre + 0.1)))
    mean_ratio = np.exp(np.mean([np.log(t["ratio"]) for t in trials]))
    cand = [t for t in trials if t["hold"] <= 1.6] or trials
    ex = min(cand, key=lambda t: abs(np.log(t["ratio"]) - np.log(mean_ratio)))

    # ── 3) figure ──
    plt.rcParams.update({"font.size": 9.5, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(14, 6.4), facecolor="white")
    gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1.0, 0.75], height_ratios=[1.1, 0.7, 1.0], wspace=0.55, hspace=0.5)
    def shade(ax, hold):
        for a, z, nm in ((-0.5, -0.2, "pre"), (0.2, 0.85, "hold"), (hold + 0.2, hold + 0.5, "post")):
            ax.axvspan(a, z, color=WIN_C, zorder=0, lw=0)
        ax.axvline(0, color="k", lw=0.9); ax.axvline(hold, color="k", lw=0.9, ls="--")
    # A1: AP trace + spikes (one trial)
    a1 = fig.add_subplot(gs[0, :2]); t_ap = np.arange(len(ex["x"])) / fs - PRE_S
    shade(a1, ex["hold"]); a1.plot(t_ap, ex["x"], color=MCOL, lw=0.3)
    a1.plot(ex["spk"], np.full(len(ex["spk"]), np.nanmax(ex["x"]) * 1.15), "|", color="k", ms=8, mew=1)
    a1.axhline(-thr, color="#c62828", lw=0.7, ls=":")
    for a, z, nm in ((-0.5, -0.2, "pre"), (0.2, 0.85, "hold"), (ex["hold"] + 0.2, ex["hold"] + 0.5, "post")):
        a1.text((a + z) / 2, 1.0, nm, transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#c62828")
    a1.text(0, -0.02, "press", transform=a1.get_xaxis_transform(), ha="center", va="top", fontsize=8.5)
    a1.text(ex["hold"], -0.02, "release", transform=a1.get_xaxis_transform(), ha="center", va="top", fontsize=8.5)
    a1.set_xlim(-PRE_S, ex["hold"] + POST_S); a1.set_xticks([]); a1.set_ylabel("AP band (µV)")
    a1.set_title(f"A  One trial — {EX_REG} channel, session {EX_SES[:8]} (rat {RAT_S[EX_SES]}); ticks = spikes", loc="left", fontsize=10, fontweight="bold", pad=16)
    # A2: high gamma of the same trial
    a2 = fig.add_subplot(gs[1, :2], sharex=a1); shade(a2, ex["hold"])
    a2.plot(ex["th"], ex["hg"], color=HGC, lw=1.4); a2.axhline(0, color="0.5", lw=0.7, ls="--")
    a2.set_ylabel("High gamma\n(% vs pre)"); a2.set_xlabel("Time from press (s)")
    # B: same channel, all trials: spike rate and high gamma, press-aligned (until each release) and release-aligned (from each press)
    def aligned(ref, lo, hi):
        edges = np.arange(lo, hi + 1e-9, BIN_S); xc = edges[:-1] + BIN_S / 2; R, H = [], []
        for t in trials:
            sh = 0 if ref == "press" else t["hold"]
            r = np.histogram(t["spk"] - sh, edges)[0] / BIN_S
            h = np.interp(xc, t["th"] - sh, t["hg"], left=np.nan, right=np.nan)
            on = (edges[1:] <= t["hold"]) if ref == "press" else (edges[:-1] >= -t["hold"])   # press: until release | release: from press
            r = np.where(on, r, np.nan); h = np.where(on, h, np.nan); R.append(r); H.append(h)
        R, H = np.array(R, float), np.array(H, float)
        return xc, np.nanmean(R, 0), np.nanstd(R, 0) / np.sqrt(np.sum(np.isfinite(R), 0)), np.nanmean(H, 0)
    gsb = gs[2, :2].subgridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.08)
    b1 = fig.add_subplot(gsb[0]); b3 = fig.add_subplot(gsb[1], sharey=b1); twins = []
    for ax, ref, (lo, hi), wins, xl in ((b1, "press", (-PRE_S, 1.0), ((-0.5, -0.2), (0.2, 0.85)), "Time from press (s)"),
                                        (b3, "release", (-0.3, POST_S), ((0.2, 0.5),), "Time from release (s)")):
        xc, rm, rse, hm = aligned(ref, lo, hi)
        for a, z in wins: ax.axvspan(a, z, color=WIN_C, lw=0, zorder=0)
        ax.axvline(0, color="k", lw=0.9)
        ax.fill_between(xc, rm - rse, rm + rse, color=MCOL, alpha=0.2, lw=0); ax.plot(xc, rm, color=MCOL, lw=1.8, label="spike rate")
        tw = ax.twinx(); tw.plot(xc, hm, color=HGC, lw=1.6, label="high gamma"); twins.append(tw)
        ax.set_xlim(lo, hi); ax.set_xlabel(xl); tw.spines["top"].set_visible(False)
    lo_ = np.nanmin([np.nanmin(t.get_ylim()) for t in twins]); hi_ = np.nanmax([np.nanmax(t.get_ylim()) for t in twins])
    for tw in twins: tw.set_ylim(lo_, hi_)
    twins[0].tick_params(labelright=False); twins[1].set_ylabel("High gamma (% vs pre)", color=HGC)
    b1.set_ylabel("Spikes/s", color=MCOL); plt.setp(b3.get_yticklabels(), visible=False)
    b1.set_title(f"B  Same channel, {len(trials)} correct long trials (mean ± SEM)", loc="left", fontsize=10, fontweight="bold")
    h1, l1 = b1.get_legend_handles_labels(); h2, l2 = twins[0].get_legend_handles_labels(); b1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="lower left")
    # C: bars, all sessions (from the MUA cell)
    c1 = fig.add_subplot(gs[:, 2]); d = SV[SV.group == "mPFC (IL+PL+Cg1)"]; fmt_q = lambda q: "q < 0.001" if q < 0.001 else f"q = {q:.3f}"
    for k, c in enumerate(("hold-pre", "hold-post")):
        v = pcf(d[c]); m, _, _ = fn.sign_flip_p(d[c].values); q = RM[(RM.group == "mPFC (IL+PL+Cg1)") & (RM.comparison == c)].q.iloc[0]
        c1.bar(k, pcf(m), width=0.6, color=MCOL, alpha=0.85, edgecolor="k", lw=0.8, zorder=2)
        c1.errorbar(k, pcf(m), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="none", color="k", capsize=5, lw=1.3, zorder=3)
        c1.text(k, 1.01, f"{pcf(m):+.1f} %\n{fmt_q(q)}", transform=c1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9.5)
    c1.axhline(0, color="k", lw=0.9); c1.set_xticks([0, 1]); c1.set_xticklabels(["Hold vs\nbefore press", "Hold vs\nafter release"]); c1.set_xlim(-0.6, 1.6)
    c1.set_ylabel("Multi-unit firing change (%)")
    c1.set_title(f"C  All sessions ({d.session.nunique()}), mPFC", loc="left", fontsize=10, fontweight="bold", pad=30)
    for ax in (a1, a2, b1, b3, c1):
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    fp = os.path.join(DESK, "mua_example_and_bars.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))
    print(f"example trial: hold {ex['hold']:.2f} s, spikes pre {((ex['spk'] >= -0.5) & (ex['spk'] < -0.2)).sum()} (0.3 s) | hold "
          f"{((ex['spk'] >= 0.2) & (ex['spk'] < 0.85)).sum()} (0.65 s) | channel mean hold/pre ratio {mean_ratio:.2f} | saved: {fp}")


@cell()
def gamma_spikes_over_time():
    """INDEPENDENT CELL (run in 06): HIGH GAMMA + SPIKES OVER TIME (all sessions) and ONE REAL TRACE WITH THE SPIKES (correct long, CMR)"""
    global BIN_S, BP_CMR, CH, CHREG, CUR_DIR, DATA_ROOT, DEAD_MS, DESK, Dd, EDG, EX_MIN_RATE, EX_SES, GCOL, H, HGc, K_THR
    global MAIN, MCOL, MIN_RATE, MUc, N_TR, Path, RAW, REG, SHORT, SHOW, T, TABLES, TRL, WCOL, WIN, X, _, a0, a1, a_, al
    global ap_bin, ax, axA, axB, axC, c, cand, cl, cnt, col, crossings, dead, drop, ex, fbp, fc, fig, final, fn, fp, fs, glob
    global gs, h, hm, hold, importlib, k_ex, l, lab, lb, load_bp, lr, m, mid, n_, n_hold, n_pre, n_tot, noise, np, ok, os, pcf
    global pd, pickle, plt, r, rate, raw, rec, rmean, rng, s, se, sel, si, sp_, st, starts, sys, thr, tr, tt, w, warnings, x
    global x_, xl, ylo, z, z_
    # ── INDEPENDENT CELL (run in 06): HIGH GAMMA + SPIKES OVER TIME (all sessions) and ONE REAL TRACE WITH THE SPIKES (correct long, CMR) ──
    #    Top (A): same as "high gamma and multi-unit firing over time": high gamma = saved 06 curves (log10 power / own pre-press, trials only while
    #      on the lever, median over all channels); spikes = cache tables\mua_curves (threshold crossings, 50-ms bins, mPFC channels ≥ 1 spike/s before
    #      the press, log10(rate / own pre rate), median over channels). Mean ± SEM over sessions, % vs before the press.
    #    Bottom (B): ONE real trial of ONE channel: spike band (300–6000 Hz, phase_shift, CMR — same processing as the MUA cells), from 0.7 s before
    #      the press to 0.7 s after the release; every detected spike (crossing of −4.5 × noise) marked; spike count per window written on top.
    #      Example session EX_SES; channel = the mPFC channel (≥ 5 spikes/s before the press) whose hold/pre drop is closest to the session median
    #      (typical, not best); trial = the trial of that channel whose hold/pre spike ratio is closest to the channel's mean (typical trial).
    EX_SES = "20260529_r0000136_wt_1318"                                       # example session (change for another)
    import os, sys, glob, pickle, importlib, warnings
    from pathlib import Path
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    MAIN = r"C:\Users\ar687\Documents\GitHub\AmirAnalysis\main"
    sys.path.insert(0, MAIN)
    import functions as fn
    DATA_ROOT = r"O:\archive\projects\2020_euSNN\Neuropixels_RT_mPFC\data"
    RAW, TABLES = os.path.join(DATA_ROOT, "0_raw"), os.path.join(MAIN, "tables")
    BP_CMR = os.path.join(DATA_ROOT, "amir_data", "data", "band power", "CMR")
    CUR_DIR = os.path.join(TABLES, "mua_curves")
    final = pd.read_csv(os.path.join(TABLES, "final_sessions.csv"), dtype={"rat": str})
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    SHORT = {"Infralimbic area": "IL", "Prelimbic area": "PL", "Cingulate area 1": "Cg1"}
    K_THR, N_TR, DEAD_MS, MIN_RATE, EX_MIN_RATE, BIN_S = 4.5, 60, 1.0, 1.0, 5.0, 0.05
    SHOW = {"Press": (-0.7, 1.2), "Release": (-0.7, 0.7)}
    EDG = {al: np.arange(lo, hi + 1e-9, BIN_S) for al, (lo, hi) in SHOW.items()}; mid = {al: EDG[al][:-1] + BIN_S / 2 for al in SHOW}
    WIN = {"Press": [("pre", -0.5, -0.2), ("hold", 0.20, 0.85)], "Release": [("post", 0.20, 0.50)]}
    GCOL, MCOL, WCOL = "#6a1b9a", "#455A64", "#F6DADD"
    pcf = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    load_bp = lambda s: pickle.load(open(os.path.join(BP_CMR, s + ".pkl"), "rb"))
    DESK = next((p for p in (os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"), os.path.join(os.path.expanduser("~"), "Desktop"))
                 if os.path.isdir(p)), os.path.expanduser("~"))

    def crossings(x, thr, dead):
        """onsets (samples) of downward crossings of -thr per channel; x (samples, channels)"""
        below = x < -thr[None, :]; on = below[1:] & ~below[:-1]; out = []
        for j in range(x.shape[1]):
            t = np.flatnonzero(on[:, j]) + 1
            if len(t) > 1: t = t[np.r_[True, np.diff(t) > dead]]
            out.append(t)
        return out

    # ── 1) session curves: high gamma (06 band power, CMR) and spikes (cache) ──
    HGc = {al: [] for al in SHOW}; MUc = {al: [] for al in SHOW}
    for s in final["session"]:
        fbp = os.path.join(BP_CMR, s + ".pkl")
        if os.path.exists(fbp):
            r = load_bp(s)
            for al in SHOW:
                c = fn.session_curve(r, al, "highG 61-100", "correct long")
                if c is not None: HGc[al].append(np.interp(mid[al], r["times"][al], c))
        fc = os.path.join(CUR_DIR, s + ".npz")
        if not os.path.exists(fc): continue
        z = np.load(fc); ok = z["pre_rate"] >= MIN_RATE
        if ok.sum() == 0: continue
        for al in SHOW:
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = z[f"{al}_counts"][ok] / (z[f"{al}_ntr"][None, :] * BIN_S)
                lr = np.log10((rate + 0.1) / (z["pre_rate"][ok][:, None] + 0.1))
            lr[:, z[f"{al}_ntr"] < 3] = np.nan
            MUc[al].append(np.nanmedian(lr, axis=0))

    # ── 2) example channel and typical trial (raw spike band) ──
    import spikeinterface.full as si
    z = np.load(os.path.join(CUR_DIR, EX_SES + ".npz")); hm = (mid["Press"] >= 0.2) & (mid["Press"] <= 0.85)
    with np.errstate(invalid="ignore", divide="ignore"):
        drop = np.log10((z["Press_counts"][:, hm].sum(1) / (z["Press_ntr"][hm].sum() * BIN_S) + 0.1) / (z["pre_rate"] + 0.1))
    cand = np.where(z["pre_rate"] >= EX_MIN_RATE)[0]; k_ex = cand[np.argmin(np.abs(drop[cand] - np.median(drop[cand])))]
    CH = str(z["channel_id"][k_ex]); REG = SHORT.get(CHREG[(CHREG.session == EX_SES) & (CHREG.channel_id == CH)].region.iloc[0], "other")
    T = load_bp(EX_SES)["trials"]; cl = T[(T["label"] == "correct") & (T["type"] == "long")]; cl = cl.sample(min(N_TR, len(cl)), random_state=0)
    ap_bin = glob.glob(os.path.join(RAW, EX_SES, "**", "*.ap.bin"), recursive=True)[0]
    raw = si.read_spikeglx(Path(ap_bin).parent, stream_name="imec0.ap")
    rec = si.common_reference(si.phase_shift(si.bandpass_filter(raw, freq_min=300, freq_max=6000, dtype="float32")), reference="global", operator="median")
    fs, n_tot = rec.get_sampling_frequency(), rec.get_num_samples(); sel = [c for c in rec.get_channel_ids() if str(c) == CH]
    rng = np.random.default_rng(0); starts = rng.integers(int(10 * fs), n_tot - int(2 * fs), 20)
    noise = np.median([np.median(np.abs(rec.get_traces(start_frame=a, end_frame=a + int(fs), channel_ids=sel, return_in_uV=True)), axis=0) / 0.6745
                       for a in starts], axis=0)
    thr, dead = K_THR * noise, int(DEAD_MS * fs / 1000)
    TRL = []
    for _, tr in cl.iterrows():
        hold = tr["release_s"] - tr["press_s"]; a0 = int((tr["press_s"] - 0.7) * fs); a1 = int((tr["release_s"] + 0.7) * fs)
        if a0 < 0 or a1 > n_tot: continue
        x = rec.get_traces(start_frame=a0, end_frame=a1, channel_ids=sel, return_in_uV=True)[:, 0]
        st = crossings(x[:, None], thr, dead)[0] / fs - 0.7
        n_pre, n_hold = ((st >= -0.5) & (st < -0.2)).sum(), ((st >= 0.2) & (st < 0.85)).sum()
        TRL.append(dict(hold=hold, x=x, st=st, ratio=np.log10((n_hold / 0.65 + 1) / (n_pre / 0.3 + 1))))
    rmean = np.mean([t["ratio"] for t in TRL]); ex = min(TRL, key=lambda t: abs(t["ratio"] - rmean))
    print(f"example: {EX_SES}, channel {CH.split('#')[-1]} ({REG}), {z['pre_rate'][k_ex]:.1f} spikes/s before the press; typical trial held {ex['hold']:.2f} s", flush=True)

    # ── 3) figure ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(12, 7.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1.4], height_ratios=[1, 1.05], hspace=0.55, wspace=0.08)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1], sharey=axA)
    for ax, al, xl in ((axA, "Press", "Time from press (s)"), (axB, "Release", "Time from release (s)")):
        for Dd, col, lab in ((HGc, GCOL, f"high gamma ({len(HGc[al])} sessions)"), (MUc, MCOL, f"multi-unit firing ({len(MUc[al])} sessions)")):
            X = np.array(Dd[al])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(X, 0); se = np.nanstd(X, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(X), 0))
            ax.fill_between(mid[al], pcf(m - se), pcf(m + se), color=col, alpha=0.2, lw=0); ax.plot(mid[al], pcf(m), color=col, lw=2, label=lab)
        ax.axhline(0, color="0.5", lw=0.8, ls="--"); ax.axvline(0, color="k", lw=0.9)
        for w, a_, z_ in WIN[al]:
            ax.axvspan(a_, z_, color=WCOL, zorder=0, lw=0)
            ax.text((a_ + z_) / 2, 1.0, w, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c62828")
        ax.set_xlim(*SHOW[al]); ax.set_xlabel(xl)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    plt.setp(axB.get_yticklabels(), visible=False)
    axA.set_ylabel("Change vs before the press (%)")
    axA.set_title("A  Average of all sessions", loc="left", fontsize=10.5, fontweight="bold", pad=16)
    h, l = axA.get_legend_handles_labels(); axB.legend(h, l, frameon=False, fontsize=9, loc="lower right")

    axC = fig.add_subplot(gs[1, :])
    tt = np.arange(len(ex["x"])) / fs - 0.7; H = ex["hold"]
    for w, a_, z_ in (("pre", -0.5, -0.2), ("hold", 0.2, 0.85), ("post", H + 0.2, H + 0.5)):
        axC.axvspan(a_, z_, color=WCOL, zorder=0, lw=0)
        n_ = ((ex["st"] >= a_) & (ex["st"] < z_)).sum()
        axC.text((a_ + z_) / 2, 1.0, f"{w}\n{n_} spikes ({n_ / (z_ - a_):.0f}/s)", transform=axC.get_xaxis_transform(), ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold", color="#c62828")
    axC.plot(tt, ex["x"], color=MCOL, lw=0.4)
    ylo = np.percentile(ex["x"], 0.05) * 1.15
    axC.plot(ex["st"], np.full(len(ex["st"]), ylo), "|", color="k", ms=9, mew=1.2, label="detected spike")
    axC.axhline(-thr[0], color="#c62828", lw=0.7, ls=":", label=f"threshold (−{K_THR} × noise)")
    for x_, lb in ((0, "press"), (H, "release")): axC.axvline(x_, color="k", lw=1); axC.text(x_, ylo * 1.12, lb, ha="center", va="top", fontsize=9.5)
    axC.set_xlim(-0.7, H + 0.7); axC.set_ylim(ylo * 1.3, np.percentile(ex["x"], 99.95) * 1.3)
    axC.set_xlabel("Time from press (s)"); axC.set_ylabel("Spike band 300–6000 Hz (µV)")
    axC.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.25), ncol=2)
    axC.set_title(f"B  One real trial, one channel ({EX_SES[:8]}, rat {EX_SES.split('_')[1][-3:]}, {CH.split('#')[-1]} {REG}; typical channel and trial)",
                  loc="left", fontsize=10.5, fontweight="bold", pad=34)
    for sp_ in ("top", "right"): axC.spines[sp_].set_visible(False)
    fig.suptitle("Fewer spikes while waiting: high gamma and multi-unit firing drop during the hold (correct long trials)", fontsize=12, fontweight="bold", y=1.0)
    fp = os.path.join(DESK, "high_gamma_spikes_average_and_trace.png"); _savefig(fig, fp); plt.show()
    cnt = lambda a, b: int(((ex["st"] >= a) & (ex["st"] < b)).sum())
    print(f"trial: held {H:.2f} s | spikes pre {cnt(-0.5, -0.2)} ({cnt(-0.5, -0.2) / 0.3:.0f}/s), hold {cnt(0.2, 0.85)} ({cnt(0.2, 0.85) / 0.65:.0f}/s), "
          f"post {cnt(H + 0.2, H + 0.5)} ({cnt(H + 0.2, H + 0.5) / 0.3:.0f}/s)")


@cell()
def theta_across_session():
    """Fig. S13. Theta across the session (pooled mPFC, CMR): before the press (dashed) and during the hold (solid), with tests"""
    global BAND, CHREG, CUTS, D, L, MIN_CH, MIN_HOLD, MIN_TR, NB, P, PHASES, REF, REGS, T, THETA_COL, W, _, a, ax, b, base
    global col, d, fig, g, gen_dir, held, hi, hold, hold_r, k, lab, lo, ls, m, masks, multipletests, n, p, part, pc, ph
    global ph_rows, pos, pre, pre_r, q, r, rows, s, sp, sup_dir, t0, t1, t_, tests, v, x, x0, x1
    # ── Fig. S13. Theta across the session (pooled mPFC, CMR): before the press (dashed) and during the hold (solid), with tests ──
    from statsmodels.stats.multitest import multipletests
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    REF, BAND, REGS, NB, MIN_CH, MIN_TR, MIN_HOLD = "CMR", "theta 4-7", ["IL", "PL", "Cg1"], 20, 5, 3, 0.85
    CUTS = (0.25, 0.85)                                                              # start / middle / end, as for beta
    PHASES = {"start": (0, CUTS[0]), "middle": CUTS, "end": (CUTS[1], 1.0001)}
    THETA_COL = "#2e7d32"
    pc = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    sup_dir = os.path.join(AMIR, "supplementary"); os.makedirs(sup_dir, exist_ok=True)
    gen_dir = os.path.join(AMIR, "Theta", "How theta behaves in general"); os.makedirs(gen_dir, exist_ok=True)

    rows, ph_rows = [], []
    for s in done_sessions(REF):
        r = load_bp(s, REF); T = r["trials"]; L = r["logpow"][BAND]
        masks, _ = fn.region_masks(r, CHREG, s)
        t0, t1 = np.nanmin(T["press_s"]), np.nanmax(T["release_s"])
        pos = (T["press_s"].values - t0) / (t1 - t0)
        part = np.clip((pos * NB).astype(int), 0, NB - 1)
        held = (T["type"].values == "long") & (T["hold_s"].values >= MIN_HOLD)
        pre_r, hold_r = [], []
        for g in REGS:
            m = masks[g]
            if m.sum() < MIN_CH: continue
            pre_r.append(np.nanmedian(L["pre"][:, m], axis=1)); hold_r.append(np.nanmedian(L["hold"][:, m], axis=1))
        if not pre_r: continue
        pre, hold = np.nanmean(pre_r, axis=0), np.nanmean(hold_r, axis=0)          # pooled mPFC: mean of the regions (log10)
        base = np.nanmean(pre)                                                       # session's average theta before the press
        for k in range(NB):                                                          # 20 parts (figure)
            a, b = part == k, (part == k) & held
            rows.append(dict(session=s, rat=str(RAT[s]), part=k,
                             pre=pc(np.nanmean(pre[a]) - base) if a.sum() >= MIN_TR else np.nan,
                             hold=pc(np.nanmean(hold[b]) - base) if b.sum() >= MIN_TR else np.nan))
        for ph, (lo, hi) in PHASES.items():                                          # 3 parts (tests), log10 vs session pre level
            b = held & (pos >= lo) & (pos < hi)
            ph_rows.append(dict(session=s, phase=ph, hold=np.nanmean(hold[b]) - base if b.sum() >= MIN_TR else np.nan))
        del r
    D, P = pd.DataFrame(rows), pd.DataFrame(ph_rows)
    _table(D, os.path.join(gen_dir, "theta_across_session_pooled_CMR_data.csv"), index=False)

    # tests: hold vs session pre level in each part; start vs middle, end vs middle (paired); FDR over the 5 tests
    W = P.pivot(index="session", columns="phase", values="hold"); tests = []
    for ph in PHASES:
        v = W[ph].dropna(); m, p, n = fn.sign_flip_p(v.values)
        tests.append({"test": f"hold vs pre level, {ph}", "n_sessions": n, "sessions_up": int((v > 0).sum()), "mean_%": round(float(pc(m)), 1), "p": p})
    for a in ["start", "end"]:
        d = (W[a] - W["middle"]).dropna(); m, p, n = fn.sign_flip_p(d.values)
        tests.append({"test": f"{a} − middle", "n_sessions": n, "sessions_up": int((d > 0).sum()), "mean_%": round(float(pc(m)), 1), "p": p})
    for t_, q in zip(tests, multipletests([x["p"] for x in tests], method="fdr_bh")[1]):
        t_["q"] = q

    # figure (paper style)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(4.4, 2.9), facecolor="white")
    for x0, x1 in [(0, CUTS[0]), (CUTS[1], 1)]:
        ax.axvspan(x0, x1, color="0.93", lw=0, zorder=0)
    for ph, (lo, hi) in PHASES.items():
        ax.text((lo + min(hi, 1)) / 2, 1.01, ph, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5, color="0.35")
    for col, ls, lab in (("hold", "-", f"During the hold (long trials held ≥ {MIN_HOLD} s)"),
                         ("pre", "--", "Before the press (all trials)")):
        g = D.groupby("part")[col].agg(["mean", "sem"]); x = (g.index + 0.5) / NB
        ax.fill_between(x, g["mean"] - g["sem"], g["mean"] + g["sem"], color=THETA_COL, alpha=0.15, lw=0)
        ax.plot(x, g["mean"], color=THETA_COL, lw=1.6, ls=ls, label=lab)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlim(0, 1); ax.set_xlabel("Position in the session (start → end)")
    ax.set_ylabel("Theta power change from the\nsession's level before the press (%)")
    ax.legend(frameon=False, fontsize=7, loc="center right")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _savefig(fig, os.path.join(gen_dir, "theta_across_session_pooled_CMR.png"))
    _savefig(fig, os.path.join(sup_dir, "Fig S12 theta across the session CMR.png"))
    plt.show()
    print(f"{D.session.nunique()} sessions (figure); parts: start 0–{CUTS[0]}, middle {CUTS[0]}–{CUTS[1]}, end {CUTS[1]}–1")
    print(pd.DataFrame(tests).round({"p": 4, "q": 4}).to_string(index=False))


@cell()
def theta_gamma_along_session():
    """THETA AND HIGH GAMMA DURING THE HOLD ALONG THE SESSION (correct long, all sessions, CMR): A) along the session, B) start vs middle and start vs end"""
    global BANDS_A, DA, DESK, Dlt, M, MIN_T, NB_A, P, PARTS_A, PP, REF_A, ST_A, T, a, a1, a2, ax, b, bins, cl, col, d, fig
    global fmt_q, fname, fp, j, k, m, mb, mu, multipletests, n, name, o_pct, other, p, p_, pcf, pos, q, r, rng_, rows, s
    global s_pct, sb, se, sp, st, t, top, x, xo, y, z
    # ── THETA AND HIGH GAMMA DURING THE HOLD ALONG THE SESSION (correct long, all sessions, CMR): A) along the session, B) start vs middle and start vs end ──
    #    Value = hold (0.2–0.85 s) vs before the press (−0.5…−0.2 s) of the same trial, log10; per channel mean over trials, median over channels (as 1e).
    #    Position = press time in the session (clock time, first → last kept press). A: 10 parts, mean ± SEM over sessions (part drawn if ≥ 3 trials).
    #    B: start 0–25 %, middle 25–85 %, end 85–100 % (as the report); paired sign-flip across sessions (sessions with ≥ 3 trials in both parts);
    #    q = FDR-BH over the 4 tests (2 bands × 2 comparisons). Saved on the desktop (needs cell 0).
    from statsmodels.stats.multitest import multipletests
    DESK = next((p for p in (os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"), os.path.join(os.path.expanduser("~"), "Desktop"))
                 if os.path.isdir(p)), os.path.expanduser("~"))
    REF_A, NB_A, MIN_T = "CMR", 10, 3
    BANDS_A = {"theta 4-7": ("Theta", "#2e7d32", "theta"), "highG 61-100": ("High gamma", "#6a1b9a", "high_gamma")}
    PARTS_A = {"start": (0.0, 0.25), "middle": (0.25, 0.85), "end": (0.85, 1.0001)}
    pcf = lambda x: 100 * (10 ** np.asarray(x, float) - 1)
    fmt_q = lambda q: "q < 0.001" if q < 0.001 else f"q = {q:.3f}"

    # ── 1) session values per part ──
    rows = []
    for s in done_sessions(REF_A):
        r = load_bp(s, REF_A); T = r["trials"]
        t = T["press_s"].values; pos = (t - np.nanmin(t)) / (np.nanmax(t) - np.nanmin(t))
        cl = ((T["label"] == "correct") & (T["type"] == "long")).values
        bins = np.minimum((pos * NB_A).astype(int), NB_A - 1)
        for b in BANDS_A:
            Dlt = r["logpow"][b]["hold"] - r["logpow"][b]["pre"]                       # (trials, channels), same trial
            for k in range(NB_A):
                m = cl & (bins == k)
                if m.sum() >= MIN_T: rows.append(dict(session=s, band=b, kind="bin", part=k, n=int(m.sum()), val=float(np.median(Dlt[m].mean(0)))))
            for p_, (a, z) in PARTS_A.items():
                m = cl & (pos >= a) & (pos < z)
                if m.sum() >= MIN_T: rows.append(dict(session=s, band=b, kind="part", part=p_, n=int(m.sum()), val=float(np.median(Dlt[m].mean(0)))))
    DA = pd.DataFrame(rows)

    # ── 2) tests: start vs middle, start vs end (paired across sessions) ──
    st = []
    for b in BANDS_A:
        P = DA[(DA.band == b) & (DA.kind == "part")].pivot(index="session", columns="part", values="val")
        for other in ("middle", "end"):
            PP = P[["start", other]].dropna(); d = (PP["start"] - PP[other]).values; m, p, n = fn.sign_flip_p(d)
            s_pct, o_pct = float(pcf(PP["start"].mean())), float(pcf(PP[other].mean()))                 # paired sessions only
            st.append(dict(band=BANDS_A[b][0], comparison=f"start − {other}", n_sessions=n, sessions_lower_at_start=int((d < 0).sum()),
                           start_pct=round(s_pct, 1), other_pct=round(o_pct, 1), diff_pct_points=round(s_pct - o_pct, 1), p=p))
    ST_A = pd.DataFrame(st); ST_A["q"] = multipletests(ST_A["p"], method="fdr_bh")[1]
    _table(ST_A, os.path.join(DESK, f"theta_highgamma_along_session_tests_{REF_A}.csv"), index=False)

    # ── 3) figures: one per band, A along the session + B bars ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    for b, (name, col, fname) in BANDS_A.items():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), facecolor="white", gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.35})
        # A
        M = DA[(DA.band == b) & (DA.kind == "bin")].pivot(index="session", columns="part", values="val").reindex(columns=range(NB_A))
        x = (np.arange(NB_A) + 0.5) / NB_A * 100; mu = M.mean(); se = M.std() / np.sqrt(M.notna().sum())
        a1.axvspan(0, 25, color="0.93", zorder=0); a1.axvspan(85, 100, color="0.93", zorder=0)
        a1.fill_between(x, pcf(mu - se), pcf(mu + se), color=col, alpha=0.2, lw=0); a1.plot(x, pcf(mu), color=col, lw=2, marker="o", ms=3.5)
        a1.axhline(0, color="0.5", lw=0.8, ls="--")
        for p_, (a, z) in PARTS_A.items():
            a1.text((a + min(z, 1)) / 2 * 100, 1.01, p_, transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="0.35")
        a1.set_xlim(0, 100); a1.set_xlabel("Position in the session (% of session time)"); a1.set_ylabel(f"{name}, hold vs before press (%)")
        # B
        P = DA[(DA.band == b) & (DA.kind == "part")].pivot(index="session", columns="part", values="val")[list(PARTS_A)]
        mb = pcf(P.mean().values); sb = np.array([pcf(P[c].dropna()).std(ddof=1) / np.sqrt(P[c].notna().sum()) for c in P])
        a2.bar(range(3), mb, width=0.6, color=col, alpha=0.85, edgecolor="k", lw=0.8, zorder=2)
        a2.errorbar(range(3), mb, yerr=sb, fmt="none", color="k", capsize=4, lw=1.3, zorder=3)
        a2.axhline(0, color="k", lw=0.9)
        top = np.nanmax(np.r_[mb + sb, 0]); rng_ = np.nanmax(np.r_[mb + sb, 0]) - np.nanmin(np.r_[mb - sb, 0])
        for j, (other, xo) in enumerate((("middle", 1), ("end", 2))):
            q = ST_A[(ST_A.band == name) & (ST_A.comparison == f"start − {other}")].q.iloc[0]
            y = top + rng_ * (0.12 + 0.17 * j)
            a2.plot([0, 0, xo, xo], [y - rng_ * 0.03, y, y, y - rng_ * 0.03], color="k", lw=0.9)
            a2.text(xo / 2, y + rng_ * 0.01, fmt_q(q), ha="center", va="bottom", fontsize=8.5, fontweight="bold" if q < 0.05 else "normal")
        a2.set_ylim(top=top + rng_ * 0.55)
        a2.set_xticks(range(3)); a2.set_xticklabels(list(PARTS_A)); a2.set_ylabel(f"{name}, hold vs before press (%)")
        for ax in (a1, a2):
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        fp = os.path.join(DESK, f"{fname}_along_the_session_{REF_A}.png")
        _savefig(fig, fp); plt.close(fig); display(Image(fp))

    print(f"correct long trials, {REF_A}: sessions per band:", DA[DA.kind == "part"].groupby("band").session.nunique().to_dict(),
          "| trials per part (median over sessions):", DA[DA.kind == "part"].groupby("part").n.median().to_dict())
    print(ST_A.round({"p": 4, "q": 4}).to_string(index=False))


@cell()
def theta_three_splits():
    """THETA DURING THE HOLD ALONG THE SESSION (correct long, all sessions, CMR): A) by trial number, B) start / middle / end with three ways to split"""
    global ALPHA, BAND_A, CUTS, DA, DEFS, DESK, Dlt, M, MB, MIN_T, NB_A, P, PHS, PP, PT, REF_A, SB, SPL, ST_A, T, TCOL, XS, a
    global a1, a2, ax, bins, cl, col, d, fig, fmt_q, fp, i, j, k, lev, low, m, mu, multipletests, n, ok, other, p, p_, pcf, ph
    global phase, pos, q, r, rng_, rows, s, se, sp, split, st, t0_, t1_, top, v, vol, w, x, x0, x1, y, z
    # ── THETA DURING THE HOLD ALONG THE SESSION (correct long, all sessions, CMR): A) by trial number, B) start / middle / end with three ways to split ──
    #    Value = hold (0.2–0.85 s) vs before the press (−0.5…−0.2 s) of the same trial, log10; per channel mean over trials, median over channels (as 1e).
    #    Positions as in beta Step 3 (trials_filtered.csv): trial number = rank of the trial in the session (first → last trial), clock time = press time
    #    (first → last press), water drunk = share of the session's reward volume received before the trial. Parts: start < 0.25, middle 0.25–0.85,
    #    end ≥ 0.85 (same cuts for the three splits, as Step 3). A: 10 parts by trial number, mean ± SEM over sessions (part used if ≥ 3 trials).
    #    B: bars = mean ± SEM over sessions (three splits); brackets only for trial number: middle or end vs start (paired sign-flip across sessions
    #    with ≥ 3 trials in both); q = FDR-BH over the 2 tests of each split. Saved on the desktop (needs cell 0).
    from statsmodels.stats.multitest import multipletests
    DESK = next((p for p in (os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"), os.path.join(os.path.expanduser("~"), "Desktop"))
                 if os.path.isdir(p)), os.path.expanduser("~"))
    REF_A, NB_A, MIN_T, BAND_A, TCOL = "CMR", 10, 3, "theta 4-7", "#2e7d32"
    PHS, CUTS = ["start", "middle", "end"], (0.25, 0.85)
    DEFS = {"trial number": "position", "clock time": "v_time", "water drunk": "v_water"}
    ALPHA = {"trial number": 0.95, "clock time": 0.6, "water drunk": 0.3}
    pcf = lambda x: 100 * (10 ** np.asarray(x, float) - 1)

    # ── trial positions (same as beta Step 3) ──
    PT = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False).sort_values(["session", "trial_index"])
    PT["position"] = PT.groupby("session")["trial_index"].rank(pct=True)
    t0_ = PT.groupby("session")["press_neural_s"].transform("min"); t1_ = PT.groupby("session")["press_neural_s"].transform("max")
    PT["v_time"] = (PT["press_neural_s"] - t0_) / (t1_ - t0_)
    vol = PT["reward_size"].fillna(0).clip(lower=0)
    PT["v_water"] = (vol.groupby(PT["session"]).cumsum() - vol) / vol.groupby(PT["session"]).transform("sum")
    phase = lambda v: np.where(v < CUTS[0], "start", np.where(v < CUTS[1], "middle", "end"))

    # ── 1) session values ──
    rows = []
    for s in done_sessions(REF_A):
        r = load_bp(s, REF_A); T = r["trials"]
        pos = T[["trial_index"]].merge(PT.loc[PT.session == s, ["trial_index", *DEFS.values()]], on="trial_index", how="left")
        cl = ((T["label"] == "correct") & (T["type"] == "long")).values
        Dlt = r["logpow"][BAND_A]["hold"] - r["logpow"][BAND_A]["pre"]                       # (trials, channels), same trial
        bins = np.minimum((np.nan_to_num(pos["position"].values, nan=-1) * NB_A).astype(int), NB_A - 1)   # NaN position -> no bin
        ok = pos["position"].notna().values
        for k in range(NB_A):
            m = cl & ok & (bins == k)
            if m.sum() >= MIN_T: rows.append(dict(session=s, split="bins", part=k, n=int(m.sum()), val=float(np.median(Dlt[m].mean(0)))))
        for split, col in DEFS.items():
            v = pos[col].values; ph = phase(v)
            for p_ in PHS:
                m = cl & np.isfinite(v) & (ph == p_)
                if m.sum() >= MIN_T: rows.append(dict(session=s, split=split, part=p_, n=int(m.sum()), val=float(np.median(Dlt[m].mean(0)))))
    DA = pd.DataFrame(rows)

    # ── 2) tests: middle and end vs start, per split ──
    st = []
    for split in DEFS:
        P = DA[DA.split == split].pivot(index="session", columns="part", values="val")
        for other in ("middle", "end"):
            PP = P[["start", other]].dropna(); d = (PP[other] - PP["start"]).values; m, p, n = fn.sign_flip_p(d)
            st.append(dict(split=split, comparison=f"{other} vs start", n_sessions=n, sessions_higher_than_start=int((d > 0).sum()),
                           start_pct=round(float(pcf(PP["start"].mean())), 1), other_pct=round(float(pcf(PP[other].mean())), 1), p=p))
    ST_A = pd.DataFrame(st); ST_A["q"] = ST_A.groupby("split")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])   # FDR over the 2 tests of each split
    _table(ST_A, os.path.join(DESK, f"theta_along_session_three_splits_{REF_A}.csv"), index=False)

    # ── 3) figure ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white", gridspec_kw={"width_ratios": [1.6, 1.2], "wspace": 0.3})
    # A: by trial number
    M = DA[DA.split == "bins"].pivot(index="session", columns="part", values="val").reindex(columns=range(NB_A))
    x = (np.arange(NB_A) + 0.5) / NB_A; mu = M.mean(); se = M.std() / np.sqrt(M.notna().sum())
    a1.axvspan(0, CUTS[0], color="0.93", zorder=0); a1.axvspan(CUTS[1], 1, color="0.93", zorder=0)
    a1.fill_between(x, pcf(mu - se), pcf(mu + se), color=TCOL, alpha=0.2, lw=0); a1.plot(x, pcf(mu), color=TCOL, lw=2, marker="o", ms=3.5)
    for p_, (a, z) in zip(PHS, ((0, CUTS[0]), CUTS, (CUTS[1], 1))):
        a1.text((a + z) / 2, 1.01, p_, transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="0.35")
    a1.set_xlim(0, 1); a1.set_xlabel("Position in the session (first → last trial)")
    a1.set_ylabel("Theta, hold vs before press (%)")
    # B: three splits, brackets = start vs middle and start vs end of the same split (q written on the bracket)
    w = 0.26; MB, SB, XS = {}, {}, {}
    fmt_q = lambda q: "q < 0.001" if q < 0.001 else f"q = {q:.3f}"
    for i, split in enumerate(DEFS):
        P = DA[DA.split == split].pivot(index="session", columns="part", values="val").reindex(columns=PHS)
        MB[split] = pcf(P.mean().values); SB[split] = np.array([pcf(P[c].dropna()).std(ddof=1) / np.sqrt(P[c].notna().sum()) for c in PHS])
        XS[split] = np.arange(3) + (i - 1) * w
        a2.bar(XS[split], MB[split], width=w, color=TCOL, alpha=ALPHA[split], edgecolor="k", lw=0.6, label=split, zorder=2)
        a2.errorbar(XS[split], MB[split], yerr=SB[split], fmt="none", color="k", capsize=3, lw=1.1, zorder=3)
    top = max(np.nanmax(np.r_[MB[k] + SB[k], 0]) for k in DEFS); low = min(np.nanmin(np.r_[MB[k] - SB[k], 0]) for k in DEFS); rng_ = top - low
    lev, SPL = 0, "trial number"                                               # brackets only for the trial-number split
    for j, other in ((1, "middle"), (2, "end")):
        q = ST_A[(ST_A.split == SPL) & (ST_A.comparison == f"{other} vs start")].q.iloc[0]
        y = top + rng_ * (0.08 + 0.13 * lev); x0, x1 = XS[SPL][0], XS[SPL][j]; lev += 1
        a2.plot([x0, x0, x1, x1], [y - rng_ * 0.02, y, y, y - rng_ * 0.02], color="k", lw=1.0)
        a2.text((x0 + x1) / 2, y + rng_ * 0.005, fmt_q(q), ha="center", va="bottom", fontsize=8.5, fontweight="bold" if q < 0.05 else "normal")
    a2.set_ylim(top=top + rng_ * (0.08 + 0.13 * lev + 0.05))
    a2.axhline(0, color="k", lw=0.8); a2.set_xticks(range(3)); a2.set_xticklabels([p_.capitalize() for p_ in PHS])
    a2.set_ylabel("Theta, hold vs before press (%)")
    a2.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in (a1, a2):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fp = os.path.join(DESK, f"theta_along_the_session_three_splits_{REF_A}.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))

    print(f"theta, correct long trials, {REF_A}: {DA.session.nunique()} sessions | trials per part (median over sessions):",
          DA[DA.split != "bins"].groupby(["split", "part"]).n.median().unstack()[PHS].to_dict("index"))
    print(ST_A.round({"p": 4, "q": 4}).to_string(index=False))


@cell()
def rt_vs_theta_drop():
    """REACTION TIME vs THETA DROP AFTER THE RELEASE: long correct + late trials pooled, same number of each per session, RT 0.1–1.5 s"""
    global BAND_RT, L, MIN_EACH, REF_RT, RTD, RT_MAX, RT_MIN, T, ax, drop, fig, g, ic, il, k, m_, n, n_, neg, ok, p_, pick, r
    global rat, rng_rt, rows, rt, s, sp, spearmanr
    # ── REACTION TIME vs THETA DROP AFTER THE RELEASE: long correct + late trials pooled, same number of each per session, RT 0.1–1.5 s ──
    from scipy.stats import spearmanr
    REF_RT, BAND_RT = "CMR", "theta 4-7"
    RT_MIN, RT_MAX, MIN_EACH = 0.1, 1.5, 5                 # RT window (s) | min. trials of EACH outcome per session (after the RT window)
    rng_rt = np.random.default_rng(0)
    rows = []
    for s in done_sessions(REF_RT):
        r = load_bp(s, REF_RT); T, L = r["trials"], r["logpow"][BAND_RT]
        rt = (T["hold_s"] - T["tone_delay_s"]).values                                    # reaction time = release − tone
        ok = (T["type"] == "long").values & np.isfinite(rt) & (rt >= RT_MIN) & (rt <= RT_MAX)
        ic, il = np.where(ok & (T["label"] == "correct").values)[0], np.where(ok & (T["label"] == "late").values)[0]
        n = min(len(ic), len(il))
        if n < MIN_EACH: continue
        pick = np.r_[rng_rt.choice(ic, n, replace=False), rng_rt.choice(il, n, replace=False)]   # same number of correct and late
        drop = np.median(L["hold"][pick] - L["post"][pick], axis=1)                     # theta drop per trial (log10 hold / post; > 0 = drop)
        rows.append(dict(session=s, rat=RAT[s], n_each=n, rho=spearmanr(drop, rt[pick])[0]))
    RTD = pd.DataFrame(rows)
    m_, p_, n_ = fn.sign_flip_p(RTD["rho"].values)
    neg = int((RTD["rho"] < 0).sum())
    print(f"{n_} sessions (≥ {MIN_EACH} correct and ≥ {MIN_EACH} late long trials with RT {RT_MIN}–{RT_MAX} s) | mean rho {m_:+.3f} | "
          f"sign-flip p = {p_:.4f} | negative in {neg}/{n_} sessions")
    print("per rat:", {RAT_LABEL[r]: f"{(g.rho < 0).sum()}/{len(g)} negative, mean {g.rho.mean():+.3f}" for r, g in RTD.groupby("rat")})
    _table(RTD, "rt_vs_theta_drop_sessions.csv", index=False)

    # ── figure: one dot = one session (rho), colour = rat; black = mean ± SEM ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(4.2, 4.4), facecolor="white", constrained_layout=True)
    for k, (rat, g) in enumerate(RTD.groupby("rat")):
        ax.scatter(k + np.random.default_rng(k).uniform(-0.12, 0.12, len(g)), g.rho, s=26, color=RCOL[rat], edgecolor="none", label=RAT_LABEL[rat])
    ax.errorbar(3.0, RTD.rho.mean(), yerr=RTD.rho.std(ddof=1) / np.sqrt(len(RTD)), fmt="o", color="k", ms=7, capsize=4, lw=1.6)
    ax.axhline(0, color="0.4", ls="--", lw=0.9)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels([RAT_LABEL[r] for r in sorted(RTD.rat.unique())] + ["all"])
    ax.set_xlim(-0.6, 3.6); ax.set_ylabel("Spearman ρ (theta drop vs RT)\none dot = one session")
    ax.set_title(f"Bigger theta drop → faster response?\nmean ρ = {m_:+.2f}, p = {p_:.3f}, negative in {neg}/{n_} sessions", fontsize=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    _savefig(fig, "rt_vs_theta_drop.png")
    plt.show()
