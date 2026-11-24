"""
fx_res_coupling.py — IL–PL theta (and delta) phase coupling: cells of 07 section 2 (and the coupling cells of section 4),
plus the coupling data steps run from 06 (overnight_tests, trial_phase_files, T8c_sums, T13_fix).

The constants and helpers below are shared by all coupling cells (they were in the old A1, A2b and "overnight" cells).
setup(B) adds the names that need the settings (paths, tables, the A run chosen with A_RUN).

Measures: PLV (phase-locking value), PLI (phase-lag index), PPC (Vinck 2010, PLV without trial-number bias),
debiased PLI² / wPLI. Chance: trial shuffle (derangement: the PL trial is never paired with its own IL trial).
Tests across sessions: strat_test = sign-flip test with every rat weighted equally (one-sided); two = two-sided.
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from fx_results import cell

# ── constants (same values as the old cells) ──
REF, MIN_RING, STEP_S, MINN = "SR30-500", 5, 0.010, 5
RAT, IL_XY, PL_XY, PAIR = "136", (282, 0), (532, 2220), "136|IL282,0|PL532,2220"
NSH, STEP_S, F0, MIN_ON, MIN_TR, MIN_RING = 500, 0.010, 5.5, 5, 10, 5
COL, COL_IL, COL_PL, PILOT_COL, DOT = "#43A047", "#ff7f0e", "#1f77b4", "#d62728", "0.45"
SHORT = {"Infralimbic area": "IL", "Prelimbic area": "PL", "Cingulate area 1": "Cg1"}
PAIRS3 = {"134": ((500, 600), (282, 2235), "134|IL500,600|PL282,2235 (134B)"),
          "135": ((250, 1290), (500, 3255), "135|IL250,1290|PL500,3255"),
          "136": ((282, 0), (532, 2220), "136|IL282,0|PL532,2220")}
RC = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}; RL = {"134": "Rat 1", "135": "Rat 2", "136": "Rat 3"}
CONDS3 = ["correct", "late", "early"]
CCOLS = {"correct": "#2CA02C", "late": "#DD8452", "early": "#E6B800"}
EARLY_MIN_HOLD, MIN_TR2 = 0.3, 5                       # ≥ 5 trials per condition, as in C11
ptxt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
fpt = lambda p: "n.a." if p is None or not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}")
two = lambda d, r: min(1.0, 2 * min(strat_test(d, r), strat_test(-np.asarray(d), r))) if len(d) > 1 else np.nan
rmean = lambda v, r: pd.Series(np.asarray(v)).groupby(np.asarray(r)).mean().mean()
A_RUNS = {1: ("SR30-500", 1.0, "press-1s"), 2: ("CMR", 1.0, "press-1s"), 3: ("SR30-500", 3.0, "press-release"), 4: ("CMR", 3.0, "press-release")}


def setup(B):
    """Names every coupling cell can use, built once from the settings (B = shared names of fx_results.setup)."""
    fn, TABLES, AMIR = B["fn"], B["TABLES"], B["AMIR"]
    REF, WEND, TAG = A_RUNS[B["A_RUN"]]                   # A1–A2d: reference | window end after the press (s) | folder tag
    FR, NCYC = np.array(fn.MAIN_BANDS["theta 4-7"]["freqs"]), fn.MAIN_BANDS["theta 4-7"]["n_cycles"]
    FR_TH = FR
    TESTS = os.path.join(AMIR, "Theta", "Phase connectivity", "Tests"); os.makedirs(TESTS, exist_ok=True)
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    K = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False); K = K[K["keep"]]
    BK = pd.read_csv(os.path.join(AMIR, "BACKUP_A_C11_C10_rows.csv"), dtype={"rat": str}, low_memory=False)
    SESS = {r: sorted(BK.loc[BK.pair == v[2], "session"].unique()) for r, v in PAIRS3.items()}
    rng = np.random.default_rng(0)
    return dict(REF=REF, WEND=WEND, TAG=TAG, FR=FR, NCYC=NCYC, FR_TH=FR_TH, TESTS=TESTS, LOGF=os.path.join(TESTS, "ALL_overnight_log.txt"),
                CHREG=CHREG, K=K, BK=BK, SESS=SESS, rng=rng)
SETUP_NAMES = ("REF", "WEND", "TAG", "FR", "NCYC", "FR_TH", "TESTS", "LOGF", "CHREG", "K", "BK", "SESS", "rng")


# ── helpers (from the old A1, A2b and overnight cells; unchanged) ──
def onesided(v, n=10000):
    null = (np.random.default_rng(0).choice([-1.0, 1.0], size=(n, len(v))) * v).mean(1)
    return (1 + np.sum(null >= v.mean())) / (n + 1)

def rayleigh(th):
    n = len(th); R = np.abs(np.exp(1j * th).mean())
    return R, float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n)))

def get_signals(session, T):
    """IL and PL epochs around the press, with the chosen reference"""
    a0, a1 = fn.MAIN_EPOCHS["Press"]
    r_ = "original" if REF == "SR30-500" else REF
    rec = fn.load_session(PROC, session, XB, refs=(r_,))[0][r_]
    ids = [str(c) for c in rec.get_channel_ids()]; fs = rec.get_sampling_frequency()
    X, ok = fn.read_epochs(rec, T["press_s"].values, a0, a1)
    regmap = CHREG[CHREG.session == session].set_index("channel_id")["region"]
    out, used = [], []
    for xy in (IL_XY, PL_XY):
        c = chan_at(session, xy)
        if c is None or c not in ids: raise ValueError(f"{session}: no channel at {xy}")
        i = ids.index(c); used.append(c)
        if REF == "SR30-500":
            loc = rec.get_channel_locations()
            reg = np.array([SHORT.get(regmap.get(k, ""), "other") for k in ids])
            dd = np.hypot(loc[:, 0] - loc[i, 0], loc[:, 1] - loc[i, 1])
            ring = np.where((reg == reg[i]) & (dd > 30) & (dd <= 500))[0]
            if len(ring) < MIN_RING: raise ValueError(f"{session}: ring of {c} has {len(ring)} channels")
            out.append((X[:, i] - np.median(X[:, ring], axis=1)).astype(np.float64))
        else:
            out.append(X[:, i].astype(np.float64))
    return out[0], out[1], np.arange(X.shape[-1]) / fs + a0, fs, ok, used

def wavelet(x, fs, idx):
    return np.stack([ss.fftconvolve(x, fn.morlet(f, NCYC, fs)[None, :], mode="same", axes=-1) for f in FR], axis=1)[..., idx]

def prepare(xI, xP, t, fs, hold):
    """press → WEND; trials sorted by hold (longest first): at time j the held trials are the first n_on[j]"""
    idx = np.where((t >= -1e-9) & (t <= WEND + 1e-9))[0][::int(round(STEP_S * fs))]
    tt, o = t[idx], np.argsort(-hold)
    A_, B_, h_ = wavelet(xI[o], fs, idx), wavelet(xP[o], fs, idx), hold[o]
    n_on = (h_[:, None] >= tt[None, :]).sum(0)
    groups = [(n_, np.where(n_on == n_)[0]) for n_ in np.unique(n_on[n_on >= MIN_ON])]
    M_ = (h_[:, None] >= tt[None, :]) & (n_on >= MIN_ON)[None, :]
    return A_, B_, h_, tt, groups, M_, o

def plv_pli(A_, B_, groups, shuffle=False):
    pv, pl = [], []
    for n_, cols in groups:
        b = B_[:n_][derangement(n_)] if shuffle else B_[:n_]
        S = A_[:n_][..., cols] * np.conj(b[..., cols])
        pv.append(np.abs((S / np.abs(S).clip(1e-30)).mean(0)).mean(0)); pl.append(np.abs(np.sign(S.imag).mean(0)).mean(0))
    return np.concatenate(pv).mean(), np.concatenate(pl).mean()

def direction(A_, B_, M_):
    E_ = A_ * np.conj(B_); E_ = E_ / np.abs(E_).clip(1e-30) * M_[:, None, :]
    return np.angle(E_.sum()), np.angle(E_.sum(axis=(1, 2)))

def get_pair(session, T, il_xy, pl_xy):
    global IL_XY, PL_XY
    keep_ = (IL_XY, PL_XY); IL_XY, PL_XY = il_xy, pl_xy
    try: return get_signals(session, T)
    finally: IL_XY, PL_XY = keep_

def log(msg):
    print(msg, flush=True); open(LOGF, "a", encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M ") + str(msg) + "\n")

def cond_trials(T, c):
    hold = T["release_s"] - T["press_s"]
    if c == "early": return (T["type"] == "long") & (T["label"] == "early") & (hold >= 0.3)
    return (T["type"] == "long") & (T["label"] == c)

def chan_at(session, xy):
    d = CHREG[(CHREG.session == session) & ((CHREG.x - xy[0]).abs() < 1) & ((CHREG.y - xy[1]).abs() < 1)]
    return d.channel_id.iloc[0] if len(d) else None

def get_ev(session, T, il_xy, pl_xy, event):
    """IL, PL epochs around 'Press' or 'Release', same-region ring reference 30–500 µm"""
    a0, a1 = fn.MAIN_EPOCHS[event]
    rec = fn.load_session(PROC, session, XB, refs=("original",))[0]["original"]
    ids = [str(c) for c in rec.get_channel_ids()]; fs = rec.get_sampling_frequency()
    X, ok = fn.read_epochs(rec, T["press_s" if event == "Press" else "release_s"].values, a0, a1)
    regmap = CHREG[CHREG.session == session].set_index("channel_id")["region"]
    reg = np.array([SHORT.get(regmap.get(k, ""), "other") for k in ids]); loc = rec.get_channel_locations(); out = []
    for xy in (il_xy, pl_xy):
        c = chan_at(session, xy)
        if c is None or c not in ids: raise ValueError(f"no channel at {xy}")
        i = ids.index(c); dd = np.hypot(loc[:, 0] - loc[i, 0], loc[:, 1] - loc[i, 1])
        ring = np.where((reg == reg[i]) & (dd > 30) & (dd <= 500))[0]
        if len(ring) < MIN_RING: raise ValueError(f"ring of {c} has {len(ring)} channels")
        out.append((X[:, i] - np.median(X[:, ring], axis=1)).astype(np.float64))
    return out[0], out[1], np.arange(X.shape[-1]) / fs + a0, fs, ok

def wl(x, fs, idx, freqs):
    return np.stack([ss.fftconvolve(x, fn.morlet(f, NCYC, fs)[None, :], mode="same", axes=-1) for f in freqs], axis=1)[..., idx]

def derangement(n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p

def strat_test(d, rats, NPERM=10000):
    d = np.asarray(d, float); rats = np.asarray(rats); u = np.unique(rats)
    W = np.stack([(rats == r) / (rats == r).sum() for r in u], 1); real = (d @ W).mean()
    null = ((np.random.default_rng(0).choice([-1.0, 1.0], size=(NPERM, len(d))) * d) @ W).mean(1)
    return (1 + np.sum(null >= real)) / (NPERM + 1)


@cell()
def overnight_tests():
    """OVERNIGHT COUPLING TESTS: T8 theta, T8 delta (Jäckel replication), T11 (amplitude / power), T13–15 (windows, equal n, ERP kept)"""
    global e, job, name, run_T11, run_T13_15, run_T8, ss, t0, time, traceback
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    if not RUN:
        print("RUN = False in the settings: nothing is computed here (the cell only computes missing sessions when RUN = True)")
        return
    # ── OVERNIGHT COUPLING TESTS: T8 theta, T8 delta (Jäckel replication), T11 (amplitude / power), T13–15 (windows, equal n, ERP kept) ──
    import time, traceback
    import scipy.signal as ss
    # ════════ PART 1–2: T8 Jäckel replication (FIR + Hilbert, PLV/PLI around press and release, correct vs early) ════════
    def run_T8(BAND, BNAME):
        OUT = os.path.join(TESTS, f"T8_Jaeckel_replication_{BNAME}_{REF}"); os.makedirs(OUT, exist_ok=True)
        WINP, WINR, NSH, NDRAW = (-1.0, 1.5), (-1.0, 1.0), 200, 50
        def fir_phase(x, fs):
            ntap = int(round((366 if BAND[0] < 4 else 183) / 244.14 * fs)) | 1
            b = ss.firwin(ntap, BAND, pass_zero=False, fs=fs)
            return ss.hilbert(ss.filtfilt(b, [1.0], x, axis=-1), axis=-1)
        def traces(A_, B_, n_sh):
            S = A_ * np.conj(B_); E = S / np.abs(S).clip(1e-30)
            plv, pli = np.abs(E.mean(0)), np.abs(np.sign(S.imag).mean(0)); sv, sl = np.zeros_like(plv), np.zeros_like(pli)
            for _ in range(n_sh):
                S2 = A_ * np.conj(B_[derangement(len(B_))]); sv += np.abs((S2 / np.abs(S2).clip(1e-30)).mean(0)); sl += np.abs(np.sign(S2.imag).mean(0))
            return np.stack([plv, pli, sv / n_sh, sl / n_sh])
        F = os.path.join(OUT, "T8_traces.npz"); store = dict(np.load(F, allow_pickle=True)) if os.path.exists(F) else {}
        t_ = time.time()
        for rat, (il_xy, pl_xy, _) in PAIRS3.items():
            for s in SESS[rat]:
                if f"{rat}|{s}|meta" in store: continue
                try:
                    T = fn.session_trials(K, s).reset_index(drop=True)
                    mc, me = cond_trials(T, "correct").values, cond_trials(T, "early").values
                    if mc.sum() < MINN or me.sum() < MINN: continue
                    T = T[mc | me].reset_index(drop=True)
                    for align, ev, (w0, w1) in (("press", "Press", WINP), ("release", "Release", WINR)):
                        xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, ev)
                        Tk = T[ok].reset_index(drop=True); m = {"correct": cond_trials(Tk, "correct").values, "early": cond_trials(Tk, "early").values}
                        idx = np.where((t >= w0 - 1e-9) & (t <= w1 + 1e-9))[0][::int(round(STEP_S * fs))]
                        for erp in ("raw", "erp"):
                            for c in ("correct", "early"):
                                a, b = xI[m[c]], xP[m[c]]
                                if erp == "erp": a, b = a - a.mean(0), b - b.mean(0)
                                A_, B_ = fir_phase(a, fs)[:, idx], fir_phase(b, fs)[:, idx]
                                store[f"{rat}|{s}|{align}|{erp}|{c}"] = traces(A_, B_, NSH)
                                if c == "correct":
                                    n = m["early"].sum()
                                    store[f"{rat}|{s}|{align}|{erp}|correct_m"] = np.mean(
                                        [traces(A_[k], B_[k], 10) for k in (rng.choice(len(A_), n, replace=False) for _ in range(NDRAW))], 0) \
                                        if len(A_) > n else store[f"{rat}|{s}|{align}|{erp}|correct"]
                        store[f"{rat}|{s}|{align}|t"] = t[idx]; del xI, xP
                    store[f"{rat}|{s}|meta"] = np.array([mc.sum(), me.sum()]); np.savez(F, **store)
                    log(f"T8 {BNAME} {RL[rat]} {s}: {mc.sum()} correct / {me.sum()} early | {(time.time() - t_) / 60:.1f} min")
                except Exception as e:
                    log(f"T8 {BNAME} {RL[rat]} {s} error: {e}")
        sess = sorted({k.rsplit("|", 1)[0] for k in store if k.endswith("|meta")}); rats = np.array([k.split("|")[0] for k in sess])
        get = lambda al, erp, c: np.stack([store[f"{k}|{al}|{erp}|{c}"] for k in sess])
        tP, tR = store[f"{sess[0]}|press|t"], store[f"{sess[0]}|release|t"]
        at0 = lambda X, tt: X[:, :, np.argmin(np.abs(tt))]
        fig, axs = plt.subplots(2, 4, figsize=(18, 7.5), sharex="col", facecolor="white", constrained_layout=True)
        for row, erp in enumerate(("raw", "erp")):
            for col, (al, j, tt) in enumerate((("press", 0, tP), ("release", 0, tR), ("press", 1, tP), ("release", 1, tR))):
                ax = axs[row, col]
                for c, lab, colr in (("correct_m", "correct (matched n)", CCOLS["correct"]), ("early", "early", CCOLS["early"])):
                    X = get(al, erp, c); mu = X[:, j].mean(0); se = X[:, j].std(0, ddof=1) / np.sqrt(len(X))
                    ax.plot(tt, mu, color=colr, lw=2, label=lab); ax.fill_between(tt, mu - se, mu + se, color=colr, alpha=0.25)
                    ax.plot(tt, X[:, j + 2].mean(0), color=colr, lw=1, ls="--")
                ax.axvline(0, color="k", lw=1); ax.set_title(f"{'PLV' if j == 0 else 'PLI'} at {al} | {'raw' if erp == 'raw' else 'ERP removed'}", fontsize=10)
                for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        axs[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"PL–IL {BNAME} {BAND[0]:g}–{BAND[1]:g} Hz around press/release ({len(sess)} sessions, dashed = chance)", fontweight="bold")
        fig.savefig(os.path.join(OUT, "F1_traces.png"), dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)
        L = [f"T8 Jäckel-style {BNAME} ({BAND[0]:g}–{BAND[1]:g} Hz, FIR+Hilbert), {len(sess)} sessions, {REF}"]
        for erp in ("raw", "erp"):
            for j, mn in ((0, "PLV"), (1, "PLI")):
                cp, cr, er, crm = at0(get("press", erp, "correct"), tP), at0(get("release", erp, "correct"), tR), at0(get("release", erp, "early"), tR), at0(get("release", erp, "correct_m"), tR)
                p1 = strat_test(cr[:, j] - cp[:, j], rats); dm = (er[:, j] - er[:, j + 2]) - (crm[:, j] - crm[:, j + 2])
                L.append(f"{erp:3s} {mn}: correct press {cp[:, j].mean():.3f} | correct release {cr[:, j].mean():.3f} | early release {er[:, j].mean():.3f} || "
                         f"vs chance p: press {fpt(strat_test(cp[:, j] - cp[:, j + 2], rats))}, release {fpt(strat_test(cr[:, j] - cr[:, j + 2], rats))}, "
                         f"early {fpt(strat_test(er[:, j] - er[:, j + 2], rats))} || correct release > press p {fpt(p1)} ({(cr[:, j] > cp[:, j]).sum()}/{len(cr)}) | "
                         f"early − correct matched (above chance) {rmean(dm, rats):+.3f}, two-sided p {fpt(two(dm, rats))}")
        open(os.path.join(OUT, "T8_summary.txt"), "w", encoding="utf-8").write("\n".join(L)); log("\n".join(L))

    # ════════ PART 3: T11 amplitude coupling + theta power, early vs correct ════════
    def run_T11():
        OUT = os.path.join(TESTS, f"T11_amplitude_power_{REF}"); os.makedirs(OUT, exist_ok=True)
        WH, PRE_A, PRE_B, EARLY_PRE, NDR = 0.5, -0.5, -0.2, 0.6, 20
        def amp_corr(a, b):
            ea, eb = np.abs(a), np.abs(b)
            bo = np.abs(np.imag(b * np.conj(a) / np.abs(a).clip(1e-30))); ao = np.abs(np.imag(a * np.conj(b) / np.abs(b).clip(1e-30)))
            def corr(x, y):
                x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
                return ((x * y).sum(-1) / np.sqrt((x ** 2).sum(-1) * (y ** 2).sum(-1)).clip(1e-30)).mean(-1)
            return corr(ea, eb), 0.5 * (corr(ea, bo) + corr(eb, ao))
        tpow = lambda a, base: 10 * np.log10((np.abs(a) ** 2).mean((1, 2)) / (np.abs(base) ** 2).mean((1, 2)).clip(1e-30))
        F = os.path.join(OUT, f"T11_sessions_{REF}.csv")
        R = pd.read_csv(F, dtype={"rat": str}) if os.path.exists(F) else pd.DataFrame(columns=["rat", "session"])
        rows, done, t_ = R.to_dict("records"), set(zip(R.rat.astype(str), R.session)), time.time()
        for rat, (il_xy, pl_xy, _) in PAIRS3.items():
            for s in SESS[rat]:
                if (rat, s) in done: continue
                try:
                    T = fn.session_trials(K, s).reset_index(drop=True)
                    T = T[cond_trials(T, "correct").values | cond_trials(T, "early").values].reset_index(drop=True)
                    xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, "Press")
                    T = T[ok].reset_index(drop=True); hold = (T["release_s"] - T["press_s"]).values
                    idx = np.where((t >= -1.0 - 1e-9) & (t <= 1.1 + 1e-9))[0][::int(round(STEP_S * fs))]; tt = t[idx]
                    cb, ch = np.where(tt < 0)[0], np.where((tt >= 0) & (tt <= WH + 1e-9))[0]; L_ = int(round((PRE_B - PRE_A) / STEP_S)) + 1
                    coef = {}
                    for c in ("correct", "early"):
                        m_ = cond_trials(T, c).values
                        if m_.sum() >= 2: coef[c] = (wl(xI[m_] - xI[m_].mean(0), fs, idx, FR_TH), wl(xP[m_] - xP[m_].mean(0), fs, idx, FR_TH), hold[m_])
                    del xI, xP
                    if len(coef) < 2: continue
                    row = dict(rat=rat, session=s)
                    for c in ("correct", "early"):
                        A_, B_, h_ = coef[c]; k = h_ >= WH
                        if k.sum() < MINN: continue
                        a_, b_ = A_[k][..., ch], B_[k][..., ch]; pl, po = amp_corr(a_, b_)
                        perm = np.array([rng.choice(np.delete(np.arange(k.sum()), i)) for i in range(k.sum())]); pls, pos_ = amp_corr(a_, b_[perm])
                        row.update({f"hold_{c}_ampc": pl.mean(), f"hold_{c}_ampc_orth": po.mean(), f"hold_{c}_ampc_ch": pls.mean(),
                                    f"hold_{c}_ampc_orth_ch": pos_.mean(), f"hold_{c}_powIL": tpow(a_, A_[k][..., cb]).mean(),
                                    f"hold_{c}_powPL": tpow(b_, B_[k][..., cb]).mean()})
                    Ae, Be, he = coef["early"]; ke = he >= EARLY_PRE; Ac, Bc, _ = coef["correct"]
                    if ke.sum() >= MINN and len(Ac) >= MINN:
                        st = np.clip(np.searchsorted(tt, he[ke] + PRE_A), 0, len(tt) - L_); ei = np.where(ke)[0]
                        seg = lambda X, rows_, s0: np.stack([X[r][:, q:q + L_] for r, q in zip(rows_, s0)])
                        a_, b_ = seg(Ae, ei, st), seg(Be, ei, st); pl, po = amp_corr(a_, b_)
                        row.update({"pre_early_ampc": pl.mean(), "pre_early_ampc_orth": po.mean(),
                                    "pre_early_powIL": tpow(a_, Ae[ei][..., cb]).mean(), "pre_early_powPL": tpow(b_, Be[ei][..., cb]).mean()})
                        acc = []
                        for _ in range(NDR):
                            k = rng.choice(len(Ac), len(ei), replace=len(Ac) < len(ei)); ac_, bc_ = seg(Ac, k, st), seg(Bc, k, st); pl, po = amp_corr(ac_, bc_)
                            acc.append([pl.mean(), po.mean(), tpow(ac_, Ac[k][..., cb]).mean(), tpow(bc_, Bc[k][..., cb]).mean()])
                        acc = np.mean(acc, 0)
                        row.update({"pre_correct_ampc": acc[0], "pre_correct_ampc_orth": acc[1], "pre_correct_powIL": acc[2], "pre_correct_powPL": acc[3]})
                    rows.append(row); done.add((rat, s)); pd.DataFrame(rows).to_csv(F, index=False)
                    log(f"T11 {RL[rat]} {s} | {(time.time() - t_) / 60:.1f} min")
                except Exception as e:
                    log(f"T11 {RL[rat]} {s} error: {e}")
        P = pd.DataFrame(rows); P["rat"] = P["rat"].astype(str)
        L = [f"T11 theta amplitude coupling and power, early vs correct ({REF}, ERP removed; session means; two-sided)"]
        for win in ("hold", "pre"):
            L.append(f"\n{win.upper()}:")
            for key, name in (("ampc", "amplitude coupling"), ("ampc_orth", "amplitude coupling orthogonalised"),
                              ("powIL", "IL theta power dB"), ("powPL", "PL theta power dB")):
                ce, cc = f"{win}_early_{key}", f"{win}_correct_{key}"
                if ce not in P or cc not in P: continue
                Q = P.dropna(subset=[ce, cc]); d = (Q[ce] - Q[cc]).values; extra = ""
                if key.startswith("ampc") and win == "hold":
                    extra = (f" | above chance: correct p {fpt(strat_test((Q[cc] - Q[f'{cc}_ch']).values, Q.rat.values))}, "
                             f"early p {fpt(strat_test((Q[ce] - Q[f'{ce}_ch']).values, Q.rat.values))}")
                L.append(f"  {name}: correct {rmean(Q[cc], Q.rat):+.4f} | early {rmean(Q[ce], Q.rat):+.4f} | early − correct {rmean(d, Q.rat):+.4f}, "
                         f"p {fpt(two(d, Q.rat.values))}, early higher {(d > 0).sum()}/{len(d)}{extra}")
        open(os.path.join(OUT, "T11_summary.txt"), "w", encoding="utf-8").write("\n".join(L)); log("\n".join(L))

    # ════════ PART 4: T13 tone (correct vs late), T14 time course, T15 frequency profile (ERP kept) ════════
    def run_T13_15():
        OUT = os.path.join(TESTS, f"Overnight_T13-15_{REF}_ERPkept"); os.makedirs(OUT, exist_ok=True)
        NSH, NDR = 100, 30
        FR_ALL = np.unique(np.round(np.geomspace(2, 30, 20), 1)); BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        C3_ = ("correct", "early", "late"); PAIRS_C = (("early", "late"), ("correct", "late"), ("correct", "early"))
        def stats_f(A_, B_, n_sh):
            S = A_ * np.conj(B_); E = S / np.abs(S).clip(1e-30); n = len(E); sg = np.sign(S.imag)
            ppc = ((np.abs(E.sum(0)) ** 2 - n) / (n * (n - 1))).mean(-1); pli = np.abs(sg.mean(0)).mean(-1); spl = sg.mean(0).mean(-1); ch = np.zeros_like(pli)
            for _ in range(n_sh):
                S2 = A_ * np.conj(B_[derangement(n)]); ch += np.abs(np.sign(S2.imag).mean(0)).mean(-1)
            return np.stack([ppc, pli, ch / n_sh, spl])
        def matched(A_, B_, n):
            if len(A_) == n: return stats_f(A_, B_, NSH)
            return np.mean([stats_f(A_[k], B_[k], max(5, NSH // 10)) for k in (rng.choice(len(A_), n, replace=False) for _ in range(NDR))], 0)
        NM = ("PPC", "PLI", "PLIch", "sPLI")
        FILES = {k: os.path.join(OUT, f"{k}_rows.csv") for k in ("T13", "T14", "T15")}
        ROWS = {k: (pd.read_csv(f, dtype={"rat": str}).to_dict("records") if os.path.exists(f) else []) for k, f in FILES.items()}
        DF = os.path.join(OUT, "done_sessions.txt"); DONE = set(open(DF).read().split()) if os.path.exists(DF) else set()
        t_ = time.time()
        for rat, (il_xy, pl_xy, _) in PAIRS3.items():
            for s in SESS[rat]:
                if f"{rat}|{s}" in DONE: continue
                try:
                    T = fn.session_trials(K, s).reset_index(drop=True)
                    T = T[np.any(np.stack([cond_trials(T, c).values for c in C3_]), 0)].reset_index(drop=True)
                    xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, "Press")
                    T = T[ok].reset_index(drop=True); hold = (T["release_s"] - T["press_s"]).values
                    tone0 = float(np.median((T["tone_s"] - T["press_s"]).values)) if "tone_s" in T.columns else 1.05
                    M = {c: cond_trials(T, c).values for c in C3_}; step = int(round(STEP_S * fs))
                except Exception as e:
                    log(f"T13-15 {RL[rat]} {s} load error: {e}"); continue
                try:                                                                   # T13
                    idx = np.where((t >= tone0 - 0.5 - 1e-9) & (t <= tone0 + 1e-9))[0][::step]
                    mc, ml = M["correct"] & (hold >= tone0), M["late"]
                    if mc.sum() >= MINN and ml.sum() >= MINN:
                        n = min(mc.sum(), ml.sum()); row = dict(rat=rat, session=s, n=int(n))
                        for c, m_ in (("correct", mc), ("late", ml)):
                            v = matched(wl(xI[m_], fs, idx, FR_TH), wl(xP[m_], fs, idx, FR_TH), n).mean(-1)
                            row.update({f"{c}_{nm}": v[i] for i, nm in enumerate(NM)})
                        ROWS["T13"].append(row)
                except Exception as e: log(f"T13 {RL[rat]} {s} error: {e}")
                try:                                                                   # T14
                    idx = np.where((t >= -1e-9) & (t <= 1.0 + 1e-9))[0][::step]; tt = t[idx]
                    W = {c: (wl(xI[M[c]], fs, idx, FR_TH), wl(xP[M[c]], fs, idx, FR_TH), hold[M[c]]) for c in C3_ if M[c].sum() >= MINN}
                    for b0, b1 in BINS:
                        cols = np.where((tt >= b0 - 1e-9) & (tt <= b1 + 1e-9))[0]
                        for ca, cb in PAIRS_C:
                            if ca not in W or cb not in W: continue
                            ka, kb = W[ca][2] >= b1, W[cb][2] >= b1; n = min(ka.sum(), kb.sum())
                            if n < MINN: continue
                            row = dict(rat=rat, session=s, bin=f"{b0:.1f}-{b1:.1f}", pair=f"{ca}-{cb}", n=int(n))
                            for side, c, k in (("A", ca, ka), ("B", cb, kb)):
                                v = matched(W[c][0][k][..., cols], W[c][1][k][..., cols], n).mean(-1)
                                row.update({f"{side}_{nm}": v[i] for i, nm in enumerate(NM)})
                            ROWS["T14"].append(row)
                    del W
                except Exception as e: log(f"T14 {RL[rat]} {s} error: {e}")
                try:                                                                   # T15
                    ih = np.where((t >= -1e-9) & (t <= 0.5 + 1e-9))[0][::step]; ip = np.where((t >= -0.5 - 1e-9) & (t <= 1e-9))[0][::step]; Fc = {}
                    for c in C3_:
                        m_ = M[c] & (hold >= 0.5)
                        if m_.sum() < MINN: continue
                        Fc[c] = (wl(xI[m_], fs, ih, FR_ALL), wl(xP[m_], fs, ih, FR_ALL))
                        if c == "correct": Fc["pre"] = (wl(xI[m_], fs, ip, FR_ALL), wl(xP[m_], fs, ip, FR_ALL))
                    for ca, cb in PAIRS_C + (("correct", "pre"),):
                        if ca not in Fc or cb not in Fc: continue
                        n = min(len(Fc[ca][0]), len(Fc[cb][0])); va, vb = matched(*Fc[ca], n), matched(*Fc[cb], n)
                        for fi, f in enumerate(FR_ALL):
                            ROWS["T15"].append(dict(rat=rat, session=s, pair=f"{ca}-{cb}", freq=f, n=int(n),
                                                    **{f"A_{nm}": va[i, fi] for i, nm in enumerate(NM)}, **{f"B_{nm}": vb[i, fi] for i, nm in enumerate(NM)}))
                    del Fc
                except Exception as e: log(f"T15 {RL[rat]} {s} error: {e}")
                del xI, xP
                for k, f in FILES.items(): pd.DataFrame(ROWS[k]).to_csv(f, index=False)
                DONE.add(f"{rat}|{s}"); open(DF, "w").write("\n".join(sorted(DONE)))
                log(f"T13-15 {RL[rat]} {s} done | {(time.time() - t_) / 60:.1f} min")
        ex = lambda df, side: df[f"{side}_PLI"] - df[f"{side}_PLIch"]
        L = [f"T13–T15 | {REF} | ERP kept | theta {FR_TH.min():g}–{FR_TH.max():g} Hz"]
        try:
            Q = pd.DataFrame(ROWS["T13"]); Q["rat"] = Q.rat.astype(str)
            L.append(f"\nT13 BEFORE THE TONE (−0.5 → 0 s), correct vs late, {len(Q)} sessions, median n {Q.n.median():.0f}")
            for m in ("PLI", "PPC", "sPLI"):
                cc = Q[f"correct_{m}"] - (Q["correct_PLIch"] if m == "PLI" else 0); ll = Q[f"late_{m}"] - (Q["late_PLIch"] if m == "PLI" else 0); d = (cc - ll).values
                L.append(f"  {m}{' − chance' if m == 'PLI' else ''}: correct {rmean(cc, Q.rat):+.4f} (vs 0 p {fpt(strat_test(cc.values, Q.rat.values))}) | "
                         f"late {rmean(ll, Q.rat):+.4f} (vs 0 p {fpt(strat_test(ll.values, Q.rat.values))}) | correct − late {rmean(d, Q.rat):+.4f}, "
                         f"two-sided p {fpt(two(d, Q.rat.values))}, correct higher {(d > 0).sum()}/{len(d)}")
        except Exception as e: log(f"T13 summary error: {e}")
        try:
            Q = pd.DataFrame(ROWS["T14"]); Q["rat"] = Q.rat.astype(str); L.append("\nT14 TIME COURSE (PLI − chance; equal n per pair; A − B two-sided)")
            for pr in [f"{a}-{b}" for a, b in PAIRS_C]:
                for b0, b1 in BINS:
                    g = Q[(Q.pair == pr) & (Q.bin == f"{b0:.1f}-{b1:.1f}")]
                    if len(g) < 3: continue
                    d = (ex(g, "A") - ex(g, "B")).values
                    L.append(f"  {pr} {b0:.1f}-{b1:.1f} s ({len(g)} sessions): A {rmean(ex(g, 'A'), g.rat):+.4f} | B {rmean(ex(g, 'B'), g.rat):+.4f} | p {fpt(two(d, g.rat.values))}")
        except Exception as e: log(f"T14 summary error: {e}")
        try:
            Q = pd.DataFrame(ROWS["T15"]); Q["rat"] = Q.rat.astype(str); L.append("\nT15 FREQUENCY PROFILE (0 → 0.5 s; PLI − chance; A − B two-sided)")
            for pr in [f"{a}-{b}" for a, b in PAIRS_C] + ["correct-pre"]:
                g0 = Q[Q.pair == pr]; res = []
                for f in sorted(g0.freq.unique()):
                    g = g0[g0.freq == f]; d = (ex(g, "A") - ex(g, "B")).values; res.append((f, rmean(d, g.rat), two(d, g.rat.values)))
                L.append(f"  {pr} ({g0.session.nunique()} sessions): " + " | ".join(f"{f:g} Hz {d:+.3f}{'*' if p < 0.05 else ''}" for f, d, p in res))
        except Exception as e: log(f"T15 summary error: {e}")
        open(os.path.join(OUT, "T13-15_summary.txt"), "w", encoding="utf-8").write("\n".join(L)); log("\n".join(L))

    # ════════ RUN ALL (each part alone; a failure is logged and the next part runs) ════════
    log("════ ALL OVERNIGHT START ════")
    for name, job in (("T8 theta", lambda: run_T8((4.0, 8.0), "theta")), ("T8 delta", lambda: run_T8((2.0, 4.0), "delta")),
                      ("T11", run_T11), ("T13-15", run_T13_15)):
        t0 = time.time()
        try:
            log(f"── {name} start ──"); job(); log(f"── {name} finished in {(time.time() - t0) / 60:.0f} min ──")
        except Exception as e:
            log(f"── {name} FAILED: {e}\n{traceback.format_exc()}")
    log("════ ALL OVERNIGHT DONE ════")


@cell()
def trial_phase_files():
    """PER-TRIAL IL / PL THETA PHASE AROUND THE PRESS, one file per session (read by the coupling cells of 07)"""
    global A_, B_, PHD, T, _, e, f, fs, idx, il_xy, lab, ok, pl_xy, rat, s, t, t_, time, xI, xP
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    if not RUN:
        print("RUN = False in the settings: nothing is computed here (the cell only computes missing sessions when RUN = True)")
        return
    # ── PER-TRIAL IL / PL THETA PHASE AROUND THE PRESS, one file per session (read by the coupling cells of 07) ──
    import time
    # ── 0) per-trial IL / PL theta phase around the press, saved once per session (reused by later tests) ──
    PHD = os.path.join(TESTS, f"Trial_phases_press_{REF}_ERPkept"); os.makedirs(PHD, exist_ok=True); t_ = time.time()
    for rat, (il_xy, pl_xy, _) in PAIRS3.items():
        for s in SESS[rat]:
            f = os.path.join(PHD, f"{s}.npz")
            if os.path.exists(f): continue
            try:
                T = fn.session_trials(K, s).reset_index(drop=True)
                lab = np.select([cond_trials(T, c).values for c in ("correct", "early", "late")], ["correct", "early", "late"], "")
                T = T[lab != ""].reset_index(drop=True)
                xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, "Press"); T = T[ok].reset_index(drop=True)
                lab = np.select([cond_trials(T, c).values for c in ("correct", "early", "late")], ["correct", "early", "late"], "")
                idx = np.where((t >= -0.5 - 1e-9) & (t <= 1.1 + 1e-9))[0][::int(round(STEP_S * fs))]
                A_, B_ = wl(xI, fs, idx, FR_TH), wl(xP, fs, idx, FR_TH); del xI, xP
                np.savez(f, uA=(A_ / np.abs(A_).clip(1e-30)).astype(np.complex64), uB=(B_ / np.abs(B_).clip(1e-30)).astype(np.complex64),
                         t=t[idx], trial_index=T["trial_index"].values, label=lab, hold=(T["release_s"] - T["press_s"]).values, rat=rat)
                log(f"phases saved {RL[rat]} {s} | {(time.time() - t_) / 60:.1f} min")
            except Exception as e:
                log(f"phases {RL[rat]} {s} error: {e}")


@cell()
def T8c_sums():
    """JÄCKEL Fig. 3C, ERP REMOVED: per session, sum of the unit phase-difference vectors (delta, theta; press, release; correct, early)"""
    global BANDS, COL, CONDS, E, EVS, OUTE, S, STEP, T, Tk, WIN, _, a, b, band, bn, c, e, ev, f, fir_phase, fs, idx, il_xy, m
    global mc, me, ok, pl_xy, rat, s, ss, store, t, t_, time, xI, xP
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    if not RUN:
        print("RUN = False in the settings: nothing is computed here (the cell only computes missing sessions when RUN = True)")
        return
    # ── JÄCKEL Fig. 3C, ERP REMOVED: per session, sum of the unit phase-difference vectors (delta, theta; press, release; correct, early) ──
    import time
    import scipy.signal as ss
    BANDS = {"delta": (2.0, 4.0), "theta": (4.0, 8.0)}; EVS = ("Press", "Release"); CONDS = ("correct", "early")
    WIN, STEP = (-1.0, 1.0), 0.01
    COL = {"correct": "#2CA02C", "early": "#E6B800"}

    def fir_phase(x, fs, band):                                       # same filters as the paper / T8
        ntap = int(round((366 if band[0] < 4 else 183) / 244.14 * fs)) | 1
        b = ss.firwin(ntap, band, pass_zero=False, fs=fs)
        return ss.hilbert(ss.filtfilt(b, [1.0], x, axis=-1), axis=-1)

    OUTE = os.path.join(TESTS, f"T8c_Jaeckel_Fig3C_replication_{REF}_ERPremoved"); os.makedirs(OUTE, exist_ok=True)
    t_ = time.time()
    for rat, (il_xy, pl_xy, _) in PAIRS3.items():
        for s in SESS[rat]:
            f = os.path.join(OUTE, f"{s}.npz")
            if os.path.exists(f): continue
            try:
                T = fn.session_trials(K, s).reset_index(drop=True)
                mc, me = cond_trials(T, "correct").values, cond_trials(T, "early").values
                if mc.sum() < MINN or me.sum() < MINN: continue
                T = T[mc | me].reset_index(drop=True); store = {}
                for ev in EVS:
                    xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, ev); Tk = T[ok].reset_index(drop=True)
                    idx = np.where((t >= WIN[0] - 1e-9) & (t <= WIN[1] + 1e-9))[0][::int(round(STEP * fs))]
                    for c in CONDS:
                        m = cond_trials(Tk, c).values; a, b = xI[m] - xI[m].mean(0), xP[m] - xP[m].mean(0)   # ERP removed within condition
                        for bn, band in BANDS.items():
                            S = fir_phase(a, fs, band)[:, idx] * np.conj(fir_phase(b, fs, band)[:, idx]); E = S / np.abs(S).clip(1e-30)
                            store[f"{bn}|{ev}|{c}|sum"] = E.sum(0); store[f"{bn}|{ev}|{c}|n"] = np.array(m.sum())
                    store[f"t|{ev}"] = t[idx]; del xI, xP
                np.savez(f, rat=rat, **store); log(f"T8c-ERPremoved {RL[rat]} {s} | {(time.time() - t_) / 60:.1f} min")
            except Exception as e:
                log(f"T8c-ERPremoved {RL[rat]} {s} error: {e}")


@cell()
def A1_rat136():
    """A1. RAT 3 (136), CORRECT, PRESS → +1 s: pair map, pilot, example trial, sessions, direction (needs cell 0)"""
    global A_, B_, CHREG, D, FILE, K, OUTD, PX, Rm, SESS, _, a, a_, amp, ang, ax, axp, axs, b, c95, ch, cr, dist, done, dp
    global dtr, e, fIL, fPL, fig, fs, glob, grp, h5, h6, h_, hi, hk, il, importlib, jit, k, lab, lb, lo, m, nIL, nPL, n_, near
    global np, ns, o, os, p, p_, p_ray, pd, pi, pl, plt, pp, prow, reg, row, rows, run_session, s, save, sh_, show, sos, sp
    global ss, t, t_, th, time, v, vi, win, x, x_, xi, yy
    # ── A1. RAT 3 (136), CORRECT, PRESS → +1 s: pair map, pilot, example trial, sessions, direction (needs cell 0) ──
    import os, glob, time, importlib
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.signal as ss

    # REF, WEND, TAG come from A_RUN in the settings cell (1 = SR30-500 press → 1 s, 2 = CMR press → 1 s, 3 / 4 = the same, press → release)
    print(f"RUNNING: {REF} | {TAG}")
    OUTD = os.path.join(AMIR, "Theta", "Phase connectivity", f"{REF}_{TAG}", "1_Rat3_correct"); os.makedirs(OUTD, exist_ok=True)
    CHREG = pd.read_csv(os.path.join(TABLES, "channel_regions_all.csv")); CHREG["channel_id"] = CHREG["channel_id"].astype(str)
    K = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False); K = K[K["keep"]]
    SESS = sorted(BK.loc[BK.pair == PAIR, "session"].unique())
    print(f"rat {RAT}: {len(SESS)} sessions | pilot in list: {TEST in SESS} | saving to {OUTD}")

    # ── helpers ──

    def run_session(session, keep=False):
        T = fn.session_trials(K, session)
        T = T[(T["type"] == "long") & (T["label"] == "correct")].reset_index(drop=True)
        if len(T) < MIN_TR: return None, None
        xI, xP, t, fs, ok, used = get_signals(session, T); T = T[ok].reset_index(drop=True)
        hold = (T["release_s"] - T["press_s"]).values
        A_, B_, h_, tt, gr, M_, o = prepare(xI, xP, t, fs, hold)
        pv, pl = plv_pli(A_, B_, gr); sh = np.array([plv_pli(A_, B_, gr, True) for _ in range(NSH)])
        dp, dtr = direction(A_, B_, M_)
        row = dict(session=session, rat=RAT, pilot=(session == TEST), ref=REF, window=TAG, condition="correct",
                   IL=used[0], PL=used[1], trials=len(A_), dphi_deg=np.degrees(dp))
        for j, m in enumerate(("PLV", "PLI")):
            real = (pv, pl)[j]
            row.update({m: real, f"{m}_shuffle_mean": sh[:, j].mean(), f"{m}_chance95": np.percentile(sh[:, j], 95),
                        f"{m}_p": (1 + (sh[:, j] >= real).sum()) / (1 + NSH)})
        extra = dict(A=A_, B=B_, dtr=dtr, dp=dp, xI=xI, xP=xP, t=t, fs=fs, o=o, h=h_) if keep else None
        return row, extra

    # ── compute (pilot first, kept for the example trial; others resume from the CSV) ──
    FILE = os.path.join(OUTD, f"sessions_{REF}_{TAG}.csv")
    done = pd.read_csv(FILE) if os.path.exists(FILE) else pd.DataFrame(columns=["session"])
    rows, t_ = [r for r in done.to_dict("records") if r["session"] != TEST], time.time()
    prow, PX = run_session(TEST, keep=True); rows.append(prow)
    print(f"pilot {TEST}: {prow['trials']} trials | PLV p {prow['PLV_p']:.3f} | PLI p {prow['PLI_p']:.3f} | {(time.time() - t_) / 60:.1f} min", flush=True)
    for n_, s in enumerate(SESS, 1):
        if s == TEST or s in set(done["session"]): continue
        try:
            row, _ = run_session(s)
        except ValueError as e:
            print(f"[{n_}] {e}, skipped"); continue
        if row is None: print(f"[{n_}] {s}: < {MIN_TR} trials, skipped"); continue
        rows.append(row); pd.DataFrame(rows).to_csv(FILE, index=False)
        print(f"[{n_}/{len(SESS)}] {s}: {row['trials']} tr | PLV {row['PLV']:.3f} (p {row['PLV_p']:.3f}) | PLI {row['PLI']:.3f} "
              f"(p {row['PLI_p']:.3f}) | Δφ {row['dphi_deg']:+.0f}° | {(time.time() - t_) / 60:.1f} min", flush=True)
    D = pd.DataFrame(rows).sort_values("session").reset_index(drop=True); D["pilot"] = D["pilot"].astype(bool)
    D.to_csv(FILE, index=False); ns = len(D)
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.9})
    save = lambda fig, name: (_savefig(fig, os.path.join(OUTD, name)),
                              plt.close(fig), display(Image(os.path.join(OUTD, name))))

    # ── FIG 1: probe map, pilot session, pair only ──
    cr = CHREG[CHREG.session == TEST].copy(); cr["reg"] = cr["region"].map(SHORT)
    cr["shank"] = (cr["x"] // 250).astype(int); cr["xp"] = cr["shank"] + (cr["x"] % 250) / 250 * 0.6
    il = cr[((cr.x - IL_XY[0]).abs() < 1) & ((cr.y - IL_XY[1]).abs() < 1)].iloc[0]
    pl = cr[((cr.x - PL_XY[0]).abs() < 1) & ((cr.y - PL_XY[1]).abs() < 1)].iloc[0]
    dist = np.linalg.norm([il.ap_mm - pl.ap_mm, il.ml_mm - pl.ml_mm, il.dv_mm - pl.dv_mm])
    fig, ax = plt.subplots(figsize=(6.2, 8), facecolor="white", constrained_layout=True)
    ax.scatter(cr.xp, cr.y, marker="s", s=14, color="0.85", edgecolor="none")
    for lo, hi in (("IL", "PL"), ("PL", "Cg1")):
        a, b = cr[cr.reg == lo].y, cr[cr.reg == hi].y
        if len(a) and len(b): ax.axhline((a.max() + b.min()) / 2, color="k", lw=0.9, ls=":")
    for reg in ("IL", "PL", "Cg1"):
        yy = cr[cr.reg == reg].y
        if len(yy): ax.text(-0.45, yy.median(), reg, va="center", fontsize=14, fontweight="bold")
    ax.plot([il.xp, pl.xp], [il.y, pl.y], color="k", lw=2.2, ls="--")
    ax.scatter([il.xp, pl.xp], [il.y, pl.y], marker="s", s=70, color="k", zorder=3)
    ax.text((il.xp + pl.xp) / 2 + 0.12, (il.y + pl.y) / 2, f"{dist:.1f} mm", fontsize=13, fontweight="bold")
    ax.text(il.xp + 0.1, il.y, f"IL ({il.channel_id.split('#')[-1]})", va="center", fontsize=10)
    ax.text(pl.xp + 0.1, pl.y, f"PL ({pl.channel_id.split('#')[-1]})", va="center", fontsize=10)
    sh_ = sorted(cr.shank.unique()); ax.set_xticks([s + 0.06 for s in sh_]); ax.set_xticklabels([f"shank {s}" for s in sh_])
    ax.set_xlim(-0.5, max(sh_) + 0.9); ax.set_ylabel("Distance from probe tip (µm)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(f"Rat 3: IL–PL pair {dist:.1f} mm apart", fontsize=13, fontweight="bold")
    save(fig, "F1_probe_map_pair.png")

    # ── FIG 2: pilot session, PLV and PLI vs 95 % of its own shuffles ──
    fig, axs = plt.subplots(1, 2, figsize=(6, 4.6), facecolor="white", constrained_layout=True)
    for ax, m in zip(axs, ("PLV", "PLI")):
        v, c95, p = prow[m], prow[f"{m}_chance95"], prow[f"{m}_p"]
        ax.bar(0, v, width=0.5, color=COL)
        ax.plot([-0.4, 0.4], [c95, c95], color="k", lw=1.8, ls="--", label="95 % of shuffles")
        ax.set_xticks([0]); ax.set_xticklabels(["IL–PL"]); ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, max(v, c95) * 1.35)
        ax.set_ylabel(f"Theta {m} (press → +{WEND:g} s)")
        ax.set_title(f"{m} {v:.2f}, {ptxt(p) if p < 0.05 else 'n.s.'}", fontsize=12, fontweight="bold" if p < 0.05 else "normal")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"Pilot session: {prow['trials']} correct long trials ({REF})", fontsize=12, fontweight="bold")
    save(fig, "F2_pilot_PLV_PLI.png")

    # ── FIG 3: example trial of the pilot (Δφ within ±20° of the mean, strongest theta) ──
    A_, B_, dtr, dp, o, h_, t, fs = PX["A"], PX["B"], PX["dtr"], PX["dp"], PX["o"], PX["h"], PX["t"], PX["fs"]
    amp = 0.5 * (np.abs(A_).mean(axis=(1, 2)) + np.abs(B_).mean(axis=(1, 2)))
    near = np.where(np.abs(np.angle(np.exp(1j * (dtr - dp)))) <= np.radians(20))[0]
    k = near[np.argmax(amp[near])] if len(near) else int(np.argmax(amp))
    sos = ss.butter(4, [4, 7], btype="bandpass", fs=fs, output="sos")
    fIL, fPL = ss.sosfiltfilt(sos, PX["xI"][o[k]]), ss.sosfiltfilt(sos, PX["xP"][o[k]])
    hk = min(h_[k], WEND)                                  # this trial's release (press → release) or 1 s (press → 1 s)
    show = (t >= -0.2) & (t <= hk + 0.2); win = (t >= 0) & (t <= hk)
    nIL, nPL = fIL / np.abs(fIL[show]).max(), fPL / np.abs(fPL[show]).max()
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="white", constrained_layout=True)
    ax.axvspan(0, hk, color="0.94", zorder=0)
    ax.plot(t[show], nIL[show] + 1.2, color=COL_IL, lw=2.2, label="IL")
    ax.plot(t[show], nPL[show] - 1.2, color=COL_PL, lw=2.2, label="PL")
    for p_ in t[win][ss.find_peaks(nIL[win], distance=int(0.1 * fs))[0]]: ax.axvline(p_, ymin=0.5, ymax=0.95, color=COL_IL, ls="--", lw=1.1)
    for p_ in t[win][ss.find_peaks(nPL[win], distance=int(0.1 * fs))[0]]: ax.axvline(p_, ymin=0.05, ymax=0.5, color=COL_PL, ls="--", lw=1.1)
    for x_, lb in ((0, "press"), (1.05, "tone"), (h_[k], "release")):
        if x_ <= hk + 0.2:
            ax.axvline(x_, color="k", lw=1)
            ax.text(x_ + 0.01, 0.02, lb, ha="left", va="bottom", fontsize=10, transform=ax.get_xaxis_transform())
    ax.set_yticks([]); ax.set_ylim(-2.6, 2.6); ax.set_xlabel("Time from press (s)")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.set_title(f"Example trial: IL ahead by {np.degrees(dtr[k]):+.0f}° ≈ {np.degrees(dtr[k]) / 360 * 1000 / F0:+.0f} ms "
                 f"(theta 4–7 Hz, dashed = peaks)", fontsize=13, fontweight="bold")
    save(fig, "F3_pilot_example_trial.png")

    # ── FIG 4: all sessions pooled (bar ± SEM, dots, pilot red same size, dashed = mean of shuffles, one-sided sign-flip) ──
    grp = {m: onesided((D[m] - D[f"{m}_shuffle_mean"]).values) for m in ("PLV", "PLI")}
    fig, axs = plt.subplots(1, 2, figsize=(6, 4.6), facecolor="white", constrained_layout=True)
    jit = np.random.default_rng(1).uniform(-0.14, 0.14, ns)
    for ax, m in zip(axs, ("PLV", "PLI")):
        v, ch = D[m].values, D[f"{m}_shuffle_mean"].mean()
        ax.bar(0, v.mean(), width=0.5, color=COL, zorder=1)
        ax.errorbar(0, v.mean(), yerr=v.std(ddof=1) / np.sqrt(ns), color="k", capsize=6, lw=1.5, zorder=3)
        ax.scatter(jit[~D.pilot], v[~D.pilot], s=18, color=DOT, zorder=2)
        ax.scatter(jit[D.pilot], v[D.pilot], s=18, color=PILOT_COL, zorder=4, label="pilot")
        ax.plot([-0.4, 0.4], [ch, ch], color="k", lw=1.8, ls="--", zorder=3, label="chance (mean of shuffles)")
        ax.set_xticks([0]); ax.set_xticklabels(["IL–PL"]); ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, max(v.max(), ch) * 1.25)
        ax.set_ylabel(f"Theta {m} (press → +{WEND:g} s)")
        ax.set_title(f"{m} {v.mean():.2f}, {ptxt(grp[m]) if grp[m] < 0.05 else 'n.s.'}", fontsize=12,
                     fontweight="bold" if grp[m] < 0.05 else "normal")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"Rat 3, {ns} sessions, correct ({REF}, one-sided)", fontsize=12, fontweight="bold")
    save(fig, "F4_sessions_pooled.png")

    # ── FIG 5: each session alone (bar, own 95 % dash, ★ if p < 0.05) ──
    x = np.arange(ns); lab = [s.split("_")[0][4:] for s in D.session]
    fig, axs = plt.subplots(1, 2, figsize=(max(9, 0.9 * ns + 3), 4.2), facecolor="white", constrained_layout=True)
    for ax, m in zip(axs, ("PLV", "PLI")):
        v, c95, p = D[m].values, D[f"{m}_chance95"].values, D[f"{m}_p"].values
        ax.bar(x, v, width=0.65, color=[PILOT_COL if pp else COL for pp in D.pilot])
        ax.hlines(c95, x - 0.33, x + 0.33, color="k", lw=1.6)
        for xi, vi, pi in zip(x, v, p):
            if pi < 0.05: ax.text(xi, vi * 1.03, "★", ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0, max(v.max(), c95.max()) * 1.25); ax.set_xticks(x); ax.set_xticklabels(lab, rotation=45)
        ax.set_xlabel("Session (MMDD)"); ax.set_ylabel(f"Theta {m}")
        ax.set_title(f"{m}: {int((p < 0.05).sum())}/{ns} sessions significant", fontsize=11, fontweight="bold")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    h5 = [plt.Line2D([], [], color="k", lw=1.6, label="95 % of own shuffles"), plt.Line2D([], [], marker="*", ls="", color="k", label="p < 0.05"),
          plt.Rectangle((0, 0), 1, 1, color=PILOT_COL, label="pilot")]
    fig.legend(handles=h5, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"Rat 3, each session alone, correct ({REF})", fontsize=12, fontweight="bold")
    save(fig, "F5_sessions_each.png")

    # ── FIG 6: direction, one arrow per session (pilot red), black = mean ──
    th = np.radians(D.dphi_deg.values); Rm, p_ray = rayleigh(th); ang = np.angle(np.exp(1j * th).mean())
    fig = plt.figure(figsize=(5, 5.8), facecolor="white"); axp = fig.add_subplot(projection="polar")
    for a_, pp in zip(th, D.pilot):
        axp.annotate("", xy=(a_, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=PILOT_COL if pp else COL, lw=1.5))
    axp.annotate("", xy=(ang, Rm), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=3))
    axp.set_ylim(0, 1.1); axp.set_yticks([]); axp.set_xticks(np.radians([0, 90, 180, 270]))
    axp.set_xticklabels(["0°", "+90°\nIL ahead", "180°", "−90°\nPL ahead"], fontsize=10)
    h6 = [plt.Line2D([], [], color=COL, lw=1.5, label="session"), plt.Line2D([], [], color=PILOT_COL, lw=1.5, label="pilot"),
          plt.Line2D([], [], color="k", lw=3, label="mean")]
    fig.legend(handles=h6, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=False, fontsize=10)
    axp.set_title(f"IL ahead in {(np.sin(th) > 0).sum()}/{ns} sessions\nmean {np.degrees(ang):+.0f}° ≈ "
                  f"{np.degrees(ang) / 360 * 1000 / F0:+.0f} ms, R = {Rm:.2f}, Rayleigh {ptxt(p_ray)}", fontsize=12, fontweight="bold", pad=22)
    save(fig, "F6_direction_circle.png")

    print(f"\nRat 3 | {REF} | press → +{WEND:g} s | correct | {ns} sessions")
    for m in ("PLV", "PLI"):
        print(f"  {m}: mean {D[m].mean():.3f} | chance {D[f'{m}_shuffle_mean'].mean():.3f} | one-sided {ptxt(grp[m])} "
              f"| sessions alone {(D[f'{m}_p'] < 0.05).sum()}/{ns}")
    print(f"  Δφ {np.degrees(ang):+.0f}° | IL ahead {(np.sin(th) > 0).sum()}/{ns} | R {Rm:.2f} | Rayleigh {ptxt(p_ray)}")


@cell()
def A2b_three_rats():
    """A2b. 3 RATS, CORRECT / LATE / EARLY, PRESS → +1 s (fixed resume, ≥ 5 trials as C11) (needs A1)"""
    global FILE, FILE2, OUT2, P2, POS, PS, Q, Qr, R2, RC, RL, Rm, Rr, TAG2, TR, TRIED, a, a1, a_, ang, ax, axp, axs, bb, bk
    global bold, both, c, ch, ci, cmp, d, e, fig, gs, h_, have, hl, hp, i, il_, il_xy, j, m, n, new, other, p, pR, p_ray, pl_
    global pl_xy, plab, pname, pr, pr2, r, rat, rm, row, rows2, rr, run_conds, s, save_, sem, sp, t_, th, todo, top, tr_
    global tried, txt, v, xyz, ymax
    # ── A2b. 3 RATS, CORRECT / LATE / EARLY, PRESS → +1 s (fixed resume, ≥ 5 trials as C11) (needs A1) ──
    TAG2 = "2_three_rats_conditions"
    OUT2 = os.path.join(AMIR, "Theta", "Phase connectivity", f"{REF}_{TAG}", TAG2); os.makedirs(OUT2, exist_ok=True)
    RC = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}; RL = {"134": "Rat 1", "135": "Rat 2", "136": "Rat 3"}

    def run_conds(rat, session, il_xy, pl_xy, todo):
        """one session: read once, compute the conditions in todo; returns rows and the conditions tried"""
        T = fn.session_trials(K, session).reset_index(drop=True)
        masks = {c: cond_trials(T, c).values for c in todo}
        if not any(m_.sum() for m_ in masks.values()): return [], list(todo)
        T = T[np.any(np.stack(list(masks.values())), 0)].reset_index(drop=True)
        masks = {c: cond_trials(T, c).values for c in todo}
        xI, xP, t, fs, ok, used = get_pair(session, T, il_xy, pl_xy)
        T = T[ok].reset_index(drop=True); masks = {c: m_[ok] for c, m_ in masks.items()}
        hold = (T["release_s"] - T["press_s"]).values; out = []
        for c, m_ in masks.items():
            if m_.sum() < MIN_TR2: continue
            A_, B_, h_, tt, gr, M_, o = prepare(xI[m_], xP[m_], t, fs, hold[m_])
            if not gr: continue
            pv, pl = plv_pli(A_, B_, gr); sh = np.array([plv_pli(A_, B_, gr, True) for _ in range(NSH)])
            dp, _ = direction(A_, B_, M_)
            row = dict(rat=rat, session=session, condition=c, IL=used[0], PL=used[1], trials=int(m_.sum()), dphi_deg=np.degrees(dp))
            for j, m in enumerate(("PLV", "PLI")):
                real = (pv, pl)[j]
                row.update({m: real, f"{m}_shuffle_mean": sh[:, j].mean(), f"{m}_chance95": np.percentile(sh[:, j], 95),
                            f"{m}_p": (1 + (sh[:, j] >= real).sum()) / (1 + NSH)})
            out.append(row)
        return out, list(todo)

    # ── compute: resume per (rat, session, condition) ──
    FILE2 = os.path.join(OUT2, f"rows_{REF}_{TAG}.csv"); TRIED = os.path.join(OUT2, "tried.csv")
    R2 = pd.read_csv(FILE2, dtype={"rat": str}) if os.path.exists(FILE2) else pd.DataFrame(columns=["rat", "session", "condition"])
    TR = pd.read_csv(TRIED, dtype={"rat": str}) if os.path.exists(TRIED) else pd.DataFrame(columns=["rat", "session", "condition"])
    rows2 = R2.to_dict("records")
    FILE = os.path.join(AMIR, "Theta", "Phase connectivity", f"{REF}_{TAG}", "1_Rat3_correct", f"sessions_{REF}_{TAG}.csv")   # rat 3 table of A1
    a1 = pd.read_csv(FILE, dtype={"rat": str}).assign(rat="136", condition="correct")
    have = set(zip(R2.rat.astype(str), R2.session, R2.condition))
    for r in a1.to_dict("records"):
        if ("136", r["session"], "correct") not in have: rows2.append(r); have.add(("136", r["session"], "correct"))
    tried = set(zip(TR.rat.astype(str), TR.session, TR.condition)) | have
    t_ = time.time()
    for rat, (il_xy, pl_xy, pname) in PAIRS3.items():
        for s in sorted(BK.loc[BK.pair == pname, "session"].unique()):
            todo = [c for c in CONDS3 if (rat, s, c) not in tried]
            if not todo: continue
            try:
                new, tr_ = run_conds(rat, s, il_xy, pl_xy, todo)
            except ValueError as e:
                print(f"{RL[rat]} {s}: {e}, skipped"); continue
            rows2 += new; tried |= {(rat, s, c) for c in tr_}
            pd.DataFrame(rows2).to_csv(FILE2, index=False)
            pd.DataFrame(sorted(tried), columns=["rat", "session", "condition"]).to_csv(TRIED, index=False)
            print(f"{RL[rat]} {s}: " + (" | ".join(f"{r['condition']} {r['trials']} tr, PLI p {r['PLI_p']:.3f}" for r in new) or "no condition with ≥ 5 trials")
                  + f" | {(time.time() - t_) / 60:.1f} min", flush=True)
    P2 = pd.DataFrame(rows2); P2["rat"] = P2["rat"].astype(str)
    P2 = P2[P2.condition.isin(CONDS3)].drop_duplicates(["rat", "session", "condition"])
    P2.to_csv(FILE2, index=False)
    print("\nsessions per rat and condition:\n", P2.groupby(["rat", "condition"]).size().unstack(fill_value=0))

    plab = lambda p: ptxt(p) if p < 0.05 else "n.s."
    save_ = lambda fig, name: (_savefig(fig, os.path.join(OUT2, name)),
                               plt.close(fig), display(Image(os.path.join(OUT2, name))))

    # ── FIG 1: 3 rats pooled, conditions ──
    PS = {(c, m): strat_test((P2[P2.condition == c][m] - P2[P2.condition == c][f"{m}_shuffle_mean"]).values,
                             P2[P2.condition == c]["rat"].values) for c in CONDS3 for m in ("PLV", "PLI")}
    both = {c: PS[(c, "PLV")] < 0.05 and PS[(c, "PLI")] < 0.05 for c in CONDS3}
    bold = lambda c, p: (p < 0.05) if c == "correct" else both[c]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.8), facecolor="white", constrained_layout=True)
    for ax, m in zip(axs, ("PLV", "PLI")):
        ymax = 0
        for i, c in enumerate(CONDS3):
            Q = P2[P2.condition == c]; rm = Q.groupby("rat")[m].mean(); ch = Q.groupby("rat")[f"{m}_shuffle_mean"].mean().mean()
            sem = rm.std(ddof=1) / np.sqrt(len(rm)) if len(rm) > 1 else 0
            ax.bar(i, rm.mean(), width=0.6, color=CCOLS[c]); ax.errorbar(i, rm.mean(), yerr=sem, color="k", capsize=5, lw=1.4)
            ax.plot([i - 0.38, i + 0.38], [ch, ch], color="k", lw=1.6, ls="--")
            ax.scatter(i + np.random.default_rng(i).uniform(-0.17, 0.17, len(Q)), Q[m], s=14, c=[RC[r] for r in Q.rat], zorder=3)
            ymax = max(ymax, Q[m].max(), rm.mean() + sem, ch)
        ax.set_ylim(0, ymax * 1.18)
        for i, c in enumerate(CONDS3):
            p = PS[(c, m)]; ax.text(i, ymax * 1.06, plab(p), ha="center", va="bottom", fontsize=10, fontweight="bold" if bold(c, p) else "normal")
        ax.set_xticks(range(3)); ax.set_xticklabels(CONDS3); ax.set_ylabel(f"Theta {m} (press → +{WEND:g} s)")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    h_ = [plt.Line2D([], [], marker="o", ls="", color=RC[r], label=RL[r]) for r in RL] + [plt.Line2D([], [], color="k", ls="--", label="chance (mean of shuffles)")]
    fig.legend(handles=h_, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"3 rats pooled (each rat equal weight), IL–PL theta, {REF}", fontsize=12, fontweight="bold")
    save_(fig, "F1_pooled_conditions.png")

    # ── FIG 2: circle, correct, all sessions of all rats ──
    Q = P2[P2.condition == "correct"]; th = np.radians(Q.dphi_deg.values); Rm, p_ray = rayleigh(th); ang = np.angle(np.exp(1j * th).mean())
    fig = plt.figure(figsize=(5, 5.8), facecolor="white"); axp = fig.add_subplot(projection="polar")
    for a_, r in zip(th, Q.rat): axp.annotate("", xy=(a_, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RC[r], lw=1.4))
    axp.annotate("", xy=(ang, Rm), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=3))
    axp.set_ylim(0, 1.1); axp.set_yticks([]); axp.set_xticks(np.radians([0, 90, 180, 270]))
    axp.set_xticklabels(["0°", "+90°\nIL ahead", "180°", "−90°\nPL ahead"], fontsize=10)
    hp = [plt.Line2D([], [], color=RC[r], lw=1.5, label=RL[r]) for r in RL] + [plt.Line2D([], [], color="k", lw=3, label="mean")]
    fig.legend(handles=hp, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=4, frameon=False, fontsize=10)
    axp.set_title(f"Correct: IL ahead in {(np.sin(th) > 0).sum()}/{len(th)} sessions\nmean {np.degrees(ang):+.0f}°, R = {Rm:.2f}, Rayleigh {ptxt(p_ray)}",
                  fontsize=12, fontweight="bold", pad=22)
    save_(fig, "F2_pooled_circle_correct.png")

    # ── FIG 3: one row per rat — pair map, PLV, PLI, circle ──
    def xyz(rat, xy):
        v = []
        for s in P2.loc[P2.rat == rat, "session"].unique():
            d = CHREG[(CHREG.session == s) & ((CHREG.x - xy[0]).abs() < 1) & ((CHREG.y - xy[1]).abs() < 1)]
            if len(d): v.append([d.ap_mm.iloc[0], abs(d.ml_mm.iloc[0]), d.dv_mm.iloc[0]])
        return np.median(v, 0)
    POS = {r: (xyz(r, PAIRS3[r][0]), xyz(r, PAIRS3[r][1])) for r in PAIRS3}
    other = lambda m: "PLV" if m == "PLI" else "PLI"
    fig = plt.figure(figsize=(18, 4.6 * 3), facecolor="white")
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.25, 1.25, 1.0], wspace=0.35, hspace=0.55)
    for row, rat in enumerate(("134", "135", "136")):
        ax = fig.add_subplot(gs[row, 0])
        for rr, (il_, pl_) in POS.items():
            hl = rr == rat
            ax.plot([il_[1], pl_[1]], [il_[2], pl_[2]], color=RC[rr], ls="--", lw=2.2 if hl else 1.1, alpha=1 if hl else 0.35)
            ax.scatter([il_[1], pl_[1]], [il_[2], pl_[2]], marker="s", s=50 if hl else 22, color=RC[rr], edgecolor="k" if hl else "none", alpha=1 if hl else 0.35)
        il_, pl_ = POS[rat]
        ax.set_title(f"{RL[rat]}: {np.linalg.norm(il_ - pl_):.1f} mm", fontsize=11, fontweight="bold")
        ax.set_xlabel("|ML| (mm)"); ax.set_ylabel("DV (mm)")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        Qr = P2[P2.rat == rat]
        for j, m in enumerate(("PLV", "PLI")):
            ax = fig.add_subplot(gs[row, 1 + j]); top = 0; txt = []
            pr = {c: onesided((Qr[Qr.condition == c][m] - Qr[Qr.condition == c][f"{m}_shuffle_mean"]).values) for c in CONDS3 if (Qr.condition == c).sum() >= 2}
            pr2 = {c: onesided((Qr[Qr.condition == c][other(m)] - Qr[Qr.condition == c][f"{other(m)}_shuffle_mean"]).values) for c in pr}
            for ci, c in enumerate(CONDS3):
                if c not in pr: continue
                d = Qr[Qr.condition == c]; v = d[m].values; ch = d[f"{m}_shuffle_mean"].mean()
                ax.bar(ci, v.mean(), width=0.6, color=CCOLS[c])
                ax.errorbar(ci, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), color="k", capsize=4, lw=1.2)
                ax.scatter(ci + np.random.default_rng(1).uniform(-0.13, 0.13, len(v)), v, s=8, color="0.35", zorder=3)
                ax.plot([ci - 0.35, ci + 0.35], [ch, ch], color="k", ls="--", lw=1.4)
                top = max(top, v.max(), ch); txt.append((ci, c, pr[c], len(v)))
            ax.set_ylim(0, top * 1.3)
            for ci, c, p, n in txt:
                bb = (p < 0.05) if c == "correct" else (p < 0.05 and pr2[c] < 0.05)
                ax.text(ci, top * 1.06, f"{plab(p)}\nn={n}", ha="center", va="bottom", fontsize=8.5, fontweight="bold" if bb else "normal")
            ax.set_xticks(range(3)); ax.set_xticklabels(CONDS3, fontsize=9); ax.set_ylabel(f"Theta {m}"); ax.set_title(m, fontsize=11, fontweight="bold")
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        axp = fig.add_subplot(gs[row, 3], projection="polar")
        d = Qr[Qr.condition == "correct"]; th = np.radians(d.dphi_deg.values); Rr, pR = rayleigh(th); a = np.angle(np.exp(1j * th).mean())
        for a_ in th: axp.annotate("", xy=(a_, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=CCOLS["correct"], lw=1.4))
        axp.annotate("", xy=(a, Rr), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=3))
        axp.set_ylim(0, 1.1); axp.set_yticks([]); axp.set_xticks(np.radians([0, 90, 180, 270]))
        axp.set_xticklabels(["0°", "+90°\nIL ahead", "180°", "−90°\nPL ahead"], fontsize=9)
        axp.set_title(f"Correct: IL ahead {(np.sin(th) > 0).sum()}/{len(th)}\n{np.degrees(a):+.0f}°, R = {Rr:.2f}, Rayleigh {ptxt(pR)}",
                      fontsize=10, fontweight="bold" if pR < 0.05 else "normal", pad=16)
    fig.suptitle(f"IL–PL theta per rat, press → +{WEND:g} s ({REF})", fontsize=13, fontweight="bold", y=0.995)
    save_(fig, "F3_per_rat_rows.png")

    print("\npooled p (one-sided, rats equal weight):")
    for c in CONDS3: print(f"  {c}: PLV {plab(PS[(c, 'PLV')])} | PLI {plab(PS[(c, 'PLI')])}")

    # ── check: trial counts vs C11 (same sessions and conditions) ──
    bk = BK[BK.pair.isin([v[2] for v in PAIRS3.values()]) & BK.condition.isin(CONDS3)].assign(rat=lambda d: d.pair.str[:3])
    cmp = P2.merge(bk[["rat", "session", "condition", "trials"]], on=["rat", "session", "condition"], how="outer", suffixes=("_new", "_C11"))
    cmp["same"] = cmp.trials_new == cmp.trials_C11
    print("\ntrial counts same as C11 (per rat and condition):\n", cmp.groupby(["rat", "condition"]).same.agg(["sum", "size"]))
    print(cmp[~cmp.same].head(15).to_string(index=False))


@cell(needs={
    'A2b_three_rats': ['OUT2', 'P2'],
})
def direction_circles_all():
    """IL–PL theta phase direction per condition, ALL trials (from A2b's P2) — arrows in the condition colour"""
    global C_OK, Q, Rm, a_, ang, axp, c, fig, fp_, hp, i, p_ray, th
    # ── IL–PL theta phase direction per condition, ALL trials (from A2b's P2) — arrows in the condition colour
    C_OK = [c for c in CONDS3 if (P2.condition == c).any()]
    fig = plt.figure(figsize=(4.2 * len(C_OK), 5.2), facecolor="white")
    for i, c in enumerate(C_OK):
        Q = P2[P2.condition == c]; th = np.radians(Q.dphi_deg.values); Rm, p_ray = rayleigh(th); ang = np.angle(np.exp(1j * th).mean())
        axp = fig.add_subplot(1, len(C_OK), i + 1, projection="polar")
        for a_ in th: axp.annotate("", xy=(a_, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=CCOLS[c], lw=1.4, alpha=0.85))
        axp.annotate("", xy=(ang, Rm), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=3))
        axp.set_ylim(0, 1.1); axp.set_yticks([]); axp.set_xticks(np.radians([0, 90, 180, 270]))
        axp.set_xticklabels(["0°", "+90°\nIL ahead", "180°", "−90°\nPL ahead"], fontsize=9)
        axp.set_title(f"{c.capitalize()} ({len(Q)} sessions)\nIL ahead {(np.sin(th) > 0).sum()}/{len(th)} | {np.degrees(ang):+.0f}°, R = {Rm:.2f}\nRayleigh {ptxt(p_ray)}",
                      fontsize=10, color=CCOLS[c], fontweight="bold" if p_ray < 0.05 else "normal", pad=20)
    hp = [plt.Line2D([], [], color="0.5", lw=1.4, label="one session"), plt.Line2D([], [], color="k", lw=3, label="mean (length = R)")]
    fig.legend(handles=hp, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"IL–PL theta phase direction per session, press → +{WEND:g} s, {REF}, all trials (≥ {MIN_TR2} per condition)",
                 fontsize=11, fontweight="bold", y=1.06)
    fig.subplots_adjust(wspace=0.55)
    fp_ = os.path.join(OUT2, "F4_circles_all_conditions_alltrials.png")
    _savefig(fig, fp_); plt.close(fig); display(Image(fp_))

    for c in C_OK:
        Q = P2[P2.condition == c]; th = np.radians(Q.dphi_deg.values); Rm, p_ray = rayleigh(th)
        print(f"{c:8s} IL ahead {(np.sin(th) > 0).sum()}/{len(th)} sessions | mean {np.degrees(np.angle(np.exp(1j * th).mean())):+.0f}° | R {Rm:.2f} | Rayleigh {ptxt(p_ray)}")


@cell()
def A2c_equal_n():
    """A2c. 3 RATS, CORRECT / LATE / EARLY, PRESS → +1 s, SAME NUMBER OF TRIALS per condition and session (needs A1 + A2b)"""
    global C_OK, FILE3, NREP, NSH_EQ, N_EQ, OUT3, P3, PS3, Q, Rm, TAG3, TRIED3, a_, ang, ax, axp, axs, both3, c, ch, e, fig
    global h_, hp, i, il_xy, m, new, nses, p, p_ray, pl_xy, plab, pname, r, rat, rm, rng_eq, rows3, run_eq, s, save3, sem, sp
    global t_, th, tried3, ymax
    # ── A2c. 3 RATS, CORRECT / LATE / EARLY, PRESS → +1 s, SAME NUMBER OF TRIALS per condition and session (needs A1 + A2b) ──
    N_EQ, NREP, NSH_EQ = 10, 20, 25                 # trials per condition per session | random draws | shuffles per draw
    TAG3 = f"2c_three_rats_conditions_equalN{N_EQ}"
    OUT3 = os.path.join(AMIR, "Theta", "Phase connectivity", f"{REF}_{TAG}", TAG3); os.makedirs(OUT3, exist_ok=True)
    FILE3, TRIED3 = os.path.join(OUT3, f"rows_{REF}_{TAG}.csv"), os.path.join(OUT3, "tried.csv")
    rng_eq = np.random.default_rng(0)

    def run_eq(rat, session, il_xy, pl_xy):
        """one session: read signals once; per condition, NREP random draws of N_EQ trials"""
        T = fn.session_trials(K, session).reset_index(drop=True)
        masks = {c: cond_trials(T, c).values for c in CONDS3}
        if not any(m_.sum() >= N_EQ for m_ in masks.values()): return []
        T = T[np.any(np.stack(list(masks.values())), 0)].reset_index(drop=True)
        masks = {c: cond_trials(T, c).values for c in CONDS3}
        xI, xP, t, fs, ok, used = get_pair(session, T, il_xy, pl_xy)
        T = T[ok].reset_index(drop=True); masks = {c: m_[ok] for c, m_ in masks.items()}
        hold = (T["release_s"] - T["press_s"]).values; out = []
        for c, m_ in masks.items():
            idx_all = np.where(m_)[0]
            if len(idx_all) < N_EQ: continue
            rv, sv, dv = [], [], []
            for _ in range(NREP):
                idx = rng_eq.choice(idx_all, N_EQ, replace=False)
                A_, B_, h_, tt, gr, M_, o = prepare(xI[idx], xP[idx], t, fs, hold[idx])
                if not gr: continue
                rv.append(plv_pli(A_, B_, gr)); sv += [plv_pli(A_, B_, gr, True) for _ in range(NSH_EQ)]
                dv.append(direction(A_, B_, M_)[0])
            if not rv: continue
            rv, sv = np.array(rv), np.array(sv)
            row = dict(rat=rat, session=session, condition=c, IL=used[0], PL=used[1], trials_available=len(idx_all), trials=N_EQ,
                       dphi_deg=np.degrees(np.angle(np.exp(1j * np.array(dv)).mean())))
            for j, m in enumerate(("PLV", "PLI")):
                row.update({m: rv[:, j].mean(), f"{m}_shuffle_mean": sv[:, j].mean(), f"{m}_chance95": np.percentile(sv[:, j], 95)})
            out.append(row)
        return out

    # ── compute (resume per session) ──
    rows3 = pd.read_csv(FILE3, dtype={"rat": str}).to_dict("records") if os.path.exists(FILE3) else []
    tried3 = set(map(tuple, pd.read_csv(TRIED3, dtype={"rat": str})[["rat", "session"]].values)) if os.path.exists(TRIED3) else set()
    t_ = time.time()
    for rat, (il_xy, pl_xy, pname) in PAIRS3.items():
        for s in sorted(BK.loc[BK.pair == pname, "session"].unique()):
            if (rat, s) in tried3: continue
            try: new = run_eq(rat, s, il_xy, pl_xy)
            except ValueError as e: print(f"{RL[rat]} {s}: {e}, skipped"); continue
            rows3 += new; tried3.add((rat, s))
            pd.DataFrame(rows3).to_csv(FILE3, index=False); pd.DataFrame(sorted(tried3), columns=["rat", "session"]).to_csv(TRIED3, index=False)
            print(f"{RL[rat]} {s}: " + (" | ".join(f"{r['condition']} ({r['trials_available']} avail.)" for r in new) or f"no condition with ≥ {N_EQ} trials")
                  + f" | {(time.time() - t_) / 60:.1f} min", flush=True)
    P3 = pd.DataFrame(rows3); P3["rat"] = P3["rat"].astype(str)
    print(f"\nsessions per rat and condition ({N_EQ} trials each):\n", P3.groupby(["rat", "condition"]).size().unstack(fill_value=0))

    plab = lambda p: ptxt(p) if p < 0.05 else "n.s."
    save3 = lambda fig, name: (_savefig(fig, os.path.join(OUT3, name)),
                               plt.close(fig), display(Image(os.path.join(OUT3, name))))

    # ── FIG 1: 3 rats pooled, conditions, equal trial numbers ──
    C_OK = [c for c in CONDS3 if (P3.condition == c).any()]
    PS3 = {(c, m): strat_test((P3[P3.condition == c][m] - P3[P3.condition == c][f"{m}_shuffle_mean"]).values,
                              P3[P3.condition == c]["rat"].values) for c in C_OK for m in ("PLV", "PLI")}
    both3 = {c: PS3[(c, "PLV")] < 0.05 and PS3[(c, "PLI")] < 0.05 for c in C_OK}
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.8), facecolor="white", constrained_layout=True)
    for ax, m in zip(axs, ("PLV", "PLI")):
        ymax = 0
        for i, c in enumerate(C_OK):
            Q = P3[P3.condition == c]; rm = Q.groupby("rat")[m].mean(); ch = Q.groupby("rat")[f"{m}_shuffle_mean"].mean().mean()
            sem = rm.std(ddof=1) / np.sqrt(len(rm)) if len(rm) > 1 else 0
            ax.bar(i, rm.mean(), width=0.6, color=CCOLS[c]); ax.errorbar(i, rm.mean(), yerr=sem, color="k", capsize=5, lw=1.4)
            ax.plot([i - 0.38, i + 0.38], [ch, ch], color="k", lw=1.6, ls="--")
            ax.scatter(i + np.random.default_rng(i).uniform(-0.17, 0.17, len(Q)), Q[m], s=14, c=[RC[r] for r in Q.rat], zorder=3)
            ymax = max(ymax, Q[m].max(), rm.mean() + sem, ch)
        ax.set_ylim(0, ymax * 1.18)
        for i, c in enumerate(C_OK):
            p = PS3[(c, m)]; nses = int((P3.condition == c).sum())
            ax.text(i, ymax * 1.05, f"{plab(p)}\n{nses} sess", ha="center", va="bottom", fontsize=9, fontweight="bold" if both3[c] else "normal")
        ax.set_xticks(range(len(C_OK))); ax.set_xticklabels(C_OK); ax.set_ylabel(f"Theta {m} (press → +{WEND:g} s)")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    h_ = [plt.Line2D([], [], marker="o", ls="", color=RC[r], label=RL[r]) for r in RL] + [plt.Line2D([], [], color="k", ls="--", label="chance (mean of shuffles)")]
    fig.legend(handles=h_, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"3 rats pooled, IL–PL theta, {REF} — {N_EQ} trials per condition and session ({NREP} random draws)", fontsize=12, fontweight="bold")
    save3(fig, f"F1_pooled_conditions_equalN{N_EQ}.png")

    # ── FIG 2: circle, correct, equal trial numbers ──
    Q = P3[P3.condition == "correct"]; th = np.radians(Q.dphi_deg.values); Rm, p_ray = rayleigh(th); ang = np.angle(np.exp(1j * th).mean())
    fig = plt.figure(figsize=(5, 5.8), facecolor="white"); axp = fig.add_subplot(projection="polar")
    for a_, r in zip(th, Q.rat): axp.annotate("", xy=(a_, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RC[r], lw=1.4))
    axp.annotate("", xy=(ang, Rm), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=3))
    axp.set_ylim(0, 1.1); axp.set_yticks([]); axp.set_xticks(np.radians([0, 90, 180, 270]))
    axp.set_xticklabels(["0°", "+90°\nIL ahead", "180°", "−90°\nPL ahead"], fontsize=10)
    hp = [plt.Line2D([], [], color=RC[r], lw=1.5, label=RL[r]) for r in RL] + [plt.Line2D([], [], color="k", lw=3, label="mean")]
    fig.legend(handles=hp, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=4, frameon=False, fontsize=10)
    axp.set_title(f"Correct ({N_EQ} trials/session): IL ahead in {(np.sin(th) > 0).sum()}/{len(th)} sessions\n"
                  f"mean {np.degrees(ang):+.0f}°, R = {Rm:.2f}, Rayleigh {ptxt(p_ray)}", fontsize=12, fontweight="bold", pad=22)
    save3(fig, f"F2_pooled_circle_correct_equalN{N_EQ}.png")

    print(f"\npooled p (one-sided, rats equal weight), {N_EQ} trials per condition and session:")
    for c in C_OK: print(f"  {c}: PLV {plab(PS3[(c, 'PLV')])} | PLI {plab(PS3[(c, 'PLI')])} | {int((P3.condition == c).sum())} sessions")


@cell(needs={
    'A2c_equal_n': ['N_EQ', 'P3', 'save3'],
})
def consistency_circles_equal_n():
    """consistency per condition (equal trial numbers, from P3): phase-direction circles + sessions above chance"""
    global C_OK, Q, Rm, a_, ang, axp, c, fig, hp, i, p_ray, r, th
    # ── consistency per condition (equal trial numbers, from P3): phase-direction circles + sessions above chance
    C_OK = [c for c in CONDS3 if (P3.condition == c).any()]
    fig = plt.figure(figsize=(4.2 * len(C_OK), 5.2), facecolor="white")
    for i, c in enumerate(C_OK):
        Q = P3[P3.condition == c]; th = np.radians(Q.dphi_deg.values); Rm, p_ray = rayleigh(th); ang = np.angle(np.exp(1j * th).mean())
        axp = fig.add_subplot(1, len(C_OK), i + 1, projection="polar")
        for a_, r in zip(th, Q.rat): axp.annotate("", xy=(a_, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RC[r], lw=1.3))
        axp.annotate("", xy=(ang, Rm), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=3))
        axp.set_ylim(0, 1.1); axp.set_yticks([]); axp.set_xticks(np.radians([0, 90, 180, 270]))
        axp.set_xticklabels(["0°", "+90°\nIL ahead", "180°", "−90°\nPL ahead"], fontsize=9)
        axp.set_title(f"{c.capitalize()} ({len(Q)} sessions)\nIL ahead {(np.sin(th) > 0).sum()}/{len(th)} | {np.degrees(ang):+.0f}°, R = {Rm:.2f}\nRayleigh {ptxt(p_ray)}",
                      fontsize=10, color=CCOLS[c], fontweight="bold" if p_ray < 0.05 else "normal", pad=20)
    hp = [plt.Line2D([], [], color=RC[r], lw=1.5, label=RL[r]) for r in RL] + [plt.Line2D([], [], color="k", lw=3, label="mean")]
    fig.legend(handles=hp, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"IL–PL theta phase direction per session, press → +{WEND:g} s, {REF}, {N_EQ} trials per condition and session",
                 fontsize=11, fontweight="bold", y=1.06)
    fig.subplots_adjust(wspace=0.55)
    save3(fig, f"F4_circles_all_conditions_equalN{N_EQ}.png")

    print(f"sessions above chance (value > mean of shuffles), {N_EQ} trials per condition and session:")
    for c in C_OK:
        Q = P3[P3.condition == c]
        print(f"  {c:8s} PLV {int((Q.PLV > Q.PLV_shuffle_mean).sum())}/{len(Q)} | PLI {int((Q.PLI > Q.PLI_shuffle_mean).sum())}/{len(Q)} | "
              + " | ".join(f"{RL[r]}: PLI {int((g.PLI > g.PLI_shuffle_mean).sum())}/{len(g)}" for r, g in Q.groupby("rat")))


@cell(needs={
    'A2c_equal_n': ['plab'],
})
def A2d_unbiased():
    """A2d. 3 RATS, CORRECT / LATE / EARLY, PRESS → +1 s, ALL TRIALS, UNBIASED MEASURES: PPC, debiased PLI², debiased wPLI (needs A1 + A2b)"""
    global C_OK, FILE4, MEAS_U, NSH_U, OUT4, P4, PU, Q, SHOW, TAG4, TRIED4, W_, a, ax, axs, b, c, ch, d, e, f_, fig, fp_, h_
    global hi, i, il_xy, lab, lo, m, new, pl_xy, pname, rat, rats_, rm, rows4, run_unb, s, s_, sem, sp_, sq, strat_two, t_
    global tried4, txt, unbiased, v
    # ── A2d. 3 RATS, CORRECT / LATE / EARLY, PRESS → +1 s, ALL TRIALS, UNBIASED MEASURES: PPC, debiased PLI², debiased wPLI (needs A1 + A2b) ──
    NSH_U = 50
    TAG4 = "2d_three_rats_conditions_unbiased"
    OUT4 = os.path.join(AMIR, "Theta", "Phase connectivity", f"{REF}_{TAG}", TAG4); os.makedirs(OUT4, exist_ok=True)
    FILE4, TRIED4 = os.path.join(OUT4, f"rows_{REF}_{TAG}.csv"), os.path.join(OUT4, "tried.csv")
    MEAS_U = ["PPC", "dPLI2", "dwPLI", "PLV", "PLI"]

    def unbiased(A_, B_, groups, shuffle=False):
        """same groups (held trials) and averaging as plv_pli; per time point and frequency: PPC, debiased PLI², debiased wPLI (+ PLV, PLI)"""
        out = {m: [] for m in MEAS_U}
        for n_, cols in groups:
            if n_ < 2: continue
            b = B_[:n_][derangement(n_)] if shuffle else B_[:n_]
            S = A_[:n_][..., cols] * np.conj(b[..., cols])                  # trials × freqs × times
            E = S / np.abs(S).clip(1e-30); sg = np.sign(S.imag); X = S.imag
            out["PPC"].append(((np.abs(E.sum(0)) ** 2 - n_) / (n_ * (n_ - 1))).ravel())
            out["dPLI2"].append(((sg.sum(0) ** 2 - (sg ** 2).sum(0)) / (n_ * (n_ - 1))).ravel())
            sx, sx2, sa = X.sum(0), (X ** 2).sum(0), np.abs(X).sum(0); den = sa ** 2 - sx2
            out["dwPLI"].append(np.where(den > 0, (sx ** 2 - sx2) / np.where(den > 0, den, 1), np.nan).ravel())
            out["PLV"].append(np.abs(E.mean(0)).ravel()); out["PLI"].append(np.abs(sg.mean(0)).ravel())
        return {m: np.nanmean(np.concatenate(v)) for m, v in out.items()}

    def run_unb(rat, session, il_xy, pl_xy):
        T = fn.session_trials(K, session).reset_index(drop=True)
        masks = {c: cond_trials(T, c).values for c in CONDS3}
        if not any(m_.sum() >= MIN_TR2 for m_ in masks.values()): return []
        T = T[np.any(np.stack(list(masks.values())), 0)].reset_index(drop=True)
        masks = {c: cond_trials(T, c).values for c in CONDS3}
        xI, xP, t, fs, ok, used = get_pair(session, T, il_xy, pl_xy)
        T = T[ok].reset_index(drop=True); masks = {c: m_[ok] for c, m_ in masks.items()}
        hold = (T["release_s"] - T["press_s"]).values; out = []
        for c, m_ in masks.items():
            if m_.sum() < MIN_TR2: continue
            A_, B_, h_, tt, gr, M_, o = prepare(xI[m_], xP[m_], t, fs, hold[m_])
            if not gr: continue
            real = unbiased(A_, B_, gr); sh = pd.DataFrame([unbiased(A_, B_, gr, True) for _ in range(NSH_U)])
            row = dict(rat=rat, session=session, condition=c, IL=used[0], PL=used[1], trials=int(m_.sum()))
            for m in MEAS_U: row.update({m: real[m], f"{m}_shuffle_mean": sh[m].mean()})
            out.append(row)
        return out

    # ── compute (resume per session) ──
    rows4 = pd.read_csv(FILE4, dtype={"rat": str}).to_dict("records") if os.path.exists(FILE4) else []
    tried4 = set(map(tuple, pd.read_csv(TRIED4, dtype={"rat": str})[["rat", "session"]].values)) if os.path.exists(TRIED4) else set()
    t_ = time.time()
    for rat, (il_xy, pl_xy, pname) in PAIRS3.items():
        for s in sorted(BK.loc[BK.pair == pname, "session"].unique()):
            if (rat, s) in tried4: continue
            try: new = run_unb(rat, s, il_xy, pl_xy)
            except ValueError as e: print(f"{RL[rat]} {s}: {e}, skipped"); continue
            rows4 += new; tried4.add((rat, s))
            pd.DataFrame(rows4).to_csv(FILE4, index=False); pd.DataFrame(sorted(tried4), columns=["rat", "session"]).to_csv(TRIED4, index=False)
            print(f"{RL[rat]} {s}: " + (" | ".join(f"{r['condition']} {r['trials']} tr" for r in new) or f"no condition with ≥ {MIN_TR2} trials")
                  + f" | {(time.time() - t_) / 60:.1f} min", flush=True)
    P4 = pd.DataFrame(rows4); P4["rat"] = P4["rat"].astype(str)
    print(f"\nsessions per rat and condition (all trials, ≥ {MIN_TR2}):\n", P4.groupby(["rat", "condition"]).size().unstack(fill_value=0))

    # ── tests: each condition vs chance (one-sided), conditions vs each other (paired, two-sided); rats equal weight ──
    def strat_two(d, rats, NPERM=10000):
        rng_ = np.random.default_rng(0); u = np.unique(rats)
        W = np.stack([(rats == r) / (rats == r).sum() for r in u], 1); real = (d @ W).mean()
        null = ((rng_.choice([-1.0, 1.0], size=(NPERM, len(d))) * d) @ W).mean(1)
        return (1 + np.sum(null >= real)) / (NPERM + 1), (1 + np.sum(np.abs(null) >= abs(real))) / (NPERM + 1)
    C_OK = [c for c in CONDS3 if (P4.condition == c).any()]
    PU = {(c, m): strat_two((P4[P4.condition == c][m] - P4[P4.condition == c][f"{m}_shuffle_mean"]).values,
                            P4[P4.condition == c].rat.values)[0] for c in C_OK for m in MEAS_U}
    print("\nvs chance (one-sided, rats equal weight):")
    for c in C_OK: print(f"  {c:8s} " + " | ".join(f"{m} {plab(PU[(c, m)])}" for m in MEAS_U) + f" | {int((P4.condition == c).sum())} sessions")
    print("\nconditions compared (paired by session, real − chance, two-sided):")
    for a, b in (("correct", "late"), ("correct", "early"), ("early", "late")):
        W_ = P4[P4.condition.isin([a, b])].pivot_table(index=["rat", "session"], columns="condition",
                                                        values=[m for m in MEAS_U] + [f"{m}_shuffle_mean" for m in MEAS_U]).dropna()
        if len(W_) < 3: continue
        rats_ = W_.index.get_level_values("rat").values; txt = []
        for m in MEAS_U:
            d = (W_[(m, a)] - W_[(f"{m}_shuffle_mean", a)]) - (W_[(m, b)] - W_[(f"{m}_shuffle_mean", b)])
            txt.append(f"{m} {'+' if d.mean() > 0 else '−'} {plab(strat_two(d.values, rats_)[1])}")
        print(f"  {a} vs {b} ({len(W_)} sessions): " + " | ".join(txt) + f"   (+ = {a} higher)")

    # ── figure: one panel per unbiased measure ──
    SHOW = [("PPC", "√PPC (bias-free PLV)"), ("dPLI2", "debiased PLI²"), ("dwPLI", "debiased wPLI")]
    sq = lambda v: np.sign(v) * np.sqrt(np.abs(v))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.8), facecolor="white", constrained_layout=True)
    for ax, (m, lab) in zip(axs, SHOW):
        f_ = sq if m == "PPC" else (lambda v: v); lo, hi = 0, 0
        for i, c in enumerate(C_OK):
            Q = P4[P4.condition == c]; v = f_(Q[m].values); rm = pd.Series(v, index=Q.rat.values).groupby(level=0).mean()
            ch = f_(Q[f"{m}_shuffle_mean"].mean()); sem = rm.std(ddof=1) / np.sqrt(len(rm)) if len(rm) > 1 else 0
            ax.bar(i, rm.mean(), width=0.6, color=CCOLS[c]); ax.errorbar(i, rm.mean(), yerr=sem, color="k", capsize=5, lw=1.4)
            ax.plot([i - 0.38, i + 0.38], [ch, ch], color="k", lw=1.6, ls="--")
            ax.scatter(i + np.random.default_rng(i).uniform(-0.17, 0.17, len(Q)), v, s=12, c=[RC[r] for r in Q.rat], zorder=3)
            lo, hi = min(lo, v.min(), ch), max(hi, v.max(), rm.mean() + sem, ch)
        sp_ = hi - lo; ax.set_ylim(lo - 0.05 * sp_, hi + 0.22 * sp_); ax.axhline(0, color="0.5", lw=0.8)
        for i, c in enumerate(C_OK):
            ax.text(i, hi + 0.06 * sp_, f"{plab(PU[(c, m)])}\n{int((P4.condition == c).sum())} sess", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if PU[(c, m)] < 0.05 else "normal")
        ax.set_xticks(range(len(C_OK))); ax.set_xticklabels(C_OK); ax.set_ylabel(f"Theta {lab}"); ax.set_title(lab, fontsize=11, fontweight="bold")
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    h_ = [plt.Line2D([], [], marker="o", ls="", color=RC[r], label=RL[r]) for r in RL] + [plt.Line2D([], [], color="k", ls="--", label="chance (trial shuffle)")]
    fig.legend(handles=h_, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"IL–PL theta, press → +{WEND:g} s while holding, {REF} — all trials, measures not biased by trial number", fontsize=12, fontweight="bold")
    fp_ = os.path.join(OUT4, "F1_pooled_conditions_unbiased.png"); _savefig(fig, fp_); plt.close(fig); display(Image(fp_))


@cell()
def T4_power_control():
    """T4. POWER CONTROL for T1: hold vs pre / post with theta amplitude matched (PPC + PLI, correct long, SR30-500) (needs A1, A2b, T1)"""
    global Eh, Eo, F4, L, Mh, Mo, OUT4, P4, Q, R4, Sh, So, T, Tp, WN4, _, ah, ao, ax, axs, col, d, done4, e, fig, fs, h_, hold
    global ih, il_xy, io, lh, lo_, m, match, new, ok, ok2, other, p, parts, per, ph, pl_xy, pname, po, ppc_masked, r, rat
    global raw_ratio, row, rows4, s, sp, t, t_, txt, window_parts, xI, xP
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ── T4. POWER CONTROL for T1: hold vs pre / post with theta amplitude matched (PPC + PLI, correct long, SR30-500) (needs A1, A2b, T1) ──
    OUT4 = os.path.join(AMIR, "Theta", "Phase connectivity", "Tests", f"T4_power_control_{REF}"); os.makedirs(OUT4, exist_ok=True)
    WN4 = {"pre": ("Press", -1.0, 0.0), "hold": ("Press", 0.0, 3.0), "post": ("Release", 0.0, 1.0)}

    def window_parts(xI, xP, t, fs, w0, w1, hold=None):
        """ERP removed; returns unit phase diffs E, sign of imag S, held mask M, amplitude per trial"""
        xI = xI - xI.mean(0, keepdims=True); xP = xP - xP.mean(0, keepdims=True)
        idx = np.where((t >= w0 - 1e-9) & (t <= w1 + 1e-9))[0][::int(round(STEP_S * fs))]; tt = t[idx]
        A_, B_ = wavelet(xI, fs, idx), wavelet(xP, fs, idx); S = A_ * np.conj(B_)
        M_ = np.ones((len(A_), len(tt)), bool) if hold is None else (hold[:, None] >= tt[None, :])
        amp = np.array([np.sqrt(np.abs(A_[i][:, M_[i]]).mean() * np.abs(B_[i][:, M_[i]]).mean()) if M_[i].any() else np.nan for i in range(len(A_))])
        return S / np.abs(S).clip(1e-30), np.sign(S.imag), M_, amp

    def ppc_masked(E, Sg, M_):
        """PPC and PLI per time point over the trials held there (≥ MIN_ON), averaged over time and freqs"""
        w = M_[:, None, :]; n = M_.sum(0); ok_ = n >= MIN_ON
        s = (E * w).sum(0)[:, ok_]; nn = n[ok_]
        ppc = ((np.abs(s) ** 2 - nn) / (nn * (nn - 1))).mean()
        pli = np.abs((Sg * w).sum(0)[:, ok_] / nn).mean()
        return ppc, pli

    def match(amp_hi, amp_lo):
        """drop highest of amp_hi and lowest of amp_lo until mean(hi) <= mean(lo); equal n"""
        hi, lo = list(np.argsort(amp_hi)), list(np.argsort(amp_lo))
        while len(hi) > MIN_TR2 and np.nanmean(amp_hi[hi]) > np.nanmean(amp_lo[lo]):
            hi.pop(-1); lo.pop(0)
        return np.array(hi), np.array(lo)

    F4 = os.path.join(OUT4, f"T4_rows_{REF}.csv")
    R4 = pd.read_csv(F4, dtype={"rat": str}) if os.path.exists(F4) else pd.DataFrame(columns=["rat", "session"])
    rows4, done4, t_ = R4.to_dict("records"), set(zip(R4.rat.astype(str), R4.session)), time.time()
    for rat, (il_xy, pl_xy, pname) in PAIRS3.items():
        for s in sorted(BK.loc[BK.pair == pname, "session"].unique()):
            if (rat, s) in done4: continue
            T = fn.session_trials(K, s); T = T[(T["type"] == "long") & (T["label"] == "correct")].reset_index(drop=True)
            try:
                xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, "Press")
                Tp = T[ok].reset_index(drop=True); hold = (Tp["release_s"] - Tp["press_s"]).values
                parts = {"pre": window_parts(xI, xP, t, fs, -1.0, 0.0), "hold": window_parts(xI, xP, t, fs, 0.0, 3.0, hold)}
                xI, xP, t, fs, ok2 = get_ev(s, T, il_xy, pl_xy, "Release")
                parts["post"] = window_parts(xI, xP, t, fs, 0.0, 1.0); del xI, xP
            except ValueError as e:
                print(f"{RL[rat]} {s}: {e}, skipped"); continue
            new = []
            for other in ("pre", "post"):
                Eh, Sh, Mh, ah = parts["hold"]; Eo, So, Mo, ao = parts[other]
                raw_ratio = np.nanmean(ah) / np.nanmean(ao)
                ih, io = match(ah, ao)
                if len(ih) < MIN_TR2: continue
                ph, lh = ppc_masked(Eh[ih], Sh[ih], Mh[ih]); po, lo_ = ppc_masked(Eo[io], So[io], Mo[io])
                new.append(dict(rat=rat, session=s, comparison=f"hold vs {other}", n=len(ih), amp_ratio_raw=raw_ratio,
                                amp_ratio_matched=np.nanmean(ah[ih]) / np.nanmean(ao[io]),
                                hold_PPC=ph, other_PPC=po, hold_PLI=lh, other_PLI=lo_))
            rows4 += new; done4.add((rat, s)); pd.DataFrame(rows4).to_csv(F4, index=False)
            print(f"{RL[rat]} {s}: " + " | ".join(f"{r['comparison']} n={r['n']} amp {r['amp_ratio_raw']:.2f}→{r['amp_ratio_matched']:.2f} "
                                                 f"PPC {r['hold_PPC']:.4f} vs {r['other_PPC']:.4f}" for r in new)
                  + f" | {(time.time() - t_) / 60:.1f} min", flush=True)
    P4 = pd.DataFrame(rows4); P4["rat"] = P4["rat"].astype(str)

    # ── figure + numbers ──
    fig, axs = plt.subplots(2, 2, figsize=(10, 9), facecolor="white", constrained_layout=True)
    L = [f"T4 power control (correct long, ERP removed, theta amplitude matched, equal n, {REF})"]
    for row, other in enumerate(("pre", "post")):
        Q = P4[P4.comparison == f"hold vs {other}"]
        L.append(f"hold vs {other}: amplitude ratio hold/{other} raw {Q.amp_ratio_raw.mean():.2f} → matched {Q.amp_ratio_matched.mean():.2f} | n per session median {Q.n.median():.0f}")
        for col, m in enumerate(("PPC", "PLI")):
            ax = axs[row, col]; d = (Q[f"hold_{m}"] - Q[f"other_{m}"]).values; p = strat_test(d, Q.rat.values)
            for _, r in Q.iterrows(): ax.plot([0, 1], [r[f"other_{m}"], r[f"hold_{m}"]], color=RC[r.rat], lw=1, alpha=0.7, marker="o", ms=3)
            ax.bar([0, 1], [Q.groupby("rat")[f"other_{m}"].mean().mean(), Q.groupby("rat")[f"hold_{m}"].mean().mean()],
                   width=0.5, color=["0.65", "#2CA02C"], alpha=0.35, zorder=0)
            if m == "PPC": ax.axhline(0, color="k", lw=0.8, ls=":")
            ax.set_xticks([0, 1]); ax.set_xticklabels([other, "hold"]); ax.set_ylabel(f"Theta {m} (power matched)")
            ax.set_title(f"hold > {other}: {ptxt(p) if p < 0.05 else 'n.s.'} ({len(Q)} sessions)", fontsize=11, fontweight="bold" if p < 0.05 else "normal")
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            per = " | ".join(f"{RL[r]} p {onesided(g[f'hold_{m}'].values - g[f'other_{m}'].values):.3f} (n={len(g)})" for r, g in Q.groupby("rat"))
            L.append(f"  {m}: hold {Q[f'hold_{m}'].mean():.4f} vs {other} {Q[f'other_{m}'].mean():.4f} | pooled p {p:.3f} | hold higher {(d > 0).sum()}/{len(d)} | {per}")
    h_ = [plt.Line2D([], [], color=RC[r], marker="o", label=RL[r]) for r in RL]
    fig.legend(handles=h_, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Does 'coupling higher during waiting' survive equal theta power?", fontsize=12, fontweight="bold")
    _savefig(fig, os.path.join(OUT4, "F1_power_control.png")); plt.close(fig)
    display(Image(os.path.join(OUT4, "F1_power_control.png")))
    txt = "\n".join(L); print(txt); open(os.path.join(OUT4, "T4_summary.txt"), "w", encoding="utf-8").write(txt)


@cell()
def T5_matched():
    """T5. TEST A: is there coupling in early / late / errors? correct matched to the same n and window (needs A1, A2b, T1)"""
    global A_, B_, E, ERRS, F5, L, MINN5, NDRAW5, NSH5, OUT5, P5, Q, R5, S, T, W5, _, a, ax, axs, b, coef, col, cross, d_pli
    global d_ppc, done5, e, ek, fig, fpt, fs, held5, idx, il_xy, k, key, lab, labels, labels_x, m_, n, new, ok, out, p, p1, p2
    global per, pl_xy, pli, pli_chance, pname, pos, ppc, rat, rng5, rows5, s, side, sp, stats5, t, t_, txt, v, vals, xI, xP
    global xpos, yl
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ── T5. TEST A: is there coupling in early / late / errors? correct matched to the same n and window (needs A1, A2b, T1) ──
    W5, NDRAW5, NSH5, MINN5 = 0.5, 100, 200, 5
    OUT5 = os.path.join(AMIR, "Theta", "Phase connectivity", "Tests", f"T5_matched_vs_chance_{REF}"); os.makedirs(OUT5, exist_ok=True)
    ERRS = {"early": ("early",), "late": ("late",), "errors": ("early", "late")}
    rng5 = np.random.default_rng(5)

    def held5(T, labels):
        """long trials of these labels still held for the whole window (early also ≥ 0.3 s, as before)"""
        hold = (T["release_s"] - T["press_s"]).values
        return np.any(np.stack([cond_trials(T, c).values for c in labels]), 0) & (hold >= W5)

    def stats5(E, Sg):
        """E unit phase diffs, Sg sign(imag) (trials, freqs, time) → PPC, PLI, PLI shuffle mean"""
        n = len(E); s = E.sum(0)
        ppc = ((np.abs(s) ** 2 - n) / (n * (n - 1))).mean(); pli = np.abs(Sg.mean(0)).mean()
        return ppc, pli

    def cross(a, b, fs, idx):
        S = wavelet(a, fs, idx) * np.conj(wavelet(b, fs, idx))
        return S

    def pli_chance(A_, B_, n_sh):
        """mean PLI with B trials shuffled (derangement), A_ B_ complex coefficients"""
        v = []
        for _ in range(n_sh):
            S = A_ * np.conj(B_[derangement(len(B_))]); v.append(np.abs(np.sign(S.imag).mean(0)).mean())
        return np.mean(v)

    F5 = os.path.join(OUT5, f"T5_rows_{REF}.csv")
    R5 = pd.read_csv(F5, dtype={"rat": str}) if os.path.exists(F5) else pd.DataFrame(columns=["rat", "session"])
    rows5, done5, t_ = R5.to_dict("records"), set(zip(R5.rat.astype(str), R5.session)), time.time()
    for rat, (il_xy, pl_xy, pname) in PAIRS3.items():
        for s in sorted(BK.loc[BK.pair == pname, "session"].unique()):
            if (rat, s) in done5: continue
            T = fn.session_trials(K, s).reset_index(drop=True)
            T = T[held5(T, ("correct", "early", "late"))].reset_index(drop=True)
            if len(T) == 0: continue
            try:
                xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, "Press")
            except ValueError as e:
                print(f"{RL[rat]} {s}: {e}, skipped"); continue
            T = T[ok].reset_index(drop=True)
            idx = np.where((t >= -1e-9) & (t <= W5 + 1e-9))[0][::int(round(STEP_S * fs))]
            coef = {}
            for key, labels in {"correct": ("correct",), **ERRS}.items():
                m_ = held5(T, labels)
                if m_.sum() < MINN5: continue
                a, b = xI[m_] - xI[m_].mean(0), xP[m_] - xP[m_].mean(0)                    # ERP removed per condition
                coef[key] = (wavelet(a, fs, idx), wavelet(b, fs, idx))
            del xI, xP
            new = []
            for ek in ERRS:
                if ek not in coef or "correct" not in coef: continue
                n = min(len(coef["correct"][0]), len(coef[ek][0])); out = {}
                for key in ("correct", ek):
                    A_, B_ = coef[key]; vals = []
                    for _ in range(NDRAW5 if len(A_) > n else 1):
                        k = rng5.choice(len(A_), n, replace=False) if len(A_) > n else np.arange(n)
                        S = A_[k] * np.conj(B_[k]); E = S / np.abs(S).clip(1e-30)
                        ppc, pli = stats5(E, np.sign(S.imag))
                        vals.append((ppc, pli, pli_chance(A_[k], B_[k], NSH5 // (10 if len(A_) > n else 1))))
                    out[key] = np.mean(vals, 0)
                new.append(dict(rat=rat, session=s, set=ek, n=n,
                                corr_PPC=out["correct"][0], corr_PLI=out["correct"][1], corr_PLIch=out["correct"][2],
                                err_PPC=out[ek][0], err_PLI=out[ek][1], err_PLIch=out[ek][2]))
            rows5 += new; done5.add((rat, s)); pd.DataFrame(rows5).to_csv(F5, index=False)
            print(f"{RL[rat]} {s}: " + (" | ".join(f"{r['set']} n={r['n']} PPC corr {r['corr_PPC']:.4f} / err {r['err_PPC']:.4f}" for r in new) or "no set")
                  + f" | {(time.time() - t_) / 60:.1f} min", flush=True)
    P5 = pd.DataFrame(rows5); P5["rat"] = P5["rat"].astype(str)

    # ── tests + figure ──
    fpt = lambda p: "<.001" if p < 0.001 else f"{p:.3f}"
    L = [f"T5 test A: coupling vs chance with EQUAL trials (0→{W5:g} s after press, held trials, ERP removed, {REF})",
         "PPC vs 0 and PLI vs shuffle chance; one-sided; pooled = rats equal weight; correct subsampled to the error n"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), facecolor="white", constrained_layout=True)
    xpos = 0
    for ek in ERRS:
        Q = P5[P5.set == ek]
        if Q.empty: continue
        L.append(f"\n{ek.upper()} ({len(Q)} sessions, median n {Q.n.median():.0f}):")
        for side, lab, col in (("corr", "correct (matched)", CCOLS["correct"]), ("err", ek, CCOLS.get(ek, "#d62728"))):
            d_ppc = Q[f"{side}_PPC"].values; d_pli = (Q[f"{side}_PLI"] - Q[f"{side}_PLIch"]).values
            p1, p2 = strat_test(d_ppc, Q.rat.values), strat_test(d_pli, Q.rat.values)
            per = " | ".join(f"{RL[r]} PPC p {fpt(onesided(g[f'{side}_PPC'].values))}, PLI p {fpt(onesided((g[f'{side}_PLI'] - g[f'{side}_PLIch']).values))} (n={len(g)})"
                             for r, g in Q.groupby("rat") if len(g) > 1)
            L.append(f"  {lab}: PPC {Q.groupby('rat')[f'{side}_PPC'].mean().mean():.4f} p {fpt(p1)} | PLI−chance "
                     f"{pd.Series(d_pli).groupby(Q.rat.values).mean().mean():+.4f} p {fpt(p2)} || {per}")
            for ax, v, p in ((axs[0], Q.groupby('rat')[f'{side}_PPC'].mean().mean(), p1),
                             (axs[1], pd.Series(d_pli).groupby(Q.rat.values).mean().mean(), p2)):
                ax.bar(xpos, v, width=0.8, color=col); ax.text(xpos, max(v, 0) * 1.05 + 1e-4, fpt(p) if p < 0.05 else "n.s.", ha="center",
                                                             va="bottom", fontsize=9, fontweight="bold" if p < 0.05 else "normal")
            xpos += 1
        xpos += 0.8
    for ax, yl in zip(axs, ("PPC (0 = no coupling)", "PLI − chance")):
        ax.axhline(0, color="k", lw=0.8); ax.set_ylabel(f"Theta {yl}"); ax.set_xticks([])
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    labels_x = [f"{a}\n{b}" for ek in ERRS if not P5[P5.set == ek].empty for a, b in (("correct", f"(n as {ek})"), (ek, ""))]
    for ax in axs:
        pos = [i + 0.8 * (i // 2) for i in range(len(labels_x))]; ax.set_xticks(pos); ax.set_xticklabels(labels_x, fontsize=9)
    fig.suptitle(f"Coupling above chance with equal trial numbers (0 → {W5:g} s after press, {REF})", fontsize=12, fontweight="bold")
    _savefig(fig, os.path.join(OUT5, "F1_matched_vs_chance.png")); plt.close(fig)
    display(Image(os.path.join(OUT5, "F1_matched_vs_chance.png")))
    txt = "\n".join(L); print(txt); open(os.path.join(OUT5, "T5_summary.txt"), "w", encoding="utf-8").write(txt)


@cell()
def T5_slide():
    """T5 result for the slide: theta coupling during the hold (0–0.5 s, held trials, ERP removed), equal trial numbers"""
    global BARS, COLS3, OUT5, P5, Q, R, _, ax, axs, ch, fig, fp, i, lab, m, n, np, ns, os, pd, pl, plc, pli, plt, plv_of, ppc
    global pv, pvc, r, r_, rat, res, rm, sd, sig, sp, stars, yl
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ T5 result for the slide: theta coupling during the hold (0–0.5 s, held trials, ERP removed), equal trial numbers ══
    #    bars = PLV and PLI per condition (all 3 rats, rats weighted equally); dashed line = chance for the same number of trials; stars = T5 tests
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    OUT5 = os.path.join(AMIR, "Theta", "Phase connectivity", "Tests", f"T5_matched_vs_chance_{REF}")
    P5 = pd.read_csv(os.path.join(OUT5, f"T5_rows_{REF}.csv"), dtype={"rat": str})
    COLS3 = {"correct": "#2CA02C", "early": "#E6B800", "late": "#DD8452"}
    rm = lambda v, r: pd.Series(np.asarray(v, float)).groupby(np.asarray(r)).mean().mean()
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    plv_of = lambda ppc, n: np.sqrt(np.clip((ppc * (n - 1) + 1) / n, 0, None))            # PLV from the saved PPC and trial number

    BARS = [("correct", P5[P5.set == "early"], "corr"), ("early", P5[P5.set == "early"], "err"), ("late", P5[P5.set == "late"], "err")]
    res = []
    for lab, Q, sd in BARS:
        r_, n = Q.rat.values, Q.n.values.astype(float); ppc, pli, ch = Q[f"{sd}_PPC"].values, Q[f"{sd}_PLI"].values, Q[f"{sd}_PLIch"].values
        res.append(dict(cond=lab, sessions=len(Q), n_med=np.median(n), plv=rm(plv_of(ppc, n), r_), plv_ch=rm(np.sqrt(1 / n), r_), p_plv=strat_test(ppc, r_),
                        pli=rm(pli, r_), pli_ch=rm(ch, r_), p_pli=strat_test(pli - ch, r_),
                        per_rat={r: (plv_of(g[f"{sd}_PPC"].values, g.n.values.astype(float)).mean(), np.sqrt(1 / g.n.values.astype(float)).mean(),
                                     g[f"{sd}_PLI"].mean(), g[f"{sd}_PLIch"].mean(), len(g)) for r, g in Q.groupby("rat")}))
    R = pd.DataFrame(res)

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.8), facecolor="white", constrained_layout=True)
    R["p_both"] = R[["p_plv", "p_pli"]].max(axis=1)                       # significant only if both PLV and PLI are
    for ax, (m, ch, yl) in zip(axs, (("plv", "plv_ch", "Theta PLV"), ("pli", "pli_ch", "Theta PLI"))):
        for i, r in R.iterrows():
            ax.bar(i, r[m], width=0.65, color=COLS3[r.cond], edgecolor="k", lw=0.6)
            ax.plot([i - 0.38, i + 0.38], [r[ch]] * 2, color="k", ls="--", lw=1.3)
            sig = r.p_both < 0.05
            ax.text(i, max(r[m], r[ch]) * 1.04, stars(r.p_both), ha="center", va="bottom", fontsize=11 if sig else 9, fontweight="bold" if sig else "normal")
        ax.set_xticks(range(len(R))); ax.set_xticklabels([c.capitalize() for c in R.cond]); ax.set_ylabel(yl)
        ax.set_ylim(0, max(R[m].max(), R[ch].max()) * 1.25)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[1].plot([], [], color="k", ls="--", lw=1.3, label="chance")
    axs[1].plot([], [], ls="", marker="$*$", color="k", ms=9, label="above chance in PLV and PLI")
    axs[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("IL–PL theta coupling during the hold (equal trial numbers, 3 rats)", fontsize=11, fontweight="bold")
    fp = os.path.join(OUT5, "F2_slide_PLV_PLI_vs_chance.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))

    for _, r in R.iterrows():
        print(f"{r.cond:7s} ({r.sessions} sessions, median n {r.n_med:.0f}): PLV {r.plv:.3f} vs chance {r.plv_ch:.3f}, p {r.p_plv:.3f} {stars(r.p_plv)} | "
              f"PLI {r.pli:.3f} vs chance {r.pli_ch:.3f}, p {r.p_pli:.3f} {stars(r.p_pli)}")
        for rat, (pv, pvc, pl, plc, ns) in r.per_rat.items():
            print(f"      {RL[rat]} ({ns} sessions): PLV {pv:.3f} vs {pvc:.3f} {'↑' if pv > pvc else '–'} | PLI {pl:.3f} vs {plc:.3f} {'↑' if pl > plc else '–'}")


@cell()
def bias_corrected_slide():
    """Slide figure: bias-corrected theta PLV (√PPC) and PLI (√ debiased PLI²) during the hold, correct / early / late, equal trial numbers"""
    global BARS, COLS3, MINT, NDR, P, PHD, Q, R, S, Sg, W, Z, _, ax, axs, c, draws, f, fig, fp, held, i, ic, idx, ix, k, lab
    global m, mcol, n, np, os, pd, pk, pli2, plt, ppc, r, r_, rat, res, rng_s, rows, s, sd, sp, sroot, stars, t, unb, vals, yl
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ Slide figure: bias-corrected theta PLV (√PPC) and PLI (√ debiased PLI²) during the hold, correct / early / late, equal trial numbers ══
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    PHD = os.path.join(TESTS, f"Trial_phases_press_{REF}_ERPkept")
    W, MINT, NDR = (0.0, 0.5), 5, 100
    COLS3 = {"correct": "#2CA02C", "early": "#E6B800", "late": "#DD8452"}
    rng_s = np.random.default_rng(7)
    sroot = lambda v: np.sign(v) * np.sqrt(np.abs(v))
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    def unb(E, Sg):                                                  # unbiased PPC and debiased PLI² (trials × freqs × time)
        n = len(E); return (((np.abs(E.sum(0)) ** 2 - n) / (n * (n - 1))).mean(), ((Sg.sum(0) ** 2 - n) / (n * (n - 1))).mean())

    rows = []
    for rat in PAIRS3:
        for s in SESS[rat]:
            f = os.path.join(PHD, f"{s}.npz")
            if not os.path.exists(f): continue
            Z = np.load(f); t = Z["t"]; k = (t >= W[0] - 1e-9) & (t <= W[1] + 1e-9); held = Z["hold"] >= W[1]
            S = Z["uA"][..., k] * np.conj(Z["uB"][..., k]); Sg = np.sign(S.imag)
            ic = np.where((Z["label"] == "correct") & held)[0]
            for c in ("early", "late"):
                ix = np.where((Z["label"] == c) & held)[0]
                if len(ix) < MINT or len(ic) < MINT: continue
                n = min(len(ix), len(ic)); vals = {}
                for lab, idx in (("correct", ic), (c, ix)):
                    draws = [rng_s.choice(idx, n, replace=False) for _ in range(NDR)] if len(idx) > n else [idx]
                    vals[lab] = np.mean([unb(S[d], Sg[d]) for d in draws], 0)
                rows.append(dict(rat=rat, session=s, set=c, n=n, corr_ppc=vals["correct"][0], corr_pli2=vals["correct"][1],
                                 err_ppc=vals[c][0], err_pli2=vals[c][1]))
    P = pd.DataFrame(rows)

    BARS = [("correct", P[P.set == "early"], "corr"), ("early", P[P.set == "early"], "err"), ("late", P[P.set == "late"], "err")]
    res = []
    for lab, Q, sd in BARS:
        r_ = Q.rat.values; ppc, pli2 = Q[f"{sd}_ppc"].values, Q[f"{sd}_pli2"].values
        m = lambda v: pd.Series(v).groupby(r_).mean().mean()
        res.append(dict(cond=lab, sessions=len(Q), n_med=Q.n.median(), plv=sroot(m(ppc)), p_plv=strat_test(ppc, r_), pli=sroot(m(pli2)), p_pli=strat_test(pli2, r_)))
    R = pd.DataFrame(res)

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.8), facecolor="white", constrained_layout=True)
    for ax, (mcol, pk, yl) in zip(axs, (("plv", "p_plv", "θ PLV (√PPC)"), ("pli", "p_pli", "θ PLI (√debiased PLI²)"))):
        for i, r in R.iterrows():
            ax.bar(i, r[mcol], width=0.65, color=COLS3[r.cond], edgecolor="k", lw=0.6)
            ax.text(i, max(r[mcol], 0) + 0.004, stars(r[pk]), ha="center", va="bottom", fontsize=11 if r[pk] < 0.05 else 9, fontweight="bold" if r[pk] < 0.05 else "normal")
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(R))); ax.set_xticklabels([c.capitalize() for c in R.cond]); ax.set_ylabel(yl)
        ax.set_ylim(min(0, R[mcol].min() * 1.3), R[mcol].max() * 1.3)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.suptitle("IL–PL theta coupling during the hold (0–0.5 s, equal trial numbers)", fontsize=11, fontweight="bold")
    fp = os.path.join(PHD, "F_slide_biascorrected_PLV_PLI_conditions.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))
    for _, r in R.iterrows():
        print(f"{r.cond:7s} ({r.sessions} sessions, median n {r.n_med:.0f}): PLV {r.plv:+.3f} (> 0 p {r.p_plv:.3f}) | PLI {r.pli:+.3f} (> 0 p {r.p_pli:.3f})")


@cell()
def coupling_over_trial():
    """Theta coupling over the trial, correct / early / late (same trials as T5 up to 0.5 s): around the press (−0.5 → 1.0 s) and the release (−0.5 → +0.5 s)"""
    global BW, COL, CONDS, F, HOLD_T5, MINT, MIN_SESS, OUTC, PHD, Q, RR, S, T, TONE, Z, _, al, ax, axs, b0, b1, c, done, e, f
    global fig, fp, fs, g, g0, hold, idx, il_xy, k, lab, m, m_, np, ok, os, pd, pl_xy, pli2, plt, rat, rel, rows, s, sg, sig
    global sp, sroot, t, t_, time, tt, x, xI, xP, xs, y, ys
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ Theta coupling over the trial, correct / early / late (same trials as T5 up to 0.5 s): around the press (−0.5 → 1.0 s) and the release (−0.5 → +0.5 s) ══
    #    measure = bias-corrected PLI (√ debiased PLI²; chance = 0 for every condition, corrects for trial number); filled marker = above chance (p < .05)
    import os, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
    PHD = os.path.join(TESTS, f"Trial_phases_press_{REF}_ERPkept"); OUTC = os.path.join(TESTS, f"Coupling_over_trial_{REF}_ERPkept"); os.makedirs(OUTC, exist_ok=True)
    BW, MINT, MIN_SESS, TONE, HOLD_T5 = 0.1, 5, 3, 1.05, 0.5
    CONDS = ("correct", "early", "late"); COL = {"correct": "#2CA02C", "early": "#E6B800", "late": "#DD8452"}
    sroot = lambda v: np.sign(v) * np.sqrt(np.abs(v))
    def pli2(S):                                                    # debiased PLI² over trials (trials × freqs × time)
        n = len(S); sg = np.sign(S.imag); return ((sg.sum(0) ** 2 - n) / (n * (n - 1))).mean()

    rows = []
    # ── press panel: saved phases; trials held ≥ 0.5 s up to 0.5 s (as T5), then held to the end of each step ──
    for rat in PAIRS3:
        for s in SESS[rat]:
            f = os.path.join(PHD, f"{s}.npz")
            if not os.path.exists(f): continue
            Z = np.load(f); t = Z["t"]; S = Z["uA"] * np.conj(Z["uB"]); hold = Z["hold"]
            for b0 in np.arange(-0.5, 1.0 - BW + 1e-9, BW):
                b1 = b0 + BW; k = (t >= b0 - 1e-9) & (t <= b1 + 1e-9)
                for c in CONDS:
                    m = (Z["label"] == c) & (hold >= max(b1, HOLD_T5))
                    if m.sum() >= MINT: rows.append(dict(rat=rat, session=s, align="press", c=c, t0=round(b0, 2), n=int(m.sum()), v=pli2(S[m][..., k])))
    # ── release panel: compute once ──
    F = os.path.join(OUTC, "release_rows.csv")
    RR = pd.read_csv(F, dtype={"rat": str}) if os.path.exists(F) else pd.DataFrame(columns=["rat", "session"])
    rel, done, t_ = RR.to_dict("records"), set(zip(RR.rat.astype(str), RR.session)), time.time()
    for rat, (il_xy, pl_xy, _) in PAIRS3.items():
        for s in SESS[rat]:
            if (rat, s) in done: continue
            try:
                T = fn.session_trials(K, s).reset_index(drop=True)
                lab = np.select([cond_trials(T, c).values for c in CONDS], list(CONDS), ""); T = T[lab != ""].reset_index(drop=True)
                xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, "Release"); T = T[ok].reset_index(drop=True)
                lab = np.select([cond_trials(T, c).values for c in CONDS], list(CONDS), "")
                idx = np.where((t >= -0.5 - 1e-9) & (t <= 0.5 + 1e-9))[0][::int(round(STEP_S * fs))]; tt = t[idx]
                S = wl(xI, fs, idx, FR_TH) * np.conj(wl(xP, fs, idx, FR_TH)); del xI, xP
                for b0 in np.arange(-0.5, 0.5 - BW + 1e-9, BW):
                    k = (tt >= b0 - 1e-9) & (tt <= b0 + BW + 1e-9)
                    for c in CONDS:
                        m = lab == c
                        if m.sum() >= MINT: rel.append(dict(rat=rat, session=s, align="release", c=c, t0=round(b0, 2), n=int(m.sum()), v=pli2(S[m][..., k])))
                done.add((rat, s)); pd.DataFrame(rel).to_csv(F, index=False); log(f"Coupling-over-trial release {RL[rat]} {s} | {(time.time() - t_) / 60:.1f} min")
            except Exception as e:
                log(f"Coupling-over-trial {RL[rat]} {s} error: {e}")
    Q = pd.concat([pd.DataFrame(rows), pd.DataFrame(rel)], ignore_index=True); Q["rat"] = Q.rat.astype(str)

    # ── figure ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor="white", gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.22}, sharey=True)
    for ax, al in zip(axs, ("press", "release")):
        for c in CONDS:
            g0 = Q[(Q["align"] == al) & (Q["c"] == c)]; xs, ys, sig = [], [], []
            for b0, g in g0.groupby("t0"):
                if g.session.nunique() < MIN_SESS: continue
                m_ = pd.Series(g.v.values).groupby(g.rat.values).mean().mean()
                xs.append(b0 + BW / 2); ys.append(sroot(m_)); sig.append(strat_test(g.v.values, g.rat.values) < 0.05)
            ax.plot(xs, ys, color=COL[c], lw=2.2, label=c, zorder=2)
            for x, y, sg in zip(xs, ys, sig): ax.plot(x, y, "o", ms=6, mec=COL[c], mfc=COL[c] if sg else "white", mew=1.5, zorder=3)
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="0.3", ls="--", lw=1.4)
        if al == "press":
            ax.axvline(TONE, color="0.3", ls=":", lw=1.4); ax.text(TONE, 1.0, "tone", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
            ax.axvspan(-0.5, HOLD_T5, color="0.94", zorder=0)
            ax.set_xlabel("time from press [s]"); ax.set_ylabel("Theta PLI (bias-corrected, √)"); ax.set_xlim(-0.5, 1.1)
        else:
            ax.set_xlabel("time from release [s]"); ax.set_xlim(-0.5, 0.5)
        ax.set_title(al, fontsize=10)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("IL–PL theta coupling over the trial (filled = above chance, p < .05; grey = same trials as T5, held ≥ 0.5 s)", fontsize=10)
    fp = os.path.join(OUTC, "F_coupling_over_trial_conditions_T5trials.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))
    print("sessions per condition and panel:"); print(Q.groupby(["align", "c"]).session.nunique().unstack().to_string())


@cell()
def T13_fix():
    """T13 fix: theta coupling before the tone (−0.5 → 0 s), correct vs late, ERP kept"""
    global F, L, NDR, NM, NSH, OUT, Q, R, T, T0, _, c, cc, d, done, e, fs, g, hold, idx, il_xy, ll, m, m0, m_, matched, mc, ml
    global n, np, ok, os, pd, pl_xy, r, rat, rel, row, rows, s, stats_f, t, t_, time, tone0, v, xI, xP
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    if not RUN:
        print("RUN = False in the settings: nothing is computed here (the cell only computes missing sessions when RUN = True)")
        return
    # ══ T13 fix: theta coupling before the tone (−0.5 → 0 s), correct vs late, ERP kept ══
    import os, time, numpy as np, pandas as pd
    OUT = os.path.join(TESTS, f"Overnight_T13-15_{REF}_ERPkept"); NSH, NDR = 100, 30

    # ── 0. check the tone column (why T13 had no rows) ──
    T0 = fn.session_trials(K, SESS["136"][0]); print("tone_s in columns:", "tone_s" in T0.columns)
    if "tone_s" in T0.columns:
        m0 = (cond_trials(T0, "correct") | cond_trials(T0, "late")).values; rel = (T0["tone_s"] - T0["press_s"])[m0]
        print(f"tone − press (correct+late, one session): NaN {rel.isna().mean():.0%} | median {np.nanmedian(rel):.3f} s | "
              f"range {np.nanmin(rel):.3f}–{np.nanmax(rel):.3f} s | plain median {np.median(rel):.3f}")

    def stats_f(A_, B_, n_sh):                                    # same as in the overnight cell
        S = A_ * np.conj(B_); E = S / np.abs(S).clip(1e-30); n = len(E); sg = np.sign(S.imag)
        ppc = ((np.abs(E.sum(0)) ** 2 - n) / (n * (n - 1))).mean(-1); pli = np.abs(sg.mean(0)).mean(-1); spl = sg.mean(0).mean(-1); ch = np.zeros_like(pli)
        for _ in range(n_sh):
            S2 = A_ * np.conj(B_[derangement(n)]); ch += np.abs(np.sign(S2.imag).mean(0)).mean(-1)
        return np.stack([ppc, pli, ch / n_sh, spl])
    def matched(A_, B_, n):
        if len(A_) == n: return stats_f(A_, B_, NSH)
        return np.mean([stats_f(A_[k], B_[k], max(5, NSH // 10)) for k in (rng.choice(len(A_), n, replace=False) for _ in range(NDR))], 0)
    NM = ("PPC", "PLI", "PLIch", "sPLI")

    # ── 1. compute ──
    F = os.path.join(OUT, "T13_rows_fix.csv")
    R = pd.read_csv(F, dtype={"rat": str}) if os.path.exists(F) else pd.DataFrame(columns=["rat", "session"])
    rows, done, t_ = R.to_dict("records"), set(zip(R.rat.astype(str), R.session)), time.time()
    for rat, (il_xy, pl_xy, _) in PAIRS3.items():
        for s in SESS[rat]:
            if (rat, s) in done: continue
            try:
                T = fn.session_trials(K, s).reset_index(drop=True)
                T = T[(cond_trials(T, "correct") | cond_trials(T, "late")).values].reset_index(drop=True)
                if cond_trials(T, "late").sum() < MINN: continue
                tone0 = np.nanmedian((T["tone_s"] - T["press_s"]).values) if "tone_s" in T.columns else np.nan
                if not np.isfinite(tone0) or not (0.5 < tone0 < 1.5): tone0 = 1.05
                xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, "Press")
                T = T[ok].reset_index(drop=True); hold = (T["release_s"] - T["press_s"]).values
                idx = np.where((t >= tone0 - 0.5 - 1e-9) & (t <= tone0 + 1e-9))[0][::int(round(STEP_S * fs))]
                mc, ml = cond_trials(T, "correct").values & (hold >= tone0), cond_trials(T, "late").values
                if mc.sum() < MINN or ml.sum() < MINN: del xI, xP; continue
                n = int(min(mc.sum(), ml.sum())); row = dict(rat=rat, session=s, n=n, tone0=round(float(tone0), 3))
                for c, m_ in (("correct", mc), ("late", ml)):
                    v = matched(wl(xI[m_], fs, idx, FR_TH), wl(xP[m_], fs, idx, FR_TH), n).mean(-1)
                    row.update({f"{c}_{nm}": v[i] for i, nm in enumerate(NM)})
                del xI, xP
                rows.append(row); done.add((rat, s)); pd.DataFrame(rows).to_csv(F, index=False)
                log(f"T13fix {RL[rat]} {s}: n {n}, tone at {tone0:.2f} s | {(time.time() - t_) / 60:.1f} min")
            except Exception as e:
                log(f"T13fix {RL[rat]} {s} error: {e}")

    # ── 2. summary ──
    Q = pd.DataFrame(rows); Q["rat"] = Q.rat.astype(str)
    L = [f"T13 BEFORE THE TONE (−0.5 → 0 s), correct vs late | {REF} | ERP kept | theta {FR_TH.min():g}–{FR_TH.max():g} Hz | "
         f"{len(Q)} sessions, median n {Q.n.median():.0f}, tone at {Q.tone0.median():.2f} s"]
    for m in ("PLI", "PPC", "sPLI"):
        cc = Q[f"correct_{m}"] - (Q["correct_PLIch"] if m == "PLI" else 0); ll = Q[f"late_{m}"] - (Q["late_PLIch"] if m == "PLI" else 0); d = (cc - ll).values
        L.append(f"  {m}{' − chance' if m == 'PLI' else ''}: correct {rmean(cc, Q.rat):+.4f} (vs 0 p {fpt(strat_test(cc.values, Q.rat.values))}) | "
                 f"late {rmean(ll, Q.rat):+.4f} (vs 0 p {fpt(strat_test(ll.values, Q.rat.values))}) | correct − late {rmean(d, Q.rat):+.4f}, "
                 f"one-sided p {fpt(strat_test(d, Q.rat.values))}, two-sided p {fpt(two(d, Q.rat.values))}, correct higher {(d > 0).sum()}/{len(d)}")
    for r in PAIRS3:
        g = Q[Q.rat == r]
        if len(g): d = (g.correct_PLI - g.correct_PLIch) - (g.late_PLI - g.late_PLIch); L.append(f"  {RL[r]}: PLI − chance correct − late {d.mean():+.4f} ({(d > 0).sum()}/{len(d)} sessions)")
    open(os.path.join(OUT, "T13_summary_fix.txt"), "w", encoding="utf-8").write("\n".join(L)); log("\n".join(L))


@cell()
def T14_vs_chance():
    """Each condition vs its own chance in each 0.2 s window after the press (from T14, equal trial numbers, ERP kept)"""
    global COL, G, OUTO, SRC, T14, TONE, ax, b, bins, c, ch, fig, fp, g, np, os, p, pair, pd, plt, ps, r_, rm, side, sp, stars
    global v, x, xs, y
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ Each condition vs its own chance in each 0.2 s window after the press (from T14, equal trial numbers, ERP kept) ══
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    OUTO = os.path.join(TESTS, f"Overnight_T13-15_{REF}_ERPkept"); T14 = pd.read_csv(os.path.join(OUTO, "T14_rows.csv"), dtype={"rat": str})
    COL = {"correct": "#2CA02C", "early": "#E6B800", "late": "#DD8452"}; TONE = 1.05
    SRC = {"correct": ("correct-early", "A"), "early": ("correct-early", "B"), "late": ("correct-late", "B")}     # where each condition is taken from
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    rm = lambda v, r: pd.Series(np.asarray(v, float)).groupby(np.asarray(r)).mean().mean()

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6, 3.9), facecolor="white")
    for c, (pair, side) in SRC.items():
        G = T14[T14["pair"] == pair]; bins = sorted(G["bin"].unique(), key=lambda b: float(b.split("-")[0])); xs, v, ch, ps = [], [], [], []
        for b in bins:
            g = G[G["bin"] == b]
            if g["session"].nunique() < 3: continue
            r_ = g["rat"].values; xs.append(np.mean([float(x) for x in b.split("-")])); v.append(rm(g[f"{side}_PLI"], r_)); ch.append(rm(g[f"{side}_PLIch"], r_))
            ps.append(strat_test((g[f"{side}_PLI"] - g[f"{side}_PLIch"]).values, r_))
            print(f"{c:7s} {b} s ({g['session'].nunique()} sessions, median n {g['n'].median():.0f}): PLI {v[-1]:.3f} vs chance {ch[-1]:.3f}, p {ps[-1]:.3f} {stars(ps[-1])}")
        ax.plot(xs, v, color=COL[c], lw=2.2, label=c, zorder=2); ax.plot(xs, ch, color=COL[c], ls="--", lw=1, alpha=0.7, zorder=1)
        for x, y, p in zip(xs, v, ps): ax.plot(x, y, "o", ms=7, mec=COL[c], mfc=COL[c] if p < 0.05 else "white", mew=1.6, zorder=3)
    ax.axvline(TONE, color="0.3", ls=":", lw=1.5); ax.text(TONE, 1.0, "tone", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlabel("time after press [s]"); ax.set_ylabel("Theta PLI")
    ax.plot([], [], color="0.4", ls="--", lw=1, label="chance (same colour)"); ax.plot([], [], "o", mec="0.4", mfc="0.4", label="above chance (p < .05)")
    ax.plot([], [], "o", mec="0.4", mfc="white", label="n.s."); ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fp = os.path.join(OUTO, "F_conditions_vs_chance_per_window.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def onset_timeline():
    """When is each condition coupled? Timeline of windows significantly above chance (T14, equal trial numbers, ERP kept)"""
    global COL, G, OUTO, SRC, T14, TONE, ax, b, c, fig, first, fp, g, np, os, p, pair, pd, plt, side, sp, stars, x, x0, yi
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ When is each condition coupled? Timeline of windows significantly above chance (T14, equal trial numbers, ERP kept) ══
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    OUTO = os.path.join(TESTS, f"Overnight_T13-15_{REF}_ERPkept"); T14 = pd.read_csv(os.path.join(OUTO, "T14_rows.csv"), dtype={"rat": str})
    COL = {"correct": "#2CA02C", "early": "#E6B800", "late": "#DD8452"}; TONE = 1.05
    SRC = {"correct": ("correct-early", "A"), "early": ("correct-early", "B"), "late": ("correct-late", "B")}
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*"

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6, 2.6), facecolor="white")
    for yi, (c, (pair, side)) in enumerate(SRC.items()):
        G = T14[T14["pair"] == pair]; first = None
        for b in sorted(G["bin"].unique(), key=lambda b: float(b.split("-")[0])):
            g = G[G["bin"] == b]
            if g["session"].nunique() < 3: continue
            p = strat_test((g[f"{side}_PLI"] - g[f"{side}_PLIch"]).values, g["rat"].values); x = np.mean([float(v) for v in b.split("-")])
            if p < 0.05 and first is None:
                first = float(b.split("-")[0])
                x0 = float(b.split("-")[0])
                ax.plot(x0, -yi, "o", ms=13, color=COL[c], mec="k", mew=0.6)
                break
        ax.text(1.14, -yi, f"coupled from {first:.1f} s" if first is not None else "not coupled", va="center", fontsize=8.5, color=COL[c])
    ax.axvline(0, color="0.3", ls="--", lw=1.2); ax.axvline(TONE, color="0.3", ls=":", lw=1.4)
    ax.text(0, 1.02, "press", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, clip_on=False)
    ax.text(TONE, 1.02, "tone", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, clip_on=False)
    ax.set_yticks([0, -1, -2]); ax.set_yticklabels([c.capitalize() for c in SRC]); ax.set_ylim(-2.6, 0.9)
    ax.set_xlim(-0.05, 1.12); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlabel("Time (s)")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Onset of significant IL–PL theta coupling after the press (first window with PLI above chance, p < .05)", fontsize=9.5, y=1.1)
    fp = os.path.join(OUTO, "F_coupling_onset_timeline.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def onset_consistency():
    """Consistency across sessions of each onset point: sessions above chance (T14) + direction circles (saved trial phases)"""
    global COL, Circle, FC, L_, MINT, ONSET, OUTO, PHD, R, T14, Z, a, above, ang, ax, axs, b, c, f, fig, fp, g, k, m, mdeg, mv
    global n, nah, np, os, p_ray, pair, pd, plt, rat, s, side, t, w0, w1
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ Consistency across sessions of each onset point: sessions above chance (T14) + direction circles (saved trial phases) ══
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    OUTO = os.path.join(TESTS, f"Overnight_T13-15_{REF}_ERPkept"); PHD = os.path.join(TESTS, f"Trial_phases_press_{REF}_ERPkept")
    T14 = pd.read_csv(os.path.join(OUTO, "T14_rows.csv"), dtype={"rat": str})
    COL = {"correct": "#2CA02C", "early": "#E6B800", "late": "#DD8452"}; FC, MINT, L_ = 5.5, 5, 0.85
    ONSET = {"correct": ("correct-early", "A", "0.0-0.2"), "early": ("correct-early", "B", "0.0-0.2"), "late": ("correct-late", "B", "0.8-1.0")}

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.9), facecolor="white")
    for ax, (c, (pair, side, b)) in zip(axs, ONSET.items()):
        g = T14[(T14["pair"] == pair) & (T14["bin"] == b)]; above = int((g[f"{side}_PLI"] > g[f"{side}_PLIch"]).sum())
        w0, w1 = [float(x) for x in b.split("-")]; ang = []
        for rat in PAIRS3:
            for s in SESS[rat]:
                f = os.path.join(PHD, f"{s}.npz")
                if not os.path.exists(f): continue
                Z = np.load(f); t = Z["t"]; k = (t >= w0 - 1e-9) & (t <= w1 + 1e-9); m = (Z["label"] == c) & (Z["hold"] >= w1)
                if m.sum() < MINT: continue
                ang.append(np.angle((Z["uA"][m][..., k] * np.conj(Z["uB"][m][..., k])).sum()))
        ang = np.array(ang); n = len(ang); mv = np.mean(np.exp(1j * ang)); R = np.abs(mv); mdeg = np.degrees(np.angle(mv))
        p_ray = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n)); nah = int((np.sin(ang) > 0).sum())
        ax.add_patch(Circle((0, 0), 1, fill=False, color="k", lw=1.2)); ax.axhline(0, color="0.7", lw=1); ax.axvline(0, color="0.7", lw=1)
        for a in ang: ax.annotate("", xy=(L_ * np.cos(a), L_ * np.sin(a)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=COL[c], lw=1.5))
        ax.annotate("", xy=(R * L_ * np.cos(np.angle(mv)), R * L_ * np.sin(np.angle(mv))), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=3.5))
        ax.text(0, 1.06, "+90°\nIL ahead", ha="center", va="bottom", fontsize=8.5); ax.text(0, -1.06, "−90°\nPL ahead", ha="center", va="top", fontsize=8.5)
        ax.text(1.06, 0, "0°", ha="left", va="center", fontsize=8.5); ax.text(-1.06, 0, "180°", ha="right", va="center", fontsize=8.5)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{c.capitalize()}, {w0:g}–{w1:g} s after press\nabove chance {above}/{len(g)} sessions | IL ahead {nah}/{n}\n"
                     f"mean {mdeg:+.0f}° ≈ {mdeg / 360 * 1000 / FC:+.0f} ms, R = {R:.2f}, Rayleigh p {'< 0.001' if p_ray < 0.001 else f'= {p_ray:.3f}'}", fontsize=9.5)
        print(f"{c:7s} {b} s: PLI above chance in {above}/{len(g)} sessions (T14) | IL ahead {nah}/{n} | mean {mdeg:+.0f}° | R {R:.2f} | Rayleigh p {p_ray:.4f}")
    fig.tight_layout(); fp = os.path.join(OUTO, "F_onset_consistency_circles.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def late_across_hold():
    """Late trials: theta coupling across the hold (T14 windows, vs chance) + direction circle for 0.6 s → tone"""
    global CORR, CWL, Circle, FC, G, LATE, L_, MINT, OUTO, PHD, R, T14, TONE, Z, a, ang, ax, b, bins, c, cch, corr, f, fig
    global fig2, fp1, fp2, g, k, late, lch, m, mdeg, mv, n, nah, np, ns, os, p, p_ray, pd, plt, ps, r_, rat, rm, s, sp, stars
    global t, v, x, xs
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ Late trials: theta coupling across the hold (T14 windows, vs chance) + direction circle for 0.6 s → tone ══
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    OUTO = os.path.join(TESTS, f"Overnight_T13-15_{REF}_ERPkept"); PHD = os.path.join(TESTS, f"Trial_phases_press_{REF}_ERPkept")
    TONE, CWL, FC, MINT = 1.05, (0.6, 1.05), 5.5, 5                 # tone time, circle window, theta centre (Hz) for ms, min late trials
    LATE, CORR = "#DD8452", "#2CA02C"
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    rm = lambda v, r: pd.Series(np.asarray(v, float)).groupby(np.asarray(r)).mean().mean()

    # ── 1. late across the hold (T14: pair correct-late, B = late, equal n) ──
    T14 = pd.read_csv(os.path.join(OUTO, "T14_rows.csv"), dtype={"rat": str}); G = T14[T14.pair == "correct-late"]
    bins = sorted(G.bin.unique(), key=lambda b: float(b.split("-")[0])); xs = [np.mean([float(v) for v in b.split("-")]) for b in bins]
    late, lch, corr, cch, ps, ns = [], [], [], [], [], []
    for b in bins:
        g = G[G.bin == b]; r_ = g.rat.values
        late.append(rm(g.B_PLI, r_)); lch.append(rm(g.B_PLIch, r_)); corr.append(rm(g.A_PLI, r_)); cch.append(rm(g.A_PLIch, r_))
        ps.append(strat_test((g.B_PLI - g.B_PLIch).values, r_)); ns.append(len(g))
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor="white")

    ax.plot(xs, late, color=LATE, lw=2.5, marker="o", ms=6, label="late")
    ax.plot(xs, lch, color="k", ls="--", lw=1.2, label="chance (late)")
    for x, v, c, p in zip(xs, late, lch, ps):
        ax.text(x, max(v, c) * 1.03, stars(p), ha="center", va="bottom", fontsize=11 if p < 0.05 else 8.5, fontweight="bold" if p < 0.05 else "normal")
    ax.axvline(TONE, color="0.3", ls=":", lw=1.5); ax.text(TONE, ax.get_ylim()[1], "tone", ha="center", va="bottom", fontsize=9)
    ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlabel("time after press [s]"); ax.set_ylabel("Theta PLI")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fp1 = os.path.join(OUTO, "F_late_PLI_across_hold.png"); _savefig(fig, fp1); plt.close(fig); display(Image(fp1))
    for b, v, c, p, n in zip(bins, late, lch, ps, ns): print(f"late {b} s ({n} sessions): PLI {v:.3f} vs chance {c:.3f}, p {p:.3f} {stars(p)}")

    # ── 2. circle: late trials, 0.6 s → tone, one arrow per session (saved trial phases, ERP kept) ──
    ang = []
    for rat in PAIRS3:
        for s in SESS[rat]:
            f = os.path.join(PHD, f"{s}.npz")
            if not os.path.exists(f): continue
            Z = np.load(f); t = Z["t"]; k = (t >= CWL[0] - 1e-9) & (t <= CWL[1] + 1e-9); m = Z["label"] == "late"
            if m.sum() < MINT: continue
            ang.append(np.angle((Z["uA"][m][..., k] * np.conj(Z["uB"][m][..., k])).sum()))
    ang = np.array(ang); n = len(ang); mv = np.mean(np.exp(1j * ang)); R = np.abs(mv); mdeg = np.degrees(np.angle(mv))
    p_ray = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n)); nah = int((np.sin(ang) > 0).sum()); L_ = 0.85
    fig2, ax = plt.subplots(figsize=(4.4, 4.8), facecolor="white")
    ax.add_patch(Circle((0, 0), 1, fill=False, color="k", lw=1.2)); ax.axhline(0, color="0.7", lw=1); ax.axvline(0, color="0.7", lw=1)
    for a in ang: ax.annotate("", xy=(L_ * np.cos(a), L_ * np.sin(a)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=LATE, lw=1.6))
    ax.annotate("", xy=(R * L_ * np.cos(np.angle(mv)), R * L_ * np.sin(np.angle(mv))), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=3.5))
    ax.text(0, 1.06, "+90°\nIL ahead", ha="center", va="bottom", fontsize=9); ax.text(0, -1.06, "−90°\nPL ahead", ha="center", va="top", fontsize=9)
    ax.text(1.06, 0, "0°", ha="left", va="center", fontsize=9); ax.text(-1.06, 0, "180°", ha="right", va="center", fontsize=9)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"θ, late trials, {CWL[0]:g} s → tone — IL ahead in {nah}/{n} sessions\nmean {mdeg:+.0f}° ≈ {mdeg / 360 * 1000 / FC:+.0f} ms, R = {R:.2f}, "
                 f"Rayleigh p {'< 0.001' if p_ray < 0.001 else f'= {p_ray:.3f}'}", fontsize=10, fontweight="bold")
    ax.legend([plt.Line2D([], [], color=LATE, lw=1.6), plt.Line2D([], [], color="k", lw=3.5)], ["session", "mean"],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=9)
    fp2 = os.path.join(OUTO, "F_late_direction_before_tone.png"); _savefig(fig2, fp2); plt.close(fig2); display(Image(fp2))
    print(f"late, {CWL[0]:g}→{CWL[1]:g} s: IL ahead {nah}/{n} | mean {mdeg:+.0f}° ≈ {mdeg / 360 * 1000 / FC:+.0f} ms (at {FC:g} Hz) | R {R:.2f} | Rayleigh p {p_ray:.4f}")


@cell()
def late_before_tone():
    """Late trials, 0.8–1.0 s after press (just before the tone): consistency across sessions + direction circle"""
    global Circle, FC, LATE, L_, MINT, OUTO, PHD, R, T14, W0, W1, Z, a, above, ang, ax, f, fig, fp, g, k, m, mdeg, mv, n, nah
    global np, os, p_ray, pd, plt, rat, s, t
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ Late trials, 0.8–1.0 s after press (just before the tone): consistency across sessions + direction circle ══
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    OUTO = os.path.join(TESTS, f"Overnight_T13-15_{REF}_ERPkept"); PHD = os.path.join(TESTS, f"Trial_phases_press_{REF}_ERPkept")
    W0, W1, FC, MINT, L_, LATE = 0.8, 1.0, 5.5, 5, 0.85, "#DD8452"

    T14 = pd.read_csv(os.path.join(OUTO, "T14_rows.csv"), dtype={"rat": str})
    g = T14[(T14["pair"] == "correct-late") & (T14["bin"] == f"{W0:.1f}-{W1:.1f}")]; above = int((g["B_PLI"] > g["B_PLIch"]).sum())
    ang = []
    for rat in PAIRS3:
        for s in SESS[rat]:
            f = os.path.join(PHD, f"{s}.npz")
            if not os.path.exists(f): continue
            Z = np.load(f); t = Z["t"]; k = (t >= W0 - 1e-9) & (t <= W1 + 1e-9); m = Z["label"] == "late"
            if m.sum() < MINT: continue
            ang.append(np.angle((Z["uA"][m][..., k] * np.conj(Z["uB"][m][..., k])).sum()))
    ang = np.array(ang); n = len(ang); mv = np.mean(np.exp(1j * ang)); R = np.abs(mv); mdeg = np.degrees(np.angle(mv))
    p_ray = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n)); nah = int((np.sin(ang) > 0).sum())

    fig, ax = plt.subplots(figsize=(4.6, 5.0), facecolor="white")
    ax.add_patch(Circle((0, 0), 1, fill=False, color="k", lw=1.2)); ax.axhline(0, color="0.7", lw=1); ax.axvline(0, color="0.7", lw=1)
    for a in ang: ax.annotate("", xy=(L_ * np.cos(a), L_ * np.sin(a)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=LATE, lw=1.6))
    ax.annotate("", xy=(R * L_ * np.cos(np.angle(mv)), R * L_ * np.sin(np.angle(mv))), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=3.5))
    ax.text(0, 1.06, "+90°\nIL ahead", ha="center", va="bottom", fontsize=9); ax.text(0, -1.06, "−90°\nPL ahead", ha="center", va="top", fontsize=9)
    ax.text(1.06, 0, "0°", ha="left", va="center", fontsize=9); ax.text(-1.06, 0, "180°", ha="right", va="center", fontsize=9)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Late trials, {W0:g}–{W1:g} s after press — IL ahead in {nah}/{n} sessions\n"
                 f"mean {mdeg:+.0f}° ≈ {mdeg / 360 * 1000 / FC:+.0f} ms, R = {R:.2f}, Rayleigh p {'< 0.001' if p_ray < 0.001 else f'= {p_ray:.3f}'}",
                 fontsize=10, fontweight="bold")
    ax.legend([plt.Line2D([], [], color=LATE, lw=1.6), plt.Line2D([], [], color="k", lw=3.5)], ["session", "mean"],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=9)
    fp = os.path.join(OUTO, "F_onset_consistency_circle_late.png"); _savefig(fig, fp); plt.close(fig); display(Image(fp))
    print(f"late {W0:g}–{W1:g} s: PLI above chance in {above}/{len(g)} sessions (T14) | IL ahead {nah}/{n} | mean {mdeg:+.0f}° ≈ {mdeg / 360 * 1000 / FC:+.0f} ms "
          f"(at {FC:g} Hz) | R {R:.2f} | Rayleigh p {p_ray:.4f}\nsaved: {fp}")


@cell()
def T8b_jaeckel():
    """T8b: Jäckel replication done their way (pooled per rat) + within-rat shuffle + session PPC"""
    global BANDS, CONDS, D, E, Er, F, L, MET, NPERM, OUT, P, Pp, Pr, R, S, T, TR, Tk, V, _, a, ax, axs, b, band, bi, bn, c
    global chi2, ci, col, cp, cr, d1, d2, done, e, er, ev, f, fig, fir_phase, fisher, fs, g, i, i0, il_xy, isc, ise, j, k, lab
    global mc, me, mn, nc, np, null, null2, ok, os, p_, pd, pl_xy, plt, ppc, r, rat, real, real2, ri, rng8, rows, rr, s, sp
    global ss, sw, t, t_, tid, time, w, wilcoxon, x, xI, xP
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ T8b: Jäckel replication done their way (pooled per rat) + within-rat shuffle + session PPC ══
    import os, time, numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.signal as ss
    from scipy.stats import wilcoxon, chi2

    OUT = os.path.join(TESTS, f"T8b_Jaeckel_exact_{REF}"); os.makedirs(OUT, exist_ok=True)
    BANDS = {"delta": (2.0, 4.0), "theta": (4.0, 8.0)}
    CONDS = ("correct_press", "correct_release", "early_release")
    NPERM = 10000; rng8 = np.random.default_rng(1)

    def fir_phase(x, fs, band):                       # same filter as T8
        ntap = int(round((366 if band[0] < 4 else 183) / 244.14 * fs)) | 1
        b = ss.firwin(ntap, band, pass_zero=False, fs=fs)
        return ss.hilbert(ss.filtfilt(b, [1.0], x, axis=-1), axis=-1)

    # ── 1. phase difference of every trial at t = 0 (ERP kept) ──
    F = os.path.join(OUT, "T8b_trials.csv")
    TR = pd.read_csv(F, dtype={"rat": str}) if os.path.exists(F) else pd.DataFrame(columns=["rat", "session"])
    done, rows, t_ = set(zip(TR.rat.astype(str), TR.session)), TR.to_dict("records"), time.time()
    for rat, (il_xy, pl_xy, _) in PAIRS3.items():
        for s in SESS[rat]:
            if (rat, s) in done: continue
            try:
                T = fn.session_trials(K, s).reset_index(drop=True)
                mc, me = cond_trials(T, "correct").values, cond_trials(T, "early").values
                if mc.sum() < MINN or me.sum() < MINN: continue                     # same sessions as T8
                T = T[mc | me].reset_index(drop=True)
                for ev in ("Press", "Release"):
                    xI, xP, t, fs, ok = get_ev(s, T, il_xy, pl_xy, ev)
                    tid = np.where(ok)[0]; Tk = T.iloc[tid].reset_index(drop=True); i0 = np.argmin(np.abs(t))
                    isc, ise = cond_trials(Tk, "correct").values, cond_trials(Tk, "early").values
                    for bn, band in BANDS.items():
                        S = fir_phase(xI, fs, band)[:, i0] * np.conj(fir_phase(xP, fs, band)[:, i0]); E = S / np.abs(S).clip(1e-30)
                        for k in range(len(Tk)):
                            if isc[k]:   c = f"correct_{ev.lower()}"
                            elif ise[k] and ev == "Release": c = "early_release"
                            else: continue
                            rows.append(dict(rat=rat, session=s, trial=int(tid[k]), band=bn, cond=c, re=E[k].real, im=E[k].imag))
                    del xI, xP
                done.add((rat, s)); pd.DataFrame(rows).to_csv(F, index=False)
                log(f"T8b {RL[rat]} {s} | {(time.time() - t_) / 60:.1f} min")
            except Exception as e:
                log(f"T8b {RL[rat]} {s} error: {e}")

    # ── 2. tests ──
    TR = pd.DataFrame(rows); TR["rat"] = TR.rat.astype(str); TR["E"] = TR.re.values + 1j * TR.im.values
    MET = {"PLV": lambda E: np.abs(np.mean(E)), "PLI": lambda E: np.abs(np.mean(np.sign(np.imag(E))))}
    ppc = lambda E: (np.abs(E.sum()) ** 2 - len(E)) / (len(E) * (len(E) - 1)) if len(E) > 1 else np.nan
    fisher = lambda ps: chi2.sf(-2 * np.sum(np.log(ps)), 2 * len(ps))
    L = [f"T8b Jäckel replication their way | {REF} | ERP kept | FIR+Hilbert | phase at t = 0 | {TR.session.nunique()} sessions",
         "one-sided everywhere (direction from the paper: correct release > early release, correct release > press)"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white", constrained_layout=True)
    for bi, bn in enumerate(BANDS):
        D = TR[TR.band == bn]; L.append(f"\n══ {bn.upper()} {BANDS[bn][0]:g}–{BANDS[bn][1]:g} Hz ══")
        for mn, f in MET.items():
            L.append(f"\n{mn}, all sessions pooled per rat (no matching, no chance level):"); V = {}
            for rat in PAIRS3:
                R = D[D.rat == rat]; g = {c: R.loc[R.cond == c, "E"].values for c in CONDS}
                V[rat] = {c: f(e) for c, e in g.items()}
                # shuffle A: correct release > early release (labels shuffled among release trials, counts kept)
                Er = np.concatenate([g["correct_release"], g["early_release"]]); nc = len(g["correct_release"])
                real = V[rat]["correct_release"] - V[rat]["early_release"]; null = np.empty(NPERM)
                for i in range(NPERM):
                    p_ = rng8.permutation(len(Er)); null[i] = f(Er[p_[:nc]]) - f(Er[p_[nc:]])
                V[rat]["p_ce"] = (1 + np.sum(null >= real)) / (NPERM + 1)
                # shuffle B: correct release > correct press (same trials, press/release swapped per trial)
                Pp = R[R.cond == "correct_press"].set_index(["session", "trial"]).E
                Pr = R[R.cond == "correct_release"].set_index(["session", "trial"]).E
                j = Pp.index.intersection(Pr.index); a, b = Pr.loc[j].values, Pp.loc[j].values
                real2 = f(a) - f(b); null2 = np.empty(NPERM)
                for i in range(NPERM):
                    sw = rng8.random(len(a)) < 0.5; null2[i] = f(np.where(sw, b, a)) - f(np.where(sw, a, b))
                V[rat]["p_rp"] = (1 + np.sum(null2 >= real2)) / (NPERM + 1)
                L.append(f"  {RL[rat]} (n press {len(g['correct_press'])}, correct rel {nc}, early rel {len(g['early_release'])}): "
                         f"press {V[rat]['correct_press']:.3f} | correct release {V[rat]['correct_release']:.3f} | early release {V[rat]['early_release']:.3f} "
                         f"|| shuffle p: correct rel > early rel {fpt(V[rat]['p_ce'])}, correct rel > press {fpt(V[rat]['p_rp'])}")
            cr = np.array([V[r]["correct_release"] for r in PAIRS3]); er = np.array([V[r]["early_release"] for r in PAIRS3])
            cp = np.array([V[r]["correct_press"] for r in PAIRS3])
            L.append(f"  ACROSS RATS (their test): correct rel > early rel in {(cr > er).sum()}/3 rats, Wilcoxon p {fpt(wilcoxon(cr, er, alternative='greater').pvalue)} "
                     f"| correct rel > press in {(cr > cp).sum()}/3 rats, Wilcoxon p {fpt(wilcoxon(cr, cp, alternative='greater').pvalue)}  (lowest possible p with 3 rats = .125)")
            L.append(f"  SHUFFLE, 3 rats combined (Fisher): correct rel > early rel p {fpt(fisher([V[r]['p_ce'] for r in PAIRS3]))} "
                     f"| correct rel > press p {fpt(fisher([V[r]['p_rp'] for r in PAIRS3]))}")
            if mn == "PLV":
                ax = axs[bi]; x = np.arange(3); w = 0.26
                for ci, (c, lab, col) in enumerate((("correct_press", "correct press", "0.6"), ("correct_release", "correct release", CCOLS["correct"]),
                                                    ("early_release", "early release", CCOLS["early"]))):
                    ax.bar(x + (ci - 1) * w, [V[r][c] for r in PAIRS3], w, color=col, label=lab)
                for ri, r in enumerate(PAIRS3):
                    ax.text(ri, max(V[r][c] for c in CONDS) * 1.05, f"rel>early p {fpt(V[r]['p_ce'])}\nrel>press p {fpt(V[r]['p_rp'])}", ha="center", fontsize=8)
                ax.set_xticks(x); ax.set_xticklabels([RL[r] for r in PAIRS3]); ax.set_ylabel("PLV (pooled sessions)")
                ax.set_title(f"{bn} {BANDS[bn][0]:g}–{BANDS[bn][1]:g} Hz", fontweight="bold")
                for sp in ("top", "right"): ax.spines[sp].set_visible(False)
                ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        # check: PPC per session (no trial-number bias), rats weighted equally, one-sided
        P = pd.DataFrame([dict(rat=r, session=s, **{c: ppc(G.loc[G.cond == c, "E"].values) for c in CONDS})
                          for (r, s), G in D.groupby(["rat", "session"])]).dropna()
        rr = P.rat.values
        L.append(f"\nPPC per session ({len(P)} sessions, rats weighted equally; vs 0 includes evoked phase locking, ERP kept):")
        L.append(f"  press {rmean(P.correct_press, rr):+.4f} (>0 p {fpt(strat_test(P.correct_press.values, rr))}) | "
                 f"correct release {rmean(P.correct_release, rr):+.4f} (>0 p {fpt(strat_test(P.correct_release.values, rr))}) | "
                 f"early release {rmean(P.early_release, rr):+.4f} (>0 p {fpt(strat_test(P.early_release.values, rr))})")
        d1, d2 = (P.correct_release - P.early_release).values, (P.correct_release - P.correct_press).values
        L.append(f"  correct rel > early rel p {fpt(strat_test(d1, rr))} ({(d1 > 0).sum()}/{len(d1)}) | correct rel > press p {fpt(strat_test(d2, rr))} ({(d2 > 0).sum()}/{len(d2)})")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("PL–IL phase locking at press / release, all sessions pooled per rat (Jäckel method)", fontweight="bold")
    _savefig(fig, os.path.join(OUT, "F1_pooled_per_rat_PLV.png")); plt.show()
    open(os.path.join(OUT, "T8b_summary.txt"), "w", encoding="utf-8").write("\n".join(L)); log("\n".join(L))


@cell()
def T8b_fig3D():
    """Replication of Jäckel et al. 2024 Fig. 3D: PLV at press / release per rat, δ and θ (sessions pooled per rat; from T8b_trials.csv)"""
    global COLS, D, F, MK, R, SIG, TR, _, ax, axs, bn, c, col, fig, fp, h, l, np, os, pd, plt, r, sp, v, x, xi
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ Replication of Jäckel et al. 2024 Fig. 3D: PLV at press / release per rat, δ and θ (sessions pooled per rat; from T8b_trials.csv) ══
    import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    F = os.path.join(TESTS, f"T8b_Jaeckel_exact_{REF}", "T8b_trials.csv")
    TR = pd.read_csv(F, dtype={"rat": str}); TR["E"] = TR.re.values + 1j * TR.im.values
    COLS = [("correct_press", "correct\npress", "#2CA02C"), ("correct_release", "correct\nrelease", "#2CA02C"), ("early_release", "early\nrelease", "#E6B800")]
    MK = {"134": "o", "135": "s", "136": "^"}
    SIG = {"delta": "p = .016 (correct rel > early rel), p = .004 (correct rel > press)", "theta": "n.s."}   # T8b shuffle, 3 rats combined

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 5.8), facecolor="white", gridspec_kw={"hspace": 0.5})
    for ax, bn in zip(axs, ("delta", "theta")):
        D = TR[TR.band == bn]; x = np.arange(len(COLS))
        for r in PAIRS3:
            R = D[D.rat == r]; v = [np.abs(R.loc[R.cond == c, "E"].mean()) for c, _, _ in COLS]
            ax.plot(x, v, color="0.7", lw=0.8, zorder=1)
            for xi, (c, _, col) in enumerate(COLS): ax.scatter(xi, v[xi], marker=MK[r], s=45, color=col, edgecolor="k", lw=0.5, zorder=2, label=RL[r] if xi == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in COLS]); ax.set_xlim(-0.5, len(COLS) - 0.5)
        ax.set_ylabel(f"{'δ' if bn == 'delta' else 'θ'} PLV"); ax.set_title(SIG[bn], fontsize=7.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    h, l = axs[0].get_legend_handles_labels(); axs[0].legend([plt.Line2D([], [], marker=MK[r], ls="", color="0.5", mec="k") for r in PAIRS3], [RL[r] for r in PAIRS3], frameon=False, fontsize=7.5, loc="upper left")
    fp = os.path.join(TESTS, f"T8b_Jaeckel_exact_{REF}", "Fig_Jaeckel3D_replication_per_rat.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))


@cell()
def T8c_fig3C():
    """Jäckel Fig. 3C replication, ERP removed — CORRECTED jackknife (t with sessions−1 df → z, then Stouffer over rats)"""
    global ALPHA_T, COL, D, Ellipse, MIN_DUR, MU, X, Z_, a, ax, axs, b, bn, c, ev, fig, fp, fpt, i, j, jk_z, k, log_, mu, norm
    global np, os, pd2, plt, ppc_raw, rD, res, rr, runs, s_, se, sp, splv, stouffer, t, tdist, tv, w0, w1, yhi, ylo, z, zD, zd
    global zz
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ══ Jäckel Fig. 3C replication, ERP removed — CORRECTED jackknife (t with sessions−1 df → z, then Stouffer over rats)
    #    curves = √PPC; coloured bars = condition coupled (PPC > 0, one-sided); black = early vs correct differ (delta, two-sided)
    import os, numpy as np, matplotlib.pyplot as plt
    from scipy.stats import norm, t as tdist
    from matplotlib.patches import Ellipse
    D = os.path.join(TESTS, f"T8c_Jaeckel_Fig3C_replication_{REF}_ERPremoved")
    ALPHA_T, MIN_DUR = 0.05, 0.05
    COL = {"correct": "#2CA02C", "early": "#E6B800"}
    fpt = lambda p: "<.001" if p < 0.001 else f"{p:.3f}"
    Z_ = {r: [np.load(os.path.join(D, f"{s}.npz")) for s in SESS[r] if os.path.exists(os.path.join(D, f"{s}.npz"))] for r in PAIRS3}

    def ppc_raw(Zs, key, k=None):                          # pooled PPC per rat (not biased by trial number); k = window mask → window mean
        S = sum(Z[f"{key}|sum"] for Z in Zs); N = sum(int(Z[f"{key}|n"]) for Z in Zs); v = (np.abs(S) ** 2 - N) / (N * (N - 1))
        return v if k is None else v[k].mean()
    splv = lambda v: np.sign(v) * np.sqrt(np.abs(v))
    def jk_z(Zs, f):                                       # jackknife over sessions → t (g−1 df) → z with the same tail probability
        full = np.asarray(f(Zs)); g = len(Zs); J = np.stack([np.asarray(f(Zs[:i] + Zs[i + 1:])) for i in range(g)])
        se = np.sqrt((g - 1) / g * ((J - J.mean(0)) ** 2).sum(0)); tv = np.where(se > 0, full / np.where(se > 0, se, 1), 0)
        return norm.isf(tdist.sf(tv, g - 1)), tv
    stouffer = lambda zs: np.sum(zs, 0) / np.sqrt(len(zs))
    def runs(mask, t, md):
        out, i = [], 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j + 1 < len(mask) and mask[j + 1]: j += 1
                if t[j] - t[i] >= md: out.append((t[i], t[j]))
                i = j + 1
            else: i += 1
        return out

    # ── figure ──
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(4, 1, figsize=(3.8, 9.0), facecolor="white", gridspec_kw={"hspace": 0.65})
    log_ = [f"sessions per rat: " + ", ".join(f"{RL[r]} {len(Z_[r])}" for r in PAIRS3)]
    for ax, (bn, ev) in zip(axs, [("delta", "Press"), ("delta", "Release"), ("theta", "Press"), ("theta", "Release")]):
        t = Z_[list(PAIRS3)[0]][0][f"t|{ev}"]; MU = {}
        for c in ("correct", "early"):
            X = np.stack([splv(ppc_raw(Z_[r], f"{bn}|{ev}|{c}")) for r in PAIRS3]); mu, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X)); MU[c] = mu
            ax.fill_between(t, mu - se, mu + se, color=COL[c], alpha=0.25, lw=0); ax.plot(t, mu, color=COL[c], lw=2, label=c)
        ax.axvline(0, color="0.4", ls="--", lw=1.5); ax.axhline(0, color="0.75", lw=0.8); ax.set_xlim(-1, 1)
        ylo, yhi = ax.get_ylim(); sp = yhi - ylo
        for j, c in enumerate(("correct", "early")):
            z = stouffer([jk_z(Z_[r], lambda Zx, c=c: ppc_raw(Zx, f"{bn}|{ev}|{c}"))[0] for r in PAIRS3]); rr = runs(norm.sf(z) < ALPHA_T, t, MIN_DUR)
            for a, b in rr: ax.plot([a, b], [yhi + (0.16 + 0.07 * j) * sp] * 2, color=COL[c], lw=2.5, solid_capstyle="butt")
            log_.append(f"{bn} {ev.lower()}: {c} coupled at " + (", ".join(f"{a:+.2f}→{b:+.2f} s" for a, b in rr) or "none"))
        if bn == "delta":
            zD = stouffer([jk_z(Z_[r], lambda Zx: ppc_raw(Zx, f"{bn}|{ev}|early") - ppc_raw(Zx, f"{bn}|{ev}|correct"))[0] for r in PAIRS3])
            rD = runs(2 * norm.sf(np.abs(zD)) < ALPHA_T, t, MIN_DUR)
            for a, b in rD: ax.plot([a, b], [yhi + 0.05 * sp] * 2, color="k", lw=3, solid_capstyle="butt")
            log_.append(f"{bn} {ev.lower()}: early vs correct differ at " + (", ".join(f"{a:+.2f}→{b:+.2f} s ({'early' if zD[(t >= a) & (t <= b)].mean() > 0 else 'correct'} higher)" for a, b in rD) or "none"))
        ax.set_ylim(ylo, yhi + 0.28 * sp)
        if (bn, ev) == ("theta", "Release"):
            k = (t >= -0.8) & (t <= -0.3); i = np.where(k)[0][np.argmax(MU["early"][k])]
            ax.add_patch(Ellipse((t[i], MU["early"][i] * 0.6), 0.32, abs(MU["early"][i]) * 1.6, fill=False, ec="red", lw=2))
        ax.set_title(ev.lower(), fontsize=9); ax.set_ylabel(f"{'δ' if bn == 'delta' else 'θ'} PLV (√PPC)")
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    axs[-1].set_xlabel("time [s]"); axs[0].legend(frameon=True, fontsize=8, loc="upper right")
    fig.text(0.5, 0.005, "coloured bar = condition coupled (PPC > 0); black bar = early vs correct differ; p < .05, ≥ 50 ms, uncorrected over time",
             ha="center", va="bottom", fontsize=6.5, color="0.35")
    fp = os.path.join(D, "Fig_Jaeckel3C_replication_ERPremoved_sqrtPPC_sig_conditions_tcorr.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))
    print("\n".join(log_))

    # ── window tests (same as before, corrected) ──
    print("\nWINDOW TESTS (pooled PPC per rat, jackknife t → z, Stouffer over rats)")
    for bn, ev, (w0, w1) in [("delta", "Press", (-0.1, 0.1)), ("delta", "Release", (-0.1, 0.1)), ("theta", "Release", (-0.1, 0.0)), ("theta", "Press", (-0.1, 0.1))]:
        k = (Z_[list(PAIRS3)[0]][0][f"t|{ev}"] >= w0 - 1e-9) & (Z_[list(PAIRS3)[0]][0][f"t|{ev}"] <= w1 + 1e-9); res, tv = {}, {}
        for c in ("correct", "early"):
            zz = [jk_z(Z_[r], lambda Zx, c=c: ppc_raw(Zx, f"{bn}|{ev}|{c}", k)) for r in PAIRS3]; res[c] = norm.sf(stouffer([z for z, _ in zz])); tv[c] = [float(t_) for _, t_ in zz]
        zz = [jk_z(Z_[r], lambda Zx: ppc_raw(Zx, f"{bn}|{ev}|early", k) - ppc_raw(Zx, f"{bn}|{ev}|correct", k)) for r in PAIRS3]
        zd = stouffer([z for z, _ in zz]); pd2 = 2 * norm.sf(abs(zd))
        print(f"  {bn} {ev.lower()} {w0:+.1f}→{w1:+.1f} s: correct > 0 p {fpt(res['correct'])} | early > 0 p {fpt(res['early'])} | "
              f"early − correct two-sided p {fpt(pd2)} ({'early' if zd > 0 else 'correct'} higher; one-sided {fpt(pd2 / 2)})")
        print("      per-rat t (sessions−1 df): " + " | ".join(f"{RL[r]} n={len(Z_[r])}: correct {tv['correct'][i]:+.2f}, early {tv['early'][i]:+.2f}, diff {float(zz[i][1]):+.2f}"
                                                   for i, r in enumerate(PAIRS3)))


@cell()
def overnight_summaries():
    """SUMMARIES OF THE OVERNIGHT TESTS (made in 06): T8 Jäckel θ / δ, T11 amplitude / power, T13–15 windows, T13 fix"""
    global SUMS, b, f, fp, name
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ── SUMMARIES OF THE OVERNIGHT TESTS (made in 06): T8 Jäckel θ / δ, T11 amplitude / power, T13–15 windows, T13 fix ──
    SUMS = {"T8 theta (Jäckel replication)": os.path.join(TESTS, f"T8_Jaeckel_replication_theta_{REF}", "T8_summary.txt"),
            "T8 delta (Jäckel replication)": os.path.join(TESTS, f"T8_Jaeckel_replication_delta_{REF}", "T8_summary.txt"),
            "T11 (coupling vs theta amplitude / power)": os.path.join(TESTS, f"T11_amplitude_power_{REF}", "T11_summary.txt"),
            "T13–15 (windows, equal trial numbers, ERP kept)": os.path.join(TESTS, f"Overnight_T13-15_{REF}_ERPkept", "T13-15_summary.txt"),
            "T13 fix (before the tone, correct vs late)": os.path.join(TESTS, f"Overnight_T13-15_{REF}_ERPkept", "T13_summary_fix.txt")}
    for name, f in SUMS.items():
        print(f"════ {name} ════")
        print(open(f, encoding="utf-8").read() if os.path.exists(f) else f"not made yet -> run 06 (overnight_tests / T13_fix): {f}")
    for b in ("theta", "delta"):
        fp = os.path.join(TESTS, f"T8_Jaeckel_replication_{b}_{REF}", "F1_traces.png")
        if os.path.exists(fp): display(Image(fp))


@cell()
def coupling_along_session():
    """THETA COUPLING DURING THE HOLD ALONG THE SESSION (correct long, SR30-500, chosen pairs): A) by trial number, B) start / middle / end, three splits"""
    global ALPHA, CUTS, DA, DEFS, DESK, HW, M, MB, MIN_T, NB_A, NDR, NSH, P, PHD, PHS, PP, PT, SB, SPL, ST_A, TCOL, XS, Z, a
    global a1, a2, ax, bins, cl, col, d, f, fig, fmt_q, fp, hw, i, j, k, lev, low, m, mu, multipletests, n, np, ok, os, other
    global p, p_, pd, ph, phase, plix, plt, pos, q, rat, rng_, rows, s, se, sel, sp, split, st, t0_, t1_, time, top, tt, uA
    global uB, v, val, vol, w, x, x0, x1, y, z
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ── THETA COUPLING DURING THE HOLD ALONG THE SESSION (correct long, SR30-500, chosen pairs): A) by trial number, B) start / middle / end, three splits ──
    #    Value = PLI − chance (hold 0.2–0.85 s) of the trials in each part; chance = trials of the same part shuffled. Positions and cuts as in the theta-power cell.
    #    A: 10 parts by trial number (part used if ≥ 5 trials). B: start / middle / end, equal n across the parts of a session (20 draws);
    #    brackets = middle or end vs start (sign-flip across sessions), q = FDR-BH over the 2 tests of each split. Saved on the desktop.
    import os, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from statsmodels.stats.multitest import multipletests
    DESK = next((p for p in (os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"), os.path.join(os.path.expanduser("~"), "Desktop"))
                 if os.path.isdir(p)), os.path.expanduser("~"))
    NB_A, MIN_T, NDR, NSH, TCOL = 10, 5, 20, 100, "#2e7d32"
    HW = (0.2, 0.85)
    PHS, CUTS = ["start", "middle", "end"], (0.25, 0.85)
    DEFS = {"trial number": "position", "clock time": "v_time", "water drunk": "v_water"}
    ALPHA = {"trial number": 0.95, "clock time": 0.6, "water drunk": 0.3}

    # ── trial positions (same as beta Step 3 / theta-power cell) ──
    PT = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False).sort_values(["session", "trial_index"])
    PT["position"] = PT.groupby("session")["trial_index"].rank(pct=True)
    t0_ = PT.groupby("session")["press_neural_s"].transform("min"); t1_ = PT.groupby("session")["press_neural_s"].transform("max")
    PT["v_time"] = (PT["press_neural_s"] - t0_) / (t1_ - t0_)
    vol = PT["reward_size"].fillna(0).clip(lower=0)
    PT["v_water"] = (vol.groupby(PT["session"]).cumsum() - vol) / vol.groupby(PT["session"]).transform("sum")
    phase = lambda v: np.where(v < CUTS[0], "start", np.where(v < CUTS[1], "middle", "end"))

    PHD = os.path.join(TESTS, f"Trial_phases_press_{REF}_ERPkept")
    # ── 0) per-trial IL / PL theta phase
    # (the per-trial phase files are made in 06: trial_phase_files)
    def plix(a, b):                                   # PLI − chance over trials (a, b: trials × freqs × time, unit phase)
        pli = np.abs(np.sign((a * np.conj(b)).imag).mean(0)).mean()
        return pli - np.mean([np.abs(np.sign((a * np.conj(b[derangement(len(a))])).imag).mean(0)).mean() for _ in range(NSH)])

    # ── 1) session values ──
    rows = []
    for rat in PAIRS3:
        for s in SESS[rat]:
            f = os.path.join(PHD, f"{s}.npz")
            if not os.path.exists(f): continue
            Z = np.load(f); tt = Z["t"]; hw = np.where((tt >= HW[0] - 1e-9) & (tt <= HW[1] + 1e-9))[0]
            cl = Z["label"] == "correct"; uA, uB = Z["uA"][cl][..., hw], Z["uB"][cl][..., hw]
            pos = pd.DataFrame({"trial_index": Z["trial_index"][cl]}).merge(PT.loc[PT.session == s, ["trial_index", *DEFS.values()]], on="trial_index", how="left")
            ok = pos["position"].notna().values
            bins = np.minimum((np.nan_to_num(pos["position"].values, nan=-1) * NB_A).astype(int), NB_A - 1)
            for k in range(NB_A):                                                    # A: 10 parts (no matching)
                m = ok & (bins == k)
                if m.sum() >= MIN_T: rows.append(dict(rat=rat, session=s, split="bins", part=k, n=int(m.sum()), val=plix(uA[m], uB[m])))
            for split, col in DEFS.items():                                          # B: start / middle / end, equal n per session
                v = pos[col].values; ph = phase(v); M = {p_: np.where(np.isfinite(v) & (ph == p_))[0] for p_ in PHS}
                M = {p_: i for p_, i in M.items() if len(i) >= MIN_T}
                if len(M) < 2: continue
                n = min(len(i) for i in M.values())
                for p_, i in M.items():
                    val = plix(uA[i], uB[i]) if len(i) == n else np.mean([plix(uA[k], uB[k]) for k in (rng.choice(i, n, replace=False) for _ in range(NDR))])
                    rows.append(dict(rat=rat, session=s, split=split, part=p_, n=n, val=val))
    DA = pd.DataFrame(rows)

    # ── 2) tests: middle and end vs start, per split ──
    st = []
    for split in DEFS:
        P = DA[DA.split == split].pivot(index="session", columns="part", values="val")
        for other in ("middle", "end"):
            if "start" not in P or other not in P: continue
            PP = P[["start", other]].dropna(); d = (PP[other] - PP["start"]).values; m, p, n = fn.sign_flip_p(d)
            st.append(dict(split=split, comparison=f"{other} vs start", n_sessions=n, sessions_higher_than_start=int((d > 0).sum()),
                           start=round(float(PP["start"].mean()), 4), other=round(float(PP[other].mean()), 4), p=p))
    ST_A = pd.DataFrame(st); ST_A["q"] = ST_A.groupby("split")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    _table(ST_A, os.path.join(DESK, f"theta_coupling_along_session_three_splits_{REF}.csv"), index=False)

    # ── 3) figure ──
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white", gridspec_kw={"width_ratios": [1.6, 1.2], "wspace": 0.3})
    M = DA[DA.split == "bins"].pivot(index="session", columns="part", values="val").reindex(columns=range(NB_A))
    x = (np.arange(NB_A) + 0.5) / NB_A; mu = M.mean(); se = M.std() / np.sqrt(M.notna().sum())
    a1.axvspan(0, CUTS[0], color="0.93", zorder=0); a1.axvspan(CUTS[1], 1, color="0.93", zorder=0); a1.axhline(0, color="k", lw=0.8, ls="--")
    a1.fill_between(x, mu - se, mu + se, color=TCOL, alpha=0.2, lw=0); a1.plot(x, mu, color=TCOL, lw=2, marker="o", ms=3.5)
    for p_, (a, z) in zip(PHS, ((0, CUTS[0]), CUTS, (CUTS[1], 1))):
        a1.text((a + z) / 2, 1.01, p_, transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="0.35")
    a1.set_xlim(0, 1); a1.set_xlabel("Position in the session (first → last trial)"); a1.set_ylabel("IL–PL theta coupling, PLI − chance")
    w = 0.26; MB, SB, XS = {}, {}, {}
    fmt_q = lambda q: "q < 0.001" if q < 0.001 else f"q = {q:.3f}"
    for i, split in enumerate(DEFS):
        P = DA[DA.split == split].pivot(index="session", columns="part", values="val").reindex(columns=PHS)
        MB[split] = P.mean().values; SB[split] = np.array([P[c].dropna().std(ddof=1) / np.sqrt(P[c].notna().sum()) for c in PHS])
        XS[split] = np.arange(3) + (i - 1) * w
        a2.bar(XS[split], MB[split], width=w, color=TCOL, alpha=ALPHA[split], edgecolor="k", lw=0.6, label=split, zorder=2)
        a2.errorbar(XS[split], MB[split], yerr=SB[split], fmt="none", color="k", capsize=3, lw=1.1, zorder=3)
    top = max(np.nanmax(np.r_[MB[k] + SB[k], 0]) for k in DEFS); low = min(np.nanmin(np.r_[MB[k] - SB[k], 0]) for k in DEFS); rng_ = top - low
    lev, SPL = 0, "trial number"
    for j, other in ((1, "middle"), (2, "end")):
        sel = ST_A[(ST_A.split == SPL) & (ST_A.comparison == f"{other} vs start")]
        if not len(sel): continue
        q = sel.q.iloc[0]; y = top + rng_ * (0.08 + 0.13 * lev); x0, x1 = XS[SPL][0], XS[SPL][j]; lev += 1
        a2.plot([x0, x0, x1, x1], [y - rng_ * 0.02, y, y, y - rng_ * 0.02], color="k", lw=1.0)
        a2.text((x0 + x1) / 2, y + rng_ * 0.005, fmt_q(q), ha="center", va="bottom", fontsize=8.5, fontweight="bold" if q < 0.05 else "normal")
    a2.set_ylim(top=top + rng_ * (0.08 + 0.13 * lev + 0.05))
    a2.axhline(0, color="k", lw=0.8); a2.set_xticks(range(3)); a2.set_xticklabels([p_.capitalize() for p_ in PHS])
    a2.set_ylabel("IL–PL theta coupling, PLI − chance"); a2.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in (a1, a2):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fp = os.path.join(DESK, f"theta_coupling_along_the_session_three_splits_{REF}.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))

    print(f"theta coupling, correct long trials, {REF}: {DA.session.nunique()} sessions | sessions per part (A):",
          DA[DA.split == "bins"].groupby("part").size().to_dict())
    print("trials per part (B, equal n per session, median):", DA[DA.split != "bins"].groupby("split").n.median().to_dict())
    print(ST_A.round({"p": 4, "q": 4}).to_string(index=False))


@cell(needs={
    'coupling_along_session': ['ALPHA', 'CUTS', 'DEFS', 'DESK', 'HW', 'MIN_T', 'NB_A', 'NDR', 'PHD', 'PHS', 'PT', 'TCOL', 'multipletests', 'phase'],
})
def raw_plv_pli_along_session():
    """RAW PLV and PLI (no chance) along the session: correct long, hold 0.2–0.85 s, SR30-500; equal n across the parts of each session"""
    global DR, M, MB, P, PP, SB, ST_R, XS, Z, a, a1, a2, ax, axs, bins, cl, col, d, eq, f, fig, fmt_q, fp, hw, i, j, k, lev
    global low, m, met, mu, n, ok, other, p, p_, parts, ph, pli, plv, pos, q, rat, raw, ri, rng_, rows, s, se, sel, sp, split
    global st, top, tt, uA, uB, v, vals, w, x, x0, x1, y, z
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ── RAW PLV and PLI (no chance) along the session: correct long, hold 0.2–0.85 s, SR30-500; equal n across the parts of each session ──
    def raw(a, b):                                    # a, b: trials × freqs × time (unit phase) → PLV, PLI
        S = a * np.conj(b); return np.abs(S.mean(0)).mean(), np.abs(np.sign(S.imag).mean(0)).mean()
    def eq(a, b, idx_list):                           # same n for every part (random draws to the smallest part)
        n = min(len(i) for i in idx_list); out = []
        for i in idx_list:
            out.append(raw(a[i], b[i]) if len(i) == n else tuple(np.mean([raw(a[k], b[k]) for k in (rng.choice(i, n, replace=False) for _ in range(NDR))], 0)))
        return out, n

    rows = []
    for rat in PAIRS3:
        for s in SESS[rat]:
            f = os.path.join(PHD, f"{s}.npz")
            if not os.path.exists(f): continue
            Z = np.load(f); tt = Z["t"]; hw = np.where((tt >= HW[0] - 1e-9) & (tt <= HW[1] + 1e-9))[0]
            cl = Z["label"] == "correct"; uA, uB = Z["uA"][cl][..., hw], Z["uB"][cl][..., hw]
            pos = pd.DataFrame({"trial_index": Z["trial_index"][cl]}).merge(PT.loc[PT.session == s, ["trial_index", *DEFS.values()]], on="trial_index", how="left")
            ok = pos["position"].notna().values
            bins = np.minimum((np.nan_to_num(pos["position"].values, nan=-1) * NB_A).astype(int), NB_A - 1)
            parts = {k: np.where(ok & (bins == k))[0] for k in range(NB_A)}; parts = {k: i for k, i in parts.items() if len(i) >= MIN_T}
            if len(parts) >= 2:
                vals, n = eq(uA, uB, list(parts.values()))
                for k, (plv, pli) in zip(parts, vals): rows.append(dict(rat=rat, session=s, split="bins", part=k, n=n, PLV=plv, PLI=pli))
            for split, col in DEFS.items():
                v = pos[col].values; ph = phase(v); M = {p_: np.where(np.isfinite(v) & (ph == p_))[0] for p_ in PHS}
                M = {p_: i for p_, i in M.items() if len(i) >= MIN_T}
                if len(M) < 2: continue
                vals, n = eq(uA, uB, list(M.values()))
                for p_, (plv, pli) in zip(M, vals): rows.append(dict(rat=rat, session=s, split=split, part=p_, n=n, PLV=plv, PLI=pli))
    DR = pd.DataFrame(rows)

    st = []
    for met in ("PLV", "PLI"):
        for split in DEFS:
            P = DR[DR.split == split].pivot(index="session", columns="part", values=met)
            for other in ("middle", "end"):
                if "start" not in P or other not in P: continue
                PP = P[["start", other]].dropna(); d = (PP[other] - PP["start"]).values; m, p, n = fn.sign_flip_p(d)
                st.append(dict(metric=met, split=split, comparison=f"{other} vs start", n_sessions=n, sessions_higher_than_start=int((d > 0).sum()),
                               start=round(float(PP["start"].mean()), 4), other=round(float(PP[other].mean()), 4), p=p))
    ST_R = pd.DataFrame(st); ST_R["q"] = ST_R.groupby(["metric", "split"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    _table(ST_R, os.path.join(DESK, "theta_coupling_RAW_along_session_three_splits_SR30-500.csv"), index=False)

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), facecolor="white", gridspec_kw={"width_ratios": [1.6, 1.2], "wspace": 0.3, "hspace": 0.45})
    fmt_q = lambda q: "q < 0.001" if q < 0.001 else f"q = {q:.3f}"
    for ri, met in enumerate(("PLV", "PLI")):
        a1, a2 = axs[ri]
        M = DR[DR.split == "bins"].pivot(index="session", columns="part", values=met).reindex(columns=range(NB_A))
        x = (np.arange(NB_A) + 0.5) / NB_A; mu = M.mean(); se = M.std() / np.sqrt(M.notna().sum())
        a1.axvspan(0, CUTS[0], color="0.93", zorder=0); a1.axvspan(CUTS[1], 1, color="0.93", zorder=0)
        a1.fill_between(x, mu - se, mu + se, color=TCOL, alpha=0.2, lw=0); a1.plot(x, mu, color=TCOL, lw=2, marker="o", ms=3.5)
        for p_, (a, z) in zip(PHS, ((0, CUTS[0]), CUTS, (CUTS[1], 1))):
            a1.text((a + z) / 2, 1.01, p_, transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="0.35")
        a1.set_xlim(0, 1); a1.set_xlabel("Position in the session (first → last trial)"); a1.set_ylabel(f"IL–PL theta {met} (raw)")
        w = 0.26; MB, SB, XS = {}, {}, {}
        for i, split in enumerate(DEFS):
            P = DR[DR.split == split].pivot(index="session", columns="part", values=met).reindex(columns=PHS)
            MB[split] = P.mean().values; SB[split] = np.array([P[c].dropna().std(ddof=1) / np.sqrt(P[c].notna().sum()) for c in PHS])
            XS[split] = np.arange(3) + (i - 1) * w
            a2.bar(XS[split], MB[split], width=w, color=TCOL, alpha=ALPHA[split], edgecolor="k", lw=0.6, label=split, zorder=2)
            a2.errorbar(XS[split], MB[split], yerr=SB[split], fmt="none", color="k", capsize=3, lw=1.1, zorder=3)
        top = max(np.nanmax(MB[k] + SB[k]) for k in DEFS); low = min(np.nanmin(MB[k] - SB[k]) for k in DEFS); rng_ = top - low
        a2.set_ylim(bottom=max(0, low - rng_ * 0.5)); lev = 0
        for j, other in ((1, "middle"), (2, "end")):
            sel = ST_R[(ST_R.metric == met) & (ST_R.split == "trial number") & (ST_R.comparison == f"{other} vs start")]
            if not len(sel): continue
            q = sel.q.iloc[0]; y = top + rng_ * (0.08 + 0.13 * lev); x0, x1 = XS["trial number"][0], XS["trial number"][j]; lev += 1
            a2.plot([x0, x0, x1, x1], [y - rng_ * 0.02, y, y, y - rng_ * 0.02], color="k", lw=1.0)
            a2.text((x0 + x1) / 2, y + rng_ * 0.005, fmt_q(q), ha="center", va="bottom", fontsize=8.5, fontweight="bold" if q < 0.05 else "normal")
        a2.set_ylim(top=top + rng_ * (0.08 + 0.13 * lev + 0.05)); a2.set_xticks(range(3)); a2.set_xticklabels([p_.capitalize() for p_ in PHS])
        a2.set_ylabel(f"IL–PL theta {met} (raw)")
        if ri == 0: a2.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        for ax in (a1, a2):
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fp = os.path.join(DESK, "theta_coupling_RAW_along_the_session_three_splits_SR30-500.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))
    print("sessions per part (A):", DR[DR.split == "bins"].groupby("part").size().to_dict())
    print("trials per part (equal n per session, median):", DR.groupby("split").n.median().to_dict())
    print(ST_R.round({"p": 4, "q": 4}).to_string(index=False))


@cell(needs={
    'coupling_along_session': ['DESK', 'HW', 'MIN_T', 'PHD', 'PT', 'TCOL'],
    'raw_plv_pli_along_session': ['eq'],
})
def pli_along_water():
    """RAW PLI along WATER DRUNK (5 parts, zoomed, no tests): correct long, hold 0.2–0.85 s, SR30-500"""
    global DW, M, NB_W, Z, ax, bins, cl, f, fig, fp, hi, hw, k, lo, mu, pad, parts, rat, rows, s, se, sp, tt, uA, uB, v, w_, x
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ── RAW PLI along WATER DRUNK (5 parts, zoomed, no tests): correct long, hold 0.2–0.85 s, SR30-500 ──
    NB_W = 5
    rows = []
    for rat in PAIRS3:
        for s in SESS[rat]:
            f = os.path.join(PHD, f"{s}.npz")
            if not os.path.exists(f): continue
            Z = np.load(f); tt = Z["t"]; hw = np.where((tt >= HW[0] - 1e-9) & (tt <= HW[1] + 1e-9))[0]
            cl = Z["label"] == "correct"; uA, uB = Z["uA"][cl][..., hw], Z["uB"][cl][..., hw]
            w_ = pd.DataFrame({"trial_index": Z["trial_index"][cl]}).merge(PT.loc[PT.session == s, ["trial_index", "v_water"]], on="trial_index", how="left")["v_water"].values
            bins = np.minimum((np.nan_to_num(w_, nan=-1) * NB_W).astype(int), NB_W - 1)
            parts = {k: np.where(np.isfinite(w_) & (bins == k))[0] for k in range(NB_W)}; parts = {k: i for k, i in parts.items() if len(i) >= MIN_T}
            if len(parts) >= 2:
                for k, v in zip(parts, eq(uA, uB, list(parts.values()))): rows.append(dict(session=s, part=k, PLI=v))
    DW = pd.DataFrame(rows)
    M = DW.pivot(index="session", columns="part", values="PLI").reindex(columns=range(NB_W))
    x = (np.arange(NB_W) + 0.5) / NB_W; mu = M.mean(); se = M.std() / np.sqrt(M.notna().sum())
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.fill_between(x, mu - se, mu + se, color=TCOL, alpha=0.2, lw=0); ax.plot(x, mu, color=TCOL, lw=2.2, marker="o", ms=6)
    lo, hi = np.nanmin(mu - se), np.nanmax(mu + se); pad = (hi - lo) * 0.15; ax.set_ylim(lo - pad, hi + pad)   # zoom to the data
    ax.set_xlim(0, 1); ax.set_xticks(x); ax.set_xticklabels([f"{int(100 * i / NB_W)}–{int(100 * (i + 1) / NB_W)}%" for i in range(NB_W)])
    ax.set_xlabel("Water drunk before the trial (share of the session's reward)"); ax.set_ylabel("IL–PL theta PLI (raw)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fp = os.path.join(DESK, "theta_PLI_raw_along_water_drunk_5parts_SR30-500.png")
    _savefig(fig, fp); plt.close(fig); display(Image(fp))
    print("sessions per part:", M.notna().sum().to_dict()); print("mean raw PLI per part:", mu.round(4).to_dict())


@cell(needs={
    'coupling_along_session': ['DESK', 'HW', 'MIN_T', 'PHD', 'PT', 'TCOL'],
    'raw_plv_pli_along_session': ['eq'],
})
def pli_along_clock_trial():
    """RAW PLI along CLOCK TIME and TRIAL POSITION (5 parts, zoomed, no tests): correct long, hold 0.2–0.85 s, SR30-500"""
    global M, NB_W, ax, col, fig, fp, hi, lo, mu, pad, pli_along, se, sp, tag, x, xlab
    REF = "SR30-500"                     # the coupling tests (T4–T15, Jäckel, along the session) always use the same-region ring
    # ── RAW PLI along CLOCK TIME and TRIAL POSITION (5 parts, zoomed, no tests): correct long, hold 0.2–0.85 s, SR30-500 ──
    NB_W = 3
    def pli_along(col):
        rows = []
        for rat in PAIRS3:
            for s in SESS[rat]:
                f = os.path.join(PHD, f"{s}.npz")
                if not os.path.exists(f): continue
                Z = np.load(f); tt = Z["t"]; hw = np.where((tt >= HW[0] - 1e-9) & (tt <= HW[1] + 1e-9))[0]
                cl = Z["label"] == "correct"; uA, uB = Z["uA"][cl][..., hw], Z["uB"][cl][..., hw]
                v_ = pd.DataFrame({"trial_index": Z["trial_index"][cl]}).merge(PT.loc[PT.session == s, ["trial_index", col]], on="trial_index", how="left")[col].values
                bins = np.minimum((np.nan_to_num(v_, nan=-1) * NB_W).astype(int), NB_W - 1)
                parts = {k: np.where(np.isfinite(v_) & (bins == k))[0] for k in range(NB_W)}; parts = {k: i for k, i in parts.items() if len(i) >= MIN_T}
                if len(parts) >= 2:
                    for k, val in zip(parts, eq(uA, uB, list(parts.values()))): rows.append(dict(session=s, part=k, PLI=val))
        return pd.DataFrame(rows).pivot(index="session", columns="part", values="PLI").reindex(columns=range(NB_W))

    for col, xlab, tag in (("v_time", "Clock time in the session (first → last press)", "clock_time"),
                           ("position", "Trial position in the session (first → last trial)", "trial_position")):
        M = pli_along(col); x = (np.arange(NB_W) + 0.5) / NB_W; mu = M.mean(); se = M.std() / np.sqrt(M.notna().sum())
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        ax.fill_between(x, mu - se, mu + se, color=TCOL, alpha=0.2, lw=0); ax.plot(x, mu, color=TCOL, lw=2.2, marker="o", ms=6)
        lo, hi = np.nanmin(mu - se), np.nanmax(mu + se); pad = (hi - lo) * 0.15; ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, 1); ax.set_xticks(x); ax.set_xticklabels([f"{int(100 * i / NB_W)}–{int(100 * (i + 1) / NB_W)}%" for i in range(NB_W)])
        ax.set_xlabel(xlab); ax.set_ylabel("IL–PL theta PLI (raw)")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        fp = os.path.join(DESK, f"theta_PLI_raw_along_{tag}_5parts_SR30-500.png")
        _savefig(fig, fp); plt.close(fig); display(Image(fp))
        print(f"{tag}: sessions per part {M.notna().sum().to_dict()} | mean raw PLI {mu.round(4).to_dict()}")
