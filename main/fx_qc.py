"""
fx_qc.py — functions of notebook 04_QC (signal quality of every channel)

Code moved out of the notebook cells without changing what it computes. It uses the shared building blocks of
functions.py (load_session, probe_table, compute_psd, psd_table, qc_session, ...).
The per-session QC files in 2_processed\\<session>\\amir_data\\QC are only written when the loop is run (RUN = True).
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as fn
import fx_common as fxc

QC_SUB = os.path.join("amir_data", "QC")


def qc_files(proc, s, local=False):
    """(folder, psd file, channel table) of one session's QC; local=True -> the LOCAL-reference files."""
    d = os.path.join(proc, s, QC_SUB)
    return (d, os.path.join(d, "psd_local.npz"), os.path.join(d, "qc_channels_local.csv")) if local else \
           (d, os.path.join(d, "psd.npz"), os.path.join(d, "qc_channels.csv"))


def run_signal_qc(final, proc, xb, run=False, overwrite=False, local=False, err_log=None):
    """Per session: load (minus extra bad channels) -> fn.qc_session (PSD 30 × 10 s, band powers, 50 Hz ratio, local_dev
    vs neighbours ±75 µm). original + CMR, or LOCAL only (local=True). Writes the per-session files only if run=True."""
    done = [s for s in final["session"] if os.path.exists(qc_files(proc, s, local)[2])]
    todo = [s for s in final["session"] if overwrite or s not in done]
    print(f"{'LOCAL' if local else 'original + CMR'} QC: {len(done)} done, {len(todo)} to do", flush=True)
    if not run: return todo
    rat_of = final.set_index("session")["rat"]; t_all = time.time()
    for i, s in enumerate(todo, 1):
        t0 = time.time()
        try:
            recs, _ = fn.load_session(proc, s, xb, refs=("LOCAL",)) if local else fn.load_session(proc, s, xb)
            key = "LOCAL" if local else "original"
            pdf = fn.probe_table(recs[key])
            freqs, psd, tab = fn.qc_session({"LOCAL": recs["LOCAL"]} if local else recs, pdf)
            tab.insert(0, "session", s); tab.insert(1, "rat", rat_of[s])
            d, f_npz, f_csv = qc_files(proc, s, local); os.makedirs(d, exist_ok=True)
            np.savez_compressed(f_npz, freqs=freqs, channel_ids=np.array(pdf["channel_id"]), **{f"psd_{r}": p for r, p in psd.items()})
            tab.to_csv(f_csv, index=False)                                   # written last = session finished
            print(f"[{i}/{len(todo)}] {s}: {(time.time() - t0) / 60:.1f} min | local outliers {tab.groupby('ref')['flag_local'].sum().to_dict()} | "
                  f"max 50 Hz ratio {tab['line_ratio'].max():.2f}", flush=True)
        except Exception as e:                                               # one broken session does not stop the run
            print(f"[{i}/{len(todo)}] {s}: ERROR {e!r}", flush=True)
            if err_log: pd.DataFrame([{"session": s, "error": repr(e)}]).to_csv(err_log, mode="a", header=not os.path.exists(err_log), index=False)
    print(f"\nfinished in {(time.time() - t_all) / 60:.0f} min", flush=True)
    return todo


def collect_qc(final, proc, local=False):
    """All finished per-session QC tables in one table (one row per session × reference × channel)."""
    parts = [pd.read_csv(qc_files(proc, s, local)[2], dtype={"rat": str}) for s in final["session"] if os.path.exists(qc_files(proc, s, local)[2])]
    if not parts:
        print("no QC results yet (run the loop with RUN = True)"); return None
    QC = pd.concat(parts, ignore_index=True)
    print(f"collected: {QC['session'].nunique()} sessions, {len(QC)} rows")
    return QC


def _repeat(QC, keys, min_n=3, min_pct=30):
    g = QC.groupby(["rat", "ref"] + keys).agg(n_sessions=("session", "nunique"), n_flagged=("flag_local", "sum"),
                                              median_dev=("local_dev", "median")).reset_index()
    g["pct_flagged"] = (100 * g["n_flagged"] / g["n_sessions"]).round(0)
    return g[(g["n_flagged"] >= min_n) & (g["pct_flagged"] >= min_pct)]


def qc_summary(QC, final):
    """Per session × reference: channels, local outliers, channels with 50 Hz ratio > 1, median 50 Hz ratio; per rat;
    repeat offenders (same readout channel / electrode position flagged in >= 3 sessions and >= 30 %). Returns (sessions, repeat)."""
    QC = QC.copy(); QC["ap"] = QC["channel_id"].astype(str).str.extract(r"(AP\d+)")[0]; QC["line_flag"] = QC["line_ratio"] > 1
    ses = QC.groupby(["rat", "session", "ref"]).agg(channels=("channel_id", "size"), local_outliers=("flag_local", "sum"),
                                                    line_gt1=("line_flag", "sum"), line_ratio_median=("line_ratio", "median")).reset_index()
    print(f"sessions with QC: {ses['session'].nunique()} of {len(final)}")
    print("\nper rat and reference (sum over sessions; median line ratio = median over sessions):")
    print(ses.groupby(["rat", "ref"]).agg(sessions=("session", "nunique"), local_outliers=("local_outliers", "sum"),
                                          max_per_session=("local_outliers", "max"), line_gt1=("line_gt1", "sum"),
                                          line_ratio_median=("line_ratio_median", "median")).round(2).to_string())
    print("\nsessions with the most local outliers:"); print(ses.sort_values("local_outliers", ascending=False).head(10).to_string(index=False))
    rep = pd.concat([_repeat(QC, ["ap"]).assign(by="readout channel"), _repeat(QC, ["x", "y"]).assign(by="electrode position")], ignore_index=True)
    rep = rep[["by"] + [c for c in rep.columns if c != "by"]].sort_values(["by", "rat", "ref", "n_flagged"], ascending=[True, True, True, False])
    print(f"\nrepeat offenders (flagged in >= 3 sessions and >= 30 % of the sessions recording them): {len(rep)}")
    print(rep.round(2).to_string(index=False) if len(rep) else "none")
    return ses, rep


def qc_local_summary(QL, QO):
    """Outliers and 50 Hz per rat for original, CMR and LOCAL side by side; LOCAL repeat offenders."""
    fin = QL["session"].unique()
    QA = pd.concat([QO[QO["session"].isin(fin)], QL], ignore_index=True); QA["line_flag"] = QA["line_ratio"] > 1
    ses = QA.groupby(["rat", "session", "ref"]).agg(local_outliers=("flag_local", "sum"), line_gt1=("line_flag", "sum"),
                                                    line_ratio_median=("line_ratio", "median")).reset_index()
    print(f"\nsignal QC ({len(fin)} sessions), per rat and reference:")
    print(ses.groupby(["rat", "ref"]).agg(sessions=("session", "nunique"), outliers_total=("local_outliers", "sum"),
                                          outliers_max_per_session=("local_outliers", "max"), line_gt1=("line_gt1", "sum"),
                                          line_ratio_median=("line_ratio_median", "median")).round(2).to_string())
    QL = QL.copy(); QL["ap"] = QL["channel_id"].astype(str).str.extract(r"(AP\d+)")[0]
    rep = _repeat(QL, ["ap"]).sort_values(["rat", "n_flagged"], ascending=[True, False])
    print(f"\nLOCAL repeat offenders (flagged in >= 3 sessions and >= 30 %): {len(rep)}")
    print(rep.round(2).to_string(index=False) if len(rep) else "none")
    return rep


def example_bad_channel(proc, ses, bad, radius_um=75):
    """One bad channel vs its neighbours (original, WITHOUT removing extra bad channels): A broadband power along the shank,
    B power spectrum, C 2 s raw LFP. Returns (fig, log10 deviation from the neighbour median)."""
    rec = fn.load_session(proc, ses, None, refs=("original",))[0]["original"]
    pdf = fn.probe_table(rec); pdf["ap"] = pdf["channel_id"].astype(str).str.extract(r"(AP\d+)")[0]
    b = pdf[pdf["ap"] == bad].iloc[0]
    sh = pdf[pdf["shank"] == b["shank"]].sort_values("y")
    nb = sh[(np.abs(sh["y"] - b["y"]) <= radius_um) & (sh["ap"] != bad)]
    freqs, psd, ids = fn.compute_psd(rec, channel_ids=list(sh["channel_id"]))
    tab = fn.psd_table(freqs, psd, ids, pdf).merge(pdf[["channel_id", "ap"]], on="channel_id")
    tab = tab.set_index("channel_id").loc[sh["channel_id"]].reset_index()
    med = tab.loc[tab["channel_id"].isin(nb["channel_id"]), "log_broadband"].median()
    dev = tab.loc[tab["ap"] == bad, "log_broadband"].iloc[0] - med
    print(f"{ses} {bad}: broadband log10 power {dev:+.2f} vs median of {len(nb)} neighbours (= {10 ** dev:.1f}x; flagged if |Δ| > 0.3 = 2x)")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), facecolor="white", gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
    ax = axs[0]; ax.axvspan(med - 0.3, med + 0.3, color="0.9", zorder=0)
    ax.scatter(tab["log_broadband"], tab["y"], s=8, color="k"); r_ = tab[tab["ap"] == bad]
    ax.scatter(r_["log_broadband"], r_["y"], s=40, color="#d62728", zorder=3); ax.set_ylim(b["y"] - 600, b["y"] + 600)
    ax.set_xlabel("broadband power (log10 µV²)"); ax.set_ylabel("distance from probe tip (µm)")
    ax.set_title("A  power along the shank", loc="left", fontsize=10, fontweight="bold")
    ax = axs[1]; idx = {c: i for i, c in enumerate(ids)}
    for c in nb["channel_id"]: ax.semilogy(freqs, psd[idx[c]], color="0.6", lw=0.8)
    ax.semilogy(freqs, psd[idx[b["channel_id"]]], color="#d62728", lw=1.6)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (µV²/Hz)"); ax.set_xlim(1, 100)
    ax.set_title("B  power spectrum", loc="left", fontsize=10, fontweight="bold")
    ax = axs[2]; near = nb.iloc[np.argsort(np.abs(nb["y"].values - b["y"]))[:4]]
    show = pd.concat([near, pdf[pdf["ap"] == bad]]).sort_values("y")
    fs = rec.get_sampling_frequency(); a = int(rec.get_num_samples() // 2)
    X = rec.get_traces(start_frame=a, end_frame=a + int(2 * fs), channel_ids=list(show["channel_id"]), return_in_uV=True).astype(float)
    X = X - X.mean(0); step = 4 * np.median(X.std(0)); t = np.arange(X.shape[0]) / fs
    for j, (_, ch) in enumerate(show.iterrows()):
        ax.plot(t, X[:, j] + j * step, color="#d62728" if ch["ap"] == bad else "k", lw=0.6)
        ax.text(-0.02, j * step, f"{int(ch['y'])} µm", ha="right", va="center", fontsize=7)
    ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_title("C  raw LFP (2 s)", loc="left", fontsize=10, fontweight="bold")
    for a_ in axs:
        for sp in ("top", "right"): a_.spines[sp].set_visible(False)
    fig.suptitle(f"Signal QC: example bad channel ({bad}, red) vs its neighbours (±{radius_um} µm); grey band in A = neighbour median ±2×",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, dev


def interference_per_rat(final, proc, ref="original", fmin=30, fmax=70):
    """Median spectrum (over channels) of every session, coloured by rat, 30–70 Hz (shows the fixed interference in rat 135)."""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white"); rat_of = final.set_index("session")["rat"]
    for s in final["session"]:
        f = qc_files(proc, s)[1]
        if not os.path.exists(f): continue
        z = np.load(f)
        ax.semilogy(z["freqs"], np.median(z[f"psd_{ref}"], axis=0), color=fxc.RATS[rat_of[s]], lw=0.6, alpha=0.7)
    ax.set_xlim(fmin, fmax); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (µV²/Hz), median over channels")
    ax.legend(handles=[plt.Line2D([], [], color=c, label=fxc.RAT_LABEL[r]) for r, c in fxc.RATS.items()], frameon=False, fontsize=9)
    ax.set_title("Median spectrum of every session (30–70 Hz): peaks at ~42.5 and ~57.5 Hz only in rat 135", fontsize=10, fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig
