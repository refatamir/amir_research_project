"""
fx_events.py — functions of notebook 05_event_check (event-related potentials: alignment and bad channels)

Code moved out of the notebook cells without changing what it computes. It uses the building blocks of functions.py
(load_session, probe_table, event_erp, erp_session, alignment_sharpness, erp_outliers).
The per-session ERP files in 2_processed\\<session>\\amir_data\\ERP are only written when the loop is run (RUN = True).
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import functions as fn

REFS = ["original", "CMR"]


def erp_files(proc, s, local=False):
    """(folder, ERP file, outlier table, sharpness table) of one session; local=True -> LOCAL-reference files."""
    d = os.path.join(proc, s, "amir_data", "ERP")
    sfx = "_local" if local else ""
    return d, os.path.join(d, f"erps{sfx}.npz"), os.path.join(d, f"erp_checks{sfx}.csv"), os.path.join(d, f"erp_sharpness{sfx}.csv")


def _events(tr):
    both = tr["press_from_ttl"] & tr["press_np_s"].notna()                    # trials with a TTL AND a behaviour press time
    return {"Press": tr["press_neural_s"].values, "Press_TTL": tr.loc[both, "press_ttl_np_s"].values,
            "Press_MA": tr.loc[both, "press_np_s"].values,
            "Tone": tr.loc[tr["label"].isin(["correct", "late"]), "tone_neural_s"].values, "Release": tr["release_neural_s"].values}


def run_event_check(final, K, proc, xb, run=False, overwrite=False, err_log=None):
    """original + CMR: fn.erp_session per session (ERPs −1..+1 s at Press, Press TTL, Press MA, Tone, Release; TTL vs MA
    sharpness; ERP outliers in CMR). Writes the per-session files only if run=True."""
    done = [s for s in final["session"] if os.path.exists(erp_files(proc, s)[3])]
    todo = [s for s in final["session"] if overwrite or s not in done]
    print(f"final sessions: {len(final)} | event check done: {len(done)} | to do: {len(todo)}")
    if not run: return todo
    t_all = time.time(); rat_of = final.set_index("session")["rat"]
    for i, s in enumerate(todo, 1):
        t0 = time.time()
        try:
            recs, _ = fn.load_session(proc, s, xb)
            e, sharp, outl = fn.erp_session(recs, K[K["session"] == s])
            d, f_npz, f_chk, f_sh = erp_files(proc, s); os.makedirs(d, exist_ok=True)
            first = next(iter(e.values()))
            np.savez_compressed(f_npz, times=first["times"], channel_ids=np.array(first["channel_ids"]),
                                **{f"{ref}__{name}": v["mean"].astype(np.float32) for (ref, name), v in e.items()},
                                **{f"n__{ref}__{name}": v["n"] for (ref, name), v in e.items()})
            for t in (outl, sharp): t.insert(0, "session", s); t.insert(1, "rat", rat_of[s])
            outl.to_csv(f_chk, index=False); sharp.to_csv(f_sh, index=False)          # sharpness written last = session finished
            print(f"[{i}/{len(todo)}] {s}: {(time.time() - t0) / 60:.1f} min | TTL/MA {sharp.set_index('ref')['ttl_ma_ratio_median'].round(3).to_dict()} | "
                  f"ERP outliers {outl.groupby('event')['flag_local'].sum().to_dict()}", flush=True)
        except Exception as err:
            print(f"[{i}/{len(todo)}] {s}: ERROR {err!r}", flush=True)
            if err_log: pd.DataFrame([{"session": s, "error": repr(err)}]).to_csv(err_log, mode="a", header=not os.path.exists(err_log), index=False)
    print(f"\nfinished in {(time.time() - t_all) / 60:.0f} min")
    return todo


def run_event_check_local(final, K, proc, xb, run=False, overwrite=False):
    """Same as run_event_check on the LOCAL reference (building blocks called directly, because fn.erp_session checks CMR only)."""
    done = [s for s in final["session"] if os.path.exists(erp_files(proc, s, True)[3])]
    todo = [s for s in final["session"] if overwrite or s not in done]
    print(f"LOCAL event check: {len(done)} done, {len(todo)} to do", flush=True)
    if not run: return todo
    rat_of = final.set_index("session")["rat"]; t_all = time.time()
    for i, s in enumerate(todo, 1):
        t0 = time.time()
        try:
            rl = fn.load_session(proc, s, xb, refs=("LOCAL",))[0]["LOCAL"]; pdf = fn.probe_table(rl)
            e = {name: fn.event_erp(rl, ts, tmin=-1.0, tmax=1.0) for name, ts in _events(K[K["session"] == s]).items()}
            shp = fn.alignment_sharpness(e["Press_TTL"], e["Press_MA"])
            sharp = pd.DataFrame([{"session": s, "rat": rat_of[s], "ref": "LOCAL", "n_trials": e["Press_TTL"]["n"],
                                   "ttl_ma_ratio_median": shp["ratio_a_b"].median(), "pct_channels_ttl_larger": 100 * (shp["ratio_a_b"] > 1).mean()}])
            outl = []
            for name in ("Press", "Tone", "Release"):
                o = fn.erp_outliers(e[name], pdf); o.insert(0, "event", name); outl.append(o)
            outl = pd.concat(outl, ignore_index=True); outl.insert(0, "session", s); outl.insert(1, "rat", rat_of[s])
            d, f_npz, f_chk, f_sh = erp_files(proc, s, True); os.makedirs(d, exist_ok=True)
            np.savez_compressed(f_npz, times=e["Press"]["times"], channel_ids=np.array(e["Press"]["channel_ids"]),
                                **{f"LOCAL__{k}": v["mean"].astype(np.float32) for k, v in e.items()}, **{f"n__LOCAL__{k}": v["n"] for k, v in e.items()})
            outl.to_csv(f_chk, index=False); sharp.to_csv(f_sh, index=False)
            print(f"[{i}/{len(todo)}] {s}: {(time.time() - t0) / 60:.1f} min | TTL/MA {sharp['ttl_ma_ratio_median'].iloc[0]:.3f} | "
                  f"ERP outliers {outl.groupby('event')['flag_local'].sum().to_dict()}", flush=True)
        except Exception as err:
            print(f"[{i}/{len(todo)}] {s}: ERROR {err!r}", flush=True)
    print(f"\nfinished in {(time.time() - t_all) / 60:.0f} min", flush=True)
    return todo


def collect_erp(final, proc, local=False):
    """All finished sessions: (sharpness table, outlier table)."""
    fin = [s for s in final["session"] if os.path.exists(erp_files(proc, s, local)[3])]
    if not fin:
        print("no event-check results yet (run the loop with RUN = True)"); return None, None
    SH = pd.concat([pd.read_csv(erp_files(proc, s, local)[3], dtype={"rat": str}) for s in fin], ignore_index=True)
    OU = pd.concat([pd.read_csv(erp_files(proc, s, local)[2], dtype={"rat": str}) for s in fin], ignore_index=True)
    print(f"collected: {len(fin)} sessions")
    return SH, OU


def event_summary(SH, OU, QC, final):
    """1 TTL vs MA per rat and reference (Wilcoxon of log ratio vs 0); 2 channels flagged in all 3 events (CMR);
    3 repeat offenders by readout channel; 4 also a PSD outlier in 04? Returns the candidate table."""
    print(f"sessions: {SH['session'].nunique()} | sessions without TTL/MA value: {SH.loc[SH['ttl_ma_ratio_median'].isna(), 'session'].unique().tolist()}")
    rows = []
    for (rat, ref), d in list(SH.groupby(["rat", "ref"])) + [(("all", r), SH[SH["ref"] == r]) for r in REFS]:
        v = np.log10(d["ttl_ma_ratio_median"].dropna())
        rows.append({"rat": rat, "ref": ref, "sessions": len(v), "median_ratio": round(10 ** v.median(), 3),
                     "sessions_TTL_sharper": int((v > 0).sum()), "wilcoxon_p": round(wilcoxon(v).pvalue, 4) if len(v) >= 5 else np.nan})
    print("\n1. TTL / MA (median ERP size ratio per session; > 1 = TTL sharper):"); print(pd.DataFrame(rows).to_string(index=False))
    OU = OU.copy(); OU["ap"] = OU["channel_id"].astype(str).str.extract(r"(AP\d+)")[0]
    all3 = (OU[OU["flag_local"]].groupby(["rat", "session", "channel_id", "ap", "shank", "y"])["event"].nunique()
            .reset_index().query("event == 3").drop(columns="event"))
    all3 = all3.merge(OU.groupby(["session", "channel_id"])["local_dev"].median().rename("median_dev").reset_index(), on=["session", "channel_id"])
    print(f"\n2. channels flagged in all 3 events: {len(all3)} (in {all3['session'].nunique()} sessions)")
    rep = (all3.groupby(["rat", "ap"]).agg(n_sessions=("session", "nunique"), median_dev=("median_dev", "median"),
                                           positions=("y", lambda v: sorted(set(v.astype(int))))).reset_index()
           .sort_values(["rat", "n_sessions"], ascending=[True, False]))
    rep["of_sessions"] = rep["rat"].map(final.groupby("rat").size())
    print("\n3. readout channels flagged in all 3 events, by rat (number of sessions):")
    print(rep[rep["n_sessions"] >= 2].to_string(index=False) if (rep["n_sessions"] >= 2).any() else "none in >= 2 sessions")
    qc_c = QC[QC["ref"] == "CMR"][["session", "channel_id", "flag_local", "local_dev"]].rename(columns={"flag_local": "psd_flag_cmr", "local_dev": "psd_dev_cmr"})
    qc_o = QC[QC["ref"] == "original"][["session", "channel_id", "flag_local"]].rename(columns={"flag_local": "psd_flag_orig"})
    cand = all3.merge(qc_c, on=["session", "channel_id"], how="left").merge(qc_o, on=["session", "channel_id"], how="left")
    print(f"\n4. of the {len(cand)} all-3-event channels: also PSD outlier in CMR {int(cand['psd_flag_cmr'].sum())}, in original {int(cand['psd_flag_orig'].sum())}")
    return cand


def event_local_summary(SL, OL, SO, OC):
    """TTL / MA for original, CMR and LOCAL; channels flagged in all 3 events, CMR vs LOCAL."""
    fin = SL["session"].unique()
    SA = pd.concat([SO[SO["session"].isin(fin)], SL], ignore_index=True)
    print(f"\n1. TTL / MA press ERP size ({len(fin)} sessions; median over sessions, > 1 = TTL sharper):")
    print(SA.groupby(["ref"])["ttl_ma_ratio_median"].agg(["median", "size"]).round(3).to_string())
    for ref, O in (("CMR", OC[OC["session"].isin(fin)]), ("LOCAL", OL)):
        a3 = O[O["flag_local"]].groupby(["rat", "session", "channel_id"])["event"].nunique().reset_index().query("event == 3")
        a3["ap"] = a3["channel_id"].astype(str).str.extract(r"(AP\d+)")[0]
        rep = a3.groupby(["rat", "ap"]).size().rename("sessions").reset_index().sort_values(["rat", "sessions"], ascending=[True, False])
        print(f"\n2. {ref}: channels flagged in all 3 events: {len(a3)} (channel × session); readout channels in ≥ 2 sessions:")
        print(rep[rep["sessions"] >= 2].to_string(index=False) if (rep["sessions"] >= 2).any() else "  none")


def candidate_check(cands, cand_table, qc_all, final, proc, xb):
    """For each candidate channel, in the session where it was flagged most: power and ERP size vs neighbours (±75 µm),
    correlation with the neighbours and RMS (60 s of raw signal), original and CMR. Returns (table, {channel: [figs]})."""
    def pick_session(ap):
        c = cand_table[cand_table["ap"] == ap]
        if len(c): return c.sort_values("median_dev", ascending=False)["session"].iloc[0], "ERP all 3 events"
        q = qc_all[(qc_all["ref"] == "CMR") & qc_all["flag_local"] & qc_all["channel_id"].str.endswith("#" + ap)]
        return q.sort_values("local_dev", ascending=False)["session"].iloc[0], "QC PSD (CMR)"
    rows, figs = [], {}
    for ap in cands:
        s, why = pick_session(ap)
        recs, _ = fn.load_session(proc, s, xb); pdf = fn.probe_table(recs["original"])
        ch = pdf[pdf["channel_id"].str.endswith("#" + ap)].iloc[0]
        nb = pdf[(pdf["shank"] == ch["shank"]) & ((pdf["y"] - ch["y"]).abs() <= 75) & (pdf["channel_id"] != ch["channel_id"])]
        ids = [ch["channel_id"]] + nb["channel_id"].tolist()
        zq = np.load(os.path.join(proc, s, "amir_data", "QC", "psd.npz"), allow_pickle=True); ze = np.load(erp_files(proc, s)[1], allow_pickle=True)
        qi, ei = [str(c) for c in zq["channel_ids"]], [str(c) for c in ze["channel_ids"]]
        row = {"channel": ap, "rat": final.set_index("session").loc[s, "rat"], "session": s, "picked_from": why,
               "shank": int(ch["shank"]), "y": int(ch["y"]), "neighbours": len(nb)}
        for ref in REFS:
            lp = np.log10(zq[f"psd_{ref}"].mean(axis=1))
            row[f"power_x_{ref}"] = 10 ** (lp[qi.index(ids[0])] - np.median([lp[qi.index(c)] for c in ids[1:]]))
            for ev in ["Press", "Tone", "Release"]:
                es = np.log10(np.nanstd(ze[f"{ref}__{ev}"], axis=0))
                row[f"erp_{ev}_x_{ref}"] = 10 ** (es[ei.index(ids[0])] - np.median([es[ei.index(c)] for c in ids[1:]]))
        fs = recs["original"].get_sampling_frequency(); mid = recs["original"].get_num_samples() // 2
        for ref in REFS:
            x = recs[ref].get_traces(start_frame=mid, end_frame=mid + int(60 * fs), channel_ids=ids, return_in_uV=True).astype(float)
            C = np.corrcoef(x.T)
            row[f"corr_ch_nb_{ref}"] = np.median(C[0, 1:]); row[f"corr_nb_nb_{ref}"] = np.median(C[1:, 1:][np.triu_indices(len(ids) - 1, 1)])
            row[f"rms_x_{ref}"] = x[:, 0].std() / np.median(x[:, 1:].std(axis=0))
        rows.append(row)
        f1, axes = plt.subplots(2, 2, figsize=(11, 7))
        for j, ref in enumerate(REFS):
            ax = axes[0, j]
            for c in ids[1:]: ax.loglog(zq["freqs"], zq[f"psd_{ref}"][qi.index(c)], color="0.6", lw=0.8)
            ax.loglog(zq["freqs"], zq[f"psd_{ref}"][qi.index(ids[0])], color="red", lw=1.5); ax.set_title(f"PSD ({ref})"); ax.set_xlabel("Hz"); ax.set_ylabel("µV²/Hz")
            ax = axes[1, j]
            for c in ids[1:]: ax.plot(ze["times"], ze[f"{ref}__Press"][:, ei.index(c)], color="0.6", lw=0.8)
            ax.plot(ze["times"], ze[f"{ref}__Press"][:, ei.index(ids[0])], color="red", lw=1.5)
            ax.axvline(0, color="k", ls="--", lw=0.8); ax.set_title(f"Press ERP ({ref})"); ax.set_xlabel("Time from press (s)"); ax.set_ylabel("µV")
        f1.suptitle(f"{s} — {ap} (red) vs {len(nb)} neighbours ±75 µm (grey), shank {int(ch['shank'])}, y {int(ch['y'])} µm", fontweight="bold"); f1.tight_layout()
        f2, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, ref in zip(axes, REFS):
            x = recs[ref].get_traces(start_frame=mid, end_frame=mid + int(4 * fs), channel_ids=ids, return_in_uV=True).astype(float)
            x -= x.mean(axis=0); sp = 4 * np.median(x.std(axis=0)); tt = np.arange(len(x)) / fs
            for i in range(x.shape[1]): ax.plot(tt, x[:, i] + i * sp, color="red" if i == 0 else "0.4", lw=0.6)
            ax.set_title(f"raw LFP ({ref}), red = {ap}"); ax.set_xlabel("s"); ax.set_yticks([])
        f2.tight_layout(); figs[ap] = [f1, f2]
        print(f"{ap}: {s} done", flush=True)
    res = pd.DataFrame(rows); print("\n" + res.round(2).T.to_string())
    return res, figs


def new_bad_channels(new, xb, prep_log, usable):
    """Rows to append to extra_bad_channels.csv: each new channel in every usable session of its rat, unless already listed
    or removed in preprocessing. Returns (rows to add, new full table)."""
    prefix = "imec0.ap#" if xb["channel_id"].astype(str).str.contains("#").any() else ""
    have = set(zip(xb["session"], xb["channel_id"].astype(str).str.extract(r"(AP\d+)")[0]))
    pre = {(r.session, c) for r in prep_log.itertuples() for c in str(r.bad_ids).split(";") if c}
    add = pd.DataFrame([{"session": s, "channel_id": prefix + ap, "reason": reason} for ap, (rat, reason) in new.items()
                        for s in usable.loc[usable["rat"] == rat, "session"] if (s, ap) not in have and (s, ap) not in pre])
    print(f"rows before: {len(xb)} | rows to add: {len(add)}")
    print(add.groupby("channel_id").size().to_string() if len(add) else "nothing to add")
    return add, (pd.concat([xb, add[xb.columns.intersection(add.columns)]], ignore_index=True) if len(add) else xb)


def example_erp(proc, xb, trials, ses, ev_win=None, base=(-0.5, -0.1), xl=(-0.5, 0.5)):
    """Picks the channel with the clearest press, tone and release response (saved mean ERPs: peak in the response window /
    SD of the baseline, the worst of the 3 events), then plots its ERP (original) as mean ± SD across trials. Returns (fig, channel, rec, event times)."""
    ev_win = ev_win or {"Press": (0.0, 0.4), "Tone": (0.0, 0.3), "Release": (-0.1, 0.3)}
    z = np.load(erp_files(proc, ses)[1]); t0 = z["times"]; bm = (t0 >= base[0]) & (t0 <= base[1])
    snr = [np.nanmax(np.abs(z[f"original__{ev}"][(t0 >= a) & (t0 <= b)]), axis=0) / np.nanstd(z[f"original__{ev}"][bm], axis=0) for ev, (a, b) in ev_win.items()]
    ch = str(z["channel_ids"][int(np.nanargmax(np.min(np.vstack(snr), axis=0)))])
    rec = fn.load_session(proc, ses, xb, refs=("original",))[0]["original"]
    tr = trials[trials["keep"] & (trials["session"] == ses)]
    ev_t = {"Press": tr["press_neural_s"].values, "Tone": tr.loc[tr["label"].isin(["correct", "late"]), "tone_neural_s"].values,
            "Release": tr["release_neural_s"].values}
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", sharey=True)
    for ax, (ev, ts) in zip(axs, ev_t.items()):
        e = fn.event_erp(rec, ts, channel_ids=[ch], tmin=xl[0], tmax=xl[1]); m = e["mean"][:, 0]; sd = e["sem"][:, 0] * np.sqrt(e["n"])
        ax.fill_between(e["times"], m - sd, m + sd, color="0.8", lw=0); ax.plot(e["times"], m, color="k", lw=1.5)
        ax.axvline(0, color="#d62728", lw=1); ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlim(*xl); ax.set_xlabel(f"time from {ev.lower()} (s)"); ax.set_title(ev, fontsize=10, fontweight="bold")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("LFP (µV), original")
    fig.suptitle(f"Event-locked LFP, channel {ch.split('#')[-1]}, session {ses}: mean ± SD across trials", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, ch, rec, ev_t


def erp_delta_part(rec, ch, ev_t, ses, bands=None, win=(0, 0.4)):
    """Mean ERP of one channel (−1..+1 s) and its delta part (1–4 Hz Butterworth, zero phase); peak-to-peak in 0–0.4 s."""
    import scipy.signal as ss
    import fx_common as fxc
    bands = bands or {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (15, 30)}
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), facecolor="white", sharex=True)
    for j, (ev, ts) in enumerate(ev_t.items()):
        e = fn.event_erp(rec, ts, channel_ids=[ch], tmin=-1.0, tmax=1.0)
        m, tt = e["mean"][:, 0], e["times"]; fs = 1 / np.median(np.diff(tt))
        bp = {bn: ss.sosfiltfilt(ss.butter(3, [lo, hi], btype="bandpass", fs=fs, output="sos"), m) for bn, (lo, hi) in bands.items()}
        w = (tt >= win[0]) & (tt <= win[1]); p_all, p_del = np.ptp(m[w]), np.ptp(bp["delta"][w])
        axs[0, j].plot(tt, m, color="k", lw=1.4); axs[1, j].plot(tt, bp["delta"], color=fxc.BAND["delta"], lw=1.8)
        axs[0, j].set_title(f"{ev}\nERP: {p_all:.0f} µV", fontsize=10, fontweight="bold")
        axs[1, j].set_title(f"delta: {p_del:.0f} µV ({100 * p_del / p_all:.0f} %)", fontsize=9, color=fxc.BAND["delta"])
        axs[1, j].set_xlabel(f"time from {ev.lower()} (s)")
        print(f"{ev}: peak-to-peak {win[0]}–{win[1]} s, full ERP {p_all:.0f} µV |", {bn: f"{np.ptp(v[w]):.0f} µV" for bn, v in bp.items()})
    for a in axs.ravel():
        a.axvline(0, color="#d62728", lw=1); a.axhline(0, color="0.5", lw=0.5); a.set_xlim(-0.5, 0.5)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    axs[0, 0].set_ylabel("ERP (µV)"); axs[1, 0].set_ylabel("delta 1–4 Hz (µV)")
    fig.suptitle(f"Event-locked LFP and its delta part, channel {ch.split('#')[-1]}, {ses} (original)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.5)
    return fig
