"""
fx_preprocessing.py — functions of notebook 02_preprocessing

Code moved out of the notebook cells without changing what it computes. Nothing here saves a figure or
table (that is done in the notebook with fx_common); the only files written here are the preprocessed
recordings and their run logs, and only when the notebook's RUN switch is True.
"""
import os, re, glob, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fx_common as fxc

SUB = os.path.join("amir_data", "down_sampled", "1000_Hz")          # 1000 Hz folder inside each session
ALL_CH = [f"AP{i}" for i in range(384)]
chan_num = lambda c: int(re.sub(r"\D", "", c))


# ── STEP 1. preprocessing chain ────────────────────────────────────────────────────────────
def preprocess_session(s, raw_dir, proc, pilot, n_jobs=4, chunk="5s"):
    """raw ap.bin → bandpass 1–300 Hz → detect_bad_channels + remove → phase_shift → notch 50 Hz (q 30)
    → resample 1000 Hz → saved un-referenced in <session>/amir_data/down_sampled/1000_Hz, DONE.txt at the end."""
    import spikeinterface.full as si
    out = Path(proc) / s / SUB
    done = out / "DONE.txt"
    if done.exists() or (s == pilot and out.exists()):              # pilot already made with the same chain
        return None
    if out.exists():                                                # half-finished from a crash -> redo
        shutil.rmtree(out)
    ap_bin = glob.glob(os.path.join(raw_dir, s, "**", "*.ap.bin"), recursive=True)[0]
    t0 = time.time()
    raw = si.read_spikeglx(Path(ap_bin).parent, stream_name="imec0.ap")
    rec1 = si.bandpass_filter(raw, freq_min=1.0, freq_max=300, dtype="float32", ignore_low_freq_error=True)
    bad, labels = si.detect_bad_channels(rec1)
    rec2 = rec1.remove_channels(bad)
    rec3 = si.phase_shift(rec2)
    rec4 = si.notch_filter(rec3, freq=50, q=30)
    rec5 = si.resample(rec4, resample_rate=1000)
    rec5.save(folder=out, n_jobs=n_jobs, chunk_duration=chunk, progress_bar=True)
    done.write_text("ok")
    return {"session": s, "minutes": round((time.time() - t0) / 60, 1), "n_bad": len(bad), "bad_ids": " ".join(map(str, bad)),
            "labels": str(pd.Series(labels).value_counts().to_dict()), "n_channels_kept": rec5.get_num_channels()}


def preprocess_all(usable, raw_dir, proc, pilot, run=False, n_jobs=4, chunk="5s", run_log=None, err_log=None):
    """Lists the sessions still to do; processes them only if run=True (one broken session does not stop the run)."""
    todo = [s for s in usable["session"]
            if not (os.path.exists(os.path.join(proc, s, SUB, "DONE.txt")) or (s == pilot and os.path.isdir(os.path.join(proc, s, SUB))))]
    print(f"usable: {len(usable)} | already done: {len(usable) - len(todo)} | still to do: {len(todo)} {todo}")
    if not (run and todo):
        print("RUN = False -> nothing processed" if todo else "all usable sessions are preprocessed"); return todo
    t_all = time.time()
    for i, s in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {s}", flush=True)
        try:
            row = preprocess_session(s, raw_dir, proc, pilot, n_jobs, chunk)
        except Exception as e:
            print(f"  ERROR: {e}")
            if err_log: pd.DataFrame([{"session": s, "error": str(e)}]).to_csv(err_log, mode="a", header=not os.path.exists(err_log), index=False)
            continue
        print(f"  {row['minutes']} min | bad: {row['n_bad']} {row['bad_ids']} | {row['labels']}", flush=True)
        if run_log: pd.DataFrame([row]).to_csv(run_log, mode="a", header=not os.path.exists(run_log), index=False)
    print(f"\nFinished in {(time.time() - t_all) / 3600:.1f} h")
    return todo


# ── STEP 2. bad channels per session ───────────────────────────────────────────────────────
def bad_channel_log(usable, proc):
    """Bad channels = AP0–AP383 minus the channels kept in each saved 1000 Hz recording."""
    load = fxc.si_load(); rows = []
    for _, u in usable.iterrows():
        s = u["session"]
        try:
            rec = load(os.path.join(proc, s, SUB))                     # only reads metadata
            kept_ids = [str(c) for c in rec.get_channel_ids()]
            kept = {m.group(0) for c in kept_ids if (m := re.search(r"AP\d+", c))}
            bad = sorted(set(ALL_CH) - kept, key=lambda c: int(c[2:]))
            rows.append({"session": s, "rat": u["rat"], "n_bad": len(bad), "bad_ids": ";".join(bad),
                         "n_channels_kept": len(kept_ids), "example_id": kept_ids[0]})
        except Exception as e:
            rows.append({"session": s, "rat": u["rat"], "error": repr(e)})
    log = pd.DataFrame(rows)
    print("sessions read:", log["n_bad"].notna().sum(), "/", len(log))
    if "error" in log:
        print("errors:"); print(log.loc[log["error"].notna(), ["session", "error"]].to_string(index=False))
    print("example channel id as stored:", log["example_id"].dropna().iat[0])
    print("kept + bad == 384 in every session:", ((log["n_bad"] + log["n_channels_kept"]) == 384).all())
    print(log.groupby("rat")["n_bad"].agg(["min", "max"]))
    return log.drop(columns=["example_id"])


# ── STEP 3. bad channels per rat ───────────────────────────────────────────────────────────
def final_bad_lists(log, final):
    """The preprocessing log restricted to the final sessions, with the bad channels as a list."""
    check = log[log["session"].isin(final["session"])].copy()
    check["bad_list"] = check["bad_ids"].fillna("").apply(lambda x: [c for c in x.split(";") if c])
    return check


def bad_channel_frequency(log, final):
    """How often each channel was removed per rat, identical lists per rat, and the figure. Returns (freq, fig)."""
    check = final_bad_lists(log, final)
    print(f"final sessions in the log: {len(check)}  (expected 60)")
    print("parsed count == n_bad in every session:", (check["bad_list"].str.len() == check["n_bad"]).all())
    n_sess = check.groupby("rat")["session"].nunique()
    long = check[["rat", "session", "bad_list"]].explode("bad_list").dropna(subset=["bad_list"]).rename(columns={"bad_list": "channel"})
    freq = long.groupby(["rat", "channel"]).size().rename("n_sessions_bad").reset_index()
    freq["n_sessions_rat"] = freq["rat"].map(n_sess)
    freq["percent"] = (100 * freq["n_sessions_bad"] / freq["n_sessions_rat"]).round(1)
    freq["chan_num"] = freq["channel"].apply(chan_num)
    freq = freq.sort_values(["rat", "n_sessions_bad", "chan_num"], ascending=[True, False, True])
    print("\nhow often each channel is bad:"); print(freq.drop(columns="chan_num").to_string(index=False))
    check["bad_set"] = check["bad_list"].apply(lambda l: ", ".join(sorted(l, key=chan_num)) or "(none)")
    print("\nidentical bad-channel lists per rat (number of sessions):"); print(check.groupby(["rat", "bad_set"]).size().to_string())
    rats = sorted(check["rat"].unique())
    fig, axes = plt.subplots(2, len(rats), figsize=(6 * len(rats), 10), gridspec_kw={"height_ratios": [1, 2]}, squeeze=False)
    for j, rat in enumerate(rats):
        f = freq[freq["rat"] == rat].sort_values("chan_num"); c = check[check["rat"] == rat].sort_values("session")
        ax = axes[0, j]; ax.set_title(f"rat {rat} ({n_sess[rat]} sessions)")
        if f.empty:
            ax.text(0.5, 0.5, "no bad channels", ha="center", transform=ax.transAxes); axes[1, j].axis("off"); continue
        ax.bar(f["channel"], f["percent"], color="tab:red")
        for x, (p, n) in enumerate(zip(f["percent"], f["n_sessions_bad"])):
            ax.text(x, p + 1, str(n), ha="center", fontsize=8)                     # number of sessions above each bar
        ax.set_ylim(0, 110); ax.set_ylabel("% of sessions bad"); ax.tick_params(axis="x", rotation=90)
        M = np.array([[ch in l for ch in f["channel"]] for l in c["bad_list"]], dtype=int)
        ax = axes[1, j]
        ax.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(f))); ax.set_xticklabels(f["channel"], rotation=90, fontsize=8)
        ax.set_yticks(range(len(c))); ax.set_yticklabels(c["session"].str[:8] + "_" + c["session"].str[-4:], fontsize=7)
        ax.set_xlabel("bad channel"); ax.set_ylabel("session")
    fig.suptitle("Bad channels removed in preprocessing (detect_bad_channels), final sessions, per rat", fontsize=13)
    fig.tight_layout()
    return freq.drop(columns="chan_num"), fig


# ── STEP 4. all excluded channels per session ──────────────────────────────────────────────
def excluded_channels(log, final, xb):
    """Preprocessing bad channels ∪ channels found later (extra_bad_channels.csv), per final session."""
    check = final_bad_lists(log, final)
    xb = xb.copy(); xb["channel"] = xb["channel_id"].astype(str).str.extract(r"(AP\d+)")[0]
    later = xb.groupby("session")["channel"].apply(set)
    rows = []
    for _, r in check.iterrows():
        pre, lat = set(r["bad_list"]), later.get(r["session"], set())
        rows.append({"session": r["session"], "rat": r["rat"], "n_preproc": len(pre), "n_later": len(lat),
                     "n_later_already_preproc": len(lat & pre), "n_total": len(pre | lat), "n_used": 384 - len(pre | lat),
                     "later_ids": ";".join(sorted(lat - pre, key=chan_num))})
    exc = pd.DataFrame(rows)
    print("per rat, per session: median [min–max]")
    for rat, d in exc.groupby("rat"):
        f = lambda c: f"{d[c].median():.0f} [{d[c].min()}–{d[c].max()}]"
        print(f"  rat {rat}: {len(d)} sessions | preprocessing {f('n_preproc')} | later {f('n_later')} | total {f('n_total')} | used {f('n_used')} of 384")
    print("\nlater-found channels that were already removed in preprocessing:", int(exc["n_later_already_preproc"].sum()))
    print("\nlater-found channels (final sessions) by rat and channel:")
    xbf = xb[xb["session"].isin(final["session"])].merge(final[["session", "rat"]], on="session")
    print(xbf.groupby(["rat", "channel"]).size().to_string())
    print("\nsessions in extra_bad_channels.csv that are not final sessions:", sorted(set(xb["session"]) - set(final["session"])))
    return exc


# ── STEP 5. example of a removed channel ───────────────────────────────────────────────────
def example_bad_channel(raw_dir, ses, bad, removed, n_neighbours=4, radius_um=75, seconds=2):
    """Raw LFP (bandpass 1–300 Hz, 1000 Hz) of a removed channel (red) and its nearest good neighbours on the same shank."""
    import spikeinterface.full as si
    ap_bin = glob.glob(os.path.join(raw_dir, ses, "**", "*.ap.bin"), recursive=True)[0]
    raw = si.read_spikeglx(Path(ap_bin).parent, stream_name="imec0.ap")
    rec = si.resample(si.bandpass_filter(raw, freq_min=1.0, freq_max=300, dtype="float32", ignore_low_freq_error=True), resample_rate=1000)
    ids = np.array([str(c) for c in rec.get_channel_ids()]); loc = rec.get_channel_locations()
    ap = pd.Series(ids).str.extract(r"(AP\d+)")[0].values
    shank = np.round(loc[:, 0] / 250).astype(int)
    k = int(np.flatnonzero(ap == bad)[0])
    cand = np.flatnonzero((shank == shank[k]) & (np.abs(loc[:, 1] - loc[k, 1]) <= radius_um) & ~np.isin(ap, list(removed)))
    near = cand[np.argsort(np.abs(loc[cand, 1] - loc[k, 1]))[:n_neighbours]]
    show = np.r_[near, k][np.argsort(loc[np.r_[near, k], 1])]                        # sorted by depth
    fs = rec.get_sampling_frequency(); a = int(rec.get_num_samples() // 2)
    X = rec.get_traces(start_frame=a, end_frame=a + int(seconds * fs), channel_ids=list(ids[show]), return_in_uV=True).astype(float)
    X = X - X.mean(0); step = 4 * np.median(X.std(0)); tt = np.arange(X.shape[0]) / fs
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    for j, c in enumerate(show):
        ax.plot(tt, X[:, j] + j * step, color="#d62728" if c == k else "k", lw=0.6)
        ax.text(-0.02, j * step, f"{int(loc[c, 1])} µm", ha="right", va="center", fontsize=8)
    ax.set_yticks([]); ax.set_xlabel("time (s)")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig
