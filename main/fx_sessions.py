"""
fx_sessions.py — functions of notebook 01_sessions_setup

Every function here is the code that used to sit inside a cell of 01, moved out unchanged in its logic.
Each function returns its result tables (and figures); printing is kept so the notebook shows the same
checks as before. Nothing in this file saves anything: saving is done in the notebook with fx_common.
"""
import os, glob, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import fx_common as fxc


def d1000(proc, s):
    """Folder of the 1000 Hz (down-sampled, no reference) recording of one session."""
    return os.path.join(proc, s, "amir_data", "down_sampled", "1000_Hz")


# ── STEP 1. data inventory ──────────────────────────────────────────────────────────────────
def list_folder(folder):
    """Files of a folder with their size in MB, as a table."""
    return pd.DataFrame([{"file": f, "MB": round(os.path.getsize(os.path.join(folder, f)) / 1e6, 2)}
                         for f in sorted(os.listdir(folder))])


def data_inventory(proc, example):
    """Shows the 1000 Hz folder and the share folder of one example session, then counts the sessions."""
    load = fxc.si_load()
    hz, share = d1000(proc, example), os.path.join(proc, example, "share")
    print(f"1000 Hz folder of {example}:"); display(list_folder(hz))
    rec = load(hz)                                                 # reads only the metadata, not the traces
    dur = rec.get_total_duration()
    print(f"\nrecording: {rec.get_sampling_frequency():.0f} Hz | {rec.get_num_channels()} channels "
          f"({384 - rec.get_num_channels()} of 384 removed) | {dur/60:.1f} min")
    print(f"\nshare folder of {example}:"); display(list_folder(share))
    beh = pd.read_csv(os.path.join(share, "trials.csv"))
    print(f"\ntrials.csv: {len(beh)} trials, {beh.shape[1]} columns")
    display(beh[["press_np_s", "press_ttl_np_s", "release_np_s", "trial_end_np_s"]].agg(["min", "max"]).round(1))
    print(f"recording length: {dur:.1f} s")
    sess = [s for s in sorted(os.listdir(proc)) if os.path.isdir(os.path.join(proc, s))]
    has_folder = [s for s in sess if os.path.isdir(d1000(proc, s))]
    has_done = [s for s in sess if os.path.exists(os.path.join(d1000(proc, s), "DONE.txt"))]
    has_trials = [s for s in sess if os.path.exists(os.path.join(proc, s, "share", "trials.csv"))]
    counts = pd.DataFrame({"sessions": [len(sess), len(has_folder), len(has_done), len(has_trials)]},
                          index=["folders in 2_processed", "with a 1000_Hz folder", "with DONE.txt", "with share/trials.csv"])
    print(); display(counts)
    print("1000_Hz folder but no DONE.txt:", sorted(set(has_folder) - set(has_done)))
    return counts


# ── STEP 2. session list ────────────────────────────────────────────────────────────────────
def session_inventory(raw, proc):
    """Every item in 0_raw: ap.bin, trials.csv, synced trials.csv, DONE.txt, electrode map, sync info.
    Usable = has an ap.bin and a synced trials.csv (column press_np_s). Returns (sessions, usable)."""
    rows = []
    for s in sorted(os.listdir(raw)):
        ap = glob.glob(os.path.join(raw, s, "**", "*.ap.bin"), recursive=True)
        sh = os.path.join(proc, s, "share", "trials.csv")
        has_trials = os.path.exists(sh)
        ok_sync = has_trials and "press_np_s" in pd.read_csv(sh, nrows=0).columns
        rat = re.search(r"r0+(\d+)", s)
        imro = ""
        meta = glob.glob(os.path.join(raw, s, "**", "*.ap.meta"), recursive=True)
        if meta:
            with open(meta[0]) as fh:
                m = dict(l.strip().split("=", 1) for l in fh if "=" in l)
            imro = m.get("~imroTbl", "")
        row = {"session": s, "rat": rat.group(1) if rat else None, "n_ap_bin": len(ap),
               "ap_GB": round(sum(os.path.getsize(a) for a in ap) / 1e9, 1),
               "has_trials_csv": has_trials, "synced_trials_csv": bool(ok_sync),
               "has_1000_done": os.path.exists(os.path.join(d1000(proc, s), "DONE.txt")), "imro": imro}
        info_file = os.path.join(proc, s, "share", "session_info.json")
        if os.path.exists(info_file):
            with open(info_file) as fh:
                info = pd.json_normalize(json.load(fh), sep=".").iloc[0]
            row["sync_ok"] = info.get("sync.sync_ok")
            row["sync_coverage"] = info.get("sync.coverage_frac")
            row["waiting_times"] = str(info.get("behaviour.waiting_times_s"))
        rows.append(row)
    sessions = pd.DataFrame(rows)
    usable = sessions[(sessions["n_ap_bin"] > 0) & sessions["synced_trials_csv"]].copy()
    usable["map_id"] = usable["imro"].astype("category").cat.codes
    sessions["reason"] = np.where(sessions["n_ap_bin"] == 0, "no ap.bin",
                         np.where(~sessions["has_trials_csv"], "no trials.csv",
                         np.where(~sessions["synced_trials_csv"], "trials.csv not synced", "usable")))
    print(f"items in 0_raw: {len(sessions)} | usable sessions: {len(usable)}  (expected 63)")
    print(usable.groupby("rat").size().to_string())
    print("\nall items in 0_raw by reason:"); print(sessions["reason"].value_counts().to_string())
    print("\n--- sync check among the usable sessions ---")
    print("sync_ok values:", usable["sync_ok"].value_counts(dropna=False).to_dict())
    display(usable.loc[usable["sync_ok"] != True, ["session", "rat", "sync_ok", "sync_coverage"]])
    print("\n--- foreperiod design among the usable sessions ---")
    print(usable.groupby(["rat", "waiting_times"]).size().to_string())
    return (sessions.drop(columns=["imro"]),
            usable.drop(columns=["imro", "has_trials_csv", "has_1000_done"]))


# ── STEP 3. completeness of the 1000 Hz recordings ─────────────────────────────────────────
def check_1000hz(usable, proc):
    """Per session: file size = channels × samples × bytes, and the last 5 s are not zero in any channel."""
    load = fxc.si_load()
    rows = []
    for _, u in usable.iterrows():
        s, f = u["session"], d1000(proc, u["session"])
        row = {"session": s, "rat": u["rat"], "has_done": os.path.exists(os.path.join(f, "DONE.txt"))}
        try:
            rec = load(f)                                              # reads only the metadata
            n_samp = (getattr(rec, "get_num_samples", None) or rec.get_num_frames)()
            n_ch = rec.get_num_channels()
            expected = n_ch * n_samp * np.dtype(rec.get_dtype()).itemsize
            actual = os.path.getsize(os.path.join(f, "traces_cached_seg0.raw"))
            tail = rec.get_traces(start_frame=n_samp - 5000, end_frame=n_samp)     # last 5 s at 1000 Hz
            row.update({"n_channels": n_ch, "minutes": round(n_samp / rec.get_sampling_frequency() / 60, 1),
                        "expected_GB": round(expected / 1e9, 3), "actual_GB": round(actual / 1e9, 3),
                        "size_ok": expected == actual, "tail_ok": bool((tail.std(axis=0) > 0).all())})
        except Exception as e:
            row.update({"size_ok": False, "tail_ok": False, "error": repr(e)[:120]})
        rows.append(row)
    chk = pd.DataFrame(rows)
    chk["complete"] = chk["size_ok"] & chk["tail_ok"]
    bad, no_done = chk[~chk["complete"]], chk[~chk["has_done"]]
    print(f"complete (size and last 5 s): {int(chk['complete'].sum())} of {len(chk)}")
    print(f"not complete: {len(bad)}")
    if len(bad): display(bad)
    print(f"sessions without DONE.txt (the save did not finish or an error happened): {len(no_done)}")
    if len(no_done): display(no_done)
    return chk


# ── STEP 4. session flags ──────────────────────────────────────────────────────────────────
def session_flags(usable, proc, short_wt=0.3, long_wt=1.0):
    """Which wt values each session contains -> two_wt / single_wt / wt_manip, include, short_vs_long."""
    rows = []
    for _, u in usable.iterrows():
        s = u["session"]
        w = np.round(pd.read_csv(os.path.join(proc, s, "share", "trials.csv"), usecols=["wt"])["wt"].astype(float), 2)
        typ = pd.Series(np.where(w == short_wt, "short", np.where(w == long_wt, "long", np.where(w.isna(), "missing", "other_wt"))))
        n = typ.value_counts()
        rows.append({"session": s, "rat": u["rat"], "wt_values": sorted(w.dropna().unique()), "n_trials": len(w),
                     "n_short": int(n.get("short", 0)), "n_long": int(n.get("long", 0)),
                     "n_other_wt": int(n.get("other_wt", 0)), "n_wt_missing": int(n.get("missing", 0)),
                     "sync_ok": u["sync_ok"], "sync_coverage": u["sync_coverage"]})
    flags = pd.DataFrame(rows)
    flags["two_wt_session"] = flags["wt_values"].apply(lambda v: short_wt in v and long_wt in v)
    flags["single_wt_session"] = flags["wt_values"].apply(lambda v: long_wt in v and short_wt not in v)
    flags["wt_manip_session"] = flags["wt_values"].apply(lambda v: any(x not in (short_wt, long_wt) for x in v))
    flags["wt_values"] = flags["wt_values"].apply(lambda v: " ".join(map(str, v)))
    flags["include"] = ~flags["wt_manip_session"]                              # all analyses
    flags["short_vs_long"] = flags["include"] & flags["two_wt_session"]        # only short-vs-long analyses
    flags["exclude_reason"] = np.where(flags["wt_manip_session"], "WT manipulation", "")
    print("design (wt values) per rat:"); print(flags.groupby(["rat", "wt_values"]).size().to_string())
    print("\nflags per rat:")
    print(flags.groupby("rat")[["two_wt_session", "single_wt_session", "wt_manip_session", "include", "short_vs_long"]].sum().to_string())
    print("\ntotal:", int(flags["include"].sum()), "sessions included |", int(flags["short_vs_long"].sum()), "for short vs long")
    print("wt_manip sessions:", flags.loc[flags["wt_manip_session"], "session"].tolist())
    print("single-wt (only 1.0) sessions:", flags.loc[flags["single_wt_session"], "session"].tolist())
    print("sessions with missing wt:", flags.loc[flags["n_wt_missing"] > 0, "session"].tolist())
    return flags


# ── STEP 5. sync details ───────────────────────────────────────────────────────────────────
def sync_details(usable, proc):
    """The 'sync' section of session_info.json for every usable session."""
    rows = []
    for s in usable["session"]:
        with open(os.path.join(proc, s, "share", "session_info.json")) as fh:
            sync = json.load(fh).get("sync", {})
        rows.append({"session": s, **{k: (v if np.isscalar(v) else str(v)) for k, v in sync.items()}})
    tab = pd.DataFrame(rows).merge(usable[["session", "rat"]], on="session")
    print("columns in the sync section:", [c for c in tab.columns if c not in ("session", "rat")])
    show = [c for c in ["session", "rat", "sync_ok", "coverage_frac", "n_matched_pulses", "lag_median_s",
                        "lag_std_s", "n_trials_outside_span"] if c in tab.columns]
    print("\nsessions with sync_ok False (lowest coverage first):")
    display(tab.loc[tab["sync_ok"] != True, show].sort_values("coverage_frac"))
    num = [c for c in ["coverage_frac", "lag_std_s", "n_trials_outside_span"] if c in tab.columns]
    print("\nsync_ok True vs False (min / median / max):")
    display(tab.groupby("sync_ok")[num].agg(["min", "median", "max"]).round(4))
    return tab


# ── STEP 6. final session list ─────────────────────────────────────────────────────────────
def final_sessions(usable, flags):
    """Usable sessions with include == True (= all except the WT-manipulation sessions)."""
    cols = ["session", "two_wt_session", "single_wt_session", "wt_manip_session", "include", "short_vs_long", "exclude_reason"]
    allf = usable.merge(flags[cols], on="session", how="left", validate="1:1")
    final = allf[allf["include"] == True].reset_index(drop=True)
    print(f"usable: {len(allf)} | excluded: {(allf['include'] != True).sum()} | final: {len(final)}  (expected 63 / 3 / 60)")
    print("\nexcluded sessions:")
    print(allf.loc[allf["include"] != True, ["session", "rat", "exclude_reason"]].to_string(index=False))
    summary = pd.DataFrame({"final": final.groupby("rat").size(), "short_vs_long": final.groupby("rat")["short_vs_long"].sum()})
    summary.loc["total"] = summary.sum()
    print("\nsessions per rat:"); print(summary.to_string())
    return final


# ── STEP 7. trial filter ───────────────────────────────────────────────────────────────────
TRIAL_RULES = ["R1_no_press", "R2_touch", "R3_early_hold", "R4_other", "R5_timing_mismatch"]


def trial_filter(final, proc, tone_delay=None, early_max_hold=1.05, resp_win=0.5):
    """All trials of the final sessions with labels, neural-clock times and the exclusion rules R1–R5.
    Returns (T = all trials with a 'keep' column, funnel)."""
    tone_delay = tone_delay or {0.3: 0.35, 1.0: 1.05}
    parts = []
    for _, s in final.iterrows():
        t = pd.read_csv(os.path.join(proc, s["session"], "share", "trials.csv"))
        t.insert(0, "session", s["session"]); t.insert(1, "rat", s["rat"])
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    # labels
    wt = T["wt"].round(2)
    T["type"] = np.select([wt == 0.3, wt == 1.0], ["short", "long"], "other_wt")
    T["label"] = np.select([T["is_touch"] == 1, T["correct"] == 1, T["early"] == 1, T["late"] == 1],
                           ["touch", "correct", "early", "late"], "other")
    T["prev_outcome"] = T.groupby("session")["label"].shift(1)
    T["latency_to_press_s"] = T["press"] - T.groupby("session")["trial_end"].shift(1)
    # times on the neural clock (press from TTL, else press_np_s; tone / release = press + MA interval)
    T["press_from_ttl"] = T["press_ttl_np_s"].notna()
    T["press_neural_s"] = T["press_ttl_np_s"].fillna(T["press_np_s"])
    tone = T["tone"].where(T["tone"] > 0)                        # trials.csv writes 0 when no tone was played
    T["tone_neural_s"] = T["press_neural_s"] + (tone - T["press"])
    T["release_neural_s"] = T["press_neural_s"] + (T["release"] - T["press"])
    T["rt_s"] = T["release"] - tone
    hold = T["hold_duration"]
    T["timing_mismatch"] = (((hold >= wt) & (hold < wt.map(tone_delay)))
                            | ((T["label"] == "correct") & (T["rt_s"] > resp_win)))
    # rules
    T["R1_no_press"] = T["press_neural_s"].isna()
    T["R2_touch"] = T["label"] == "touch"
    T["R3_early_hold"] = (T["label"] == "early") & (hold > early_max_hold)
    T["R4_other"] = T["label"] == "other"
    T["R5_timing_mismatch"] = T["timing_mismatch"]
    T["keep"] = ~T[TRIAL_RULES].any(axis=1)
    # funnel (rules applied one after the other)
    left = pd.Series(True, index=T.index)
    fun = [{"step": "start", **T.groupby("rat").size().to_dict(), "total": len(T), "sessions": T["session"].nunique()}]
    for r in TRIAL_RULES:
        left &= ~T[r]
        k = T[left]
        fun.append({"step": f"− {r}", **k.groupby("rat").size().to_dict(), "total": len(k), "sessions": k["session"].nunique()})
    fun = pd.DataFrame(fun)
    print("trial funnel:"); print(fun.to_string(index=False))
    print("\nonly-this-rule counts:", {r: int((T[r] & (T[TRIAL_RULES].sum(axis=1) == 1)).sum()) for r in TRIAL_RULES})
    K = T[T["keep"]]
    print(f"\nkept: {len(K)} of {len(T)} trials | press from TTL: {K['press_from_ttl'].sum()} | from press_np_s: {(~K['press_from_ttl']).sum()}")
    print("final sessions with 0 kept trials:", sorted(set(final["session"]) - set(K["session"])))
    print("\nkept trials per rat, type x outcome:")
    print(pd.crosstab(K["rat"], [K["type"], K["label"]], margins=True).to_string())
    print("kept trials with a tone, by label:", K.groupby("label")["tone_neural_s"].apply(lambda v: v.notna().sum()).to_dict())
    return T, fun


# ── STEP 8. electrode maps and depth coverage ──────────────────────────────────────────────
def _probe_of_session(raw, s):
    import probeinterface as pi
    meta = glob.glob(os.path.join(raw, s, "**", "*.ap.meta"), recursive=True)[0]
    return pi.read_spikeglx(meta)


def electrode_maps(final, raw):
    """Per electrode map (map_id) used by the final sessions: rats, sessions, channels and depth range per shank."""
    rows = []
    for mid, g in final.groupby("map_id"):
        probe = _probe_of_session(raw, g["session"].iloc[0])          # one example session per map
        pos, shank = probe.contact_positions, probe.shank_ids.astype(int)
        for sh in np.unique(shank):
            y = pos[shank == sh, 1]
            rows.append({"map_id": mid, "rats": " ".join(sorted(g["rat"].unique())), "n_sessions": len(g),
                         "shank": sh, "n_channels": len(y), "y_min_um": int(y.min()), "y_max_um": int(y.max())})
    maps = pd.DataFrame(rows)
    per_map = maps.groupby("map_id").agg(n_sessions=("n_sessions", "first"), n_channels=("n_channels", "sum"))
    print(f"electrode maps in final sessions: {maps['map_id'].nunique()} | sessions: {per_map['n_sessions'].sum()}  (expected 60)")
    print("maps not used by final sessions:", sorted(set(range(12)) - set(maps["map_id"])))
    print("\n", maps.to_string(index=False))
    return maps


def depth_coverage(final, raw, bin_um=100, min_sess=5):
    """Number of final sessions recording each rat × shank × depth bin; returns (table, summary, figure)."""
    map_bins = {}
    for mid, g in final.groupby("map_id"):
        probe = _probe_of_session(raw, g["session"].iloc[0])
        map_bins[mid] = pd.DataFrame({"shank": probe.shank_ids.astype(int),
                                      "bin": (probe.contact_positions[:, 1] // bin_um).astype(int) * bin_um}).drop_duplicates()
    cov = pd.concat([map_bins[r["map_id"]].assign(rat=r["rat"], session=r["session"]) for _, r in final.iterrows()])
    n_cov = cov.groupby(["rat", "shank", "bin"])["session"].nunique().rename("n_sessions").reset_index()
    summ = n_cov.assign(ok=n_cov["n_sessions"] >= min_sess).groupby("rat").agg(bins_recorded=("bin", "size"), bins_testable=("ok", "sum"))
    summ["sessions"] = final.groupby("rat").size()
    print(f"bins recorded and bins with >= {min_sess} sessions, per rat:"); print(summ.to_string())
    rats = sorted(n_cov["rat"].unique()); full = np.arange(0, n_cov["bin"].max() + bin_um, bin_um)
    fig, axes = plt.subplots(1, len(rats), figsize=(4 * len(rats), 7), sharey=True)
    for ax, rat in zip(np.atleast_1d(axes), rats):
        piv = n_cov[n_cov["rat"] == rat].pivot(index="bin", columns="shank", values="n_sessions").reindex(index=full, columns=range(4))
        n_rat = int(summ.loc[rat, "sessions"])
        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=n_rat, extent=[-0.5, 3.5, 0, full[-1] + bin_um])
        ax.set_xticks(range(4)); ax.set_xlabel("shank"); ax.set_title(f"rat {rat} ({n_rat} sessions)", fontweight="bold")
        fig.colorbar(im, ax=ax, label="n sessions recording this bin")
    np.atleast_1d(axes)[0].set_ylabel("depth from tip (µm)")
    fig.suptitle(f"How many final sessions recorded each shank × {bin_um} µm bin", fontweight="bold")
    fig.tight_layout()
    return n_cov, summ, fig


# ── STEP 9. task settings and tone delays ──────────────────────────────────────────────────
SETTING_COLS = ["catch", "vibration", "long_prob", "wt", "real_wt", "reward_size", "tone_frequency", "tone_amplitude", "tone_duration"]


def task_settings(final, proc, cols=SETTING_COLS):
    """Distinct values of the task-setting columns per session, plus % outcomes (all trials, before the filter)."""
    rows = []
    for _, u in final.iterrows():
        s = u["session"]
        beh = pd.read_csv(os.path.join(proc, s, "share", "trials.csv"))
        row = {"session": s, "rat": u["rat"], "n_trials": len(beh),
               "licks_npz": os.path.exists(os.path.join(proc, s, "licks.npz")),
               "pct_correct": round(100 * beh["correct"].mean(), 1), "pct_early": round(100 * beh["early"].mean(), 1),
               "pct_late": round(100 * beh["late"].mean(), 1),
               "pct_long": round(100 * (beh["wt"].round(2) == 1.0).mean(), 1),        # from wt, not the 'long' column
               "n_touch": int(beh["is_touch"].astype(bool).sum())}
        for c in cols:
            v = np.sort(beh[c].dropna().unique())
            row[c] = (", ".join(map(str, np.round(v.astype(float), 3))) if len(v) <= 4 else f"{len(v)} values, {np.min(v):.3g}–{np.max(v):.3g}")
        rows.append(row)
    tasks = pd.DataFrame(rows)
    print(f"sessions: {len(tasks)}  (expected 60) | missing licks.npz: {(~tasks['licks_npz']).sum()}")
    print("\nper rat (median, min, max):")
    print(tasks.groupby("rat")[["n_trials", "pct_correct", "pct_early", "pct_late", "pct_long"]].agg(["median", "min", "max"]).round(1))
    for c in cols:
        print(f"\n{c}:"); print(tasks.groupby("rat")[c].value_counts().to_string())
    return tasks


def tone_delays(final, proc):
    """Measured tone − press delays and the long_prob block structure per session."""
    rows = []
    for _, u in final.iterrows():
        beh = pd.read_csv(os.path.join(proc, u["session"], "share", "trials.csv"))
        tone_ok = beh["tone"] > 0
        delay = ((beh["tone"] - beh["press"])[tone_ok]).round(2)
        lp = beh["long_prob"].round(2)
        rows.append({"session": u["session"], "rat": u["rat"],
                     "tone_delays_s": ", ".join(f"{d}({n})" for d, n in delay.value_counts().sort_index().items()),
                     "other_delays": int((~delay.isin([0.35, 1.05])).sum()),
                     "n_lp_switches": int((lp != lp.shift()).sum() - 1),
                     "trials_per_long_prob": ", ".join(f"{v}:{n}" for v, n in lp.value_counts().sort_index().items())})
    chk = pd.DataFrame(rows)
    print(chk.to_string(index=False))
    print(f"\nsessions: {len(chk)} | trials with a delay other than 0.35 / 1.05 s: {chk['other_delays'].sum()}")
    print("long_prob switches per session, per rat (median, min, max):")
    print(chk.groupby("rat")["n_lp_switches"].agg(["median", "min", "max"]))
    return chk


# ── STEP 10. probe location ────────────────────────────────────────────────────────────────
def channel_regions(final, proc, x_offset_um=27):
    """Brain region of every recorded channel (matched by position: file x = probe x + 27 µm, same depth).
    Returns (inventory per session, one row per recorded channel)."""
    load = fxc.si_load()
    rat_of = final.set_index("session")["rat"]
    inv, maps = [], []
    for s in final["session"]:
        row = {"session": s, "rat": rat_of[s]}
        try:
            d = os.path.join(proc, s)
            csv = glob.glob(os.path.join(d, "**", "*channel_regions*.csv"), recursive=True)
            xls = glob.glob(os.path.join(d, "**", "*channel_locations*.xlsx"), recursive=True)
            row.update(n_csv=len(csv), n_xlsx=len(xls), csv_file=csv[0] if csv else "", xlsx_file=xls[0] if xls else "")
            if len(csv) != 1:
                inv.append(row); print(s, "-> channel_regions.csv found:", len(csv)); continue
            reg = pd.read_csv(csv[0])
            row.update(file_rows=len(reg), file_depth_min=reg["depth_um"].min(), file_depth_max=reg["depth_um"].max(),
                       file_regions="; ".join(f"{k} {v}" for k, v in reg["brain_region"].value_counts().items()))
            if len(xls) == 1:
                lx = pd.read_excel(xls[0]).set_index("channel_id")["location"]
                row["xlsx_agrees_%"] = round(100 * (reg["brain_region"].values == lx.reindex(reg["channel"]).values).mean(), 1)
            rec = load(d1000(proc, s))                                      # only positions are used
            loc = rec.get_channel_locations()
            row["rec_x"] = str(sorted(np.unique(np.round(loc[:, 0])).astype(int).tolist()))
            row["file_x"] = str(sorted(np.unique(np.round(reg["x_um"])).astype(int).tolist()))
            key = {(round(x) - x_offset_um, round(y)): i for i, (x, y) in enumerate(zip(reg["x_um"], reg["depth_um"]))}
            row["file_dup_positions"] = len(reg) - len(key)                 # should be 0
            idx = np.array([key.get((round(x), round(y)), -1) for x, y in loc])
            ok_ = idx >= 0
            sub = reg.iloc[np.where(ok_, idx, 0)]
            m = pd.DataFrame({"session": s, "rat": row["rat"], "channel_id": rec.get_channel_ids(), "x": loc[:, 0], "y": loc[:, 1],
                              "region": np.where(ok_, sub["brain_region"].values, "NOT IN FILE"),
                              "ap_mm": np.where(ok_, sub["ap_mm"].values, np.nan), "ml_mm": np.where(ok_, sub["ml_mm"].values, np.nan),
                              "dv_mm": np.where(ok_, sub["dv_mm"].values, np.nan)})
            num = pd.Series(rec.get_channel_ids()).astype(str).str.extract(r"(\d+)$")[0].astype(float)
            fy = reg.set_index("channel")["depth_um"].reindex(num).values
            row["chan_number_same_y_%"] = round(100 * np.mean(np.isclose(fy, loc[:, 1])), 1)
            maps.append(m)
            vc = m["region"].value_counts()
            row.update(rec_channels=len(m), rec_y_min=loc[:, 1].min(), rec_y_max=loc[:, 1].max(),
                       matched_pct=round(100 * (m["region"] != "NOT IN FILE").mean(), 1),
                       n_PL=int(vc.get("Prelimbic area", 0)), n_IL=int(vc.get("Infralimbic area", 0)),
                       n_other=int((~m["region"].isin(["Prelimbic area", "Infralimbic area", "NOT IN FILE"])).sum()),
                       n_not_in_file=int(vc.get("NOT IN FILE", 0)))
        except Exception as e:
            row["error"] = repr(e)
        inv.append(row)
        print(s, {k: row.get(k) for k in ("n_csv", "n_xlsx", "matched_pct", "n_PL", "n_IL", "n_other", "error") if k in row}, flush=True)
    INV = pd.DataFrame(inv)
    CH = pd.concat(maps, ignore_index=True) if maps else pd.DataFrame()
    print("\nmissing / more than one channel_regions.csv:", INV.loc[INV["n_csv"].fillna(0) != 1, "session"].tolist())
    if "matched_pct" in INV:
        print("not 100% matched by position:", INV.loc[INV["matched_pct"].fillna(0) < 100, "session"].tolist())
    print("sessions with an error:", INV.loc[INV["error"].notna(), "session"].tolist() if "error" in INV else [])
    return INV, CH


def region_consistency(ch):
    """Same electrode position -> same region in every session of a rat? + y range per rat, shank, region."""
    g = ch[ch["region"] != "NOT IN FILE"].groupby(["rat", "x", "y"])["region"]
    cons = g.agg(n_sessions="size", n_regions="nunique",
                 regions=lambda r: "; ".join(f"{k} {v}" for k, v in r.value_counts().items())).reset_index()
    for rat, c in cons.groupby("rat"):
        bad = c[c["n_regions"] > 1]
        print(f"rat {rat}: {len(c)} electrode positions recorded, {len(bad)} with different regions across sessions")
        if len(bad): print(bad.sort_values(["x", "y"]).head(30).to_string(index=False))
    ch = ch.assign(shank=(ch["x"] // 250).astype(int))
    yr = ch.groupby(["rat", "shank", "region"])["y"].agg(["min", "max"]).unstack("region")
    print("\ny range (µm) per rat, shank and region:\n", yr.to_string())
    return cons, yr


REG_COL_PROBE = {"Infralimbic area": "tab:orange", "Prelimbic area": "tab:blue", "Cingulate area 1": "tab:purple",
                 "Secondary motor area": "tab:green", "Basal forebrain region, unspecified": "tab:brown",
                 "Not found": "black", "NOT IN FILE": "lightgrey"}


def _probe_panel(ax, m, title, reg_col):
    for r, mm in m.groupby("region"):
        ax.scatter(mm["x"], mm["y"], s=4, color=reg_col.get(r, "tab:green"), marker="s", linewidths=0)
    for sh in range(4):
        ax.axvspan(sh * 250 - 5, sh * 250 + 75, color="k", alpha=0.04, lw=0)
    ax.set_xlim(-30, 860); ax.set_xticks([30, 280, 530, 780]); ax.set_xticklabels(["sh0", "sh1", "sh2", "sh3"], fontsize=7)
    ax.set_title(title, fontsize=8); ax.tick_params(labelsize=7)


def plot_probe_regions(ch, reg_col=REG_COL_PROBE):
    """One figure per rat: one panel per session (channels coloured by region) + all sessions. Returns {rat: fig}."""
    figs = {}
    for rat, mr in ch.groupby("rat"):
        sess = sorted(mr["session"].unique())
        ncol = min(8, len(sess) + 1); nrow = int(np.ceil((len(sess) + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 4.2 * nrow), sharey=True, squeeze=False)
        axes = axes.ravel()
        for ax, s in zip(axes, sess):
            m = mr[mr["session"] == s]
            _probe_panel(ax, m, f"{s.split('_')[0]}\nPL {(m.region == 'Prelimbic area').sum()} IL {(m.region == 'Infralimbic area').sum()}", reg_col)
        _probe_panel(axes[len(sess)], mr.drop_duplicates(["x", "y", "region"]), "all sessions", reg_col)
        for ax in axes[len(sess) + 1:]: ax.axis("off")
        for ax in axes[::ncol]: ax.set_ylabel("depth from tip (µm)", fontsize=8)
        h = [plt.Line2D([], [], marker="s", ls="", color=reg_col.get(r, "tab:green"), label=r) for r in sorted(mr["region"].unique())]
        fig.legend(handles=h, loc="upper center", ncol=len(h), fontsize=8, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"rat {rat} — recorded channels coloured by region (each session's channel_regions.csv)", fontsize=10, y=1.03)
        fig.tight_layout(); figs[rat] = fig
    return figs


def probe_orientation(ch):
    """Per session and shank: median ML / AP, DV range, tilt; then ML and AP spread between shanks."""
    C = ch.copy()
    C["rat"] = C["session"].str.extract(r"r0000(\d{3})")[0]
    C["shank"] = (C["x"] // 250).astype(int)
    rows = []
    for (rat, s), d in C.groupby(["rat", "session"]):
        for sh, g in d.groupby("shank"):
            g = g.dropna(subset=["ml_mm", "dv_mm", "ap_mm"])
            if len(g) < 10: continue
            dml = np.polyfit(g["dv_mm"], g["ml_mm"], 1)[0]                   # change in ML per mm DV along the shank
            rows.append(dict(rat=rat, session=s, shank=sh, n_ch=len(g), ml_mm=g["ml_mm"].median(), ap_mm=g["ap_mm"].median(),
                             dv_top=g["dv_mm"].max(), dv_bottom=g["dv_mm"].min(), tilt_deg=np.degrees(np.arctan(abs(dml)))))
    OR = pd.DataFrame(rows)
    print("per rat (median over sessions) — one row per shank:")
    print(OR.groupby(["rat", "shank"])[["ml_mm", "ap_mm", "dv_top", "dv_bottom", "tilt_deg"]].median().round(2).to_string())
    sp = OR.groupby(["rat", "session"]).agg(ml_span=("ml_mm", lambda v: v.max() - v.min()),
                                            ap_span=("ap_mm", lambda v: v.max() - v.min())).groupby("rat").median().round(2)
    print("\nspread between shanks within a session (median over sessions):"); print(sp.to_string())
    return C, OR, sp


def plot_probe_brain(C, reg_col=None):
    """One column per rat: coronal (ML × DV, top) and sagittal (AP × DV, bottom), electrodes coloured by region."""
    from matplotlib.lines import Line2D
    reg_col = reg_col or fxc.REGION
    views = (("ml_mm", "ML (mm)", "Coronal (ML × DV)"), ("ap_mm", "AP (mm)", "Sagittal (AP × DV)"))
    rats = sorted(C["rat"].unique())
    fig, axes = plt.subplots(2, len(rats), figsize=(4.2 * len(rats), 10), squeeze=False)
    for j, rat in enumerate(rats):
        d = C[C.rat == rat].dropna(subset=["ml_mm", "dv_mm", "ap_mm"]).drop_duplicates(["x", "y"])   # every position once
        for i, (xc, xl, _) in enumerate(views):
            ax = axes[i, j]
            for reg in reg_col:                                                 # fixed order -> same legend order
                g = d[d["region"] == reg]
                if len(g): ax.scatter(g[xc], g["dv_mm"], s=6, color=reg_col[reg], label=reg)
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("DV (mm)", fontsize=8); ax.tick_params(labelsize=7)
            ax.set_aspect("equal", adjustable="datalim")
            for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
            if i == 0:
                ax.axvline(0, color="k", lw=0.6, ls="--")                       # midline (coronal only)
                ax.set_title(f"Rat {rat}", fontsize=12, fontweight="bold")
    for i, (_, _, view) in enumerate(views):
        axes[i, 0].annotate(view, xy=(0, 0.5), xycoords="axes fraction", xytext=(-55, 0), textcoords="offset points",
                            rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
    shown = [r for r in reg_col if (C["region"] == r).any()]
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=reg_col[r], label=r) for r in shown],
               loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Probe position in brain coordinates", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.03, 0.08, 1, 0.95), h_pad=3, w_pad=4)
    return fig


# ── STEP 11. histology ─────────────────────────────────────────────────────────────────────
def _norm(im, p_lo=1, p_hi=99.5):
    lo, hi = np.percentile(im, [p_lo, p_hi])
    return np.clip((im - lo) / max(hi - lo, 1e-9), 0, 1)


def czi_preview(path, target=2500):
    """One .czi slide -> RGB preview (DAPI blue, DiI red) at most `target` px on the long side; returns (rgb, µm per px)."""
    import czifile
    with czifile.CziFile(path) as czi:
        xa = czi.asxarray().squeeze(); root = czi.xml_element
    px = None
    for d in root.iter("Distance"):
        if d.get("Id") == "X":
            px = float(d.findtext("Value")) * 1e6; break
    print(os.path.basename(str(path)), xa.dims, xa.shape, f"{px:.2f} µm/px" if px else "")
    for dim in list(xa.dims):
        if dim not in ("C", "Y", "X"):
            xa = xa.max(dim) if dim == "Z" else xa.isel({dim: 0})
    step = max(1, int(np.ceil(max(xa.sizes["Y"], xa.sizes["X"]) / target)))
    small = xa.isel(Y=slice(None, None, step), X=slice(None, None, step)).values.astype(float)
    if small.ndim == 2: small = small[None]
    rgb = np.zeros(small.shape[1:] + (3,))
    rgb[..., 2] = _norm(small[0])                                               # C0 = DAPI -> blue
    if small.shape[0] > 1:
        rgb[..., 0] = _norm(small[1]); rgb[..., 1] = 0.5 * _norm(small[1])      # C1 = DiI -> red
    return rgb, (px * step if px else None)


def plot_czi(rgb, px_small, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgb); ax.axis("off"); ax.set_title(title)
    if px_small:                                                                # 1 mm scale bar
        bar = 1000 / px_small; y0, x0 = rgb.shape[0] * 0.95, rgb.shape[1] * 0.05
        ax.plot([x0, x0 + bar], [y0, y0], color="white", lw=4)
        ax.text(x0 + bar / 2, y0 - 20, "1 mm", color="white", ha="center")
    return fig


def recolor_atlas(src, scale=4, new=None):
    """Recolours PL / IL / Cg1 in an atlas-overlay image to the region colours of the probe figures; returns a PIL image."""
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    hexrgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)
    new = new or {"PL": hexrgb("#1f77b4"), "IL": hexrgb("#ff7f0e"), "Cg1": hexrgb("#9467bd")}
    old = {"PL": np.array([93, 187, 238.]), "IL": np.array([242, 201, 99.])}
    im = np.array(Image.open(src).convert("RGB")).astype(float)
    H, W, _ = im.shape
    lum = im @ [0.299, 0.587, 0.114]; out = im.copy()

    def remap(mask, o, n):
        lo = o @ [0.299, 0.587, 0.114]; l = lum[mask]
        dark = np.clip(l / lo, 0, 1)[:, None] * n
        t = np.clip((l - lo) / (255 - lo), 0, 1)[:, None]
        out[mask] = np.where((l <= lo)[:, None], dark, n + (255 - n) * t)

    for key in ("PL", "IL"):
        other = "PL" if key == "IL" else "IL"
        core = np.abs(im - old[key]).sum(2) < 40
        zone = ndi.binary_dilation(core, iterations=2) & ~(np.abs(im - old[other]).sum(2) < 40)
        remap(zone & (np.abs(im - old[key]).sum(2) < 200), old[key], new[key])
    poly = [(283.5, 205), (338.5, 118.5), (366.5, 139.5), (366.5, 173.5), (288.5, 210)]   # Cg1 outline (pixels of the original image)
    pm = Image.new("L", (W, H), 0); ImageDraw.Draw(pm).polygon(poly, fill=255)
    pm = ndi.binary_dilation(np.array(pm) > 0, iterations=1)
    m = pm & ~ndi.binary_dilation(np.abs(im - old["PL"]).sum(2) < 40, iterations=1)
    out[m] = new["Cg1"] * (lum[m] / 255.0)[:, None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((W * scale, H * scale), Image.Resampling.LANCZOS)
