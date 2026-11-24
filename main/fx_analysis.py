"""
fx_analysis.py — functions of notebook 06_analysis (the heavy computations that produce the data read by 07)

Code moved out of the notebook cells without changing what it computes. Every `run_*` function loops over the
final sessions, skips sessions whose output file exists (unless overwrite=True) and ONLY computes and writes
when run=True; with run=False it just reports what is done. Output files appear only when complete
(written as .tmp, then renamed). Building blocks come from functions.py (fn).
"""
import os, glob, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import functions as fn

SHORT = {"Infralimbic area": "IL", "Prelimbic area": "PL", "Cingulate area 1": "Cg1"}


def _todo(names, out_file, overwrite, label):
    todo = [s for s in names if overwrite or not os.path.exists(out_file(s))]
    print(f"{label}: {len(names) - len(todo)} done, {len(todo)} to do", flush=True)
    return todo


def _log_error(err_log, **row):
    if err_log: pd.DataFrame([row]).to_csv(err_log, mode="a", header=not os.path.exists(err_log), index=False)


def _save_npz(path, compressed=True, **arrays):
    tmp = path[:-4] + ".tmp.npz"
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)                                                     # file appears only when complete


# ── band power ────────────────────────────────────────────────────────────────────────────
def run_band_power(final, K, proc, xb, bp_dirs, run=False, overwrite=False, err_log=None):
    """fn.band_power_session for every final session and reference (one .pkl each)."""
    jobs = [(s, ref) for ref in bp_dirs for s in final["session"] if overwrite or not os.path.exists(os.path.join(bp_dirs[ref], s + ".pkl"))]
    print(f"band power: {len(bp_dirs) * len(final) - len(jobs)} done, {len(jobs)} to do", flush=True)
    if not run: return jobs
    t_all = time.time()
    for i, (s, ref) in enumerate(jobs, 1):
        t0 = time.time()
        try:
            recs, removed = fn.load_session(proc, s, xb, refs=(ref,))
            os.makedirs(bp_dirs[ref], exist_ok=True)
            fn.band_power_session(recs[ref], fn.session_trials(K, s), os.path.join(bp_dirs[ref], s + ".pkl"), verbose=False,
                                  meta_extra={"session": s, "ref": ref, "extra_bad_removed": removed})
            print(f"[{i}/{len(jobs)}] {s} {ref}: ok ({(time.time() - t0) / 60:.1f} min)", flush=True)
        except Exception as e:
            print(f"[{i}/{len(jobs)}] {s} {ref}: ERROR {e!r}", flush=True); _log_error(err_log, session=s, ref=ref, error=repr(e))
    print(f"band power finished in {(time.time() - t_all) / 60:.0f} min", flush=True)
    return jobs


# ── bursts (50-ms bins) and bursts with exact times ───────────────────────────────────────
def run_bursts(final, K, proc, xb, rep, chreg, out_dir, run=False, overwrite=False, exact=False, err_log=None):
    """fn.bursts_session (50-ms bins) or fn.bursts_session_exact (exact times) for every final session, CMR + LOCAL."""
    f = lambda s: os.path.join(out_dir, s + ".npz")
    todo = _todo(list(final["session"]), f, overwrite, "exact bursts" if exact else "bursts")
    if not run: return todo
    os.makedirs(out_dir, exist_ok=True); rat_of = final.set_index("session")["rat"]; t_all = time.time()
    for i, s in enumerate(todo, 1):
        t0 = time.time()
        try:
            recs, _ = fn.load_session(proc, s, xb, refs=("CMR", "LOCAL"))
            (fn.bursts_session_exact if exact else fn.bursts_session)(recs, fn.session_trials(K, s), rep, chreg, s, rat_of[s], f(s), verbose=False)
            print(f"[{i}/{len(todo)}] {s}: ok ({(time.time() - t0) / 60:.1f} min)", flush=True)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {s}: ERROR {e!r}", flush=True); _log_error(err_log, session=s, error=repr(e))
    print(f"finished in {(time.time() - t_all) / 60:.0f} min", flush=True)
    return todo


# ── bursts over the whole probe ───────────────────────────────────────────────────────────
def _grid_channels(rec, s, chreg, burst_dir, step_um):
    """One channel per shank and depth bin (closest to the bin centre), all labelled regions, plus the 3 representative channels."""
    ids = np.array([str(c) for c in rec.get_channel_ids()]); loc = rec.get_channel_locations()
    reg = chreg[chreg["session"] == s].set_index("channel_id")["region"].reindex(ids).values
    d = pd.DataFrame({"channel_id": ids, "x": loc[:, 0], "y": loc[:, 1], "region": reg})
    d = d[d["region"].notna() & (d["region"] != "Not found")].copy()
    d["ybin"] = (d["y"] // step_um).astype(int); d["dist"] = (d["y"] - (d["ybin"] + 0.5) * step_um).abs()
    g = d.sort_values("dist").groupby(["x", "ybin"]).head(1)
    zb = np.load(os.path.join(burst_dir, s + ".npz"))
    rep = {r: str(zb[f"LOCAL|{r}|channel"][0]) for r in ("PL", "IL", "Cg1") if f"LOCAL|{r}|channel" in zb.files}
    g = pd.concat([g, d[d["channel_id"].isin(rep.values()) & ~d["channel_id"].isin(g["channel_id"])]])
    return g.sort_values(["x", "y"]).reset_index(drop=True), rep, zb


def _read_full32(rec, ids, chunk_s=60):
    """Whole session for the given channels, in chunks -> (samples, channels) float32."""
    fs, n = rec.get_sampling_frequency(), rec.get_num_samples(); step, out = int(chunk_s * fs), np.empty((n, len(ids)), np.float32)
    for a in range(0, n, step):
        out[a:min(n, a + step)] = rec.get_traces(start_frame=a, end_frame=min(n, a + step), channel_ids=ids, return_in_uV=True)
    return out


def _whole_probe_session(s, proc, xb, chreg, burst_dir, out_dir, bands, step_um):
    T = pd.read_csv(os.path.join(burst_dir, s + "_trials.csv"))                 # same rows as the main burst file
    rec = fn.load_session(proc, s, xb, refs=("LOCAL",))[0]["LOCAL"]; fs = rec.get_sampling_frequency()
    G, rep, zb = _grid_channels(rec, s, chreg, burst_dir, step_um)
    X = _read_full32(rec, G["channel_id"].tolist()); nC = len(G)
    arrs = {f"{b}|{al}|{k}": [] for b in bands for al in fn.BURST_ALIGN for k in ("occ", "cnt")}
    sess_occ = {b: np.full(nC, np.nan) for b in bands}; art_frac = np.full(nC, np.nan)
    for j in range(nC):
        x = X[:, j].astype(np.float64)
        art = fn._ndi_dilate(np.abs(x) > fn.BURST_ART_UV, fn.BURST_ART_PAD); art_frac[j] = art.mean()
        for b in bands:
            cfg = fn.BURST_BANDS[b]
            on, ev = fn.karvat_bursts(x, fs, cfg["range"], cfg["n_cycles"], art, skip=cfg["skip"])
            if (~art).any(): sess_occ[b][j] = on[~art].mean()
            for al, (col, a, z) in fn.BURST_ALIGN.items():
                o, c = fn.burst_binned(on, ev, fs, T[col].values.astype(float), a, z)
                arrs[f"{b}|{al}|occ"].append(o.astype(np.float16)); arrs[f"{b}|{al}|cnt"].append(c.astype(np.float16))
    del X
    res = {k: np.stack(v) for k, v in arrs.items()}
    res.update({f"{b}|session_occ": sess_occ[b] for b in bands})
    res.update(channel_id=G["channel_id"].values.astype(str), x=G["x"].values, y=G["y"].values, region=G["region"].values.astype(str),
               artifact_frac=art_frac, bands=np.array(bands), step_um=np.array([step_um]))
    _save_npz(os.path.join(out_dir, s + ".npz"), **res)
    diff = {}                                                                    # check: representative channels reproduce the main file
    for b in bands:
        for r, cid in rep.items():
            j = np.flatnonzero(res["channel_id"] == cid)
            if len(j):
                a_new = res[f"{b}|tone|occ"][j[0]].astype(float); a_old = zb[f"LOCAL|{b}|{r}|tone|occ"]
                diff[f"{b.split()[0]} {r}"] = round(float(np.nanmax(np.abs(a_new - a_old))), 4) if np.isfinite(a_old).any() else np.nan
    return nC, diff


def run_whole_probe_bursts(final, proc, xb, chreg, burst_dir, out_dir, bands=("beta 15-30", "gamma 30-100"), step_um=90,
                           max_hours=13, run=False, err_log=None):
    """Beta + gamma bursts (same rule as the main burst files, LOCAL) on a grid of channels over the whole probe."""
    ss_ = final[["session", "rat"]].copy(); ss_["k"] = ss_.groupby("rat").cumcount()            # alternate rats
    order = [s for s in ss_.sort_values(["k", "rat"])["session"]
             if os.path.exists(os.path.join(burst_dir, s + ".npz")) and not os.path.exists(os.path.join(out_dir, s + ".npz"))]
    print(f"whole-probe bursts ({', '.join(bands)}): {len(final) - len(order)} done / skipped, {len(order)} to do | grid {step_um} µm", flush=True)
    if not run: return order
    os.makedirs(out_dir, exist_ok=True); t_all, times = time.time(), []
    for i, s in enumerate(order, 1):
        if (time.time() - t_all) / 3600 > max_hours:
            print(f"stopped: {max_hours} h reached; run the cell again to continue", flush=True); break
        t0 = time.time()
        try:
            nC, diff = _whole_probe_session(s, proc, xb, chreg, burst_dir, out_dir, list(bands), step_um)
            times.append(time.time() - t0)
            print(f"[{i}/{len(order)}] {s}: {nC} channels, {times[-1] / 60:.1f} min | check vs main file (max |diff|): {diff} | "
                  f"~{np.mean(times) * (len(order) - i) / 3600:.1f} h left", flush=True)
        except Exception as e:
            print(f"[{i}/{len(order)}] {s}: ERROR {e!r}", flush=True); _log_error(err_log, session=s, error=repr(e))
    print(f"finished in {(time.time() - t_all) / 3600:.1f} h", flush=True)
    return order


# ── referencing figure ────────────────────────────────────────────────────────────────────
def referencing_figure(proc, xb, chreg, ses, pick_xy=(282.0, 1875.0), r_in=30.0, r_out=200.0, min_nb=5, x_stretch=2.5, zoom=260):
    """What CMR and LOCAL subtract for one example channel; checks both references by hand against SpikeInterface (1 s)."""
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    reg_short = {**SHORT, "Secondary motor area": "M2", "Basal forebrain region, unspecified": "BF"}
    recs, removed = fn.load_session(proc, ses, xb, refs=("original", "CMR", "LOCAL")); orig = recs["original"]
    ids = np.array([str(c) for c in orig.get_channel_ids()]); loc = orig.get_channel_locations()
    k = int(np.argmin(np.hypot(loc[:, 0] - pick_xy[0], loc[:, 1] - pick_xy[1]))); cid = ids[k]
    reg = reg_short.get(chreg.loc[(chreg["session"] == ses) & (chreg["channel_id"].astype(str) == cid), "region"].squeeze(), "?")
    seg = recs["LOCAL"]._recording_segments[0]; nb = None
    for name in ("neighbors", "local_neighbors"):
        if hasattr(seg, name): nb = np.asarray(getattr(seg, name)[k]); src = f"SpikeInterface ({name})"; break
    if nb is None:
        d = np.linalg.norm(loc - loc[k], axis=1); nb = np.flatnonzero((d > r_in) & (d <= r_out))
        if len(nb) < min_nb: cand = np.flatnonzero(d > r_in); nb = cand[np.argsort(d[cand])[:min_nb]]
        src = f"rule recomputed ({r_in:.0f} < d ≤ {r_out:.0f} µm)"
    print(f"{ses}: {len(ids)} channels ({len(removed)} extra bad removed) | red channel {cid} ({reg}) at x={loc[k, 0]:.0f}, y={loc[k, 1]:.0f} µm")
    print(f"CMR subtracts the median of {len(ids)} channels | LOCAL subtracts the median of {len(nb)} channels ({src}); "
          f"{int((np.abs(loc[nb, 0] - loc[k, 0]) > 1).sum())} in the other column")
    fs = orig.get_sampling_frequency(); a = int(orig.get_num_samples() // 2); b = a + int(fs)
    X = orig.get_traces(start_frame=a, end_frame=b, return_in_uV=True).astype(float)
    cmr_si = recs["CMR"].get_traces(start_frame=a, end_frame=b, channel_ids=[cid], return_in_uV=True)[:, 0]
    loc_si = recs["LOCAL"].get_traces(start_frame=a, end_frame=b, channel_ids=[cid], return_in_uV=True)[:, 0]
    print(f"check (max |by hand − SpikeInterface|): CMR {np.abs(X[:, k] - np.median(X, axis=1) - cmr_si).max():.2f} µV, "
          f"LOCAL {np.abs(X[:, k] - np.median(X[:, nb], axis=1) - loc_si).max():.2f} µV (1 bit = {float(np.median(orig.get_channel_gains())):.2f} µV)")
    allx = np.array([sh * 250 + col for sh in range(4) for col in (0, 32)]); ymax = loc[:, 1].max() + 200
    gx, gy = np.meshgrid(allx, np.arange(0, 640) * 15.0); gx, gy = gx[gy <= ymax], gy[gy <= ymax]

    def sites(ax, blue, s_bg, s_blue, s_red):
        ax.scatter(gx, gy, s=s_bg, marker="s", color="0.85", edgecolors="none", rasterized=True)
        ax.scatter(loc[:, 0], loc[:, 1], s=s_bg, marker="s", color="k", edgecolors="none", rasterized=True)
        ax.scatter(loc[blue, 0], loc[blue, 1], s=s_blue, marker="s", color="#1f77b4", edgecolors="none", rasterized=True)
        ax.scatter(loc[k, 0], loc[k, 1], s=s_red, marker="s", color="#d62728", edgecolors="k", linewidths=0.6, zorder=5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 8), facecolor="white", gridspec_kw={"width_ratios": [1, 1.2]})
    sites(a1, np.arange(len(ids)), 4, 10, 60)
    a1.set_aspect(1 / x_stretch); a1.set_xlim(-80, 3 * 250 + 112); a1.set_ylim(-50, ymax)
    a1.set_xticks([sh * 250 + 16 for sh in range(4)]); a1.set_xticklabels([str(sh + 1) for sh in range(4)]); a1.set_xlabel("shank")
    a1.set_ylabel("distance from probe tip (µm)")
    a1.set_title(f"CMR: channel − median of all {len(ids)}\nrecorded channels (x-axis stretched {x_stretch}×)", fontsize=10, fontweight="bold")
    sites(a2, nb, 30, 40, 50)
    for r, ls in ((r_in, "-"), (r_out, "--")): a2.add_patch(Circle((loc[k, 0], loc[k, 1]), r, fill=False, color="k", lw=1.2, ls=ls, zorder=6))
    a2.set_xlim(loc[k, 0] - zoom, loc[k, 0] + zoom); a2.set_ylim(loc[k, 1] - zoom, loc[k, 1] + zoom); a2.set_aspect("equal")
    a2.set_xlabel("x on probe (µm)"); a2.set_ylabel("distance from probe tip (µm)")
    a2.set_title(f"LOCAL: channel − median of {len(nb)} channels\n{r_in:.0f}–{r_out:.0f} µm around it (true scale)", fontsize=10, fontweight="bold")
    for ax, lab in ((a1, "A"), (a2, "B")):
        ax.text(-0.12, 1.06, lab, transform=ax.transAxes, fontsize=12, fontweight="bold")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    handles = [Line2D([], [], marker="s", ls="", ms=6, color="0.85", label="all sites on the probe"),
               Line2D([], [], marker="s", ls="", ms=6, color="k", label="recorded channels"),
               Line2D([], [], marker="s", ls="", ms=7, color="#d62728", mec="k", mew=0.6, label=f"channel being referenced ({reg})"),
               Line2D([], [], marker="s", ls="", ms=6, color="#1f77b4", label="channels in the subtracted median"),
               Line2D([], [], color="k", ls="-", lw=1.2, label=f"LOCAL inner radius {r_in:.0f} µm"),
               Line2D([], [], color="k", ls="--", lw=1.2, label=f"LOCAL outer radius {r_out:.0f} µm")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Re-referencing, example session {ses}", fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), w_pad=3)
    return fig


# ── burst properties (98th percentile, Karvat 2020) ────────────────────────────────────────
def burst_properties_pct(x, fs, art, pct, band_name="beta 15-30"):
    """karvat_bursts steps at percentile pct; per peak: sample, freq, power × threshold, power × median, span (Hz), duration (s);
    plus the burst-on mask (runs at the peak frequency that contain a peak, as karvat_bursts)."""
    from skimage.morphology import local_maxima
    cfg = fn.BURST_BANDS[band_name]; band, ncyc = cfg["range"], cfg["n_cycles"]
    foi = np.arange(max(1, band[0] - 1), band[1] + 2)
    P = np.empty((len(foi), len(x)), np.float32)
    for i, f in enumerate(foi): P[i] = np.abs(ss.fftconvolve(x, fn.morlet(f, ncyc, fs), mode="same")) ** 2
    P[:, art] = np.nan
    thr = np.nanpercentile(P, pct, axis=1, keepdims=True); med = np.nanmedian(P, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"): M = P / thr
    thr_over_med = (thr / med)[:, 0]; del P
    M[~(M >= 1)] = 0
    pk = local_maxima(M, connectivity=2, allow_borders=True) & (M > 0); pk[[0, -1], :] = False
    on = np.zeros(len(x), bool); rows = []
    for i in range(1, len(foi) - 1):
        js = np.flatnonzero(pk[i])
        if not len(js): continue
        row = M[i] > 0
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]])); st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        run = np.cumsum(np.diff(np.concatenate([[0], row.astype(np.int8)])) == 1)
        on |= np.isin(run, np.unique(run[js])) & row
        k = np.searchsorted(st, js, "right") - 1
        for j, kk in zip(js, k):
            col = M[:, j] > 0; lo = hi = i
            while lo - 1 >= 0 and col[lo - 1]: lo -= 1
            while hi + 1 < len(foi) and col[hi + 1]: hi += 1
            rows.append((j, foi[i], M[i, j], M[i, j] * thr_over_med[i], foi[hi] - foi[lo], (en[kk] - st[kk]) / fs))
    rows.sort()
    return np.array(rows, dtype=float).reshape(-1, 6), on


def run_burst_properties(final, K, proc, xb, rep, chreg, out_root, pcts=(98,), test=None, only_test=False, region="IL", ref="LOCAL",
                         band="beta 15-30", half_s=0.5, n_top=50, n_ex=100, run=False, overwrite=False):
    """Per session: burst peaks of the representative IL channel at the given percentile(s) with frequency, power, span, duration;
    burst start/end; artifact periods; trough-aligned ±0.5 s waveforms (mean of all, mean of the 50 strongest)."""
    sessions = [test] if only_test else list(final["session"]); rat_of = final.set_index("session")["rat"]
    todo_all = {}
    for pct in pcts:
        outd = os.path.join(out_root, f"Burst properties p{pct}")
        todo_all[pct] = _todo(sessions, lambda s: os.path.join(outd, s + ".npz"), overwrite, f"burst properties p{pct}")
    if not run: return todo_all
    t_all = time.time()
    for pct, todo in todo_all.items():
        outd = os.path.join(out_root, f"Burst properties p{pct}"); os.makedirs(outd, exist_ok=True)
        for n_, s in enumerate(todo, 1):
            t0 = time.time()
            try:
                rec = fn.load_session(proc, s, xb, refs=(ref,))[0][ref]; fs = rec.get_sampling_frequency()
                pick, _ = fn.pick_region_channels(rec, s, rep, chreg, rat_of[s])
                if region not in pick: print(f"[{n_}/{len(todo)}] {s}: no {region} channel"); continue
                x = fn.read_full(rec, [pick[region]])[:, 0]
                art = fn._ndi_dilate(np.abs(x) > fn.BURST_ART_UV, fn.BURST_ART_PAD)
                PR, on = burst_properties_pct(x, fs, art, pct, band)
                bstart, bend = fn._runs(on, fs)
                xf = ss.sosfiltfilt(ss.butter(4, [15, 30], btype="bandpass", fs=fs, output="sos"), np.where(art, 0, x))
                tr = ss.find_peaks(-xf)[0]; h = int(round(half_s * fs)); W, keep = [], []
                for q, (j, f) in enumerate(PR[:, :2]):
                    j = int(j); w_ = int(round(fs / (2 * f))); c = tr[(tr >= j - w_) & (tr <= j + w_)]
                    if not len(c): continue
                    t_ = c[np.argmin(np.abs(c - j))]
                    if t_ - h < 0 or t_ + h >= len(x) or art[t_ - h:t_ + h + 1].any(): continue
                    W.append(x[t_ - h:t_ + h + 1]); keep.append(q)
                W = np.asarray(W, np.float32); keep = np.asarray(keep)
                top = np.argsort(PR[keep, 2])[::-1][:n_top] if len(keep) else np.array([], int)
                st_ = fn.session_trials(K, s)
                out = dict(pct=np.array([pct]), peak_s=PR[:, 0] / fs, peak_hz=PR[:, 1], power_x_thr=PR[:, 2], power_x_median=PR[:, 3],
                           span_hz=PR[:, 4], dur_s=PR[:, 5], burst_start_s=bstart, burst_end_s=bend,
                           art_start_s=fn._runs(art, fs)[0], art_end_s=fn._runs(art, fs)[1], fs=np.array([fs]), channel=np.array([pick[region]]),
                           task_start_s=np.array([np.nanmin(st_["press_s"])]), task_end_s=np.array([np.nanmax(st_["release_s"])]),
                           wave_t=np.arange(-h, h + 1) / fs, wave_mean_all=W.mean(0) if len(W) else np.array([]), wave_n_all=np.array([len(W)]),
                           wave_mean_top=W[top].mean(0) if len(top) else np.array([]), wave_keep_idx=keep)
                if s == test and len(W):
                    rng = np.random.default_rng(0); out["wave_top"] = W[top]; out["wave_random"] = W[rng.choice(len(W), min(n_ex, len(W)), replace=False)]
                _save_npz(os.path.join(outd, s + ".npz"), **out)
                print(f"[{pct}th {n_}/{len(todo)}] {s}: {len(PR)} peaks, {len(bstart)} bursts | median freq {np.median(PR[:, 1]):.0f} Hz, "
                      f"span {np.median(PR[:, 4]):.0f} Hz, duration {1000 * np.median(PR[:, 5]):.0f} ms, power {np.median(PR[:, 3]):.1f}× median | "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            except Exception as e:
                print(f"[{pct}th {n_}/{len(todo)}] {s}: ERROR {e!r}", flush=True)
    print(f"\nall done in {(time.time() - t_all) / 60:.0f} min")
    return todo_all


# ── average IL beta burst per session ─────────────────────────────────────────────────────
def _align(x, fs, art, P, h, tr):
    """Trough nearest each burst peak -> sample indices of the aligned troughs (no artifact in ±h)."""
    K_ = []
    for t, f in zip(P.t.values, P.f.values):
        j = int(round(t * fs)); w_ = int(round(fs / (2 * f))); c = tr[(tr >= j - w_) & (tr <= j + w_)]
        if not len(c): continue
        k = c[np.argmin(np.abs(c - j))]
        if k - h >= 0 and k + h < len(x) and not art[k - h:k + h + 1].any(): K_.append(k)
    return np.array(K_, int)


def _burst_metrics(m, tsec, fs, base, f_mid):
    """Trough depth (µV), oscillation length (15–30 Hz envelope above half max around 0; ms and cycles), clarity (depth / baseline SD)."""
    c = len(m) // 2; bl = np.abs(tsec) > base; depth = m[c] - np.median(m[bl])
    env = np.abs(ss.hilbert(ss.sosfiltfilt(ss.butter(4, [15, 30], btype="bandpass", fs=fs, output="sos"), m - np.median(m[bl]))))
    above = env >= 0.5 * env[np.abs(tsec) <= 0.05].max(); lo = hi = c
    while lo > 0 and above[lo - 1]: lo -= 1
    while hi < len(m) - 1 and above[hi + 1]: hi += 1
    width_ms = 1000 * (hi - lo + 1) / fs
    return depth, width_ms, width_ms / 1000 * f_mid, abs(depth) / np.std(m[bl])


def run_burst_averages(proc, xb, prop_dir, avg_dir, ref="LOCAL", half=0.30, base=0.20, min_dur=0.070, band=(20, 25), f_mid=22.5,
                       n_traces=10, run=False, overwrite=False):
    """Per session: bursts as in 07 Step 8 (strongest peak of each merged burst, task period), aligned to the nearest beta trough (±300 ms);
    mean of the selected bursts (≥ 70 ms, peak 20–25 Hz, Karvat 2020) and of all bursts; 10 single traces; metrics; beta share of the signal."""
    sessions = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(prop_dir, "*.npz")) if not f.endswith(".tmp.npz"))
    todo = _todo(sessions, lambda s: os.path.join(avg_dir, s + ".npz"), overwrite, "burst averages")
    if run:
        os.makedirs(avg_dir, exist_ok=True); t_start = time.time()
        for n_, s in enumerate(todo, 1):
            t_s = time.time()
            try:
                z = np.load(os.path.join(prop_dir, s + ".npz")); ch = str(z["channel"][0])
                rec = fn.load_session(proc, s, xb, refs=(ref,))[0][ref]; fs = rec.get_sampling_frequency()
                x = fn.read_full(rec, [ch])[:, 0]; n = len(x)
                art = np.zeros(n, bool); on = np.zeros(n, bool)
                for a, b in zip(z["art_start_s"], z["art_end_s"]): art[int(a * fs):int(np.ceil(b * fs)) + 1] = True
                for a, b in zip(z["burst_start_s"], z["burst_end_s"]): on[int(a * fs):int(np.ceil(b * fs)) + 1] = True
                t0, t1 = float(z["task_start_s"][0]), float(z["task_end_s"][0])
                P = pd.DataFrame(dict(t=z["peak_s"], f=z["peak_hz"], d=z["dur_s"], p=z["power_x_median"])); P = P[(P.t >= t0) & (P.t <= t1)]
                bs, be = z["burst_start_s"], z["burst_end_s"]; i = np.searchsorted(bs, P.t.values, "right") - 1
                ok = (i >= 0) & (P.t.values <= be[np.clip(i, 0, None)])
                P = P[ok].assign(b=i[ok]).sort_values("p").groupby("b").last().sort_values("t")
                Psel = P[(P.d >= min_dur) & P.f.between(*band)]
                xf = ss.sosfiltfilt(ss.butter(4, [15, 30], btype="bandpass", fs=fs, output="sos"), np.where(art, 0, x))
                tr = ss.find_peaks(-xf)[0]; h = int(round(half * fs)); tsec = np.arange(-h, h + 1) / fs
                Kall, Ksel = _align(x, fs, art, P, h, tr), _align(x, fs, art, Psel, h, tr)
                m_all = np.mean([x[k - h:k + h + 1] for k in Kall], 0); m_sel = np.mean([x[k - h:k + h + 1] for k in Ksel], 0)
                pick = np.sort(np.random.default_rng(0).choice(len(Ksel), min(n_traces, len(Ksel)), replace=False))
                f_, pxx = ss.welch(np.where(art, 0, x), fs=fs, nperseg=int(2 * fs))
                beta_pct = 100 * pxx[(f_ >= 15) & (f_ <= 30)].sum() / pxx[(f_ >= 1) & (f_ <= 200)].sum()
                d_s, w_s, c_s, q_s = _burst_metrics(m_sel, tsec, fs, base, f_mid); d_a, w_a, c_a, q_a = _burst_metrics(m_all, tsec, fs, base, f_mid)
                _save_npz(os.path.join(avg_dir, s + ".npz"), t_s=tsec, fs=np.array([fs]), channel=np.array([ch]), mean_sel=m_sel, mean_all=m_all,
                          n_sel=np.array([len(Ksel)]), n_all=np.array([len(Kall)]),
                          traces_sel=np.array([x[Ksel[q] - h:Ksel[q] + h + 1] for q in pick], np.float32), on_sel=np.array([on[Ksel[q] - h:Ksel[q] + h + 1] for q in pick]),
                          raw_rms_uV=np.array([np.std(x[~art])]), beta_pct=np.array([beta_pct]),
                          sel_depth_uV=np.array([d_s]), sel_width_ms=np.array([w_s]), sel_cycles=np.array([c_s]), sel_clarity=np.array([q_s]),
                          all_depth_uV=np.array([d_a]), all_width_ms=np.array([w_a]), all_cycles=np.array([c_a]), all_clarity=np.array([q_a]),
                          task_min=np.array([(t1 - t0) / 60]))
                print(f"[{n_}/{len(todo)}] {s}: {len(Ksel)} selected / {len(Kall)} all | trough {d_s:.1f} µV, {w_s:.0f} ms ({c_s:.1f} cycles), "
                      f"clarity {q_s:.1f} | {(time.time() - t_s) / 60:.1f} min", flush=True)
            except Exception as e:
                print(f"[{n_}/{len(todo)}] {s}: ERROR {e!r}", flush=True)
    rows = []
    for s in sessions:
        f = os.path.join(avg_dir, s + ".npz")
        if not os.path.exists(f): continue
        z = np.load(f)
        rows.append(dict(session=s, rat=s.split("_r0000")[1][:3], n_sel=int(z["n_sel"][0]), n_all=int(z["n_all"][0]),
                         sel_per_30min=30 * int(z["n_sel"][0]) / float(z["task_min"][0]),
                         **{k: float(z[k][0]) for k in ("sel_depth_uV", "sel_width_ms", "sel_cycles", "sel_clarity", "all_depth_uV", "all_width_ms",
                                                        "all_cycles", "all_clarity", "raw_rms_uV", "beta_pct")}))
    M = pd.DataFrame(rows)
    if len(M):
        print(f"\n{len(M)} sessions | median (min–max) over sessions:")
        for c in ("sel_depth_uV", "sel_width_ms", "sel_cycles", "sel_clarity", "all_depth_uV", "all_width_ms", "all_cycles", "all_clarity", "raw_rms_uV", "beta_pct"):
            print(f"   {c:14s} {M[c].median():7.2f}   ({M[c].min():.2f} – {M[c].max():.2f})")
        print("\nper rat (median):"); print(M.groupby("rat")[["sel_depth_uV", "sel_width_ms", "sel_cycles", "sel_clarity", "beta_pct"]].median().round(2).to_string())
    return M


# ── theta signal of all channels ──────────────────────────────────────────────────────────
def run_theta_all_channels(final, K, proc, xb, chreg, out_root, refs=("CMR", "LOCAL", "original"), types=("long", "short"),
                           conds=("correct", "early", "late"), t0=-1.5, t1=2.0, step_s=0.020, chunk=32, tbatch=120, run=False, overwrite=False):
    """Per session and reference: every channel, every kept trial, −1.5 to 2.0 s around the press, 20-ms steps:
    theta = sum of the complex Morlet coefficients at 4, 5, 6, 7 Hz (5 cycles) -> amplitude and phase. Saved with the channel and trial tables."""
    fr, ncyc = fn.MAIN_BANDS["theta 4-7"]["freqs"], fn.MAIN_BANDS["theta 4-7"]["n_cycles"]
    sessions = list(final["session"])
    jobs = [(s, r) for r in refs for s in sessions if overwrite or not os.path.exists(os.path.join(out_root, r, s + ".npz"))]
    print(f"{len(sessions)} sessions × {len(refs)} references: {len(jobs)} files to do", flush=True)
    if not run: return jobs
    t_start, done = time.time(), 0
    for ref in refs:
        os.makedirs(os.path.join(out_root, ref), exist_ok=True)
        for s in sessions:
            if (s, ref) not in jobs: continue
            t_s = time.time()
            try:
                tr = fn.session_trials(K, s); tr = tr[tr["type"].isin(types) & tr["label"].isin(conds)].reset_index(drop=True)
                rec = fn.load_session(proc, s, xb, refs=(ref,))[0][ref]; fs = rec.get_sampling_frequency()
                reg = chreg[chreg.session == s].set_index("channel_id")["region"]
                ids = [str(c) for c in rec.get_channel_ids()]; loc = rec.get_channel_locations(); a0, a1 = fn.MAIN_EPOCHS["Press"]
                parts, keep = [], []
                for b0 in range(0, len(tr), tbatch):
                    X, ok = fn.read_epochs(rec, tr["press_s"].values[b0:b0 + tbatch], a0, a1)
                    keep.append(np.arange(b0, min(b0 + tbatch, len(tr)))[ok])
                    t = np.arange(X.shape[-1]) / fs + a0; idx = np.where((t >= t0 - 1e-9) & (t <= t1 + 1e-9))[0][::int(round(step_s * fs))]; td = t[idx]
                    C = np.empty((X.shape[0], X.shape[1], len(idx)), np.complex64)
                    for c0 in range(0, X.shape[1], chunk):
                        xs = X[:, c0:c0 + chunk].astype(np.float64); acc = np.zeros(xs.shape[:2] + (len(idx),), np.complex128)
                        for f in fr: acc += ss.fftconvolve(xs, fn.morlet(f, ncyc, fs)[None, None, :], mode="same", axes=-1)[..., idx]
                        C[:, c0:c0 + chunk] = acc.astype(np.complex64)
                    parts.append(C); del X
                C = np.concatenate(parts, axis=0); T = tr.iloc[np.concatenate(keep)].reset_index(drop=True)
                out = os.path.join(out_root, ref, s); T.to_csv(out + "_trials.csv", index=False)
                _save_npz(out + ".npz", compressed=False, coef=C, t=td, fs=np.array([fs]), channel_ids=np.array(ids), x=loc[:, 0], y=loc[:, 1],
                          region=np.array([str(reg.get(c, "")) for c in ids]), freqs=np.array(fr), n_cycles=np.array([ncyc]))
                done += 1; el = (time.time() - t_start) / 60
                print(f"[{done}/{len(jobs)}] {ref} {s}: {C.shape[0]} trials × {C.shape[1]} channels × {C.shape[2]} times | "
                      f"{(time.time() - t_s) / 60:.1f} min, ~{el / done * (len(jobs) - done):.0f} min left", flush=True)
                del C, parts, rec
            except Exception as e:
                print(f"{ref} {s}: ERROR {e!r}", flush=True)
    print(f"\nall done in {(time.time() - t_start) / 60:.0f} min")
    return jobs


# ── re-referenced signals of the representative channels (CMR, SR30-500) ──────────────────
def run_rep_channel_signals(final, K, proc, xb, chreg, burst_dir, out_dir, regs=("IL", "PL", "Cg1"), ring=(30, 500), tbatch=100,
                            types=("long", "short"), conds=("correct", "early", "late"), run=False, overwrite=False):
    """Representative IL / PL / Cg1 channels, all kept trials, −2.5 to +3.5 s around the press (1000 Hz), in two references:
    CMR (channel − median of all channels) and SR30-500 (channel − median of the SAME-region channels 30–500 µm away)."""
    sessions = [s for s in final["session"] if os.path.exists(os.path.join(burst_dir, s + ".npz"))]
    todo = _todo(sessions, lambda s: os.path.join(out_dir, s + ".npz"), overwrite, "representative-channel signals")
    if not run: return todo
    os.makedirs(out_dir, exist_ok=True); a0, a1 = fn.MAIN_EPOCHS["Press"]; t_start = time.time()
    for n_, s in enumerate(todo, 1):
        t_s = time.time()
        try:
            zb = np.load(os.path.join(burst_dir, s + ".npz"))
            rep = {g: (str(zb[f"CMR|{g}|channel"][0]), float(zb[f"CMR|{g}|distance_um"][0])) for g in regs if f"CMR|{g}|channel" in zb.files}
            if not rep: print(f"[{n_}/{len(todo)}] {s}: no representative channel, skipped", flush=True); continue
            tr = fn.session_trials(K, s); tr = tr[tr["type"].isin(types) & tr["label"].isin(conds)].reset_index(drop=True)
            rec = fn.load_session(proc, s, xb, refs=("original",))[0]["original"]; fs = rec.get_sampling_frequency()
            ids = [str(c) for c in rec.get_channel_ids()]; loc = rec.get_channel_locations()
            reg = np.array([SHORT.get(r, "other") for r in pd.Series(ids).map(chreg[chreg.session == s].set_index("channel_id")["region"]).fillna("")])
            rep = {g: v for g, v in rep.items() if v[0] in ids}; seed = {g: ids.index(c) for g, (c, _) in rep.items()}
            rings = {g: np.where((reg == g) & (np.hypot(loc[:, 0] - loc[i, 0], loc[:, 1] - loc[i, 1]) > ring[0])
                                 & (np.hypot(loc[:, 0] - loc[i, 0], loc[:, 1] - loc[i, 1]) <= ring[1]))[0] for g, i in seed.items()}
            out = {f"{r}|{g}": [] for r in ("CMR", "SR30-500") for g in seed}; keep = []
            for b0 in range(0, len(tr), tbatch):
                X, ok = fn.read_epochs(rec, tr["press_s"].values[b0:b0 + tbatch], a0, a1)
                keep.append(np.arange(b0, min(b0 + tbatch, len(tr)))[ok]); med = np.median(X, axis=1)
                for g, i in seed.items():
                    out[f"CMR|{g}"].append((X[:, i] - med).astype(np.float32)); rs = rings[g]
                    out[f"SR30-500|{g}"].append((X[:, i] - np.median(X[:, rs], axis=1)).astype(np.float32) if len(rs) else np.full_like(X[:, i], np.nan))
                del X, med
            T = tr.iloc[np.concatenate(keep)].reset_index(drop=True)
            res = {"t0": np.array([a0]), "fs": np.array([fs]), "ring_um": np.array(ring), **{f"{k}|signal": np.concatenate(v, axis=0) for k, v in out.items()}}
            for g, (c, d) in rep.items():
                res[f"{g}|channel"] = np.array([c]); res[f"{g}|distance_um"] = np.array([d]); res[f"{g}|ring_n"] = np.array([len(rings[g])])
            T.to_csv(os.path.join(out_dir, s + "_trials.csv"), index=False); _save_npz(os.path.join(out_dir, s + ".npz"), compressed=False, **res)
            info = " | ".join(f"{g} {rep[g][0].split('#')[-1]} ({rep[g][1]:.0f} µm, ring {len(rings[g])})" for g in seed)
            print(f"[{n_}/{len(todo)}] {s}: {len(T)} trials | {info} | {(time.time() - t_s) / 60:.1f} min", flush=True)
            del rec, out
        except Exception as e:
            print(f"[{n_}/{len(todo)}] {s}: ERROR {e!r}", flush=True)
    print(f"\nall done in {(time.time() - t_start) / 60:.0f} min")
    return todo


# ── beta power normalised to the 90th percentile ──────────────────────────────────────────
def run_beta_power_p90(proc, xb, burst_dir, out_dir, ref="LOCAL", regs=("IL", "PL", "Cg1"), freqs=np.arange(15, 31), ncyc=7,
                       t0=-1.0, t1=0.5, step=0.010, run=False, overwrite=False):
    """Representative channels, LOCAL: power at 15–30 Hz, each frequency divided by its artifact-free 90th percentile, mean over
    frequencies; cut −1.0 to 0.5 s around the tone of every trial (same trial rows as the burst files)."""
    sessions = sorted(f[:-4] for f in os.listdir(burst_dir) if f.endswith(".npz") and not f.endswith(".tmp.npz"))
    todo = _todo(sessions, lambda s: os.path.join(out_dir, s + ".npz"), overwrite, "beta power p90")
    if not run: return todo
    os.makedirs(out_dir, exist_ok=True); t_start = time.time()
    for n_, s in enumerate(todo, 1):
        t_s = time.time()
        try:
            zb = np.load(os.path.join(burst_dir, s + ".npz"))
            ch = {g: str(zb[f"{ref}|{g}|channel"][0]) for g in regs if f"{ref}|{g}|channel" in zb.files}
            T = pd.read_csv(os.path.join(burst_dir, s + "_trials.csv"))
            rec = fn.load_session(proc, s, xb, refs=(ref,))[0][ref]; fs = rec.get_sampling_frequency(); X = fn.read_full(rec, list(ch.values()))
            off = np.round(np.arange(t0, t1 + 1e-9, step) * fs).astype(int); res = {"times": off / fs, "freqs": freqs}
            for j, (g, cid) in enumerate(ch.items()):
                x = X[:, j]; art = fn._ndi_dilate(np.abs(x) > fn.BURST_ART_UV, fn.BURST_ART_PAD); B = np.zeros(len(x), np.float32)
                for f in freqs:
                    p = (np.abs(ss.fftconvolve(x, fn.morlet(f, ncyc, fs), mode="same")) ** 2).astype(np.float32); p[art] = np.nan
                    B += p / np.nanpercentile(p, 90)
                B /= len(freqs); M = np.full((len(T), len(off)), np.nan, np.float32)
                for k, t in enumerate(T["tone_s"].values.astype(float)):
                    if not np.isfinite(t): continue
                    idx = int(round(t * fs)) + off
                    if idx[0] >= 0 and idx[-1] < len(B): M[k] = B[idx]
                res[f"{g}|tone"], res[f"{g}|channel"], res[f"{g}|artifact_frac"] = M, np.array([cid]), np.array([art.mean()])
            del X; _save_npz(os.path.join(out_dir, s + ".npz"), **res)
            print(f"[{n_}/{len(todo)}] {s}: ok {list(ch)} ({(time.time() - t_s) / 60:.1f} min)", flush=True)
        except Exception as e:
            print(f"[{n_}/{len(todo)}] {s}: ERROR {e!r}", flush=True)
    print(f"finished in {(time.time() - t_start) / 60:.0f} min", flush=True)
    return todo


# ── channel selection for the all-pairs IL–PL test ─────────────────────────────────────────
def all_pairs_selection(chreg, xb, final, ring=(30, 500), min_ring=5, stride=4, min_pair_um=1000, shank_w=250):
    """Per session: usable IL / PL channels (≥ min_ring same-region channels in the 30–500 µm ring), every stride-th per region,
    and only those with a partner ≥ min_pair_um away. Returns (table, geometry per session for the figure)."""
    C = chreg[chreg.session.isin(final["session"])].copy(); C["channel_id"] = C["channel_id"].astype(str)
    xb = xb.copy(); xb["channel_id"] = xb["channel_id"].astype(str); bad = set(zip(xb.session, xb.channel_id))
    C = C[[(s, c) not in bad for s, c in zip(C.session, C.channel_id)]]
    C["reg"] = C["region"].map(SHORT).fillna("other"); C["rat"] = C["rat"].astype(str)
    rows, geo = [], {}
    for s, d in C.groupby("session"):
        d = d.reset_index(drop=True); x, y, reg = d.x.values, d.y.values, d.reg.values; use = {}
        for g in ("IL", "PL"):
            idx = [i for i in np.where(reg == g)[0]
                   if ((reg == g) & (np.hypot(x - x[i], y - y[i]) > ring[0]) & (np.hypot(x - x[i], y - y[i]) <= ring[1])).sum() >= min_ring]
            use[g] = (idx, sorted(idx, key=lambda i: (x[i] // shank_w, y[i], x[i]))[::stride])
        il, pl = use["IL"][1], use["PL"][1]
        D = np.hypot(x[il][:, None] - x[pl][None, :], y[il][:, None] - y[pl][None, :]) if il and pl else np.zeros((len(il), len(pl)))
        M = D >= min_pair_um
        il_t = [i for k, i in enumerate(il) if M[k].any()] if M.size else []; pl_t = [j for k, j in enumerate(pl) if M[:, k].any()] if M.size else []
        rows.append(dict(session=s, rat=d.rat.iloc[0], IL_usable=len(use["IL"][0]), PL_usable=len(use["PL"][0]),
                         IL_tested=len(il_t), PL_tested=len(pl_t), pairs=int(M.sum())))
        geo[s] = (d, use, il_t, pl_t)
    sel = pd.DataFrame(rows)
    print(sel.to_string(index=False))
    print("\nper rat:"); print(sel.groupby("rat").agg(sessions=("session", "size"), with_pairs=("pairs", lambda v: int((v > 0).sum())),
                                                     median_IL=("IL_tested", "median"), median_PL=("PL_tested", "median"), median_pairs=("pairs", "median")).to_string())
    print("\nsessions with 0 pairs:", list(sel.loc[sel.pairs == 0, "session"]) or "none")
    return sel, geo


def plot_all_pairs_selection(sel, geo, min_ring=5, stride=4, min_pair_um=1000):
    """One panel per session (grouped by rat): IL / PL channels tested (dark), usable but not tested (light), others (grey)."""
    dcol, lcol = {"IL": "#e65100", "PL": "#0d47a1"}, {"IL": "#ffcc80", "PL": "#90caf9"}
    ss_ = sel.sort_values(["rat", "session"]).session.tolist(); nc = 10; nr = int(np.ceil(len(ss_) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.2 * nc, 3.6 * nr), facecolor="white", sharex=True, sharey=True)
    for ax, s in zip(axs.flat, ss_):
        d, use, il_t, pl_t = geo[s]; ax.scatter(d.x, d.y, marker="s", s=2, color="0.88", zorder=1)
        for g, t in (("IL", il_t), ("PL", pl_t)):
            u = [i for i in use[g][0] if i not in t]
            ax.scatter(d.x[u], d.y[u], marker="s", s=2, color=lcol[g], zorder=2); ax.scatter(d.x[t], d.y[t], marker="s", s=6, color=dcol[g], zorder=3)
        r_ = sel[sel.session == s].iloc[0]
        ax.set_title(f"r{r_.rat} {s[4:6]}/{s[6:8]} {s[-4:]}\n{r_.IL_tested} IL × {r_.PL_tested} PL, {r_.pairs} pairs", fontsize=7)
        ax.set_xticks([16, 266, 516, 766]); ax.set_xticklabels(["0", "1", "2", "3"], fontsize=6); ax.tick_params(labelsize=6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for ax in axs.flat[len(ss_):]: ax.axis("off")
    fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=dcol["IL"], label="IL tested"), plt.Line2D([], [], marker="s", ls="", color=lcol["IL"], label="IL usable, not tested"),
                        plt.Line2D([], [], marker="s", ls="", color=dcol["PL"], label="PL tested"), plt.Line2D([], [], marker="s", ls="", color=lcol["PL"], label="PL usable, not tested"),
                        plt.Line2D([], [], marker="s", ls="", color="0.88", label="other")], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Channels for the all-pairs IL–PL test (ring ≥ {min_ring}, every {stride}th, pairs ≥ {min_pair_um} µm), {len(ss_)} sessions",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig
