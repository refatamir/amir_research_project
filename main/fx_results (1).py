"""
fx_results.py — the engine behind 07_results (and the coupling data steps of 06)

What is in here
---------------
1. setup(...)   : called once by the settings cell. Builds the shared names every result cell may use
                  (paths, session list, load_bp, colours, ...), switches saving on / off.
2. run(cell)    : runs ONE result cell (a function in fx_res_power / fx_res_pilot / fx_res_coupling / fx_res_bursts).
                  If the cell needs results of an earlier cell (written in its @cell(needs=...) line), that earlier cell is run
                  first, silently (no figures, no prints, nothing saved). So every cell in 07 runs right after the settings cell.
3. saving       : figures -> FIG_DIR/<cell name>/<name>.png, tables -> OUT_TABLES/<cell name>/<name>.csv,
                  only when SAVE = True (see fx_common.save_fig / save_table). Data files that a cell re-uses later
                  (caches, "rows" files, per-session .npz) are always written, because the next run reads them instead of recomputing.

How a result cell is written
----------------------------
Each result cell is a function with no arguments. Its first line, `global ...`, lists the names the cell creates, so that later cells
can use them (exactly as in a notebook). `run` executes the cell in its own name space: the shared names of `setup` + the names it needs
from earlier cells + the helpers of its own fx_res_* file. Cells therefore cannot overwrite each other's variables by accident.
"""
import os
import sys
import io
import types
import pickle
import warnings
import contextlib
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fx_common as fxc

REGISTRY = {}            # cell name -> function
_STORE = {}              # cell name -> its name space after running
_BASE = {}               # shared names (built by setup)
_MODBASE = {}            # module name -> extra names from that module's setup(B)
_STATE = {"cell": None, "quiet": False, "save_to": None}
_SHOWN = {}              # path the old cell wanted -> PNG really written (so display(Image(path)) still works when SAVE is off)
_TMP = os.path.join(__import__("tempfile").gettempdir(), "fx_results_figures")


# ─────────────────────────────────────────────────────────────────────────────────────────────
# 1. the @cell decorator (used in the fx_res_* files)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def cell(needs=None):
    """@cell(needs={"other_cell": ["name1", "name2"]}): this cell uses name1, name2 created by other_cell."""
    def deco(f):
        f._needs = dict(needs or {})
        f._is_cell = True
        REGISTRY[f.__name__] = f
        return f
    return deco


# ─────────────────────────────────────────────────────────────────────────────────────────────
# 2. setup (called by the settings cell)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def setup(CODE_DIR, DATA_ROOT, TABLES, FIG_DIR, OUT_TABLES, SAVE=False, RUN=False, A_RUN=1, **extra):
    """Builds the shared names (what the old 07 cell 0 did) and switches saving on / off."""
    sys.path.insert(0, CODE_DIR)
    import functions as fn
    warnings.filterwarnings("ignore", message="Mean of empty slice|All-NaN slice")
    warnings.filterwarnings("ignore", message="The following channels did not have enough neighbors")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    fxc.set_saving(SAVE, fig_dir=FIG_DIR, table_dir=OUT_TABLES)

    AMIR = os.path.join(DATA_ROOT, "amir_data")
    DATA_OUT = os.path.join(AMIR, "data")
    BP_DIRS = {"CMR": os.path.join(DATA_OUT, "band power", "CMR"), "LOCAL": os.path.join(DATA_OUT, "band power", "LOCAL")}
    final = pd.read_csv(os.path.join(TABLES, "final_sessions.csv"), dtype={"rat": str})

    def load_bp(session, ref="CMR"):
        """One band-power file (06 step 1) -> dict (meta, trials, logpow, curves, counts, times, spectra, spec_n)."""
        with open(os.path.join(BP_DIRS[ref], session + ".pkl"), "rb") as f:
            return pickle.load(f)

    def done_sessions(ref="CMR"):
        """Final sessions whose band-power file exists."""
        return [s for s in final["session"] if os.path.exists(os.path.join(BP_DIRS[ref], s + ".pkl"))]

    TR = pd.read_csv(os.path.join(TABLES, "trials_filtered.csv"), low_memory=False)
    B = dict(
        os=os, np=np, pd=pd, plt=plt, glob=__import__("glob"), pickle=pickle, importlib=importlib, warnings=warnings, sys=sys,
        fn=fn, fxc=fxc,
        MAIN=CODE_DIR, CODE_DIR=CODE_DIR, DATA_ROOT=DATA_ROOT, AMIR=AMIR, TABLES=TABLES, DATA_OUT=DATA_OUT, BP_DIRS=BP_DIRS,
        FIGS=FIG_DIR, FIG_DIR=FIG_DIR, OUT_TABLES=OUT_TABLES, SAVE=SAVE, RUN=RUN, A_RUN=A_RUN,
        BURST_DIR=os.path.join(DATA_OUT, "Bursts detection"),                          # Karvat 2021, 50-ms bins, beta + gamma
        EXACT_DIR=os.path.join(DATA_OUT, "Bursts detection exact times"),               # same detection, exact burst times (beta)
        ALL_DIR=os.path.join(DATA_OUT, "Beta and gamma bursts across channels"),        # whole probe, LOCAL
        BETA_DIR=os.path.join(AMIR, "Beta bursts"),
        PROC=os.path.join(DATA_ROOT, "2_processed"),
        XB=pd.read_csv(os.path.join(TABLES, "extra_bad_channels.csv")),
        THETA_DIR=os.path.join(AMIR, "Theta"),
        final=final, RAT=final.set_index("session")["rat"].astype(str), BANDS=list(fn.MAIN_BANDS),
        TEST="20260529_r0000136_wt_1318",
        K=TR[TR["keep"]].copy(),                                                        # kept trials (as 06)
        load_bp=load_bp, done_sessions=done_sessions,
        # colours (fx_common): outcomes, short / long, rats (three pinks), bands
        COL={**fxc.COND, "wait": fxc.TYPE["long"], "go": fxc.TYPE["short"]},           # wait = correct long, go = correct short
        RCOL=dict(fxc.RATS), COND=fxc.COND, TYPE=fxc.TYPE, RATS=fxc.RATS, RAT_LABEL=fxc.RAT_LABEL, BANDCOL=fxc.BAND,
        pc=lambda x: 100 * (10 ** np.asarray(x, float) - 1),                            # log10 ratio -> % change
        display=_display, Image=image,
    )
    B.update(extra)
    _BASE.clear(); _BASE.update(B)
    _MODBASE.clear(); _STORE.clear()
    print(f"band power done: { {r: len(done_sessions(r)) for r in BP_DIRS} } of {len(final)} | A runs use run {A_RUN}")
    return B


def image(path, *a, **k):
    """IPython Image of a figure a cell has just saved: if SAVE is off, the figure was written to a temporary folder instead."""
    p = _SHOWN.get(str(path), str(path))
    if not os.path.exists(p):
        print("(figure file not found:", p, ")")
        return None
    try:
        from IPython.display import Image
        return Image(p, *a, **k)
    except Exception:            # outside Jupyter
        return None


def _display(*a, **k):
    if _STATE["quiet"] or (a and a[0] is None):
        return
    try:
        from IPython.display import display
        display(*a, **k)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────────────────────
# 3. saving (called from inside the cells)
# ─────────────────────────────────────────────────────────────────────────────────────────────
def savefig(fig, path, *args, **kw):
    """Replaces fig.savefig(path) in the old cells: PNG, FIG_DIR/<cell>/<name>.png, only if SAVE is on."""
    if _STATE["quiet"] or fig is None:
        return None
    name = os.path.basename(str(path))
    if _STATE["save_to"]:                                       # run(cell, save_to=...): this cell only, saved there
        out = fxc.save_fig(fig, name, folder=_STATE["save_to"], force=True)
    elif fxc._SAVE["on"]:
        out = fxc.save_fig(fig, name, folder=os.path.join(_BASE["FIG_DIR"], _STATE["cell"] or ""))
    else:                         # not saved; a temporary copy so that a later display(Image(path)) in the cell still shows it
        out = fxc.save_fig(fig, name, folder=os.path.join(_TMP, _STATE["cell"] or ""), force=True, quiet=True)
    _SHOWN[str(path)] = out
    return out


def table(df, path, *args, index=True, **kw):
    """Replaces df.to_csv(os.path.join(...)) for result tables: OUT_TABLES/<cell>/<name>.csv, only if SAVE is on."""
    if _STATE["quiet"] or df is None:
        return None
    if _STATE["save_to"]:
        return fxc.save_table(df, os.path.basename(str(path)), folder=_STATE["save_to"], force=True, index=index)
    folder = os.path.join(_BASE["OUT_TABLES"], _STATE["cell"] or "")
    return fxc.save_table(df, os.path.basename(str(path)), folder=folder if fxc._SAVE["on"] else None, index=index)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# 4. run one cell
# ─────────────────────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _quiet():
    """Earlier cells that are run only because a later cell needs them: no prints, no figures, nothing saved."""
    old_show, old_q = plt.show, _STATE["quiet"]
    figs_before = set(plt.get_fignums())
    plt.show = lambda *a, **k: None
    _STATE["quiet"] = True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        plt.show = old_show
        _STATE["quiet"] = old_q
        for n in set(plt.get_fignums()) - figs_before:
            plt.close(n)


def _module_names(mod):
    """Helpers and constants of the cell's own fx_res_* file (functions re-bound so they see the cell's name space)."""
    if mod.__name__ not in _MODBASE:
        extra = mod.setup(dict(_BASE)) if hasattr(mod, "setup") else {}
        _MODBASE[mod.__name__] = extra
    names = {k: v for k, v in vars(mod).items()
             if not k.startswith("__") and k != "setup" and not getattr(v, "_is_cell", False)}
    names.update(_MODBASE[mod.__name__])
    return names


def _bind(ns, mod):
    """Functions defined in the fx_res_* file read their settings from the cell's name space (as in a notebook)."""
    for k, v in list(ns.items()):
        if isinstance(v, types.FunctionType) and v.__module__ == mod.__name__ and not getattr(v, "_is_cell", False):
            ns[k] = types.FunctionType(v.__code__, ns, v.__name__, v.__defaults__, v.__closure__)


def run(f, quiet=False, save_to=None, again=False):
    """Run one result cell. Earlier cells it needs are run first (silently) if they have not run yet in this session.
    save_to = r"D:\\folder": save the figures and tables of THIS cell there (even if SAVE is False).
    again = True: also re-run the earlier cells it needs (e.g. after changing a file)."""
    if not _BASE:
        raise RuntimeError("run the settings cell first (fx_results.setup)")
    if isinstance(f, str):
        f = REGISTRY[f]
    mod = sys.modules[f.__module__]
    ns = dict(_BASE)
    ns["__builtins__"] = __builtins__
    ns["__name__"] = mod.__name__
    ns.update(_module_names(mod))
    for dep, names in f._needs.items():
        if dep not in _STORE or again:
            run(REGISTRY[dep], quiet=True)
        for n in names:
            if n not in _STORE[dep]:
                raise NameError(f"{f.__name__} needs '{n}' from {dep}, but {dep} did not make it")
            ns[n] = _STORE[dep][n]
    ns["_savefig"], ns["_table"] = savefig, table
    _bind(ns, mod)
    g = types.FunctionType(f.__code__, ns, f.__name__)
    old, old_to = _STATE["cell"], _STATE["save_to"]
    _STATE["cell"], _STATE["save_to"] = f.__name__, (save_to if not quiet else None)
    try:
        if quiet:
            with _quiet():
                g()
        else:
            g()
    finally:
        _STATE["cell"], _STATE["save_to"] = old, old_to
    _STORE[f.__name__] = ns
    return None


def forget(name=None):
    """Forget the results of one cell (or all), so the next run computes them again."""
    if name is None:
        _STORE.clear()
    else:
        _STORE.pop(name if isinstance(name, str) else name.__name__, None)
