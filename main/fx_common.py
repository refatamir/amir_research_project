"""
fx_common.py — shared by every notebook (01–07)

What is in here
---------------
1. COLOURS  : one place for all figure colours (conditions, short/long, rats, bands, regions).
2. SAVING   : save_fig / save_table. They write ONLY when saving is switched on, so a notebook can be
              run from top to bottom without changing any file on disk.
3. small helpers used by several notebooks (path check, spikeinterface loader).

How saving works
----------------
Each notebook's settings cell calls

    fxc.set_saving(save=SAVE, fig_dir=FIG_DIR, table_dir=OUT_TABLES)

After that, every cell saves with

    fxc.save_fig(fig, "name")          -> FIG_DIR/name.png      (only if SAVE is True)
    fxc.save_table(df, "name")         -> OUT_TABLES/name.csv   (only if SAVE is True)

To save ONE cell somewhere else (or even when SAVE is False), add folder= and force=True in that cell:

    fxc.save_fig(fig, "name", folder=r"D:\\my\\folder", force=True)

Figures are always PNG (never PDF).
"""
import os
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────────────────────
# 1. COLOURS (change them here and every notebook follows)
# ─────────────────────────────────────────────────────────────────────────────────────────────
COND = {"correct": "#2CA02C", "late": "#DD8452", "early": "#E6B800"}      # trial outcome
TYPE = {"long": "#2CA02C", "short": "#C62828"}                            # short vs long of the SAME outcome
RATS = {"134": "#F48FB1", "135": "#EC407A", "136": "#AD1457"}             # three pink shades (Rat 1, 2, 3)
RAT_LABEL = {"134": "Rat 1", "135": "Rat 2", "136": "Rat 3"}
BAND = {"theta": "#2e7d32", "delta": "#FFA000", "gamma": "#6a1b9a",       # delta = amber yellow
        "beta": "#1565c0", "bursts": "#1565c0"}
REGION = {"Infralimbic area": "#ff7f0e", "Prelimbic area": "#1f77b4",     # brain regions (probe figures)
          "Cingulate area 1": "#9467bd", "Basal forebrain region": "#8c564b"}


def pair_colors(a, b):
    """Colours for two groups that are compared in one figure.
    a, b = (outcome, type), e.g. ("correct", "long"), ("correct", "short").
    Same outcome, different type -> long green / short red; otherwise each outcome keeps its own colour."""
    if a[0] == b[0] and a[1] != b[1]:
        return TYPE[a[1]], TYPE[b[1]]
    return COND[a[0]], COND[b[0]]


# ─────────────────────────────────────────────────────────────────────────────────────────────
# 2. SAVING (off unless switched on)
# ─────────────────────────────────────────────────────────────────────────────────────────────
_SAVE = {"on": False, "fig_dir": None, "table_dir": None}


def set_saving(save=False, fig_dir=None, table_dir=None):
    """Called once in the settings cell of each notebook."""
    _SAVE.update(on=bool(save), fig_dir=fig_dir, table_dir=table_dir)
    print(f"saving is {'ON' if save else 'OFF'}" + (f" | figures -> {fig_dir} | tables -> {table_dir}" if save else
          " (nothing is written; set SAVE = True in the settings to write figures and tables)"))


def _target(folder, default, force):
    if not (force or _SAVE["on"]):
        return None
    folder = folder or _SAVE[default]
    if folder is None:
        raise ValueError(f"no folder for saving: set {default} in the settings or pass folder=")
    os.makedirs(folder, exist_ok=True)
    return folder


def save_fig(fig, name, folder=None, force=False, dpi=300, quiet=False):
    """PNG only. Writes only if saving is on (or force=True)."""
    folder = _target(folder, "fig_dir", force)
    if folder is None:
        return None
    path = os.path.join(folder, os.path.splitext(name)[0] + ".png")
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    if not quiet:
        print("saved:", path)
    return path


def save_table(df, name, folder=None, force=False, index=False):
    """CSV. Writes only if saving is on (or force=True)."""
    folder = _target(folder, "table_dir", force)
    if folder is None:
        return None
    path = os.path.join(folder, os.path.splitext(name)[0] + ".csv")
    df.to_csv(path, index=index)
    print("saved:", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────────────────────
# 3. SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────────────────────
def check_paths(**paths):
    """Stops with a clear message if a folder in the settings does not exist."""
    for name, p in paths.items():
        assert os.path.isdir(p), f"{name} not found: {p}  -> change it in the settings cell"
    print("paths ok:", ", ".join(paths))


def si_load():
    """spikeinterface loader: si.load (version 0.105+) or si.load_extractor (older versions)."""
    import spikeinterface as si
    return getattr(si, "load", None) or si.load_extractor
