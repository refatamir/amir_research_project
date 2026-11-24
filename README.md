# amir_research_project

Analysis code of the master's project **"mPFC theta, beta bursts and phase coupling while rats wait for a cue"**
(Neuropixels LFP in IL, PL and Cg1 of 3 rats, 60 sessions; lab of Prof. Ilka Diester, University of Freiburg).

The code is a chain of 7 Jupyter notebooks. Each notebook only has short cells; the code lives in `.py` files next to it.

## Run order

| # | Notebook | What it does | Main output |
|---|---|---|---|
| 01 | `01_sessions_setup.ipynb` | which sessions and trials are used (inventory, 1000 Hz check, sync, trial filter R1–R5), probe positions and brain regions | `final_sessions.csv`, `trials_filtered.csv`, `channel_regions_all.csv` |
| 02 | `02_preprocessing.ipynb` | 1000 Hz LFP per session (SpikeInterface), bad channels | `2_processed\<session>`, bad-channel tables |
| 03 | `03_behaviour.ipynb` | behaviour: post-error slowing, early after error, latency to press, outcome shares along the session | behaviour tables and figures |
| 04 | `04_QC.ipynb` | signal quality per channel (noise, interference) | QC tables |
| 05 | `05_event_check.ipynb` | evoked potentials (ERP) around press / release, last check of bad channels | ERP tables, extra bad channels |
| 06 | `06_analysis.ipynb` | **heavy computations** (hours, `RUN = True`): band power, bursts, burst properties, theta signals, coupling inputs, multi-unit firing | one file per session in `amir_data\data\...` and `Theta\Phase connectivity\Tests\...` |
| 07 | `07_results.ipynb` | **all results**: power during the hold, phase coupling, beta bursts, effects along the session | figures (PNG) and tables (only if `SAVE = True`) |

## Code files

| File | Used by | Contents |
|---|---|---|
| `functions.py` | all | shared low-level functions (loading sessions, referencing, epochs, Morlet power, bursts, statistics such as `sign_flip_p`) |
| `fx_common.py` | all | **colours** (one place for all figures) and **saving** (`save_fig`, `save_table`; nothing is written unless saving is on) |
| `fx_sessions.py`, `fx_preprocessing.py`, `fx_behaviour.py`, `fx_qc.py`, `fx_events.py`, `fx_analysis.py` | 01–06 | the functions of each notebook |
| `fx_results.py` | 06 (steps 12–16), 07 | the engine of the results: `setup` (settings), `run(cell)` (runs one result cell; earlier cells it needs are run first, silently) |
| `fx_res_power.py` | 07 section 1 (+ theta along the session) | theta, delta (total vs induced), high gamma, multi-unit firing |
| `fx_res_pilot.py` | 07 section 2 (pilot) | choosing the IL–PL pair and the reference, rat 3 with the pilot channels, all pairs |
| `fx_res_coupling.py` | 07 section 2 (+ coupling along the session), 06 steps 12–15 | IL–PL phase coupling (A1–A2d, T4–T15, Jäckel replication) |
| `fx_res_bursts.py` | 07 section 3 (+ bursts along the session) | beta / gamma bursts, Steps 0–10, Gamma Step 1 |

## How every notebook is built

1. A **guide** (first markdown): what is inside, the steps, colours, run order, what to change.
2. A **settings** markdown (every setting explained) and the **settings cell** — the only cell you change (paths, `SAVE`, `RUN`, ...).
3. One short cell per step. In 06 and 07 every step has a **method** markdown: what it computes, how, which test and why.

**Saving:** `SAVE = False` by default — figures are only shown. `SAVE = True` writes all figures (PNG only) and tables.
To save one cell somewhere else: in 01–06 add `folder=r"...", force=True` to its save line; in 07 use `fxr.run(<cell>, save_to=r"...")`.
Data files of the computations (06, caches) are always written, because they are re-used.

**Paths to change** (settings cell of each notebook): `CODE_DIR` (this folder), `DATA_ROOT` (the server data folder), `TABLES`
(temporarily the `tables` folder here; later `amir_data\tables`).

## Colours

| What | Colour |
|---|---|
| outcome | correct green `#2CA02C`, late orange `#DD8452`, early yellow `#E6B800` |
| short vs long of the same outcome (e.g. go vs wait) | long green `#2CA02C`, short red `#C62828` |
| rats | three pinks: Rat 1 `#F48FB1`, Rat 2 `#EC407A`, Rat 3 `#AD1457` |
| bands (figures and the headings of 06 / 07) | theta green `#2e7d32`, delta amber `#FFA000`, gamma purple `#6a1b9a`, beta / bursts blue `#1565c0` |
| regions (probe figures) | IL orange, PL blue, Cg1 violet, basal forebrain brown |

## Results in 07

| Section | Question |
|---|---|
| 1. Power during the hold | Theta rises while the rat holds the lever (all sessions, every rat, both references); delta changes are mostly the evoked potential; high gamma and multi-unit firing drop during the hold |
| 2. Phase coupling | IL–PL theta phases are locked during the hold (pilot → rat 3 → three rats; equal trial numbers; unbiased measures; power control); IL's phase is ahead of PL's; coupling over the trial; replication of Jäckel et al. 2024 (δ, θ) |
| 3. Beta bursts | more beta bursts before late responses (IL); burst occupancy and success; last burst and RT; more beta after an error; burst shape (Karvat 2020); gamma as control band |
| 4. Along the session | how the session is split (trial number, clock time, water drunk); theta, coupling and bursts at the start / middle / end; reaction time vs theta drop |

## Environment

Python with numpy, scipy, pandas, matplotlib, statsmodels, spikeinterface, mne / mne-connectivity, fooof.
To record the exact environment (run once in the Anaconda prompt, in the environment you use, inside this folder):

```
conda env export --from-history > environment.yml
pip freeze > requirements.txt
```

To make the same environment on another computer: `conda env create -f environment.yml`.

## Data

The data are **not** in this repository (they are on the lab server, `Neuropixels_RT_mPFC\data`). Figures and tables are not kept here either:
they are made again by the notebooks. (For now the `tables` folder is still here, because the notebooks read the tables of 01–05 from it;
it will move to `amir_data\tables`, and then `tables` and `figures` are removed from the repository.)
