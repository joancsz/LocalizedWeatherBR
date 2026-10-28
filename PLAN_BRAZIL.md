# PLAN_BRAZIL.md — Porting LocalizedWeather to Northeast Brazil

Working plan for Claude Code. Read this file fully before editing anything. Work phase by phase; each phase ends with an acceptance check that must pass before moving on. Commit at the end of each phase.

---

## 0. Context

**Paper.** Yang et al., *Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data* (arXiv:2410.12938).

**Model.** Station observations $\mathbf{w}_{t-b:t}$ (MADIS, $b=48$ h) plus a gridded product $\mathbf{g}_{t-b:t+l}$ (ERA5 or HRRR) are mapped to station values at lead $l$:

$$\hat{\mathbf{w}}_{t+l}=F_\theta(\mathbf{w}_{t-b:t},\,\mathbf{g}_{t-b:t+l}),\qquad \mathbf{w}\in\mathbb{R}^{n\times 4}=(u_{10},v_{10},T_{2m},T_{d,2m}).$$

Two architectures:
- Transformer (`ModelType.ViT`): one token per station.
- Heterogeneous MPNN (`ModelType.GNN`): station↔station edges, plus grid→station edges.

**Original setup.**
- Northeastern US, 358 MADIS stations, 2019–2023.
- Year split: train 2019–21, val 2022, test 2023.

**Goal of this fork.**
1. Reproduce the NE-US ERA5 result from the Zenodo data (record 15346612, `WindDataNE-US.zip`, 1.9 GB).
2. Remove hardcoded years and paths, and fix the missing observation mask in the loss.
3. Build a data pipeline for **Northeast Brazil**: INMET automatic stations replace MADIS, ERA5 stays as the gridded input. There is no HRRR outside the US, so `n_neighbors_h2m = 0` always.
4. Add evaluation scripts for the planned experiments (§7).

**Non-goals.**
- Do not change the model architectures.
- Do not add new model types.
- Do not touch the HRRR code paths beyond keeping them importable.

**Environment.**
- Local development.
- Python ≥ 3.10, PyTorch, torch_geometric, xarray, netCDF4, geopandas, shapely ≥ 2, scipy, pandas, cdsapi.
- Pin versions in `requirements.txt` (the repo has none). Mirror whatever imports `Source/` actually uses.

---

## 1. How the current code works (verified by reading the source)

Entry point:
- `Source/LocalLauncher.py` builds an `AttrDict` of args with a hardcoded `data_path`.
- It calls `PostProcessInputs.PostProcessArgs` (int → Enum), then `Main.Main(args).Run()`.
- `Source/Arg_Parser.py` is the SLURM argparse variant, with the same keys.

Data flow in `Main.Run()`:
1. `MetaStation(lat_low, lat_up, lon_low, lon_up, n_years, control_ratio, shapefile_path, data_path)` reads two station shapefiles (all stations, plus filtered), or builds them from raw MADIS.
   - The bbox comes from `gpd.read_file(shapefile).bounds`, or from `args.coords`.
   - Filtering rule: `num >= n_years * 366 * 24 * control_ratio`.
2. `MadisNetwork(meta_station, ...)` reads station positions from `meta_station.stations.geometry`, **in shapefile row order**.
3. `years = list(range(2024 - n_years, 2024))`
4. `BuildExternalNetwork`: for each year, construct `ERA5Interpolated` (or `ERA5`).
   - `ERA5.load_ERA5_monthly` reads `data_path/ERA5/<region>/surface_{year}_{month}.nc`, where `region='World'` is the default and never overridden.
   - It slices `longitude=slice(lon_low-1.5, lon_up+1.5)` and `latitude=slice(lat_up+1.5, lat_low-1.5)`, using floor/ceil. **Latitude must be descending.**
   - It renames `valid_time→time`, keeps `u10,v10,t2m,d2m,ssr`, and writes `ERA5/Processed/era5_{year}_{lonlo}_{lonup}_{latlo}_{latup}.nc`.
   - `ERA5Interpolated.make_interpolated` interpolates to station positions (Nearest/BiCubic/Stacked) and writes `ERA5/Interpolated/era5interpolated_e2m_{k}_{Interp}_{year}_{filtered_file_name}.nc`.
   - It calls `self.data.drop(['expver','number'])`, which **crashes if those variables are absent**.
5. `GetDataList`: one `MixData(year, ...)` per year.
   - `MixData` builds an hourly `time_line` for the calendar year (naive datetimes, treated as UTC).
   - It instantiates `Madis(...)`.
6. `Madis.__init__`:
   - If `madis/processed/Meta-{2024-n_years}-2023/madis_{year}_{filtered_file_name}.nc` exists, it is opened and loaded.
   - Otherwise it rebuilds from `madis/raw_monthly/mesonet/{year}/{month}.nc`.
   - **The Brazil adapter writes this processed file directly, so raw MADIS is never needed.**
7. `CreateDataLoaders`: train = `Data_List[:int(0.7n)]`, val = next `int(0.2n)`, test = rest. For `n_years=5` this gives 3/1/1.
8. Training uses `EvaluateModel`. The loss is from `Utils/LossFunctions.GetLossFunction`.
9. `SaveTest` pickles `Preds_*`, `Targets_*`, `Times_*` and attention maps into `data_path/output_saving_path`.

### Processed station NetCDF schema (the contract the Brazil adapter must satisfy)

Produced by `Madis.createRawFile` / `Madis.createFile`:

```
dims:   stations (int, 1..n), time (datetime64[ns], hourly, full calendar year, UTC-naive)
vars:   u, v, temp, dewpoint, solar_radiation          (stations, time) float — NaN-free (filled)
        u_is_real, v_is_real, temp_is_real,
        dewpoint_is_real, solar_radiation_is_real       (stations, time) bool — True where observed
        elv                                             (stations) float, metres
        lon, lat                                        (stations) float, degrees
```

Units follow MADIS:
- `u`, `v`: m/s, with `u = cos(270°−wd)·ws` and `v = sin(270°−wd)·ws` (wd in degrees, meteorological convention).
- `temp`, `dewpoint`: **Kelvin**.
- `solar_radiation`: W/m².

Fill rule:
- Per station and variable, forward-fill along time, then back-fill.
- Any remaining NaN takes the variable's global mean.
- `*_is_real` records the pre-fill availability.

### Station shapefile schema

Files: `madis/stations/stations_{start}_{end}_{file_name}.shp` (all) and `..._{filtered_file_name}.shp` (filtered).
- Columns: `num` (int, count of valid hours over the period) and `geometry` (Point lon/lat).
- CRS: EPSG:4326.

Name stems are built in `MetaStation.__init__`:
- `file_name = '%.2f_%.2f_%.2f_%.2f' % (lon_low, lon_up, lat_low, lat_up) + shapefile.stem`
- `filtered_file_name = file_name + f'_filtered_{control_ratio}'`

### Invariants (breaking any of these silently corrupts training)

- **I1 — station order.** The station order in the processed NetCDF must equal the row order of the *filtered* shapefile.
  - `Madis.createFile` selects with `coords_raw in coords`, so point equality must survive the shapefile round-trip.
  - Write coordinates rounded to 4 decimals in both places.
- **I2 — time axis.** The time axis is exactly `pd.date_range(f'{y}-01-01', f'{y}-12-31 23:00', freq='h')`, with 8760 or 8784 hours.
- **I3 — ERA5 alignment.** ERA5 `time` covers the same hours and has the same UTC convention.
- **I4 — no NaN.** No NaN in filled variables. Normalizers (`MixData` min/max/mean/std) use `np.min`/`np.mean` and will return NaN otherwise.
- **I5 — latitude order.** ERA5 latitude must be descending, and longitude must be in [−180, 180].

---

## 2. Known defects to fix (Phase 1)

| # | Location | Defect | Fix |
|---|---|---|---|
| D1 | `Main.py` (`years = range(2024-n_years, 2024)`), `MetaStation.py` (`start_year = 2023-n_years+1`, `2023` in filenames, `range(start_year, 2024)`), `Madis.py` (`Meta-{2024-n_years}-2023`) | End year hardcoded to 2023 | Add args `start_year`, `end_year`. Derive `years = range(start_year, end_year+1)`. Thread both into `MetaStation` and `Madis`. Keep `n_years` as a derived value for backward compatibility. |
| D2 | `Main.CreateDataLoaders` | Split by `int(0.7n)`, `int(0.2n)` | Add explicit `train_years`, `val_years`, `test_years` lists. Default: reproduce the old behaviour exactly (2019–21 / 2022 / 2023). Assert the three are disjoint and ⊆ `years`. |
| D3 | `MetaStation.generate_filtered_station_table` | `n_years*366*24` | Use the true hour count of the period: `len(pd.date_range(start, end_of_period, freq='h'))`. |
| D4 | `ERA5.load_ERA5_monthly` | `region='World'` never configurable | Add arg `era5_region` (default `'World'`) and pass it through `BuildExternalNetwork` → loader. |
| D5 | `ERA5Interpolated.make_interpolated` (×3) | `.drop(['expver','number'])` crashes when absent | `.drop_vars(['expver','number'], errors='ignore')`. |
| D6 | `Utils/LossFunctions.py` (`GetLossFunction`, `GetLossFunctionReport`, `SetupSaveMetrics`) | `is_real` (`logical`) is computed in `EvaluateModel.GetIsReal()` but the CUSTOM lambdas drop it, so **the loss and metrics include filled values**. | Add arg `use_is_real_mask` (default `False`, so the NE-US reproduction is bit-identical). When `True`, pass `logical` into `custom_loss`/`wind_loss`. `GetIsReal()` returns a NumPy array: convert it to a `torch.bool` tensor on the output's device. The report and save metrics must also normalise by the **count of real entries**, not `out.numel()`. Add a unit test with a synthetic mask. |
| D7 | `Main.SaveTest` | Test `is_real` masks are not saved | Also pickle `IsReal_{metric}_min.pkl`, aligned with `Preds`. This is needed for the hour-of-day and per-station analysis. |
| D8 | `LocalLauncher.py` | Hardcoded `data_path` and settings | Replace with a YAML config loader: `python Source/run.py --config configs/<name>.yaml`. Keep `LocalLauncher.py` working by delegating to it. |
| D9 | — | No `requirements.txt` | Add pinned requirements, plus a `notebooks/colab_setup.ipynb` cell block (install, mount Drive, run). |

Rules for Phase 1:
- Every new arg needs a default that reproduces the current behaviour.
- Write the arg into `params.json` (as `LocalLauncher` already does).
- Add it to `Arg_Parser.py` for parity.

**Acceptance P1:**
- `pytest tests/` passes.
- On the NE-US Zenodo data, with `configs/neus_repro.yaml` (Transformer, ERA5 Nearest, `n_neighbors_e2m=1`, `lead_hrs=1`, `epochs=2`), the run completes. The first-epoch train loss must match an unmodified-repo run with the same seed to within floating-point noise. Record both numbers in `docs/repro_log.md`.

---

## 3. Phase 2 — Reproduce NE-US (smoke test plus a short run)

1. `scripts/neus/download_zenodo.py` downloads `https://zenodo.org/records/15346612/files/WindDataNE-US.zip?download=1`.
   - Verify md5 `6d95a6e785d7249d697cec29ed6d7c67` and unzip to `$DATA_ROOT/neus/`.
   - Inspect the extracted tree and reconcile it with the README layout. If the zip nests an extra folder, point `data_path` at the right level and do not move files.
2. Run `configs/neus_repro.yaml` for 20 epochs on the Transformer with ERA5.
3. Run the same config again on the GNN with ERA5.
4. Log the test metrics.
   - Paper reference, Transformer + ERA5, averaged over leads: wind-vector error 0.50 m/s, T RMSE 1.44 °C, Td RMSE 1.55 °C.
   - A single 1 h lead and 20 epochs (vs 200 in the paper) will not match exactly. Record the gap; do not tune to close it.

**Acceptance P2:** both runs finish, and `docs/repro_log.md` holds a metrics table plus the exact command lines used.

---

## 4. Phase 3 — Brazil data acquisition scripts

All scripts live in `scripts/brazil/`. Requirements for every script:
- Idempotent: skip files that already exist and validate.
- argparse CLI with `--data-root` and `--years`.
- `logging` with no `print`.
- Exit non-zero on failure.

### 4.1 Region definition — `build_region.py`

- Default study region is **coastal and inner NE Brazil**: CE, RN, PB and PE.
  - Default bbox: lon ∈ [−41.5, −34.5], lat ∈ [−9.5, −2.5].
  - Make the states a CLI list.
- Source: IBGE state boundaries.
  - Preferred: the `geobr` Python package (`read_state`, year 2020).
  - Fallback: IBGE *malhas* API GeoJSON.
- Dissolve the states, buffer by 0.1° (mirrors `northeastern_buffered`), and write `Shapefiles/Regions/ne_brazil_buffered.{shp,...}` in EPSG:4326.
- Print the resulting bounds: they define `file_name`.

### 4.2 INMET download — `download_inmet.py`

- Source: annual archives from the INMET historical data portal (`https://portal.inmet.gov.br/dadoshistoricos`).
  - Expected URL pattern: `https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip`. **Verify by requesting one year first.** If it fails, stop and report; do not scrape other mirrors silently.
- Years: 2019–2023 by default (CLI configurable).
- Retries with backoff, streamed download, and a size check.
- Extract to `$DATA_ROOT/brazil/inmet/raw/{year}/`.

### 4.3 INMET → LocalizedWeather adapter — `inmet_to_localized.py` (core deliverable)

#### Parsing — `scripts/brazil/inmet_parser.py`, a separately testable module

Write it defensively, because the format has changed across years. **Inspect real files before finalising**, and log the header of one file per year.

Expected format (verify):
- Encoding `latin-1`, separator `;`, decimal `,`.
- The first ~8 lines are metadata (`REGIAO`, `UF`, `ESTACAO`, `CODIGO (WMO)`, `LATITUDE`, `LONGITUDE`, `ALTITUDE`, `DATA DE FUNDACAO`), then the header row.
- Detect the header row by searching for the line that contains a date column. Do not hardcode the skip count.
- Date/time columns vary by year:
  - ≥ 2019: `Data` (`YYYY/MM/DD`) and `Hora UTC` (`HHMM UTC`)
  - older files: `DATA (YYYY-MM-DD)` and `HORA (UTC)` (`HH:MM`)
- Normalise column names: strip accents, upper-case, collapse whitespace, then match by **substring**:

| target | substring match (after normalisation) | unit in file |
|---|---|---|
| ws | `VENTO, VELOCIDADE HORARIA` | m/s |
| wd | `VENTO, DIRECAO HORARIA` | degrees |
| temp | `TEMPERATURA DO AR - BULBO SECO` | °C |
| dewpoint | `TEMPERATURA DO PONTO DE ORVALHO` | °C |
| solar | `RADIACAO GLOBAL` | kJ/m² per hour |

- Missing-value sentinel is `-9999`; also treat empty strings as missing.

**Timezone.** The ≥ 2019 header says UTC, but third-party packages disagree.
- Add a check: for each station, compute the mean diurnal temperature cycle.
- The maximum must fall at 15–19 UTC (12–16 local).
- Log the distribution of argmax hours across stations and **fail loudly** if the median falls outside 14–20 UTC.

#### QC — after parsing, before conversion

Set an observation to NaN if any of these hold:
- `ws < 0` or `ws > 50`
- `wd ∉ [0, 360]`
- `temp ∉ [−5, 50] °C`
- `dewpoint > temp + 0.5`
- `dewpoint ∉ [−20, 35] °C`
- radiation `< 0`

Additional rules:
- If ws = 0, set u = v = 0 regardless of direction.
- **Flat-line check:** if a variable holds an identical value for ≥ 6 consecutive hours (excluding ws = 0 and night-time radiation), set those hours to NaN.
- Log the per-station rejection fraction.

#### Conversion

- `u = cos(270°−wd)·ws`, `v = sin(270°−wd)·ws`. Reuse the repo's exact formula.
- `temp`, `dewpoint`: + 273.15.
- `solar_radiation = kJ/m²/h ÷ 3.6` → W/m².
- Station metadata: lon/lat in decimal degrees (comma decimals), `elv` from `ALTITUDE`.
- Key stations by **station code** (e.g. `A305`). If coordinates of the same code differ across years by more than 0.01°, log it and take the most recent value.

#### Assembly — per region, for all years together

1. Keep stations whose point lies within the dissolved region polygon. Use the same polygon as the shapefile; `MetaStation` uses `within`.
2. Reindex each station to the I2 time axis for each year.
3. Valid-hour count: `num` = number of hours over the whole period where **u, v and temp** are all observed. This mirrors MADIS validobs, which checks wind speed, direction and temperature.
4. Write:
   - the all-stations shapefile (`num`, `geometry`, plus extra columns `code`, `name`; extra columns are harmless)
   - the filtered shapefile, `num >= control_ratio * total_hours`
5. `--control-ratio` (default 0.8). Also emit a CSV table of retained station counts for ratios {0.6, 0.7, 0.8, 0.9}, for the report.
6. Per year, write `madis/processed/Meta-{start}-{end}/madis_{year}_{filtered_file_name}.nc` following the §1 schema exactly, in filtered-shapefile row order (I1).
7. **Compute all filenames by importing the naming helper from `Source/`** (add `Source/Utils/Paths.py`, used by both `MetaStation`/`Madis` and the adapter). Never re-implement the f-strings.

#### Optional extra variables

`elv` and the raw (unfilled) series can go to `brazil/inmet/clean/{code}.parquet` for analysis.

### 4.4 ERA5 download — `download_era5.py`

- Primary: CDS API (`cdsapi`, dataset `reanalysis-era5-single-levels`), reusing the request of `Source/DataDownload/ERA5/DownloadERA5.ipynb`.
  - Variables: `10m_u_component_of_wind`, `10m_v_component_of_wind`, `2m_dewpoint_temperature`, `2m_temperature`, `surface_net_solar_radiation`, `surface_pressure`.
  - Hourly, NetCDF.
  - `area = [N, W, S, E]` = region bounds expanded by **2°**, so the loader's floor/ceil ±1.5° slice is fully covered.
- One file per month: `$DATA_ROOT/brazil/ERA5/<era5_region>/surface_{year}_{month}.nc`. Set `era5_region: NEBrazil` in the config (D4).
- Requires `~/.cdsapirc`. Fail with a clear message if it is missing.
- Fallback flag `--source arco`: read the public ARCO-ERA5 Zarr on GCS with xarray + gcsfs and write files in the **same** CDS-like format. Required conversions:
  - rename `time→valid_time`
  - names `u10,v10,t2m,d2m,ssr`
  - longitude 0..360 → −180..180, then sort
  - latitude descending
  - accumulation of `ssr` consistent with the CDS hourly field (document it)
  - Thanks to D5, missing `expver`/`number` is fine.
- Post-check per file: time length equals the hours in the month, no NaN, lat descending, bounds cover the region ±1.5°.

### 4.5 Dataset validation — `validate_dataset.py`

Run after 4.3 and 4.4 and write `docs/brazil_data_report.md`. It reports:
- **Counts:** station count per control ratio, and the map of retained stations (PNG).
- **Per-variable availability:** fraction of `is_real` per station and year, as a heatmap PNG.
- **Diurnal sanity (timezone, I3):** hour-of-day of the max temperature for stations and for ERA5 at the nearest grid point. The two must agree within ±1 h.
- **Paper Fig. 1 analogue:** for one date per season, ρ² between ERA5-nearest and INMET for u, v, T, Td. This gives a direct motivation slide for Brazil.
- **Schema check:** open each processed file and assert I1–I5.

**Acceptance P3:**
- All scripts run end to end for 2019–2023.
- `validate_dataset.py` passes every assertion.
- The report exists, and the retained station count at ratio 0.8 is stated.
- If fewer than ~25 stations survive at 0.8, report it and propose a ratio. Do not choose one silently.

---

## 5. Phase 4 — Train on Brazil

Config `configs/ne_brazil.yaml` starts from `neus_repro.yaml` and changes:

```yaml
data_path: $DATA_ROOT/brazil
shapefile_path: Shapefiles/Regions/ne_brazil_buffered.shp
start_year: 2019
end_year: 2023
train_years: [2019, 2020, 2021]
val_years: [2022]
test_years: [2023]
madis_control_ratio: 0.8        # from P3 report
era5_region: NEBrazil
n_neighbors_e2m: 1              # 8 for GNN runs with InterpolationType.none
n_neighbors_h2m: 0
interpolation_type: 0           # Nearest
use_is_real_mask: true          # D6 — required for INMET
lead_hrs: 1
epochs: 50
```

- Network construction: `FULLY_CONNECTED` for the Transformer (the default in LocalLauncher). Use `DELAUNAY` for the GNN, as in the paper.
- First run: `epochs: 2` with the Transformer, to check memory/time per epoch on Colab. Then the full runs.

**Acceptance P4:** Transformer + ERA5 and GNN + ERA5 both trained, with test metrics logged in `docs/results_brazil.md`.

---

## 6. Phase 5 — Evaluation scripts (`scripts/eval/`)

All metrics are computed on **real observations only** (using the `IsReal_*` pickles from D7), in physical units.

- **`baselines.py`.** On the test year, compute:
  - (a) ERA5-nearest interpolation at the target time
  - (b) persistence, $\hat{\mathbf{w}}_{t+l} = \mathbf{w}_t$
  - Report the same metrics as the model.
- **`metrics.py`.** Shared functions:
  - wind-vector error $\overline{\lVert(\hat u-u,\hat v-v)\rVert_2}$
  - RMSE and MAE for $T$ and $T_d$
  - bias
  - Bootstrap 95% CI by resampling **days** (block bootstrap; do not resample hours independently).
- **`diurnal_analysis.py`** (experiment E3).
  - Error by local hour of day (UTC−3) for model vs ERA5 vs persistence.
  - Stratify coastal stations (distance to coastline < 20 km, from the IBGE coastline or the Natural Earth 10m coastline) versus inland.
  - Output a PNG and a CSV.
- **`station_density.py`** (experiment E4).
  - Add config arg `station_keep_fraction` with `station_subsample_seed`.
  - Applied in `MetaStation` after filtering: a deterministic random subset of rows.
  - The filtered shapefile name must include the fraction and seed, so files do not collide.
  - Driver script runs fractions {1.0, 0.75, 0.5, 0.25} × seeds {0, 1, 2} and evaluates on a **fixed held-out set of stations**, so the metric is comparable across fractions. Implement held-out stations as stations present in the subset network whose loss is masked during training via `is_real`. Discuss the design in the PR before implementing it if this is unclear.
- **`lead_time_sweep.py`** (experiment E5, optional). Configs for `lead_hrs` ∈ {1, 6, 24}.
- **`make_tables.py`.** Collect all results into LaTeX tables (booktabs) for the presentation and report.

---

## 7. Experiment map (what the outputs serve)

| ID | Question | Produced by |
|---|---|---|
| E1 | Does the fork reproduce NE-US? | Phase 2 |
| E2 | Gain of Transformer/GNN + ERA5 over ERA5 interpolation and persistence in NE Brazil | Phase 4 + `baselines.py` |
| E3 | Does the model correct the ERA5 diurnal (sea-breeze) error at coastal stations? | `diurnal_analysis.py` |
| E4 | Sensitivity to station density | `station_density.py` |
| E5 | Decay of gain with lead time (optional) | `lead_time_sweep.py` |

---

## 8. Repository layout to create

```
configs/                  neus_repro.yaml, ne_brazil.yaml, (generated) density/*.yaml
Source/run.py             YAML-driven entry point (D8)
Source/Utils/Paths.py     single source of truth for all data filenames
scripts/neus/             download_zenodo.py
scripts/brazil/           build_region.py, download_inmet.py, inmet_parser.py,
                          inmet_to_localized.py, download_era5.py, validate_dataset.py
scripts/eval/             metrics.py, baselines.py, diurnal_analysis.py,
                          station_density.py, lead_time_sweep.py, make_tables.py
tests/                    test_loss_mask.py, test_inmet_parser.py (fixture: small synthetic CSV
                          in INMET format, both header variants), test_schema.py (I1–I5 on a
                          tiny synthetic processed file), test_paths.py
docs/                     repro_log.md, brazil_data_report.md, results_brazil.md
notebooks/colab_setup.ipynb
requirements.txt
```

---

## 9. Working rules for Claude Code

- Read the relevant `Source/` file before editing it. Keep the authors' code style, and keep edits minimal and local.
- Every behaviour change has a default that preserves the original results (D1–D8). The reproduction must not drift.
- Never hand-write data filenames outside `Source/Utils/Paths.py`.
- Do not commit data. Add `data/`, `*.nc`, `*.zip`, `ModelOutputs/` and `*.pkl` to `.gitignore`.
- Downloads: never disable TLS verification. If a source is unreachable, stop and report; do not switch to an unofficial mirror.
- When the real INMET files contradict §4.3 (column names, header layout, timezone), trust the files, update this plan's §4.3 table, and note the change in `docs/brazil_data_report.md`.
- Ask before choosing among scientifically meaningful options that this plan leaves open: control ratio, held-out station design (E4), coastal distance threshold.
- Commit after each phase with a message `phase N: <summary>`.

## 10. Order of execution (checklist)

- [ ] P1 — D1–D9 plus tests; old-vs-new first-epoch loss matches
- [ ] P2 — NE-US Transformer/GNN + ERA5 reproduced, `repro_log.md`
- [ ] P3 — region, INMET, adapter, ERA5, validation report
- [ ] P4 — Brazil training runs
- [ ] P5 — baselines, diurnal, density (and optionally lead-time) analyses; LaTeX tables
