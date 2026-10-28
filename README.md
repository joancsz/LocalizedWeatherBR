[![arXiv](https://img.shields.io/badge/arXiv-2410.12938-b31b1b.svg)](https://arxiv.org/abs/2410.12938)
[![DOI](https://zenodo.org/badge/872621898.svg)](https://doi.org/10.5281/zenodo.16933936)

# LocalizedWeatherBR

Fork of LocalizedWeather ([code DOI](https://doi.org/10.5281/zenodo.16933936), Yang et al., arXiv:2410.12938)
that ports the station-downscaling models from the Northeastern US to **Northeast Brazil** (CE, RN, PB, PE):
INMET automatic stations take the place of MADIS, ERA5 stays the gridded input, and there is no HRRR.
It is a standalone research fork and will not be merged back upstream. The working plan is `PLAN_BRAZIL.md`.

## Running it

Everything runs from the notebooks, in order, on Colab (GPU for training) or on a PC. The first cell of each
notebook detects which: on Colab it clones the repo, installs `requirements.txt` and mounts Drive
(`DATA_ROOT=/content/drive/MyDrive/LocalizedWeather`); on a PC set the `DATA_ROOT` environment variable before
starting Jupyter. `DATA_ROOT` holds one folder per dataset: `neus/` and `brazil/`.

| notebook | phase | what it does |
|---|---|---|
| `notebooks/01_neus_reproduction.ipynb` | 1–2 | Zenodo NE-US data, unit tests, 20-epoch Transformer and GNN runs, test metrics |
| `notebooks/02_brazil_data.ipynb` | 3 | region shapefile, INMET download + conversion, ERA5 download |

Results and decisions are logged in `docs/repro_log.md`.

## Changes from upstream

- **Config-driven runs**: `python Source/run.py --config configs/<name>.yaml` replaces editing `LocalLauncher.py`
  (which still works). Unset keys take the old launcher defaults; `${DATA_ROOT}` is expanded in `data_path`.
- **Years and splits**: `start_year`, `end_year`, `train_years`, `val_years`, `test_years` replace the hardcoded
  2019–2023 and the 70/20/10 split; `era5_region` picks `ERA5/<region>/`.
- **Observation mask**: `use_is_real_mask: true` excludes gap-filled station values from the loss and metrics
  (off by default, so the NE-US numbers are unchanged). Test runs also save `IsReal_*.pkl` next to the predictions.
- **Fixes**: ERA5 files without `expver`/`number` load; the GNN + interpolated ERA5 combination (which crashed
  upstream) is rejected up front; k-NN uses `torch.cdist` so `pyg-lib` is not needed.
- **File names** come from one place, `Source/Utils/Paths.py`, shared by `Source/` and the scripts.
- **Scripts**: `scripts/neus/` (Zenodo download), `scripts/brazil/` (Brazil data pipeline),
  `scripts/eval/metrics.py` (test metrics in physical units, real observations only).
- Pinned `requirements.txt`; tests in `tests/` (`pytest tests/`).

## Brazil data pipeline (`scripts/brazil/`)

Each script takes `--data-root` and `--years`, logs what it does, skips finished work and exits non-zero on failure.

1. `build_region.py`: IBGE 2020 state boundaries for CE, RN, PB, PE, clipped to lon [-41.5, -34.5],
   lat [-9.5, -2.5] (drops Fernando de Noronha), buffered 0.1°. Bounds: lon [-41.52, -34.69], lat [-9.58, -2.68].
2. `download_inmet.py`: INMET annual archives (`portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip`) to
   `inmet/raw/{year}/`. The portal resets scripted connections from some networks; zips placed by hand in
   `inmet/zips/` are used as they are.
3. `inmet_to_localized.py` (parser in `inmet_parser.py`): parses both INMET CSV header variants, applies QC (range
   limits, dewpoint ≤ T + 0.5 °C, 6 h flat lines), converts to MADIS units (u/v m/s, K, W/m²), checks that the
   diurnal temperature peak falls at 14–20 UTC, and writes the station shapefiles and yearly processed NetCDFs
   that `MetaStation`/`Madis` load directly, plus `inmet/station_counts.csv` (stations kept per control ratio).
4. `download_era5.py`: monthly ERA5 from the Copernicus CDS (`.cdsapirc` in `~`, the repo root or `$DATA_ROOT`),
   over exactly the integer box the ERA5 loader slices (lon [-44, -33], lat [-12, -1]), with per-file checks.

Brazil data layout (`$DATA_ROOT/brazil/`, used as `data_path`):
```
Shapefiles/Regions/ne_brazil_buffered.shp
inmet/zips/{year}.zip, inmet/raw/{year}/, inmet/station_counts.csv
madis/stations/stations_{start}_{end}_<file_name>[_filtered_<ratio>].shp
madis/processed/Meta-{start}-{end}/madis_{year}_<file_name>[_filtered_<ratio>].nc
ERA5/NEBrazil/surface_{year}_{month}.nc
```
with `<file_name>` = `-41.52_-34.69_-9.58_-2.68ne_brazil_buffered`.

---

# Original README (upstream paper)

## Localized Weather

This repository contains the code and data for the paper ["Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data"](https://arxiv.org/abs/2410.12938).

The paper presents a novel multi-modal method that downscales gridded weather forecasts, such as ERA5 and HRRR, to provide accurate off-grid predictions. The model leverages both global data and local weather station observations from MADIS to make predictions that reflect both large-scale atmospheric dynamics and local weather patterns.

The model is evaluated on a surface wind prediction task and shows significant improvement over baseline methods, including ERA5 and HRRR interpolation and a multi-layer perceptron.

Use the following citation when the data or the model are used:
> Yang, Q.; Giezendanner, J.; Civitarese, D. S.; Jakubik, J.; Schmitt E.; Chandra, A.; Vila, J.; Hohl, D.; Hill, C.; Watson, C.; Wang, S.; Local Off-Grid Weather Forecasting with Multi-Modal Earth Observation Data. arXiv, October 2024. https://doi.org/10.48550/arXiv.2410.12938



# Model and data
## Data

The data for training, testing and validation can be found on [Zenodo](https://zenodo.org/records/15346612).

The following data is available:
- Shapefile of the Northeastern United States (NE-US, extracted from [NWS](https://www.weather.gov/gis/USStates))
- Shapefile containing the location and number of observations (2019-2023) of the MADIS stations in NE-US
- Processed hourly averaged [MADIS](https://madis.ncep.noaa.gov/) data for the NE-US (2019-2023)
- [ERA5](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation) re-analysis data for the NE-US (2019-2023), gridded and interpolated
- [HRRR](https://rapidrefresh.noaa.gov/hrrr/) analysis and forecast data for the NE-US (2019-2023), gridded and interpolated

For MADIS, ERA5 and HRRR, the following variables are available:
- u and v component of wind vector at 10 meters above ground
- temperature at 2 meters above ground
- dewpoint at 2 meters above ground

The data can also be generated from scratch.
For this, code is available under `Source/DataDownload/{ERA5/MADIS/HRRR}` for the raw data download, and in the respective data loaders for the data processing (`Source/Dataloader/{ERA5/HRRR/MetaStation/Madis/MixDataMLP}.py`).

## Code
The code is organised as follows (in `Source/`):
- `LocalLauncher.py`, and `SLURM_job_launcher.py` are two different launchers all eventually pointing at `Main.py` (the arguments from the slurm job launcher are parsed by `Arg_Parser.py`)
- `Main.py` contains the main code loop
- `EvaluateModel` contains the code for the evaluation of the model, as well as the back propagation
- the folder `Dataloader/` contains the data loaders for MADIS, HRRR and ERA5, and the combination of them, and the folder `Network/` the code for the network construction, for both the internal (MADIS) and external (ERA5 to MADIS) connections
- `Modules/GNN/MPNN.py` contains the code for the heterogeneous message passing neural network and calls `GNN_Layer_Internal/External.py`, the message passing sequences between the networks
- `Modules/Transformer/ViT.py` contains the code for the transformer and calls `Transformer/StationsEmbedding.py`.

### Code inputs
The code expects the following data structure, you need to specify the root data path in the launcher:
```
RootDataPath/
├── ERA5
│   ├── Interpolated
│   │   ├── era5interpolated_e2m_1_Nearest_2019_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   │   ├── era5interpolated_e2m_1_Nearest_2020_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   │   ├── era5interpolated_e2m_1_Nearest_2021_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   │   ├── era5interpolated_e2m_1_Nearest_2022_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   │   └── era5interpolated_e2m_1_Nearest_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│   └── Processed
│       ├── era5_2019_-83_-65_37_49.nc
│       ├── era5_2019_e2m_1_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       ├── era5_2019_e2m_8_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       ├── era5_2020_-83_-65_37_49.nc
│       ├── era5_2020_e2m_8_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       ├── era5_2021_-83_-65_37_49.nc
│       ├── era5_2021_e2m_8_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       ├── era5_2022_-83_-65_37_49.nc
│       ├── era5_2022_e2m_8_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       ├── era5_2023_-83_-65_37_49.nc
│       └── era5_2023_e2m_8_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
├── HRRR
│   └── Northeastern
│       ├── Interpolated
│       │   ├── HRRRinterpolated_h2m_1_Nearest_2019_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       │   ├── HRRRinterpolated_h2m_1_Nearest_2020_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       │   ├── HRRRinterpolated_h2m_1_Nearest_2021_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       │   ├── HRRRinterpolated_h2m_1_Nearest_2022_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       │   └── HRRRinterpolated_h2m_1_Nearest_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
│       └── Processed
│           └── Yearly
│               └── Madis
│                   └── -80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9
│                       ├── 2019.nc
│                       ├── 2020.nc
│                       ├── 2021.nc
│                       ├── 2022.nc
│                       └── 2023.nc
├── Shapefiles
│   └── Regions
│       ├── northeastern_buffered.cpg
│       ├── northeastern_buffered.dbf
│       ├── northeastern_buffered.geojson
│       ├── northeastern_buffered.prj
│       ├── northeastern_buffered.qmd
│       ├── northeastern_buffered.shp
│       └── northeastern_buffered.shx
└── madis
    ├── processed
    │   └── Meta-2019-2023
    │       ├── madis_2019_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    │       ├── madis_2020_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    │       ├── madis_2021_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    │       ├── madis_2022_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    │       └── madis_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc
    └── stations
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.cpg
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.dbf
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.prj
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.shp
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered.shx
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.cpg
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.dbf
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.prj
        ├── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.shp
        └── stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.shx
```
