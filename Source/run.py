# YAML-driven entry point: python Source/run.py --config configs/<name>.yaml

import argparse
import json
import os
from pathlib import Path

import yaml

import Main
from LocalLauncher import AttrDict
from PostProcessInputs import PostProcessArgs
from Settings.Settings import *

# The former LocalLauncher settings; a config overrides any of them.
DEFAULTS = dict(
    lead_hrs=1,
    n_neighbors_e2m=1,
    n_neighbors_h2m=0,
    hrrr_analysis_only=False,
    interpolation_type=InterpolationType.Nearest.value,
    output_saving_path='ModelOutputs/Transformer',
    coords=None,
    shapefile_path='Shapefiles/Regions/northeastern_buffered.shp',
    back_hrs=48,
    hidden_dim=128,
    lr=1e-4,
    epochs=20,
    batch_size=128,
    weight_decay=1e-4,
    model_type=ModelType.GNN.value,
    network_construction_method=NetworkConstructionMethod.FULLY_CONNECTED.value,
    n_years=5,
    start_year=None,
    end_year=2023,
    train_years=None,
    val_years=None,
    test_years=None,
    era5_region='World',
    use_is_real_mask=False,
    madis_control_ratio=.9,
    n_passing=4,
    n_neighbors_m2m=0,
    loss_function_type=LossFunctionType.CUSTOM.value,
    madis_vars_i=[EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value, EnvVariables.dewpoint.value],
    madis_vars_o=[EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value, EnvVariables.dewpoint.value],
    external_vars=[EnvVariables.u.value, EnvVariables.v.value, EnvVariables.temp.value],
    data_path=None,
    show_progress_bar=True,
    use_wb=False,
)


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown config keys in {config_path}: {sorted(unknown)}')

    args = AttrDict(DEFAULTS)
    args.update(config)
    if args.data_path is None:
        raise ValueError(f'data_path must be set in {config_path}')
    args.data_path = os.path.expandvars(args.data_path)
    return args


def run(args):
    outputPath = Path(args.data_path) / args.output_saving_path / 'params.json'
    outputPath.parent.mkdir(exist_ok=True, parents=True)
    with open(outputPath, 'w') as f:
        json.dump(dict(args), f)

    Main.Main(PostProcessArgs(args)).Run()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    run(load_config(parser.parse_args(argv).config))


if __name__ == '__main__':
    main()
