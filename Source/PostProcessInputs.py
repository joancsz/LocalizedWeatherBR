from Settings.Settings import *


def PostProcessArgs(args):
    args.model_type = ModelType(args.model_type)
    args.network_construction_method = NetworkConstructionMethod(args.network_construction_method)
    args.loss_function_type = LossFunctionType(args.loss_function_type)
    args.interpolation_type = InterpolationType(args.interpolation_type)
    args.hrrr_analysis_only = args.hrrr_analysis_only == 1

    args.madis_vars_i = [EnvVariables(variable_k) for variable_k in args.madis_vars_i]
    args.madis_vars_o = [EnvVariables(variable_k) for variable_k in args.madis_vars_o]
    args.external_vars = [EnvVariables(variable_k) for variable_k in args.external_vars]
    args.use_is_real_mask = args.use_is_real_mask == 1

    if args.start_year is None:
        args.start_year = args.end_year - args.n_years + 1
    args.n_years = args.end_year - args.start_year + 1
    years = list(range(args.start_year, args.end_year + 1))

    # legacy 70/20/10 split by year order, e.g. 2019-21 / 2022 / 2023
    n_train, n_val = int(args.n_years * 0.7), int(args.n_years * 0.2)
    if args.train_years is None:
        args.train_years = years[:n_train]
    if args.val_years is None:
        args.val_years = years[n_train:n_train + n_val]
    if args.test_years is None:
        args.test_years = years[n_train + n_val:]

    splits = [set(args.train_years), set(args.val_years), set(args.test_years)]
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2]), 'year splits overlap'
    assert set.union(*splits) <= set(years), f'year splits must be within {years}'

    return args
