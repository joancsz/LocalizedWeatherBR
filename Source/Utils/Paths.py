# Single source of truth for data filenames shared by Source/ and scripts/.

from pathlib import Path


def region_file_name(lon_low, lon_up, lat_low, lat_up, shapefile_path=None):
    file_name = '%.2f_%.2f_%.2f_%.2f' % (lon_low, lon_up, lat_low, lat_up)
    if shapefile_path is not None:
        file_name += Path(shapefile_path).stem
    return file_name


def filtered_file_name(file_name, control_ratio):
    return file_name + f'_filtered_{control_ratio}'


def station_file(data_path, start_year, end_year, file_name):
    return Path(data_path) / f'madis/stations/stations_{start_year}_{end_year}_{file_name}.shp'


def madis_processed_dir(data_path, start_year, end_year):
    return Path(data_path) / 'madis' / 'processed' / f'Meta-{start_year}-{end_year}'


def madis_processed_file(data_path, start_year, end_year, year, file_name):
    return madis_processed_dir(data_path, start_year, end_year) / f'madis_{year}_{file_name}.nc'


def era5_monthly_file(data_path, region, year, month):
    return Path(data_path) / 'ERA5' / region / f'surface_{year}_{month}.nc'
