# Author: Qidong Yang & Jonathan Giezendanner

import os
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely import points

import Utils.Paths as Paths


class MetaStation(object):
    def __init__(self, lat_low, lat_up, lon_low, lon_up, n_years=5, control_ratio=0.9, shapefile_path=None,
                 data_path=Path(''), start_year=None, end_year=2023):

        self.lat_low = lat_low
        self.lat_up = lat_up
        self.lon_low = lon_low
        self.lon_up = lon_up
        self.control_ratio = control_ratio
        self.end_year = end_year
        self.start_year = end_year - n_years + 1 if start_year is None else start_year
        self.n_years = self.end_year - self.start_year + 1
        self.data_path = data_path
        self.shapefile_path = shapefile_path
        self.file_name = Paths.region_file_name(self.lon_low, self.lon_up, self.lat_low, self.lat_up,
                                                self.shapefile_path)
        self.filtered_file_name = Paths.filtered_file_name(self.file_name, self.control_ratio)

        self.all_station_file = Paths.station_file(data_path, self.start_year, self.end_year, self.file_name)
        self.station_file = Paths.station_file(data_path, self.start_year, self.end_year, self.filtered_file_name)
        self.data_dir = lambda year, month: data_path / f'madis/raw_monthly/mesonet/{year}/{month}.nc'

        Path(self.all_station_file).parent.mkdir(exist_ok=True, parents=True)

        if os.path.exists(self.all_station_file):
            self.stations_raw = gpd.read_file(self.all_station_file)
        else:
            print(f'All Station files does not exist at {self.all_station_file}')
            self.stations_raw = self.generate_station_table()

        if os.path.exists(self.station_file):
            self.stations = gpd.read_file(self.station_file)
        elif os.path.exists(self.all_station_file):
            print(f'Station files does not exist at {self.all_station_file}')
            self.stations = self.generate_filtered_station_table(self.stations_raw)

    def generate_station_table(self):
        print('Generating station table')
        counter = Counter([])

        for year in range(self.start_year, self.end_year + 1):
            print(f'Generating station table for year {year}', flush=True)
            for month in range(1, 13):
                print(f'Generating station table for month {month}', flush=True)

                data_dir = self.data_dir(year, month)
                data = xr.open_mfdataset(data_dir)

                wind_speed_check = ((data.windSpeedDD == b'S') + (data.windSpeedDD == b'V'))
                wind_direction_check = ((data.windDirDD == b'S') + (data.windDirDD == b'V'))
                temperature_check = ((data.temperatureDD == b'S') + (data.temperatureDD == b'V'))
                wind_speed_amplitude_check = (data.windSpeed < 50)
                validobs = wind_speed_check & wind_direction_check & temperature_check & wind_speed_amplitude_check

                data = data.sel(index=validobs)

                lat_obs = data.latitude.values
                lon_obs = data.longitude.values

                ind = (self.lat_low <= lat_obs) & (lat_obs <= self.lat_up)
                ind = ind & ((self.lon_low <= lon_obs) & (lon_obs <= self.lon_up))

                data = data.sel(index=ind)

                if len(data.index) > 0:
                    coords_sub = \
                        data[['longitude', 'latitude', 'year', 'month', 'day', 'hour']].to_pandas().reset_index(
                            drop=True).drop_duplicates()[['longitude', 'latitude']].values
                    counter = counter + Counter(points(coords_sub))

        counter = gpd.GeoDataFrame(pd.Series(counter).reset_index().rename(columns={'index': 'geometry', 0: 'num'}))

        counter = counter.set_crs(epsg=4326)

        if self.shapefile_path is not None:
            roi = gpd.read_file(self.shapefile_path).dissolve()
            counter = counter[counter.geometry.within(roi.iloc[0].geometry)]

        counter.to_file(self.all_station_file, index=False)

        return counter

    def generate_filtered_station_table(self, counter):

        n_hours = len(pd.date_range(f'{self.start_year}-01-01', f'{self.end_year}-12-31 23:00', freq='h'))
        counter = counter[counter.num >= n_hours * self.control_ratio]

        counter.to_file(self.station_file, index=False)

        return counter
