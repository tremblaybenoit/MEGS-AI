import argparse
import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset


def _load(eve_nc_path, matches_csv, output_path, output_path_normalization, output_path_names):
    """ Load EVE data, select matches and save as numpy file.

    Avoid loading the full dataset for each model training.
    Parameters
    ----------
    eve_nc_path: path to the NETCDF file.
    matches_csv: path to the CSV matches file.
    output_path: output path for the numpy files.

    Returns
    -------
    None
    """
    line_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])

    # load eve data
    eve_data_db = Dataset(eve_nc_path)
    eve_data = eve_data_db.variables['irradiance'][:]
    eve_data = eve_data[:, line_indices]

    # load matches
    matches_csv = pd.read_csv(matches_csv)
    eve_data = eve_data[matches_csv.eve_indices]

    # normalize data between 0 and max
    eve_mean = np.nanmean(eve_data, 0)
    eve_std = np.nanstd(eve_data, 0)
    eve_data = (eve_data - eve_mean[None]) / eve_std[None] # mean = -10 / stdev = 0.25

    # save eve data
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    np.save(output_path, eve_data.data)

    # save normalization
    normalizations = (eve_mean, eve_std)
    np.save(output_path_normalization, normalizations)

    # save wl names
    names = eve_data_db.variables['name'][:][line_indices]
    np.save(output_path_names, names.data)

    eve_data_db.close()

if __name__ == "__main__":
  p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('-eve_path', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_irradiance.nc",
                  help='path to the NETCDF file.')
  p.add_argument('-matches_table', type=str, default="/mnt/disks/preprocessed_data/AIA/matches_eve_aia.csv",
                  help='path to the CSV matches file.')
  p.add_argument('-output_data', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_converted.npy",
                  help='output path for the numpy files.')
  p.add_argument('-output_norm', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_normalization.npy",
                  help='output path for the numpy files.')
  p.add_argument('-output_wl', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_wl_names.npy",
                  help='output path for the wavelength names.')
  args = p.parse_args()

  _load(args.eve_path, args.matches_table, args.output_data, args.output_norm, args.output_wl)