
import numpy as np
import dateutil.parser as dt
from tqdm import tqdm
import argparse
from netCDF4 import Dataset
import os

# TODO: Update variable names
def create_eve_netcdf(eve_raw_path, netcdf_outpath):
  """_summary_

  Args:
      eve_raw_path (str): path to directory of eve files
      netcdf_outpath (str): path to output cdf file

  Returns:
      netcdf file containing EVE data and metadata
  """
  os.makedirs(os.path.split(netcdf_outpath)[0], exist_ok=True)

  eve_date = np.load(eve_raw_path + "/iso.npy",allow_pickle=True)
  eve_irr = np.load(eve_raw_path + "/irradiance.npy",allow_pickle=True)
  eve_jd = np.load(eve_raw_path + "/jd.npy",allow_pickle=True)
  eve_logt = np.load(eve_raw_path + "/logt.npy",allow_pickle=True)
  eve_name = np.load(eve_raw_path + "/name.npy",allow_pickle=True)
  eve_wl = np.load(eve_raw_path + "/wavelength.npy",allow_pickle=True)

  ###################################################################################################################################################
  # CREATE NETCDF4 FILE
  ###################################################################################################################################################
  netcdfDB = Dataset(netcdf_outpath, "w", format="NETCDF4")
  netcdfDB.title = 'EVE spectral irradiance for specific spectral lines'

  # Create dimensions
  isoDate = netcdfDB.createDimension("isoDate", None)
  name = netcdfDB.createDimension("name", eve_name.shape[0])

  # Create variables and atributes
  isoDates = netcdfDB.createVariable('isoDate', 'S2', ('isoDate',))
  isoDates.units = 'string date in ISO format'

  julianDates = netcdfDB.createVariable('julianDate', 'f4', ('isoDate',))
  julianDates.units = 'days since the beginning of the Julian Period (January 1, 4713 BC)'

  names = netcdfDB.createVariable('name', 'S2', ('name',))
  names.units = 'strings with the line names'

  wavelength = netcdfDB.createVariable('wavelength', 'f4', ('name',))
  wavelength.units = 'line wavelength in nm'

  logt = netcdfDB.createVariable('logt', 'f4', ('name',))
  logt.units = 'log10 of the temperature'

  irradiance = netcdfDB.createVariable('irradiance', 'f4', ('isoDate','name',))
  irradiance.units = 'spectal irradiance in the specific line (w/m^2)'

  # Intialize variables
  isoDates[:] = eve_date
  julianDates[:] = eve_jd 
  names[:] = eve_name
  wavelength[:] = eve_wl
  logt[:] = eve_logt
  irradiance[:] = eve_irr

  netcdfDB.close()




if __name__ == "__main__":
  p = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('-eve_path', type=str, default="/mnt/disks/observational_data/EVE",
                  help='eve_path')
  p.add_argument('-output', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_irradiance.nc",
                  help='output')
  args = p.parse_args()

  eve_path = args.eve_path
  output = args.output
  create_eve_netcdf(eve_path, output)