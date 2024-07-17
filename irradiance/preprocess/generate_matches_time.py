import argparse
import glob
import logging
import os
from multiprocessing import Pool

import datetime
import dateutil.parser as dt
import numpy as np
import pandas as pd
from astropy.io.fits import getheader
from netCDF4 import Dataset
from tqdm import tqdm


flare_class_mapping = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}

def find_match(aia_date):
    """Worker function for multithreading.

    Find matches for a given date.
    Parameters
    ----------
    aia_date: date to match.

    Returns
    -------
    The match or None if no match was found.
    """
    nb_wavelengths = len(aia_iso_dates)
    # find min diff to aia observation
    eve_idx = np.argmin(np.abs(eve_dates - aia_date))

    aia_ans = [min(aia_iso_dates[i], key=lambda sub: abs(sub - aia_date)) for i in range(nb_wavelengths)]
    eve_ans = eve_dates[eve_idx]

    if abs(eve_ans - aia_date) <= threshold_eve and np.amax(
            [abs(aia_ans[i] - aia_date) for i in range(nb_wavelengths)]) <= threshold_aia:
        # save time difference between both observations
        time_delta = abs(eve_ans - aia_date)
        # get index from original array
        eve_index_match = eve_indices[eve_idx]
        eve_date_match = eve_dates[eve_idx]
        # get wavelength files
        aia_file_match = []
        for wl in range(nb_wavelengths):
            # find index in array
            aia_idx = aia_iso_dates[wl].index(aia_ans[wl])
            aia_file = aia_filenames[wl][aia_idx]
            if getheader(aia_file, 1)['QUALITY'] != 0:
                LOG.error('Invalid quality flag encountered')
                return None
            aia_file_match += [aia_file]
        return (eve_date_match, eve_index_match, time_delta, aia_file_match)
    return None


def _remove_flares(aia_matches, goes_path):
    """ Select AIA dates and remove flaring times (>C4)


    Parameters
    ----------
    aia_iso_dates: list of datetimes.
    goes_path: path to the GOES CSV file.

    Returns
    -------
    List of datetimes without flaring events.
    """
    flare_df = pd.read_csv(goes_path, parse_dates=['event_date', 'start_time', 'peak_time', 'end_time'])
    # parse goes classes to peak flux
    peak_flux = _to_goes_peak_flux(flare_df)
    # only flares >C4 have a significant impact on the full-disk images
    flare_df = flare_df[peak_flux >= 4e-6]

    # Create a mask to capture dates outside flaring times.  Uses the index of aia_matches, which is a date
    non_flare_mask = None    
    for i, row in flare_df.iterrows():
        if non_flare_mask is None:
            non_flare_mask = np.logical_or(aia_matches.index<row.start_time, aia_matches.index>row.end_time)
        else:
            temp_mask = np.logical_or(aia_matches.index<row.start_time, aia_matches.index>row.end_time)
            non_flare_mask = np.logical_and(non_flare_mask, temp_mask)

    # remove all dates during flaring times
    return aia_matches.loc[non_flare_mask, :]


def _load_aia_dates(aia_filenames, one_hour_cad = False):
    """ load AIA dates from filenames.

    Parameters
    ----------
    aia_filenames: path names of the AIA files.
    nb_wavelengths: total number
    debug: if True select only the first 10 dates.
    one_hour_cad: if True follow different naming convention

    Returns
    -------
    List of AIA datetimes.
    """
    if one_hour_cad:
        aia_dates = [[name.split(".")[-4][:-1] for name in wl_files] for wl_files in aia_filenames]
    else:
        aia_dates = [[name.split("_")[-1].split('.')[0] for name in wl_files] for wl_files in aia_filenames]

    aia_iso_dates = [[dt.isoparse(date) for date in wl_dates] for wl_dates in aia_dates]
    return aia_iso_dates


def load_valid_eve_dates(eve, line_indices=None):
    """ load all valid dates from EVE

    Parameters
    ----------
    eve: NETCDF dataset of EVE.
    line_indices: wavelengths used for data check (optional).

    Returns
    -------
    numpy array of all valid EVE dates and corresponding indices
    """
    # load and parse eve dates
    eve_date_str = eve.variables['isoDate'][:]
    # convert to naive datetime object
    eve_dates = np.array([dt.isoparse(d).replace(tzinfo=None) for d in eve_date_str])
    # get all indices
    eve_indices = np.indices(eve_dates.shape)[0]
    # find invalid eve data points
    eve_data = eve.variables['irradiance'][:]
    if line_indices is not None:
        eve_data = eve_data[:, line_indices]
    # set -1 entries to nan
    eve_data[eve_data < 0] = np.nan
    # set outliers to nan
    outlier_threshold = np.nanmedian(eve_data, 0) + 3 * np.nanstd(eve_data, 0)
    eve_data[eve_data > outlier_threshold[None]] = np.nan
    # filter eve dates and indices
    eve_dates = eve_dates[~np.any(np.isnan(eve_data), 1)]
    eve_indices = eve_indices[~np.any(np.isnan(eve_data), 1)]

    return eve_dates, eve_indices


def create_date_file_df(dates, files, sufix, dt_round='3min', debug=False):
    """ Parse a list of datetimes and files into dataframe

    Parameters
    ----------
    dates: list of dates
    files: list of filepaths
    sufix: string to use in the creation of the columns of the df.  Typically an
            AIA wavelength or the name 'hmi'
    dt_round: frequency alias to round dates
        see https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.round.html
    debug: Whether to use only a small set of the files (10)


    Returns
    -------
    pandas df with datetime index and paths
    """
    df1 = pd.DataFrame(data={'dates':dates, f'{sufix}':files, f'dates_{sufix}':dates})
    df1['dates'] = df1['dates'].dt.round(dt_round)
    # Drop duplictaes in case datetimes round to the same value
    df1 = df1.drop_duplicates(subset='dates', keep='first')
    df1 = df1.set_index('dates', drop=True)

    if debug:
        df1 = df1.iloc[::debug,:]

    return df1


def match_aia_times(all_iso_dates, all_filenames, all_sufixes, joint_df=None, debug=False, dt_round='3min'):
    """ Parses aia_iso_dates and compile lists at the same time"

    Parameters
    ----------
    all_iso_dates: list of AIA channel datetimes
    all_filenames: filenames of AIA files
    all_sufixes: list of strings to use in the creation of the columns of the df.  Typically
            AIA wavelengths or the name 'hmi'
    joint_df: pandas dataframe to use as a starting point
    debug: Whether to use only a small set of the files (10)
    dt_round: frequency alias to round dates
        see https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.round.html

    Returns
    -------
    pandas dataframe of matching datetimes
    """

    date_columns = []
    columns_to_keep = ['median_dates']
    for n, (aia_iso_dates, aia_filenames, sufix) in enumerate(zip(all_iso_dates, all_filenames, all_sufixes)):
        date_columns.append(f'dates_{sufix}')
        columns_to_keep.append(f'{sufix}')
        df = create_date_file_df(aia_iso_dates, aia_filenames, sufix, debug=debug)
        if n == 0 and joint_df is None:
            joint_df = df
        else:
            joint_df = joint_df.join(df, how='inner')

    joint_df['median_dates'] = joint_df.loc[:, date_columns].median(numeric_only=False, axis=1)
    joint_df = joint_df.loc[:, columns_to_keep]

    return joint_df


def _to_goes_peak_flux(flare_df):
    """ Parses goes class (str) to peak flux (float)

    Parameters
    ----------
    flare_df: goes data frame.

    Returns
    -------
    Numpy array with the peak flux values.
    """
    peak_flux = [flare_class_mapping[fc[0]] * float(fc[1:]) for fc in flare_df.goes_class]
    return np.array(peak_flux)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)-4s '
                            '[%(module)s:%(funcName)s:%(lineno)d]'
                            ' %(message)s')

    LOG = logging.getLogger()
    LOG.setLevel(logging.INFO)

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-eve_path', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_irradiance.nc", help='path to cdf file')
    p.add_argument('-aia_path', type=str, default="/mnt/disks/observational_data/AIA", help='path to directory of aia files')
    p.add_argument('-goes_path', type=str, default="/mnt/disks/observational_data/GOES/goes.csv", help='path to GOES csv file')
    p.add_argument('-output_path', type=str, default="/mnt/disks/preprocessed_data/AIA", help='path to store output csv file')
    p.add_argument('-aia_wl', type=str, nargs="+", default=["94", "131", "171", "193", "211", "304", "335", "1600", "1700"],
                   help='list of wavelengths that are needed')
    p.add_argument('-eve_cutoff', type=int, default=600,
                   help='cutoff for time delta (difference between AIA and EVE file in time) in seconds')
    p.add_argument('-aia_cutoff', type=int, default=600,
                   help='cutoff for time delta (difference between AIA images in different wavelengths) in seconds')
    p.add_argument('-debug', type=str2bool, default=False, help='Only process a few files (10)')

    args = p.parse_args()

    eve_path = args.eve_path
    aia_path = args.aia_path
    goes_path = args.goes_path
    output_path = args.output_path
    wavelengths = args.aia_wl
    eve_cutoff = args.eve_cutoff
    aia_cutoff = args.aia_cutoff
    debug = args.debug

    ###########################################################################
    # MAIN -- Create pairs of AIA and EVE observations
    # keep in __main__ to use global variables for multithreading --> much faster

    available_wavelengths = [d for d in os.listdir(aia_path) if os.path.isdir(aia_path+'/'+d)]
    intersection_wavelengths = [str(wl) for wl in sorted([int(wl) for wl in list(set(available_wavelengths).intersection(wavelengths))])]
    aia_sufixes = [f'AIA{wl}' for wl in intersection_wavelengths]

    if len(intersection_wavelengths) < len(wavelengths):
        LOG.log(f'Found only {available_wavelengths}, but the user request is {wavelengths}')

    nb_wavelengths = len(intersection_wavelengths)

    # LOADING AIA data
    # List of filenames, per wavelength

    aia_filenames = [[f for f in sorted(glob.glob(aia_path + '/%s/aia%s_*.fits' % (wl, wl)))] for wl in intersection_wavelengths] 
    # load aia dates
    aia_iso_dates = _load_aia_dates(aia_filenames)

    # Find matches between AIA wavelengths
    aia_matches = match_aia_times(aia_iso_dates, aia_filenames, aia_sufixes, dt_round=f'{aia_cutoff}s')

    if debug:
        aia_matches = aia_matches.iloc[0:100,:]

    # filter solar flares
    aia_matches = _remove_flares(aia_matches, goes_path)
    aia_ref_dates = aia_matches['median_dates'].to_numpy().astype('datetime64[us]').tolist()

    # LOADING EVE data --------------
    eve = Dataset(eve_path, "r", format="NETCDF4")
    line_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])
    eve_dates, eve_indices = load_valid_eve_dates(eve, line_indices)


    # define thresholds for max diff
    threshold_eve = datetime.timedelta(seconds=eve_cutoff)  # cutoff time for time match
    threshold_aia = datetime.timedelta(seconds=aia_cutoff)
    
    # looping through AIA filenames to find matching EVE files
    num_workers = os.cpu_count()
    with Pool(num_workers) as p:
        matches = [result for result in tqdm(p.imap(find_match, aia_ref_dates), total=len(aia_ref_dates)) if
                   result is not None]

    # unpack and create result data frame --> save as CSV
    result_matches = pd.DataFrame({"eve_dates": [eve_dates for eve_dates, _, _, _ in matches],
                                   "eve_indices": [eve_indices for _, eve_indices, _, _ in matches],
                                   "time_delta": [time_delta for _, _, time_delta, _ in matches], })
    
    result_matches['dates'] = result_matches['eve_dates'].dt.round(f'{aia_cutoff}s')
    result_matches = result_matches.set_index('dates', drop=True)
    result_matches = result_matches.join(aia_matches, how='inner')

    
    # Save csv with aia filenames, aia iso dates, eve iso dates, eve indices, and time deltas
    if not os.path.exists(output_path[:output_path.rfind("/")]):
        os.makedirs(output_path[:output_path.rfind("/")], exist_ok=True)
            
    result_matches.to_csv(output_path, index=False)

    eve.close()
