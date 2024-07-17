import glob
import os
from multiprocessing import Pool

import numpy as np
from dateutil.parser import parse
# load all files
from sunpy.map import Map
from tqdm import tqdm

sdo_files = np.array(sorted(glob.glob('/mnt/sdo-jsoc/193/*.fits')))
stereo_a_files = np.array(sorted(glob.glob('/mnt/data-download/stereo_synchronic_prep/195/*_A.fits')))
stereo_b_files = np.array(sorted(glob.glob('/mnt/data-download/stereo_synchronic_prep/195/*_B.fits')))

# parse dates (hours)
dates_sdo = [parse(os.path.basename(f)[7:-11]) for f in sdo_files]
dates_stereo_a = [parse(os.path.basename(f)[:-7]) for f in stereo_a_files]
dates_stereo_b = [parse(os.path.basename(f)[:-7]) for f in stereo_b_files]

# find intersecting
intersecting_dates = set.intersection(*map(set, [dates_sdo, dates_stereo_a, dates_stereo_b]))

# filter
filterd_sdo_files = sdo_files[[d in intersecting_dates for d in dates_sdo]]
filterd_stereo_a_files = stereo_a_files[[d in intersecting_dates for d in dates_stereo_a]]
filterd_stereo_b_files = stereo_b_files[[d in intersecting_dates for d in dates_stereo_b]]

assert len(filterd_sdo_files) == len(filterd_stereo_a_files) == len(
    filterd_stereo_b_files), 'Invalid processing. Number of files need to match!'


def _get_separation(d):
    sdo_file, stereo_a_file, stereo_b_file = d
    sdo_map = Map(sdo_file)
    stereo_a_map = Map(stereo_a_file)
    stereo_b_map = Map(stereo_b_file)
    lon_diff = [np.abs((sdo_map.heliographic_longitude - stereo_a_map.heliographic_longitude).value),
                np.abs((sdo_map.heliographic_longitude - stereo_b_map.heliographic_longitude).value),
                np.abs((stereo_a_map.heliographic_longitude - stereo_b_map.heliographic_longitude).value)]
    lat_diff = [np.abs((sdo_map.heliographic_latitude - stereo_a_map.heliographic_latitude).value),
                np.abs((sdo_map.heliographic_latitude - stereo_b_map.heliographic_latitude).value),
                np.abs((stereo_a_map.heliographic_latitude - stereo_b_map.heliographic_latitude).value)]
    return sum(lat_diff), sum(lon_diff), sdo_map.date


with Pool(os.cpu_count()) as p:
    iter_data = list(zip(filterd_sdo_files, filterd_stereo_a_files, filterd_stereo_b_files))
    separation = [r for r in tqdm(p.imap(_get_separation, iter_data), total=len(filterd_sdo_files))]

separation = np.array(separation)

print('Max latitudinal separation', separation[np.argmax(separation[:, 0])])
print('Max longitudinal separation', separation[np.argmax(separation[:, 1])])

print('STD latitudinal separation', np.std(separation[:, 0]))
print('STD longitudinal separation', np.std(separation[:, 1]))
