import argparse
import logging
import os
import sys
from os.path import exists
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from s4pi.data.utils import loadMapStack, str2bool

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

# Order seems to be:  [0:94, 1:131, 2:171, 3:193, 4:211, 5:304, 6:335, 7:1600]


def standardize_stack(data_zip):

    stack_file = data_zip[0]
    mean = data_zip[1]
    std = data_zip[2]
    stack = (np.load(stack_file) - mean) / std
    np.save(stack_file, stack)
    return


def load_map_stack(aia_stack):
    # Extract filename from index_aia_i (remove aia_path)
    filename = (aia_stack[0].replace(aia_path, '')).split('_')[1]
    # Replace .fits by .npy
    filename = filename.replace('.fits', '.npy')

    output_file = matches_stacks + '/aia_' + filename
    # print(output_file)

    if exists(output_file):
        LOG.info(f'{filename} exists.')
        aia_stack = np.load(output_file)
        return aia_stack
    else:
        # if aia_extension_path is not None:
        #     calibration ='aiapy'
        # else:
        #     calibration = 'auto'
        calibration = 'aiapy'
        aia_stack = loadMapStack(aia_stack, resolution=aia_resolution, remove_nans=True,
                                 map_reproject=aia_reproject, aia_preprocessing=True, calibration=calibration)
        # Save stack
        np.save(output_file, aia_stack)
        data = np.asarray(aia_stack)
        aia_stack = None

        return data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    # Commands 
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-aia_path', dest='aia_path', type=str, default="/mnt/disks/observational_data/AIA",
                   help='aia_path')
    p.add_argument('-aia_resolution', dest='aia_resolution', type=int, default=256, help='Resolution of the output images')    
    p.add_argument('-aia_reproject', dest='aia_reproject', type=str2bool, default=False,
                   help='Use reprojection from heliographic map (remove off-limb).')
    p.add_argument('-aia_stats', dest='aia_stats', type=str,
                   default="/mnt/disks/preprocessed_data/AIA/256/stats.npz",
                   help='Stats.')
    p.add_argument('-matches_table', dest='matches_table', type=str,
                   default="/mnt/disks/preprocessed_data/AIA/matches_eve_aia.csv",
                   help='matches_table')
    p.add_argument('-matches_output', dest='matches_output', type=str,
                   default="/mnt/disks/preprocessed_data/AIA/256/matches_eve_aia.csv",
                   help='Updated matches')
    p.add_argument('-matches_stacks', dest='matches_stacks', type=str,
                   default="/mnt/disks/preprocessed_data/AIA/256",
                   help='Stacks for matches')
    p.add_argument('-debug', type=str2bool, default=False, help='Only process a few files (10)')
    args = p.parse_args()
    return args


if __name__ == "__main__":
    # Parser
    args = parse_args()
    aia_path = args.aia_path
    aia_resolution = args.aia_resolution
    aia_stats = args.aia_stats
    aia_reproject = args.aia_reproject  # Global variable?
    matches_file = args.matches_table
    matches_stacks = args.matches_stacks
    matches_output = args.matches_output
    debug = args.debug

    stats = {}

    # Load indices
    matches = pd.read_csv(matches_file)
    # if debug:
    # matches = matches.loc[0:10, :]
    # print(len(matches))

    # Extract filenames for stacks
    aia_files = []
    aia_columns = [col for col in matches.columns if 'AIA' in col]

    for index, row in tqdm(matches.iterrows()):
        aia_files.append(row[aia_columns].tolist())  # (channel, files)

    # Path for output
    os.makedirs(matches_stacks, exist_ok=True)

    # Extract filename from index_aia_i (remove aia_path)
    converted_file_paths = [matches_stacks + '/aia_' +
                            ((aia_files[i][0].replace(aia_path, '')).split('_')[1]).replace('.fits', '.npy')
                            for i in range(len(aia_files))]

    # Stacks
    print('Saving stacks')
    data = np.stack(process_map(load_map_stack, aia_files, max_workers=8, chunksize=5,
                                total=len(aia_files)))
    aia_min = np.min(data, axis=(0, 2, 3), keepdims=False)
    aia_max = np.max(data, axis=(0, 2, 3), keepdims=False)
    aia_mean = np.mean(data, axis=(0, 2, 3), keepdims=False)
    aia_std = np.stack([np.std(data[:, wl, :, :], keepdims=False) for wl in range(data.shape[1])])
    stats['AIA'] = {'mean': aia_mean, 'std': aia_std, 'min': aia_min, 'max': aia_max}
    data = None

    process_map(standardize_stack, zip(converted_file_paths,
                                       [aia_mean[:, None, None]] * len(converted_file_paths),
                                       [aia_std[:, None, None]] * len(converted_file_paths)), max_workers=8, chunksize=5)

    # Save
    if debug:
        matches = matches.loc[0:len(converted_file_paths), :]

    print('Saving Matches')
    np.savez(aia_stats, **stats)
    matches['aia_stack'] = converted_file_paths
    matches.to_csv(matches_output, index=False)
