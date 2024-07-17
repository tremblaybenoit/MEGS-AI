import argparse
import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sunpy.map import Map
from astropy import units as u
from datetime import timedelta, datetime
from dateutil.parser import parse, isoparse

from netCDF4 import Dataset

from s4pi.irradiance.models.model import IrradianceModel
from s4pi.irradiance.utilities.data_loader import FITSDataset
from s4pi.irradiance.inference import ipredict, ipredict_uncertainty

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-chk_path', type=str,
                default='/mnt/training/data_1hr_cad/3110/epoch=65-step=45540.ckpt',
                help='path to the model checkpoint.')
p.add_argument('-normalization_path', type=str,
                default='/mnt/converted_data_1hr/eve_normalization.npy',
                help='path to the EVE normalization.')
p.add_argument('-eve_wl_names', type=str,
               default='/mnt/converted_data_1hr/eve_wl_names.npy',
               help='path to the EVE norm')
p.add_argument('-output_path', type=str,
               default='/mnt/results/irradiance_pred_results',
               help='path to save the results.')
args = p.parse_args()

os.makedirs(args.output_path, exist_ok=True)

# Load measurements from stereo B
dirs = ['171', '195', '284', '304']
path = '/mnt/nerf_data/stereo_2014_04_converted/'
eve_nc_path = '/mnt/EVE_irradiance.nc'
stereo_b_files = [sorted(glob.glob(f'{path}{dir}/*_B.fits')) for dir in dirs]
stereo_b_files = np.array(stereo_b_files).transpose()[::10]
dataset = FITSDataset(stereo_b_files, aia_preprocessing=False, resolution=256)

normalization = np.load(args.normalization_path)

# Find irradiance at STEREO B position
stereo_total_irradiance = [irr for irr in tqdm(ipredict_uncertainty(args.chk_path, dataset, normalization, return_images=False), total=len(dataset))]
stereo_total_irradiance = np.stack(stereo_total_irradiance)

# load eve data
line_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])
eve_data_db = Dataset(eve_nc_path)
eve_data = eve_data_db.variables['irradiance'][:]
eve_data = eve_data[:, line_indices]
# eve_date = eve_data_db.variables['isoDate'][:]
# eve_date = np.array([parse(d[:-1]) for d in eve_date])

eve_date_str = eve_data_db.variables['isoDate'][:]
# convert to naive datetime object
eve_date = np.array([isoparse(d).replace(tzinfo=None) for d in eve_date_str])

rot_freq = 360 / 27.26 

eve_mean, eve_std = normalization
wl_names = np.load(args.eve_wl_names, allow_pickle=True)

def plot_irradiance(mean, path, std=None):
    fig, ax = plt.subplots(1, 1, figsize=(4, 5))
    if std is not None:
        ax.bar(np.arange(0, len(mean)), mean / eve_mean, yerr=std / eve_mean,
               width=0.8, ecolor='red', capsize=8, color='#1288FF')
    else:
        ax.bar(np.arange(0, len(mean)), mean / eve_mean,
               width=0.8, capsize=8, color='#1288FF')
    ax.set_xticks(np.arange(0, len(mean)))
    ax.set_xticklabels(wl_names, rotation=45)
    # ax.set_yticks([])
    ax.set_ylim(.7, 1.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

plt.style.use('dark_background')
# Compare distance to Earth (longitudes)
# Find expected date and EVE data point (SDO Measurement)
for stereo_b_file, (pred_irradiance, std, _) in zip(stereo_b_files[:, 0], stereo_total_irradiance):
    s_map = Map(stereo_b_file)
    lon = s_map.heliographic_longitude.to(u.deg).value # lon is negative
    map_date = s_map.date.to_datetime()
    time_difference = timedelta(days=-lon / rot_freq)
    target_date = map_date + time_difference
    eve_idx = np.argmin(np.abs(eve_date - target_date))
    gt_irradiance = eve_data[eve_idx]
    # Compare predicted STEREO (B) irradiance to the actual EVE irradiance
    rae = np.abs(gt_irradiance - pred_irradiance) / (gt_irradiance + 1e-8) * 100
    print(rae.mean())
    plot_irradiance(pred_irradiance, os.path.join(args.output_path, f'{target_date.isoformat(timespec="hours")}_forecast.jpg'), std)
    plot_irradiance(gt_irradiance, os.path.join(args.output_path, f'{target_date.isoformat(timespec="hours")}_gt.jpg'))


