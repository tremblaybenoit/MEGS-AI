# predictions for stereo A, B, SDO --> timeseries
# ground-truth EVE
# everything per wavelength
# shifted correlation 

# Output: x - Time, y - Irradiance (one channel) for stereo A/B/ SDO prediction


import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from dateutil.parser import parse, isoparse
from netCDF4 import Dataset
from tqdm import tqdm
from multiprocessing import Pool

import datetime
import dateutil.parser as dt

from s4pi.irradiance.inference import ipredict
from s4pi.irradiance.utilities.data_loader import IrradianceDataModule
#from matplotlib import rcParams
#rcParams['font.family'] = 'sans-serif'
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['text.usetex'] = True

flare_class_mapping = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}

def unnormalize(y, eve_norm):
    eve_norm = torch.tensor(eve_norm).float()
    norm_mean = eve_norm[0]
    norm_stdev = eve_norm[1]
    y = y * norm_stdev[None].to(y) + norm_mean[None].to(y)
    return y

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
    # eve_dates = eve_dates[~np.any(np.isnan(eve_data), 1)]
    # eve_indices = eve_indices[~np.any(np.isnan(eve_data), 1)]
    # eve_data = eve_data[~np.any(np.isnan(eve_data), 1)]

    return eve_data, eve_dates, eve_indices

def _find_reference_dates(aia_iso_dates, goes_path="/mnt/goes.csv", goes_class=4e-6):
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
    flare_df = flare_df[peak_flux >= goes_class]
    # use the first wl as reference (AIA dates should be close)
    aia_ref_dates = aia_iso_dates
    # remove all dates during flaring times
    aia_ref_dates = [d for d in tqdm(aia_ref_dates) if not ((d >= flare_df.start_time) & (d <= flare_df.end_time)).any()]
    return aia_ref_dates

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

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-checkpoint_path', type=str,
                default='/home/benoit_tremblay_23/training/ef3_circe/3110',
                help='path to the model checkpoint.')
p.add_argument('-checkpoint_file', type=str, 
                default='best_model_0.ckpt', 
                help='Filename to load checkpoint.')
p.add_argument('-goes_path', type=str,
               default='/mnt/goes.csv',
               help='path to the GOES data.')
p.add_argument('-goes_class', type=float,
               default=4e-6,
               help='Threshold applied to GOES data.')
p.add_argument('-eve_wl', type=str,
               default='/mnt/disks/aia_data/converted_data_8_channels/eve_wl_names.npy',
               help='path to the EVE norm')
p.add_argument('-eve_npy', type=str,
               default='/mnt/disks/aia_data/converted_data_8_channels/eve_converted.npy',
               help='path to converted EVE data')
p.add_argument('-aia_csv', type=str,
               default='/mnt/disks/aia-raw-6h/aia-linear-6hr-aiapy/matches/matches_256_limb.csv',
               help='path to the CSV with the AIA image stack paths.')
p.add_argument('-output_path', type=str,
               default='/home/benoit_tremblay_23/results/ef3_circe',
               help='path to save the results.')
args = p.parse_args()

result_path = args.output_path
os.makedirs(result_path, exist_ok=True)

# Initalize model
full_chkpt_path = os.path.join(args.checkpoint_path, args.checkpoint_file)
state = torch.load(full_chkpt_path)
model = state['model']
aia_wl = state['aia_wl']
eve_norm = model.eve_norm

# Init our model
data_module = IrradianceDataModule(args.aia_csv, args.eve_npy, aia_wl, 
                                   num_workers=os.cpu_count() // 2)
data_module.setup()

# Extract EVE data from dataloader
train_eve = torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module.train_ds]).numpy()
valid_eve = torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module.valid_ds]).numpy()
test_eve = torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module.test_ds]).numpy()

# Extract EUV images from dataloader
train_aia = torch.stack([torch.tensor(image) for image, eve in data_module.train_ds])
valid_aia = torch.stack([torch.tensor(image) for image, eve in data_module.valid_ds])
test_aia = torch.stack([torch.tensor(image) for image, eve in data_module.test_ds])

# Extract EVE dates from dataloader
train_dates = [date for date in data_module.train_dates]
valid_dates = [date for date in data_module.valid_dates]
test_dates = [date for date in data_module.test_dates]

# TODO: BT: HERE MODIFY PREDICTION STEP
# TODO: Add option to modify normalization
train_irradiance = [irr for irr in tqdm(ipredict(model, train_aia, return_images=False), total=len(train_aia))]
train_irradiance = torch.stack(train_irradiance).numpy()
np.save(os.path.join(result_path, 'train_irradiance.npy'), train_irradiance)

valid_irradiance = [irr for irr in tqdm(ipredict(model, valid_aia, return_images=False), total=len(valid_aia))]
valid_irradiance = torch.stack(valid_irradiance).numpy()
np.save(os.path.join(result_path, 'valid_irradiance.npy'), train_irradiance)

test_irradiance = [irr for irr in tqdm(ipredict(model, test_aia, return_images=False), total=len(test_aia))]
test_irradiance = torch.stack(test_irradiance).numpy()
np.save(os.path.join(result_path, 'test_irradiance.npy'), test_irradiance)

wl_names = np.load(args.eve_wl, allow_pickle=True)

fig, ax = plt.subplots(1, 1, figsize=(12, 4), sharex=True)
rows = [5, 10, 12, 0]
titles = ['(a) ', '(b) ', '(c) ', '(d) ']
factors = [1e5, 1e4, 1e4, 1e5]

ncols = 1
nrows = 4
# Font
font_size = 13
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
title_labels = ['(a) Fe XI', '(b) Fe XV', '(c) Fe XVI', '(d) Fe XVIII']
y_labels = [r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)', r'Spectral Irradiance ($10^{-4}$ W m$^{-2}$ nm$^{-1}$)', r'Spectral Irradiance ($10^{-4}$ W m$^{-2}$ nm$^{-1}$)', r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)']
filename = '/home/benoit_tremblay_23/results/ef3_circe/circe.png'
figsize_x = 2*5.25 * ncols
figsize_y = 4.5 * nrows
fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout = False)
'''
for j in enumerate(wl_names):
    i = rows[j]
    wl_name = wl_names[i]
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), sharex=True)
    ax.legend()
    ax.scatter(train_dates, train_eve[:,0,i], s = 2, label='MEGS-A Data', color='black')
    ax.scatter(valid_dates, valid_eve[:,0,i], s = 2, color='black')
    ax.scatter(test_dates, test_eve[:,0,i], s = 2, color='black')
    
    ax.scatter(train_dates, train_irradiance[:,i], s = 2, label='MEGS-AI (Training Set)')
    ax.scatter(valid_dates, valid_irradiance[:,i], s = 2, label='MEGS-AI (Validation Set)')
    ax.scatter(test_dates, test_irradiance[:,i], s = 2, label='MEGS-AI (Test Set)')
    
    ax.set_ylabel('Irradiance')
    ax.set_xlabel('Date')
    ax.legend()
    # plt.semilogy()
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, f'overview_{wl_name.strip()}.jpg'), dpi=300)
    plt.close(fig)
'''

for i, wl in enumerate(wl_names):
    err_rel = np.abs((test_eve[:,0,i]-test_irradiance[:,i])/test_eve[:,0,i])
    print(wl, np.mean(err_rel)*100.)

exit()

for i in range(nrows):
    for j in range(ncols):
        l = rows[i]
        wl_name = wl_names[l]
        spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols + 0.023,
                                right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 + 0.023,
                                bottom=0.10 / nrows + float(nrows-i-1) / nrows, top=float(nrows-i) / nrows - 0.10 / nrows,
                                wspace=0.00)
        ax = fig.add_subplot(spec[:, :])

        # vmax1 = 1.05*np.nanmax([np.nanmax(k[:, i*nrows+j, 0]) for k in metrics])
        # vmin1 = 0.95*np.nanmin([np.nanmin(k[:, i*nrows+j, 0]) for k in metrics])
        # ax.set_aspect((xmax-xmin)/(vmax1-vmin1))
        # ax.set_xlim([xmin, xmax])
        # ax.set_ylim([vmin1, vmax1])

        #for k in range(len(metrics)):
        #    ax.plot(x, metrics[k][:, i*ncols+j, 0], c=colors[k], label=labels[k], linewidth=1)
        
        ax.grid(True, linewidth=0.25)
        ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
        ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
        # ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
        # ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)            
        # ax.set_title(title_labels[i*ncols+j], fontsize=font_size, y=1.005)
        # ax.set_xlabel('Timestep', fontsize=font_size, labelpad=0.5)
        # ax.set_ylabel('Mean value', fontsize=font_size, labelpad=5.0)
        ax.scatter(train_dates, train_eve[:,0,l]*factors[i*ncols+j], s = 2, label='MEGS-A Data', color='black')
        ax.scatter(valid_dates, valid_eve[:,0,l]*factors[i*ncols+j], s = 2, color='black')
        ax.scatter(test_dates, test_eve[:,0,l]*factors[i*ncols+j], s = 2, color='black')
        ax.scatter(train_dates, train_irradiance[:,l]*factors[i*ncols+j], s = 2, label='MEGS-AI (Training Set)')
        ax.scatter(valid_dates, valid_irradiance[:,l]*factors[i*ncols+j], s = 2, label='MEGS-AI (Validation Set)')
        ax.scatter(test_dates, test_irradiance[:,l]*factors[i*ncols+j], s = 2, label='MEGS-AI (Test Set)')
        ax.set_title(title_labels[i*ncols+j], fontsize=font_size, y=1.005)
        ax.set_ylabel(y_labels[i*ncols+j], fontsize=font_size, labelpad=5.0)
        ax.set_xlabel('Date', fontsize=font_size, labelpad=0.5)
        ax.legend(loc='best', numpoints=1, markerscale=4)

# Draw
plt.draw()
plt.savefig(filename, format='png', dpi=300)
plt.close('all')