import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import datetime
from netCDF4 import Dataset
import matplotlib.dates as mdates
import dateutil.parser as dt
from s4pi.irradiance.inference import ipredict


def hist_cdf(data, nb_bins=2000):

    # Dimensions
    nb_files, ny, nx = data.shape

    # Bins for histograms (of the normalized quantities)
    bins = np.linspace(np.nanmin(data), np.nanmax(data), nb_bins)
    width_bins = bins[1]-bins[0]

    # Compute histograms
    hist = np.zeros((nb_files, nb_bins-1))
    for i in range(nb_files):
        hist[i, :], _ = np.histogram(data[i, ...].flatten(), bins=bins, density=True)

    # Cumulative histogram
    cdf = np.cumsum(np.nanmean(hist*width_bins, axis=0))

    return bins, cdf


def hist_matching0(data_in, cdf_in, bins_in, cdf_out, bins_out, dtype=np.float32):

    # Points for interpolation (input bins contain the edges)
    bins_in = 0.5*(bins_in[:-1] + bins_in[1:])
    bins_out = 0.5*(bins_out[:-1] + bins_out[1:])

    # Interpolation
    cdf_tmp = np.interp(data_in.flatten(), bins_in.flatten(),
                        cdf_in.flatten())
    data_out = np.interp(cdf_tmp, cdf_out.flatten(), bins_out.flatten()).astype(dtype)

    return data_out.reshape(data_in.shape)


def hist_matching(data_in, data_tmp, nb_bins=2000, dtype=np.float32):

    # Standardize data
    data_in_mean = np.nanmean(data_in, dtype=dtype)
    data_in_std = np.nanstd(data_in, dtype=dtype)
    data_tmp_mean = np.nanmean(data_tmp, dtype=dtype)
    data_tmp_std = np.nanstd(data_tmp, dtype=dtype)
    data_in = (data_in - data_in_mean) / data_in_std
    data_tmp = (data_tmp - data_tmp_mean) / data_tmp_std

    # CDFs
    bins_in, cdf_in = hist_cdf(data_in, nb_bins=nb_bins)
    bins_out, cdf_out = hist_cdf(data_tmp, nb_bins=nb_bins)

    # Points for interpolation (input bins contain the edges)
    bins_in = 0.5 * (bins_in[:-1] + bins_in[1:])
    bins_out = 0.5 * (bins_out[:-1] + bins_out[1:])

    # Interpolation
    cdf_tmp = np.interp(data_in.flatten(), bins_in.flatten(),
                        cdf_in.flatten()).astype(dtype)
    data_out = np.interp(cdf_tmp, cdf_out.flatten(), bins_out.flatten()).astype(dtype)

    return data_out.reshape(data_in.shape)*data_tmp_std + data_tmp_mean


def unnormalize(y, eve_norm):
    eve_norm = torch.tensor(eve_norm).float()
    norm_mean = eve_norm[0]
    norm_stddev = eve_norm[1]
    y = y * norm_stddev[None].to(y) + norm_mean[None].to(y)
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-checkpoint', type=str, required=True, help='Path to checkpoint.')
    p.add_argument('-eve_data', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_converted.npy",
                help='Path to converted SDO/EVE data.')
    p.add_argument('-eve_norm', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_normalization.npy",
                help='Path to converted SDO/EVE normalization.')
    p.add_argument('-eve_wl', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_wl_names.npy",
                help='Path to SDO/EVE wavelength names.')
    p.add_argument('-matches_table', dest='matches_table', type=str,
                default="/mnt/disks/preprocessed_data/AIA/matches_eve_aia.csv",
                help='matches_table')
    p.add_argument('-output_path', type=str,
                default='/home/benoit_tremblay_23/MEGS_AI',
                help='path to save the results.')
    args = p.parse_args()

    # Selection of plot colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Output path
    result_path = args.output_path
    os.makedirs(result_path, exist_ok=True)

    # Initalize model
    state = torch.load(args.checkpoint)
    model = state['model']
    input_wl = state[state['instrument']]
    eve_norm = model.eve_norm

    # load eve data
    wl_names = np.load(args.eve_wl, allow_pickle=True)
    eve_wl = ['90', '131', '132', '171', '177', '180', '195', '202', '211', '255',
              '284', '304', '335', '368']
    eve_label = [f'{f.strip()} ({eve_wl[i]}Å)' for i, f in enumerate(wl_names)]
    line_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])
    eve_data_db = Dataset(args.eve_data)
    eve_data = eve_data_db.variables['irradiance'][:]
    eve_data = eve_data[:, line_indices]
    df = pd.read_csv(args.matches_table)
    eve_indices = np.array(df['eve_indices'])
    eve_dates = df['eve_dates']
    eve_delay = df['time_delta']
    eve_delay = [pd.to_timedelta(e).total_seconds() / 3600 / 24 for e in eve_delay]
    eve_data = eve_data[eve_indices]

    # AIA
    eve_data_aia = eve_data_db.variables['irradiance'][:]
    eve_data_aia = eve_data_aia[:, line_indices]
    df_aia = pd.read_csv("/mnt/disks/preprocessed_data/AIA/256/matches_eve_aia.csv")
    eve_indices_aia = np.array(df_aia['eve_indices'])
    eve_dates_aia = df_aia['eve_dates']
    eve_data_aia = eve_data_aia[eve_indices_aia]

    factors = [1e5, 1e6, 1e6, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e4, 1e4, 1e4, 1e5]
    y_labels = [r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-6}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-6}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)',
                r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)',
                r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-4}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-4}$ W m$^{-2}$ nm$^{-1}$)',
                r'Spectral Irradiance ($10^{-4}$ W m$^{-2}$ nm$^{-1}$)', 
                r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)']

    # AIA
    aia_path = np.array(df_aia['aia_stack'])
    test_dates_aia = [eve_dates_aia[i] for i in range(len(eve_dates_aia)) if
                      eve_dates_aia[i] < eve_dates[len(eve_dates) - 1]]
    img_aia = (torch.stack([torch.tensor(np.load(aia_path[i])[input_wl, ...].transpose(0, 2, 1)) for i in
                            range(len(test_dates_aia))])).numpy()
    aia_mean = np.mean(img_aia, axis=(0, 2, 3), keepdims=True)
    aia_std = np.std(img_aia, axis=(0, 2, 3), keepdims=True)
    test_irradiance_aia = [irr for irr in tqdm(ipredict(model, img_aia, return_images=False), total=len(img_aia))]
    test_irradiance_aia = torch.stack(test_irradiance_aia).numpy()
    # img_aia = None

    # EVE (reference)
    test_eve = (torch.stack([torch.tensor(eve) for eve in eve_data])).numpy()
    # STEREO-B
    euvib_path = np.array(df['B/EUVI_stack'])
    test_dates = [date for date in eve_dates]
    img_euvib = (torch.stack([torch.tensor(np.load(image).transpose(0, 2, 1)) for image in euvib_path])).numpy()
    euvib_mean = np.mean(img_euvib, axis=(0, 2, 3), keepdims=True)
    euvib_std = np.std(img_euvib, axis=(0, 2, 3), keepdims=True)
    test_irradiance = [irr for irr in tqdm(ipredict(model, img_euvib, return_images=False), total=len(img_euvib))]
    test_irradiance = torch.stack(test_irradiance).numpy()
    img_tmp = (img_euvib.copy()-euvib_mean)/euvib_std*aia_std+aia_mean
    test_irradiance2 = [irr for irr in tqdm(ipredict(model, img_tmp, return_images=False), total=len(img_euvib))]
    test_irradiance2 = torch.stack(test_irradiance2).numpy()
    nb_bins = 10000
    img_matched = (torch.stack([torch.tensor(hist_matching(img_euvib[:, i, ...], img_aia[:, i, ...],
                                                           nb_bins=nb_bins)) for i in range(len(input_wl))])).numpy().transpose(1, 0, 2, 3)
    test_irradiance_3 = [irr for irr in tqdm(ipredict(model, img_matched, return_images=False), total=len(img_matched))]
    test_irradiance_3 = torch.stack(test_irradiance_3).numpy()
    # breakpoint()
    # img_aia = np.stack([(img_aia[:, i, ...]-aia_mean[:, i, ...])/aia_std[:, i, ...] for i in range(len(input_wl))]).transpose(1, 0, 2, 3)
    # img_euvib = (img_euvib-euvib_mean)/euvib_std
    # img_tmp = (img_tmp-aia_mean)/aia_std
    # img_matched = (img_matched-aia_mean)/aia_std
    cdf_aia = [hist_cdf(img_aia[:, i, ...], nb_bins=nb_bins) for i in range(len(input_wl))]
    cdf_euvib = [hist_cdf(img_euvib[:, i, ...], nb_bins=nb_bins) for i in range(len(input_wl))]
    cdf_tmp = [hist_cdf(img_tmp[:, i, ...], nb_bins=nb_bins) for i in range(len(input_wl))]
    cdf_matched = [hist_cdf(img_matched[:, i, ...], nb_bins=nb_bins) for i in range(len(input_wl))]
    img_aia = None
    img_euvib = None
    img_tmp = None
    img_matched = None

    for l, wl_name in enumerate(input_wl):
        fig, ax = plt.subplots(figsize=(5.25, 4.5))
        x = (cdf_aia[l][0][:-1] + cdf_aia[l][0][1:]) / 2
        ax.plot(x, cdf_aia[l][1], linestyle="-", c=colors[0], label='SDO/AIA')
        x = (cdf_euvib[l][0][:-1] + cdf_euvib[l][0][1:]) / 2
        ax.plot(x, cdf_euvib[l][1], linestyle="dashed", c=colors[1], label='STEREO-B')
        x = (cdf_tmp[l][0][:-1] + cdf_tmp[l][0][1:]) / 2
        ax.plot(x, cdf_tmp[l][1], linestyle="dashed", c=colors[2], label='Correction')
        x = (cdf_matched[l][0][:-1] + cdf_matched[l][0][1:]) / 2
        ax.plot(x, cdf_matched[l][1], linestyle="dashed", c=colors[3], label='Histogram matching')
        ax.grid(True)
        ax.set_title('Cumulative Histograms')
        ax.set_xlabel('Normalized intensity')
        ax.set_ylabel('Cumulative Probability')
        ax.legend(loc='best', numpoints=1)
        plt.draw()
        filename_output = os.path.join(result_path, f'forecast_aia1_{str(l)}_cdf.png')
        plt.savefig(filename_output, format='png', dpi=300)
        plt.close('all')


    # Plot general properties
    fig_format = 'png'
    fig_dpi = 300
    fig_transparent = False
    fig_lx = 4.0
    fig_ly = 4.0
    fig_lcb = 5
    fig_font = 13
    fig_left = 0.8
    fig_right = 0.8  # 0.4
    fig_bottom = 0.48
    fig_top = 0.32
    fig_wspace = 0.0
    fig_hspace = 0.0
    # Panel properties
    nrows = 1
    ncols = 1
    font_size = fig_font
    fig_sizex = 0. 
    fig_sizey = 0.
    fig_specx = np.zeros((nrows, ncols+1)).tolist() 
    fig_specy = np.zeros((nrows+1, ncols)).tolist()
    img_ratio = 1.0/2.25
    ncol_leg = 1
    for row in range(nrows):
        for col in range(ncols):
            if img_ratio > 1:
                fig_specy[row+1][col] = fig_specy[row][col] + fig_bottom + fig_ly*img_ratio + fig_top
                fig_specx[row][col+1] = fig_specx[row][col] + fig_left + fig_lx + fig_right
            else:
                fig_specy[row+1][col] = fig_specy[row][col] + fig_bottom + fig_ly + fig_top
                fig_specx[row][col+1] = fig_specx[row][col] + fig_left + fig_lx/img_ratio + fig_right
            if row == 0 and col == ncols-1:
                fig_sizex = fig_specx[row][col+1]
            if row == nrows-1 and col == 0:
                fig_sizey = fig_specy[row+1][col]
    for l, wl_name in enumerate(wl_names):
        fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)
        spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left/fig_sizex + fig_specx[0][0]/fig_sizex, 
                                right=-fig_right/fig_sizex + fig_specx[0][1]/fig_sizex,
                                bottom=fig_bottom/fig_sizey + fig_specy[0][0]/fig_sizey, 
                                top=-fig_top/fig_sizey + fig_specy[1][0]/fig_sizey,
                                wspace=0.00)
        ax = fig.add_subplot(spec[:, :])
        ax2 = ax.twinx()
        ax.grid(True, linewidth=0.25)
        ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
        ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
        ax2.plot(pd.to_datetime(test_dates), eve_delay, color=colors[7], linestyle='-', zorder=0)
        ax.scatter(pd.to_datetime(test_dates), eve_data[:, l]*factors[l], s=2, label='MEGS-A - ' + wl_name.strip(),
                   color='black', zorder=10)
        ax.scatter(pd.to_datetime(test_dates_aia), test_irradiance_aia[:, l]*factors[l], s=2,
                   label='MEGS-AI (SDO/AIA forecast)', color=colors[0], zorder=20)
        ax.scatter(pd.to_datetime(test_dates), test_irradiance[:, l] * factors[l], s=2,
                   label='MEGS-AI (STEREO-B forecast)', color=colors[1], zorder=30)
        ax.scatter(pd.to_datetime(test_dates), test_irradiance2[:, l] * factors[l], s=2,
                   label='Correction (STEREO-B forecast)', color=colors[2], zorder=40)
        ax.scatter(pd.to_datetime(test_dates), test_irradiance_3[:, l] * factors[l], s=2,
                   label='Histogram (SDO/AIA forecast)', color=colors[3], zorder=50)
        ax2.set_ylabel(r'Delay $\Delta t$ (days)', fontsize=font_size+2, labelpad=5.0, rotation=90)
        ax.set_title(f'Irradiance - {eve_label[l]}', fontsize=font_size+2, y=1.005)
        ax.set_ylabel(y_labels[l], fontsize=font_size, labelpad=5.0)
        ax.set_xlabel('Date', fontsize=font_size, labelpad=0.5)
        ax.get_xaxis().set_major_locator(plt.MaxNLocator(9))
        ax.legend(loc='best', numpoints=1, markerscale=4, fontsize=font_size-1, labelspacing=0.2, ncol=ncol_leg)
        plt.draw()
        filename = os.path.join(result_path, f'forecast_aia1_{str(l)}_{wl_name.replace(" ", "")}.png')
        plt.savefig(filename, format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
        plt.close('all')
