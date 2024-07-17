import argparse
import os
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import scipy as sp
import torch
from tqdm import tqdm
import pandas as pd
import datetime
from netCDF4 import Dataset
import matplotlib.dates as mdates
import dateutil.parser as dt
from s4pi.irradiance.inference import ipredict
from s4pi.irradiance.utilities.data_loader import IrradianceDataModule
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


# Absolute errors
def absolute_errors_scalar(v1, v2):
    return np.sqrt((v1 - v2) ** 2)


# Relative errors
def relative_errors_scalar(v1, v2):
    return np.sqrt((v1 - v2) ** 2) / np.sqrt(v1 ** 2 + 1.e-12)


# Absolute errors
def absolute_errors_vector(v1_x, v1_y, v2_x, v2_y):
    return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2)


# Relative errors
def relative_errors_vector(v1_x, v1_y, v2_x, v2_y):
    return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2) / np.sqrt(v1_x ** 2 + v1_y ** 2)


# Cosine similarity
def cosine_similarity_vector(v1_x, v1_y, v2_x, v2_y):
    return (v2_x * v1_x + v2_y * v1_y) / (np.sqrt(v2_x ** 2 + v2_y ** 2) * np.sqrt(v1_x ** 2 + v1_y ** 2))


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


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-checkpoint', type=str, required=True, help='Path to checkpoint.')
    p.add_argument('-eve_data', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_converted.npy",
                   help='Path to converted SDO/EVE data.')
    p.add_argument('-eve_norm', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_normalization.npy",
                   help='Path to converted SDO/EVE normalization.')
    p.add_argument('-eve_wl', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_wl_names.npy",
                   help='Path to SDO/EVE wavelength names.')
    p.add_argument('-matches_table', dest='matches_table', type=str,
                   default="/mnt/disks/preprocessed_data/AIA/matches_eve_aia.csv", help='matches_table')
    p.add_argument('-output_path', type=str, default='/home/benoit_tremblay_23/MEGS_AI',
                   help='path to save the results.')
    args = p.parse_args()

    # Selection of plot colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Output path
    result_path = args.output_path
    os.makedirs(result_path, exist_ok=True)

    # Initialize model
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
    eve_data = eve_data[eve_indices]
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

    # EVE (reference)
    test_eve = (torch.stack([torch.tensor(eve) for eve in eve_data])).numpy()
    # STEREO-B
    euvib_path = np.array(df['B/EUVI_stack'])
    test_dates = [date for date in eve_dates]
    img_euvib = (torch.stack([torch.tensor(np.load(image).transpose(0, 2, 1)) for image in euvib_path])).numpy()
    test_irradiance = [irr for irr in tqdm(ipredict(model, img_euvib, return_images=False), total=len(img_euvib))]
    test_irradiance = torch.stack(test_irradiance).numpy()

    # Plot general properties
    fig_format = 'png'
    fig_dpi = 300
    fig_transparent = False
    fig_lx = 4.0
    fig_ly = 4.0
    fig_lcb = 5
    fig_font = 13
    fig_left = 0.8
    fig_right = 0.8
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
    img_ratio = 1.0
    ncol_leg = 1
    scat_proj = 'scatter_density'
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [(0, '#ffffff'), (1e-20, '#440053'),
                                                                        (0.2, '#404388'), (0.4, '#2a788e'),
                                                                        (0.6, '#21a784'), (0.8, '#78d151'),
                                                                        (1, '#fde624'), ], N=256)
    scat_alpha = 1.0
    scat_ticks = (1, 1) 
    scat_axscale = ('linear', 'linear')
    scat_color = colors[0]
    ref_color = 'black'
    fit_color = colors[1]
    fit = True
    legend_loc = 'best'
    cb_font = fig_font
    ref_linew = 0.5
    ref_lines = '--'
    fit_linew = 0.25
    fit_lines = '-'
    scat_marker = '.'
    scat_markersize = 0.9
    scat_titlepad = 1.005
    scat_labelspad = (5, 3)
    scat_tickw = 1
    scat_tickl = 2.5
    scat_tickdir = 'out'
    legend_npoints = 1
    legend_scale = 4.0
    legend_spacing = 0.05
    legend_font = 10
    cb_pad = 0
    cb_tickw = 1
    cb_tickl = 2.5
    cb_dir = 'out'
    cb_rot = 270
    cb_labelpad = 16
    cb_side = 'right'
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
        
        # Figure dimensions
        fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)
        spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left/fig_sizex + fig_specx[0][0]/fig_sizex, 
                                right= -fig_right/fig_sizex + fig_specx[0][1]/fig_sizex,
                                bottom=fig_bottom/fig_sizey + fig_specy[0][0]/fig_sizey, 
                                top= -fig_top/fig_sizey + fig_specy[1][0]/fig_sizey, 
                                wspace=0.00)

        # Labels
        wl_name = wl_names[l]
        x_label = 'MEGS-A - ' + wl_name.strip()
        y_label = 'MEGS-AI (STEREO-B forecast)'

        # Coordinates
        x_val = eve_data[:, l].copy().flatten()*factors[l]
        y_val = test_irradiance[:, l].copy().flatten()*factors[l]
        x_min, x_max = np.nanmin(x_val), np.nanmax(x_val)
        y_min, y_max = np.nanmin(y_val), np.nanmax(y_val)
        xy_range = [np.nanmin([x_min, y_min]), np.nanmax([x_max, y_max])]

        # Plot scatter
        if scat_proj == 'scatter_density':
            ax = fig.add_subplot(spec[:, :], projection=scat_proj)
            I = ax.scatter_density(x_val, y_val, cmap=white_viridis)

        # Reference
        ax.set_aspect(1)
        ref_label = 'Reference (1:1)'
        ax.plot(xy_range, xy_range, label=ref_label, color=ref_color, linewidth=ref_linew, linestyle=ref_lines)
        # Fit
        if fit is True:
            # Fit
            slope, origin = np.polyfit(x_val.flatten(), y_val.flatten(), 1)
            if origin >= 0:
                fit_label = r'y = {0:.3f}x + {1:.3f}'.format(slope, origin)
            else:
                fit_label = r'y = {0:.3f}x - {1:.3f}'.format(slope, np.abs(origin))
            ax.plot(xy_range, [xy_range[0] * slope + origin, xy_range[1] * slope + origin],
                    label=fit_label, color=fit_color, linewidth=fit_linew, linestyle=fit_lines)

        # Grid
        ax.grid(True, linewidth=0.25)
        ax.set_xlim(xy_range)
        ax.set_ylim(xy_range)
        ax.set_xscale(scat_axscale[1])
        ax.set_yscale(scat_axscale[0])
            
        # Title
        ax.set_title(f'Irradiance - {eve_label[l]}', fontsize=font_size, y=scat_titlepad, wrap=True)
        # x/y-axis layout
        ax.get_yaxis().set_tick_params(which='both', direction=scat_tickdir, width=scat_tickw, length=scat_tickl,
                                       labelsize=font_size, left=True, right=True)
        ax.get_xaxis().set_tick_params(which='both', direction=scat_tickdir, width=scat_tickw, length=scat_tickl,
                                       labelsize=font_size, bottom=True, top=True)
        ax.set_ylabel(y_label, fontsize=font_size, labelpad=scat_labelspad[0])
        ax.set_xlabel(x_label, fontsize=font_size, labelpad=scat_labelspad[1])
        # Number of ticks
        ax.get_yaxis().set_major_locator(plt.MultipleLocator(scat_ticks[0]))
        ax.get_xaxis().set_major_locator(plt.MultipleLocator(scat_ticks[1]))

        # Legend
        plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(sp.stats.spearmanr(x_val, y_val)[0]))
        plt.plot([], [], ' ', label=r"MAE: {0:.3f}".format(np.nanmean(absolute_errors_scalar(x_val, y_val))))
        plt.plot([], [], ' ', label=r"MAPE: {0:.3f}%".format(100.*np.nanmedian(relative_errors_scalar(x_val, y_val))))
        ax.legend(loc=legend_loc, fontsize=legend_font, numpoints=legend_npoints, markerscale=legend_scale, 
                  labelspacing=legend_spacing, ncol=1, fancybox=False)
            
        # Colorbar
        divider = make_axes_locatable(ax)
        if scat_proj == 'scatter_density':
            cax = divider.append_axes(cb_side, size="{0}%".format(fig_lcb*fig_lx*ncols/fig_sizex), pad=cb_pad)
            cb = colorbar(I, extend='neither', cax=cax)
            cb.ax.tick_params(axis='y', direction=cb_dir, labelsize=cb_font, width=cb_tickw, length=cb_tickl)
            cb_label = 'Density of points'
            cb.set_label(cb_label, labelpad=cb_labelpad, rotation=cb_rot, size=cb_font)

        plt.draw()
        filename = os.path.join(result_path, f'forecast_scatterplot_{str(l)}_{wl_name.replace(" ", "")}.png')
        plt.savefig(filename, format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
        plt.close('all')

