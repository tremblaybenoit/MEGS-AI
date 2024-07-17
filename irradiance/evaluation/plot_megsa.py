import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import dateutil.parser as dt
from s4pi.irradiance.inference import ipredict
from s4pi.irradiance.utilities.data_loader import IrradianceDataModule



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
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-checkpoint', type=str, required=True, help='Path to checkpoint.')
    p.add_argument('-goes_data', type=str, default='/mnt/disks/observational_data/GOES/goes.csv',
                help='Path to the GOES data.')
    p.add_argument('-goes_class', type=float,
                default=4e-6,
                help='Threshold applied to GOES data.')
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

    result_path = args.output_path
    os.makedirs(result_path, exist_ok=True)

    # Initalize model
    state = torch.load(args.checkpoint)
    model = state['model']
    input_wl = state[state['instrument']]
    eve_norm = model.eve_norm

    # Init our model
    data_module = IrradianceDataModule(args.matches_table, args.eve_data, input_wl, 
                                    num_workers=os.cpu_count() // 2)
    data_module.setup()

    # Extract EVE data from dataloader
    train_eve = torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module.train_ds])
    valid_eve = torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module.valid_ds])
    test_eve = torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module.test_ds])

    # Extract EUV images from dataloader
    train_aia = torch.stack([torch.tensor(image) for image, eve in data_module.train_ds])
    valid_aia = torch.stack([torch.tensor(image) for image, eve in data_module.valid_ds])
    test_aia = torch.stack([torch.tensor(image) for image, eve in data_module.test_ds])

    # Extract EVE dates from dataloader
    train_dates = [date for date in data_module.train_dates]
    valid_dates = [date for date in data_module.valid_dates]
    test_dates = [date for date in data_module.test_dates]

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
    eve_wl = ['90', '131', '132', '171', '177', '180', '195', '202', '211', '255',
              '284', '304', '335', '368']
    eve_label = [f'{f.strip()} ({eve_wl[i]}Å)' for i, f in enumerate(wl_names)]


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf']
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


    # Plot general properties
    fig_format='png'
    fig_dpi=300
    fig_transparent=False 
    fig_lx=4.0 
    fig_ly=4.0 
    fig_lcb=5 
    fig_font=13
    fig_left=0.8 
    fig_right=0.4 
    fig_bottom=0.48 
    fig_top=0.32 
    fig_wspace=0.0 
    fig_hspace=0.0
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
            if row == 0 and col == ncols-1: fig_sizex = fig_specx[row][col+1]
            if row == nrows-1 and col == 0: fig_sizey = fig_specy[row+1][col]
    for l, wl_name in enumerate(wl_names):
        wl_name = wl_names[l]
        fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)
        spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left/fig_sizex + fig_specx[0][0]/fig_sizex, 
                                right= -fig_right/fig_sizex + fig_specx[0][1]/fig_sizex,
                                bottom=fig_bottom/fig_sizey + fig_specy[0][0]/fig_sizey, 
                                top= -fig_top/fig_sizey + fig_specy[1][0]/fig_sizey, 
                                wspace=0.00)
        ax = fig.add_subplot(spec[:, :])
        ax.grid(True, linewidth=0.25)
        ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
        ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
        ax.scatter(train_dates, train_eve[:,0,l]*factors[l], s = 2, label='MEGS-A - ' + wl_name.strip(), color='black')
        ax.scatter(valid_dates, valid_eve[:,0,l]*factors[l], s = 2, color='black')
        ax.scatter(test_dates, test_eve[:,0,l]*factors[l], s = 2, color='black')
        ax.scatter(train_dates, train_irradiance[:,l]*factors[l], s = 2, label='MEGS-AI (Training Set)')
        ax.scatter(valid_dates, valid_irradiance[:,l]*factors[l], s = 2, label='MEGS-AI (Validation Set)')
        ax.scatter(test_dates, test_irradiance[:,l]*factors[l], s = 2, label='MEGS-AI (Test Set)')
        ax.set_title(f'Irradiance - {eve_label[l]}', fontsize=font_size+2, y=1.005)
        ax.set_ylabel(y_labels[l], fontsize=font_size, labelpad=5.0)
        ax.set_xlabel('Date', fontsize=font_size, labelpad=0.5)
        ax.legend(loc='best', numpoints=1, markerscale=4, fontsize=font_size-1, labelspacing=0.2, ncol=ncol_leg)
        plt.draw()
        filename = os.path.join(result_path, f'overview_{str(l)}_{wl_name.replace(" ", "")}.png')
        plt.savefig(filename, format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
        plt.close('all')
