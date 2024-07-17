import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import dateutil.parser as dt
from s4pi.irradiance.inference import ipredict, ipredict_uncertainty
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

    return eve_data, eve_dates, eve_indices


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-checkpoint_path', type=str, required=True, help='Path to checkpoint.')
    p.add_argument('-instrument', type=str, nargs='+', required=True, help='Instruments.')
    p.add_argument('-mission', type=str, default=None, nargs='+', 
                   help='Mission associated with instrument.')
    p.add_argument('-aia_wl', type=str, nargs='+', 
                   default=['94', '131', '171', '193', '211', '304', '335', '1600', '1700'],
                   help='SDO/AIA wavelengths.')
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
    cmaps_all = {'94': 'sdoaia94', '131': 'sdoaia131', '171': 'sdoaia171', '193': 'sdoaia193', '211': 'sdoaia211', 
                 '304': 'sdoaia304', '335': 'sdoaia335', '1600': 'sdoaia1600', '1700': 'sdoaia1700'}
    cmaps = {}

    # Output path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    
    # Models
    aia_wl = args.aia_wl
    mission = args.mission
    instrument = args.instrument
    nb_instrument = len(instrument)
    instrument_label = []
    input_label = {}
    checkpoint_path = args.checkpoint_path
    checkpoints = [f'{checkpoint_path}/{i}/{f}' for i in instrument for f in os.listdir(f'{checkpoint_path}/{i}/') if f.endswith('.ckpt')]

    # EVE
    eve_ln = np.load(args.eve_wl, allow_pickle=True)
    eve_wl = ['90', '131', '132', '171', '177', '180', '195', '202', '211', '255',
              '284', '304', '335', '368']
    nb_channels = len(eve_ln)
    eve_factors = [1e5, 1e6, 1e6, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e4, 1e4, 1e4, 1e5]
    eve_units = [r'Spectral Irradiance ($10^{-5}$ W m$^{-2}$ nm$^{-1}$)', 
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
    eve_label = [f'{f.strip()}\n{eve_wl[i]}Å' for i, f in enumerate(eve_ln)]
    megs_ai = {}

    # Infer irradiance and compute metrics
    for i in range(nb_instrument):
        # Load model checkpoint
        state = torch.load(checkpoints[i])
        model = state['model']
        wl = state[state['instrument']]

        input_label[instrument[i]] = [aia_wl[w]+' Å' for w in wl]

        # Create instrument label for legend
        wl_label = 'Å, '.join([aia_wl[w] for w in wl])
        if len(wl) > 1:
            if mission is not None:
                instrument_label.append(f'{mission[i]}/{instrument[i]} ({wl_label}Å)')
            else:
                instrument_label.append(f'{instrument[i]} ({wl_label}Å)')
        else:
            instrument_label.append(f'{wl_label}Å')
        cmaps[instrument[i]] = [cmaps_all[i] for i in [aia_wl[w] for w in wl]]

        # Initialize dataloader
        data_module = IrradianceDataModule(args.matches_table, args.eve_data, wl, num_workers=os.cpu_count() // 2)
        data_module.setup()
        if i == 0:
            # Extract EVE data from dataloader
            eve_norm = model.eve_norm
            test_eve = (torch.stack([unnormalize(eve, eve_norm) for image, eve in [data_module.test_ds[s] for s in range(2)]])).numpy()
            # Extract dates from dataloader
            test_dates = [date for date in data_module.test_dates.iloc[0:2]]
            megs_ai['EVE'] = test_eve
        # Extract EUV images from dataloader
        test_aia = (torch.stack([torch.tensor(image) for image, eve in [data_module.test_ds[s] for s in range(2)]])).numpy()
        # Infer irradiance
        megs_ai[instrument[i]] = [irr for irr in tqdm(ipredict_uncertainty(model, test_aia, return_images=True), total=len(test_aia))]

    # Plot general properties
    fig_format='png'
    fig_dpi=300
    fig_transparent=False 
    fig_lx=4.0 
    fig_ly=[[4.0], [4.0*3.0/5.0*2.0]] 
    fig_lcb=5 
    fig_font=13
    fig_left=0.8 
    fig_right=0.4 
    fig_bottom=0.48 
    fig_top=0.32 
    fig_wspace=0.0 
    fig_hspace=0.0
    # Panel properties
    nrows = 2
    ncols = 1
    font_size = fig_font
    fig_sizex = 0. 
    fig_sizey = 0.
    fig_specx = np.zeros((nrows, ncols+1)).tolist() 
    fig_specy = np.zeros((nrows+1, ncols)).tolist()
    img_ratio = [[1.0/3.0], [1.0/3.0]]
    width = 0.15*5/(nb_instrument+1)
    for row in range(nrows):
        for col in range(ncols):
            if img_ratio[row][col] > 1:
                fig_specy[row+1][col] = fig_specy[row][col] + fig_bottom + fig_ly[row][col]*img_ratio[row][col] + fig_top
                fig_specx[row][col+1] = fig_specx[row][col] + fig_left + fig_lx + fig_right
            else:
                fig_specy[row+1][col] = fig_specy[row][col] + fig_bottom + fig_ly[row][col] + fig_top
                fig_specx[row][col+1] = fig_specx[row][col] + fig_left + fig_lx/img_ratio[row][col] + fig_right
            if row == 0 and col == ncols-1: fig_sizex = fig_specx[row][col+1]
            if row == nrows-1 and col == 0: fig_sizey = fig_specy[row+1][col]
    ind = np.arange(nb_channels)
    # Loop over samples
    sample = 0
    for i, inst in enumerate(instrument):
        for s in range(1):
            # Context image
            fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)
            nb_img = len(input_label[inst])
            
            # Irradiance
            row = 1
            col = 0
            img = 0
            if nb_img > 5:
                nsubrows = 2
            else:
                nsubrows = 1
            for subrow in range(nsubrows):
                if nsubrows == 1:
                    nsubcols = nb_img
                else:
                    nsubcols = np.amin([nb_img - img, 5])
                    if subrow == 0:
                          rx = nsubcols
                for subcol in range(nsubcols):
                    #spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left/fig_sizex + fig_specx[row][col]/fig_sizex + subcol/np.sqrt(nb_img)*fig_specx[row][col]/fig_sizex, 
                    #                        right= -fig_right/fig_sizex + fig_specx[row][col+1]/fig_sizex - (2-subcol)/np.sqrt(nb_img)*fig_specx[row][col+1]/fig_sizex,
                    #                        bottom=fig_bottom/fig_sizey + fig_specy[row][col]/fig_sizey + (2-subrow)/np.sqrt(nb_img)*fig_specy[row][col]/fig_sizey, 
                    #                        top= -fig_top/fig_sizey + fig_specy[row+1][col]/fig_sizey - (2-subrow)/np.sqrt(nb_img)*fig_specy[row][col]/fig_sizey, 
                    spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left/fig_sizex + fig_specx[row][col]/fig_sizex + subcol/rx*fig_lx/img_ratio[row][col]/fig_sizex, 
                                            right= fig_left/fig_sizex + fig_specx[row][col]/fig_sizex + (subcol+1)/rx*fig_lx/img_ratio[row][col]/fig_sizex,
                                            bottom=fig_bottom/fig_sizey + fig_specy[row][col]/fig_sizey + (nsubrows-1-subrow)/nsubrows*fig_ly[row][col]/fig_sizey, 
                                            top=fig_bottom/fig_sizey + fig_specy[row][col]/fig_sizey + (nsubrows-1-subrow+1)/nsubrows*fig_ly[row][col]/fig_sizey, 
                                            wspace=0.00)
                    ax = fig.add_subplot(spec[:, :])
                    ax.imshow(megs_ai[inst][s][1][img], cmap = plt.get_cmap(cmaps[inst][img]), norm='asinh')
                    ax.text(0.025, 0.975, input_label[inst][img], horizontalalignment='left', verticalalignment='top', color = 'w', transform=ax.transAxes, fontsize=font_size-1)
                    ax.text(0.025, 0.075, f'{test_dates[s]} (UT)', horizontalalignment='left', verticalalignment='top', color = 'w', transform=ax.transAxes, fontsize=font_size-1)
                    ax.set_axis_off()
                    img = img + 1
            # Irradiance
            row = 0
            col = 0
            spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left/fig_sizex + fig_specx[row][col]/fig_sizex, 
                                    right= -fig_right/fig_sizex + fig_specx[row][col+1]/fig_sizex,
                                    bottom=fig_bottom/fig_sizey + fig_specy[row][col]/fig_sizey, 
                                    top= -fig_top/fig_sizey + fig_specy[row+1][col]/fig_sizey, 
                                    wspace=0.00)
            ax = fig.add_subplot(spec[:, :])
            for i, inst in enumerate(instrument):
                for l, wl in enumerate(eve_wl):
                    if l == 0: 
                        if i == 0: plt.bar(l + i*width, megs_ai['EVE'][s, 0, l], width, label='MEGS-A - Reference', color=colors[i])
                        plt.bar(l + (i+1)*width, megs_ai[inst][s][0][0][l], width, yerr=megs_ai[inst][s][0][1][l], label='MEGS-AI - '+instrument_label[i], color=colors[i+1], ecolor ='black', capsize=6)
                    else:
                        if i == 0: plt.bar(l + i*width, megs_ai['EVE'][s, 0, l], width, color=colors[i])
                        plt.bar(l + (i+1)*width, megs_ai[inst][s][0][0][l], width, yerr=megs_ai[inst][s][0][1][l], color=colors[i+1], ecolor ='black', capsize=6)
            plt.xticks(ind + 0.5*nb_instrument*width, eve_label)
            ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
            ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
            # ax.set_title(f'MEGS-AI irradiance derived from {mission[i]}/{inst} on {test_dates[s]} (UT)', fontsize=font_size+2, y=1.005)
            ax.set_title(f'MEGS-AI Virtual Irradiance Instrument for {mission[i]}/{inst} - {test_dates[s]} (UT)', fontsize=font_size+2, y=1.005)
            ax.set_ylabel(r'Spectral Irradiance (W m$^{-2}$ nm$^{-1}$)', fontsize=font_size, labelpad=5.0)
            ax.set_yscale('log')
            ax.legend(loc='best', numpoints=1, markerscale=4, fontsize=font_size-1, labelspacing=0.2, ncol=1)
            plt.draw()
            plt.savefig(output_path + f'{inst}_{sample}.png', format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
            plt.close('all')
            sample += 1    
            
