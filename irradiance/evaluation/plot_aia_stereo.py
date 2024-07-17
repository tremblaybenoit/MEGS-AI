import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    p.add_argument('-frames', type=int, nargs='+', required=True, help='First/last frame.')
    p.add_argument('-aia_wl', type=str, nargs='+', 
                   default=['94', '131', '171', '193', '211', '304', '335', '1600', '1700'],
                   help='SDO/AIA wavelengths.')
    p.add_argument('-eve_data', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_converted.npy",
                   help='Path to converted SDO/EVE data.')
    p.add_argument('-eve_norm', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_normalization.npy",
                   help='Path to converted SDO/EVE normalization.')
    p.add_argument('-eve_wl', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_wl_names.npy",
                   help='Path to SDO/EVE wavelength names.')
    # Update with correct table
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
    t1 = args.frames[0]
    t2 = args.frames[1]
    aia_wl = args.aia_wl
    checkpoint_path = args.checkpoint_path
    # checkpoints = [f'{checkpoint_path}/{i}/{f}' for i in instrument for f in os.listdir(f'{checkpoint_path}/{i}/') if f.endswith('.ckpt')]
    checkpoint_aia = f'{checkpoint_path}/AIA/model.ckpt'
    checkpoint_euvi = f'{checkpoint_path}/EUVI/model.ckpt'
    # States
    state_aia = torch.load(checkpoint_aia)
    state_euvi = torch.load(checkpoint_euvi)
    # Models
    model_aia = state_aia['model']
    model_euvi = state_euvi['model']
    # Wavelengths
    wl_aia = state_aia[state_aia['instrument']]
    wl_euvi = state_euvi[state_euvi['instrument']]
    label_aia = 'SDO/AIA'
    label_euvia = 'STEREO-A/EUVI'
    label_euvib = 'STEREO-B/EUVI'
    wl_label = [aia_wl[w]+' Å' for w in wl_euvi] 
    cmaps = [cmaps_all[i] for i in [aia_wl[w] for w in wl_euvi]]

    data_module_aia = IrradianceDataModule(args.matches_table, args.eve_data, wl_aia, num_workers=os.cpu_count()//2)
    data_module_aia.setup()
    data_module_euvia = IrradianceDataModule(args.matches_table, args.eve_data, wl_euvi, num_workers=os.cpu_count()//2)
    data_module_euvia.setup()
    data_module_euvib = IrradianceDataModule(args.matches_table, args.eve_data, wl_euvi, num_workers=os.cpu_count()//2)
    data_module_euvib.setup()
    # Extract EVE data from dataloader
    eve_norm = model_aia.eve_norm

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
    eve_label = [f'{eve_wl[i]}Å' for i, f in enumerate(eve_ln)]  # Shorter labels


    # Plot properties
    fig_format='png'
    fig_dpi=300
    fig_transparent=False 
    fig_lx=4.0 
    fig_ly=4.0
    fig_lcb=5 
    fig_font=13
    fig_top=0.32 
    fig_wspace=0.0 
    fig_hspace=0.0
    nrows = 2
    ncols = 3
    font_size = fig_font
    fig_sizex = 0. 
    fig_sizey = 0.
    fig_specx = np.zeros((nrows, ncols+1)).tolist() 
    fig_specy = np.zeros((nrows+1, ncols)).tolist()
    fig_specleft = np.zeros((nrows, ncols)).tolist()
    fig_specright = np.zeros((nrows, ncols)).tolist() 
    fig_specbottom = np.zeros((nrows, ncols)).tolist()
    fig_spectop = np.zeros((nrows, ncols)).tolist() 
    fig_specy = np.zeros((nrows+1, ncols)).tolist()
    img_ratio = 1.0
    width = 0.75
    for row in range(nrows):
        for col in range(ncols):
            if col == 0:
                fig_left=0.8 
                fig_right=0.4 
            elif col == 1:
                fig_left=0. 
                fig_right=0.
            elif col == 2:
                fig_left=0.4 
                fig_right=0.8
            if row == 0:
                fig_bottom = 0.6
            elif row == 1:
                fig_bottom = 0.3
            if img_ratio > 1:
                # Horizontal
                fig_specleft[row][col] = fig_left + fig_specx[row][col]
                fig_specright[row][col] = fig_specleft[row][col] + fig_lx
                fig_specx[row][col+1] = fig_specright[row][col]+ fig_right
                # Vertical
                fig_specbottom[row][col] = fig_bottom + fig_specy[row][col]
                fig_spectop[row][col] = fig_specbottom[row][col] + fig_ly*img_ratio
                fig_specy[row+1][col] = fig_spectop[row][col] + fig_top
            else:
                # Horizontal
                fig_specleft[row][col] = fig_left + fig_specx[row][col]
                fig_specright[row][col] = fig_specleft[row][col] + fig_lx/img_ratio
                fig_specx[row][col+1] = fig_specright[row][col]+ fig_right
                # Vertical
                fig_specbottom[row][col] = fig_bottom + fig_specy[row][col]
                fig_spectop[row][col] = fig_specbottom[row][col] + fig_ly
                fig_specy[row+1][col] = fig_spectop[row][col] + fig_top
            if row == 0 and col == ncols-1: fig_sizex = fig_specx[row][col+1]
            if row == nrows-1 and col == 0: fig_sizey = fig_specy[row+1][col]

    # Loop over frames
    for t in range(t1, t2):
        
        # EVE (reference)
        test_eve = (torch.stack([unnormalize(eve, eve_norm) for image, eve in [data_module_aia.test_ds[t]]])).numpy()
        print(test_eve)
        #test_eve = (torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module_aia.test_ds])).numpy()
        #print(test_eve[t])
        #exit()
        test_dates_aia = data_module_aia.test_dates.iloc[t]
        #print(eve_norm)
        #print(data_module_aia.test_dates.iloc[t])
        # SDO
        test_aia = (torch.stack([torch.tensor(image) for image, eve in [data_module_aia.test_ds[t]]])).numpy()
        #print(test_aia.shape)
        #exit()
        #for i in range(9):
        #    print(np.nanmin(test_aia[0, i, :, :]), np.nanmax(test_aia[0, i, :, :]))
        #print(test_dates_aia)
        megs_aia = [irr for irr in tqdm(ipredict_uncertainty(model_aia, test_aia, return_images=True), total=len(test_aia))][0]
        # STEREO-A
        test_dates_euvia = data_module_euvia.test_dates.iloc[t]
        test_euvia = (torch.stack([torch.tensor(image) for image, eve in [data_module_euvia.test_ds[t]]])).numpy()
        megs_euvia = [irr for irr in tqdm(ipredict_uncertainty(model_euvi, test_euvia, return_images=True), total=len(test_euvia))][0]
        # STEREO-B
        test_dates_euvib = data_module_euvib.test_dates.iloc[t]
        test_euvib = (torch.stack([torch.tensor(image) for image, eve in [data_module_euvib.test_ds[t]]])).numpy()
        megs_euvib = [irr for irr in tqdm(ipredict_uncertainty(model_euvi, test_euvib, return_images=True), total=len(test_euvib))][0]

        # Figure
        fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)

        # Subplots
        for col in range(ncols):
            if col == 0:
                title = f'STEREO-B/EUVI - {test_dates_euvib}'
                instrument_label = label_euvib
                test_dates = test_dates_euvib
                inputs = megs_euvib[1]
                irr_mean, irr_err, irr_mi = megs_euvib[0]
                ylabel = r'Spectral Irradiance (W m$^{-2}$ nm$^{-1}$)'
                label_left = True
                label_right = False
                ylabel_position = "left"
                ylabel_rotation = 90
                ylabel_pad = 5.0
                iti_note = '(ITI-calibrated)'
            elif col == 1:
                title = f'SDO/AIA - {test_dates_aia}'
                instrument_label = label_aia
                test_dates = test_dates_aia
                inputs = [megs_aia[1][i] for i in wl_euvi]
                irr_mean, irr_err, irr_mi = megs_aia[0]
                ylabel = r''
                label_left=False
                label_right=False
                iti_note = ''
            elif col == 2:
                title = f'STEREO-A/EUVI - {test_dates_euvia}'
                instrument_label = label_euvia
                test_dates = test_dates_euvia
                inputs = megs_euvia[1]
                irr_mean, irr_err, irr_mi = megs_euvia[0]
                ylabel = r'Spectral Irradiance (W m$^{-2}$ nm$^{-1}$)'
                label_left=False
                label_right=True
                ylabel_position="right"
                ylabel_rotation=-90
                ylabel_pad = 20
                iti_note = '(ITI-calibrated)'

            # Context image
            row = 1
            spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_specleft[row][col]/fig_sizex, 
                                    right=fig_specright[row][col]/fig_sizex,
                                    bottom=fig_specbottom[row][col]/fig_sizey, 
                                    top=fig_spectop[row][col]/fig_sizey, wspace=0.00)
            ax0 = fig.add_subplot(spec[:, :])
            ax0.set_title(title, fontsize=font_size+2, y=1.005)
            ax0.set_axis_off()

            # Images
            nsubcols = 2
            nsubrows = 2
            for subrow in range(nsubrows):
                for subcol in range(nsubcols):
                    sublx = (fig_specright[row][col]-fig_specleft[row][col])
                    subly = (fig_spectop[row][col]-fig_specbottom[row][col])
                    subspec = fig.add_gridspec(nrows=1, ncols=1, left=fig_specleft[row][col]/fig_sizex+subcol/nsubcols*sublx/fig_sizex, 
                                               right=fig_specleft[row][col]/fig_sizex+(subcol+1)/nsubcols*sublx/fig_sizex,
                                               bottom=fig_specbottom[row][col]/fig_sizey+(nsubrows-subrow-1)/nsubrows*subly/fig_sizey, 
                                               top=fig_specbottom[row][col]/fig_sizey+(nsubrows-subrow)/nsubrows*subly/fig_sizey, wspace=0.00)
                    ax = fig.add_subplot(subspec[:, :])
                    ax.imshow(inputs[subrow*nsubcols+subcol], cmap=plt.get_cmap(cmaps[subrow*nsubcols+subcol]), norm='asinh')
                    ax.text(0.02, 0.98, wl_label[subrow*nsubcols+subcol], horizontalalignment='left', verticalalignment='top', 
                            color = 'w', transform=ax.transAxes, fontsize=font_size-1)
                    ax.text(0.98, 0.0, iti_note, horizontalalignment='right', verticalalignment='bottom', 
                            color = 'w', transform=ax.transAxes, fontsize=font_size-3)
                    ax.set_axis_off()

            # Irradiance
            row = 0
            ind = np.arange(nb_channels)
            spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_specleft[row][col]/fig_sizex, 
                                    right=fig_specright[row][col]/fig_sizex,
                                    bottom=fig_specbottom[row][col]/fig_sizey, 
                                    top=fig_spectop[row][col]/fig_sizey, wspace=0.00)
            ax = fig.add_subplot(spec[:, :])
            for l in range(nb_channels):
                if l == 0: 
                    plt.bar(l, irr_mean[l], width, yerr=irr_err[l], label='MEGS-AI - '+instrument_label, 
                            color=colors[0], ecolor ='black', capsize=6)
                else:
                    plt.bar(l, irr_mean[l], width, yerr=irr_err[l], color=colors[0], ecolor ='black', capsize=6)
            plt.xticks(ind, eve_label, rotation = 45)
            ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
            ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
            ax.tick_params(axis='y', which='both', left=True, right=True, labelleft=label_left, labelright=label_right)
            ax.set_title(f'MEGS-AI - {test_dates}', fontsize=font_size+2, y=1.005)
            ax.set_ylabel(ylabel, fontsize=font_size, labelpad=ylabel_pad, rotation=ylabel_rotation)
            ax.set_yscale('log')
            ax.yaxis.set_label_position(ylabel_position)
            ax.legend(loc='best', numpoints=1, markerscale=4, fontsize=font_size-1, labelspacing=0.2, ncol=1)
        plt.draw()
        plt.savefig(output_path + f'_AIA_EUVI_{t}.png', format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
        plt.close('all')

    
    # Loop over frames
    for t in range(t1, t2):
        
        # EVE (reference)
        if t == t1:
            test_eve = (torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module_aia.test_ds])).numpy()
            max_eve = [np.amax(test_eve[:, :, l]) for l in range(nb_channels)]
        test_dates_aia = data_module_aia.test_dates.iloc[t]
        # SDO
        test_aia = (torch.stack([torch.tensor(image) for image, eve in [data_module_aia.test_ds[t]]])).numpy()
        megs_aia = [irr for irr in tqdm(ipredict_uncertainty(model_aia, test_aia, return_images=True), total=len(test_aia))][0]
        # STEREO-A
        test_dates_euvia = data_module_euvia.test_dates.iloc[t]
        test_euvia = (torch.stack([torch.tensor(image) for image, eve in [data_module_euvia.test_ds[t]]])).numpy()
        megs_euvia = [irr for irr in tqdm(ipredict_uncertainty(model_euvi, test_euvia, return_images=True), total=len(test_euvia))][0]
        # STEREO-B
        test_dates_euvib = data_module_euvib.test_dates.iloc[t]
        test_euvib = (torch.stack([torch.tensor(image) for image, eve in [data_module_euvib.test_ds[t]]])).numpy()
        megs_euvib = [irr for irr in tqdm(ipredict_uncertainty(model_euvi, test_euvib, return_images=True), total=len(test_euvib))][0]

        # Figure
        fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)

        # Subplots
        for col in range(ncols):
            if col == 0:
                title = f'STEREO-B/EUVI - {test_dates_euvib}'
                instrument_label = label_euvib
                test_dates = test_dates_euvib
                inputs = megs_euvib[1]
                irr_mean, irr_err, irr_mi = megs_euvib[0]
                ylabel = r'Spectral Irradiance (W m$^{-2}$ nm$^{-1}$)'
                label_left = True
                label_right = False
                ylabel_position = "left"
                ylabel_rotation = 90
                ylabel_pad = 5.0
                iti_note = 'ITI-calibrated'
            elif col == 1:
                title = f'SDO/AIA - {test_dates_aia}'
                instrument_label = label_aia
                test_dates = test_dates_aia
                inputs = [megs_aia[1][i] for i in wl_euvi]
                irr_mean, irr_err, irr_mi = megs_aia[0]
                ylabel = r''
                label_left=False
                label_right=False
                iti_note = ''
            elif col == 2:
                title = f'STEREO-A/EUVI - {test_dates_euvia}'
                instrument_label = label_euvia
                test_dates = test_dates_euvia
                inputs = megs_euvia[1]
                irr_mean, irr_err, irr_mi = megs_euvia[0]
                ylabel = r'Spectral Irradiance (W m$^{-2}$ nm$^{-1}$)'
                label_left=False
                label_right=True
                ylabel_position="right"
                ylabel_rotation=-90
                ylabel_pad = 20
                iti_note = 'ITI-calibrated'

            # Context image
            row = 1
            spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_specleft[row][col]/fig_sizex, 
                                    right=fig_specright[row][col]/fig_sizex,
                                    bottom=fig_specbottom[row][col]/fig_sizey, 
                                    top=fig_spectop[row][col]/fig_sizey, wspace=0.00)
            ax0 = fig.add_subplot(spec[:, :])
            ax0.set_title(title, fontsize=font_size+2, y=1.005)
            ax0.set_axis_off()

            # Images
            nsubcols = 2
            nsubrows = 2
            for subrow in range(nsubrows):
                for subcol in range(nsubcols):
                    sublx = (fig_specright[row][col]-fig_specleft[row][col])
                    subly = (fig_spectop[row][col]-fig_specbottom[row][col])
                    subspec = fig.add_gridspec(nrows=1, ncols=1, left=fig_specleft[row][col]/fig_sizex+subcol/nsubcols*sublx/fig_sizex, 
                                               right=fig_specleft[row][col]/fig_sizex+(subcol+1)/nsubcols*sublx/fig_sizex,
                                               bottom=fig_specbottom[row][col]/fig_sizey+(nsubrows-subrow-1)/nsubrows*subly/fig_sizey, 
                                               top=fig_specbottom[row][col]/fig_sizey+(nsubrows-subrow)/nsubrows*subly/fig_sizey, wspace=0.00)
                    ax = fig.add_subplot(subspec[:, :])
                    ax.imshow(inputs[subrow*nsubcols+subcol], cmap=plt.get_cmap(cmaps[subrow*nsubcols+subcol]), norm='asinh')
                    ax.text(0.02, 0.98, wl_label[subrow*nsubcols+subcol], horizontalalignment='left', verticalalignment='top', 
                            color = 'w', transform=ax.transAxes, fontsize=font_size-1)
                    ax.text(0.99, 0.0, iti_note, horizontalalignment='right', verticalalignment='bottom', 
                            color = 'w', transform=ax.transAxes, fontsize=font_size-3)
                    ax.set_axis_off()

            # Irradiance
            row = 0
            ind = np.arange(nb_channels)
            spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_specleft[row][col]/fig_sizex, 
                                    right=fig_specright[row][col]/fig_sizex,
                                    bottom=fig_specbottom[row][col]/fig_sizey, 
                                    top=fig_spectop[row][col]/fig_sizey, wspace=0.00)
            ax = fig.add_subplot(spec[:, :])
            for l in range(nb_channels):
                print(irr_mean[l], max_eve[l], irr_mean[l]/max_eve[l])
                if l == 0: 
                    plt.bar(l, irr_mean[l]/max_eve[l], width, yerr=irr_err[l]/max_eve[l], label='MEGS-AI - '+instrument_label, 
                            color=colors[0], ecolor ='black', capsize=6)
                else:
                    plt.bar(l, irr_mean[l]/max_eve[l], width, yerr=irr_err[l]/max_eve[l], color=colors[0], ecolor ='black', capsize=6)
            plt.xticks(ind, eve_label, rotation = 45)
            ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
            ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
            ax.tick_params(axis='y', which='both', left=True, right=True, labelleft=label_left, labelright=label_right)
            ax.set_title(f'MEGS-AI - {test_dates}', fontsize=font_size+2, y=1.005)
            ax.set_ylabel(ylabel, fontsize=font_size, labelpad=ylabel_pad, rotation=ylabel_rotation)
            ax.set_ylim([0, 1])
            # ax.set_yscale('log')
            ax.yaxis.set_label_position(ylabel_position)
            ax.legend(loc='best', numpoints=1, markerscale=4, fontsize=font_size-1, labelspacing=0.2, ncol=1)
        plt.draw()
        plt.savefig(output_path + f'_AIA_EUVI_rel_{t}.png', format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
        plt.close('all')



