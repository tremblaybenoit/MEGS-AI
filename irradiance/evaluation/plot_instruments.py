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


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-checkpoint_path', type=str, required=True, help='Path to checkpoints.')
    p.add_argument('-instrument', type=str, nargs='+', required=True, 
                   help='Instruments.')
    p.add_argument('-mission', type=str, default=None, nargs='+', 
                   help='Mission associated with instrument.')
    p.add_argument('-aia_wl', type=str, nargs='+', 
                   default=['94', '131', '171', '193', '211', '304', '335', '1600', '1700'],
                   help='SDO/AIA wavelengths.')
    p.add_argument('-eve_data', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_converted.npy",
                   help='Path to converted SDO/EVE data.')
    p.add_argument('-eve_norm', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_normalization.npy",
                   help='Path to converted SDO/EVE normalization.')
    p.add_argument('-eve_wl', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_eve_wl.npy",
                   help='Path to SDO/EVE wavelength names.')
    p.add_argument('-matches_table', dest='matches_table', type=str, 
                   default="/mnt/disks/preprocessed_data/AIA/matches_eve_aia.csv",
                   help='matches_table')
    p.add_argument('-output_path', type=str, default='/home/benoit_tremblay_23/MEGS_AI',
                   help='Path to save plots.')
    args = p.parse_args()

    # Selection of plot colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    # Output path
    output_path = args.output_path

    # Models
    aia_wl = args.aia_wl
    mission = args.mission
    instrument = args.instrument
    nb_instrument = len(instrument)
    instrument_label = []
    checkpoint_path = args.checkpoint_path
    checkpoints = [f'{checkpoint_path}/{i}/{f}' for i in instrument for f in os.listdir(f'{checkpoint_path}/{i}/') if f.endswith('.ckpt')]

    # EVE: Wavelengths and channels
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

    # Metrics:  
    # mee = Median Absolute Error, mae: Mean Absolute Error 
    # mere = Median Absolute Relative Error, mare: Mean Absolute Relative Error 
    metrics = {}
    metrics_keys = ['mape', 'mepe', 'mae', 'mee']
    metrics['label'] = {'mape': 'Mean Absolute Percentage Errors', 'mepe': 'Median Absolute Percentage Errors', 
                        'mae': 'Mean Absolute Errors', 'mee': 'Median Absolute Errors'}
    metrics['units'] = {'mape': '(%)', 'mepe': '(%)', 'mae': r'($\sigma$)', 'mee': r'($\sigma$)'}  # TODO: Fix units. Use sigma?

    # Infer irradiance and compute metrics
    for i in range(nb_instrument):
        # Load model checkpoint
        state = torch.load(checkpoints[i])
        model = state['model']
        wl = state[state['instrument']]

        # Create instrument label for legend
        wl_label = 'Å, '.join([aia_wl[w] for w in wl])
        if len(wl) > 1:
            if mission is not None:
                instrument_label.append(f'{mission[i]}/{instrument[i]} ({wl_label}Å)')
            else:
                instrument_label.append(f'{instrument[i]} ({wl_label}Å)')
        else:
            instrument_label.append(f'{wl_label}Å')

        # Initialize dataloader
        data_module = IrradianceDataModule(args.matches_table, args.eve_data, wl, num_workers=os.cpu_count() // 2)
        data_module.setup()
        # Extract EVE data from dataloader
        eve_norm = model.eve_norm
        test_eve = (torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module.test_ds])).numpy()
        # Extract dates from dataloader
        test_dates = [date for date in data_module.test_dates]
        # Extract EUV images from dataloader
        test_aia = (torch.stack([torch.tensor(image) for image, eve in data_module.test_ds])).numpy()
        # Infer irradiance
        test_irradiance = [irr for irr in tqdm(ipredict(model, test_aia, return_images=False), total=len(test_aia))]
        test_irradiance = torch.stack(test_irradiance).numpy()
        # Compute metrics: Standard deviation
        sigma = [np.std(test_eve[:, 0, j]) for j in range(nb_channels)]
        # Compute metrics: Mean and Median Absolute Relative Errors
        mae = [np.mean(np.abs(test_eve[:, 0, j]-test_irradiance[:, j]))/sigma[j] for j in range(nb_channels)]
        mae_all = np.mean(np.abs(test_eve[:, 0, :]-test_irradiance[:, :])/sigma[:])
        mee = [np.median(np.abs(test_eve[:, 0, j]-test_irradiance[:, j]))/sigma[j] for j in range(nb_channels)]
        mee_all = np.median(np.abs(test_eve[:, 0, :]-test_irradiance[:, :])/sigma[:])
        # Compute metrics: Mean and Median Absolute Relative Errors
        mape = [np.mean(np.abs((test_eve[:, 0, j]-test_irradiance[:, j])/test_eve[:, 0, j]))*100. for j in range(nb_channels)]
        mape_all = np.mean(np.abs((test_eve[:, 0, :]-test_irradiance[:, :])/test_eve[:, 0, :]))*100.
        mepe = [np.median(np.abs((test_eve[:, 0, j]-test_irradiance[:, j])/test_eve[:, 0, j]))*100. for j in range(nb_channels)]
        mepe_all = np.median(np.abs((test_eve[:, 0, :]-test_irradiance[:, :])/test_eve[:, 0, :]))*100.
        metrics[instrument[i]] = {'mape': mape, 'mape_all': mape_all, 'mepe': mepe, 'mepe_all': mepe_all, 'mae': mae, 'mae_all': mae_all, 
                                  'mee': mee, 'mee_all': mee_all, 'sigma': sigma}


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
    img_ratio = 1.0/3.
    width = 0.15*5/nb_instrument
    if nb_instrument < 6:
        ncol_leg = 1
    elif nb_instrument < 9:
        ncol_leg = 2
    else:
        ncol_leg = 3
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
    ind = np.arange(nb_channels+1)
    # Loop over metrics
    for m, metric in enumerate(metrics_keys):
        fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)
        spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left/fig_sizex + fig_specx[0][0]/fig_sizex, 
                                right= -fig_right/fig_sizex + fig_specx[0][1]/fig_sizex,
                                bottom=fig_bottom/fig_sizey + fig_specy[0][0]/fig_sizey, 
                                top= -fig_top/fig_sizey + fig_specy[1][0]/fig_sizey, 
                                wspace=0.00)
        ax = fig.add_subplot(spec[:, :])
        for i, inst in enumerate(instrument):
            for l, wl in enumerate(eve_wl):
                if l == 0: 
                    plt.bar(l + i*width, metrics[inst][metric][l], width, label=instrument_label[i], color=colors[i])
                else:
                    plt.bar(l + i*width, metrics[inst][metric][l], width, color=colors[i])
            plt.bar(nb_channels + i*width, metrics[inst][metric+'_all'], width, color=colors[i])
        plt.xticks(ind + 0.5*(nb_instrument-1)*width, np.append(eve_label, 'All'))
        ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
        ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
        label = metrics['label'][metric]
        unit = metrics['units'][metric]
        ax.set_title(f'Test set: {label} between MEGS-A and MEGS-AI Virtual Instruments', fontsize=font_size+2, y=1.005)
        ax.set_ylabel(f'{label} {unit}', fontsize=font_size, labelpad=5.0)
        ax.legend(loc='best', numpoints=1, markerscale=4, fontsize=font_size-1, labelspacing=0.2, ncol=ncol_leg)
        plt.draw()
        plt.savefig(output_path + f'_{metric}.png', format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
        plt.close('all')

