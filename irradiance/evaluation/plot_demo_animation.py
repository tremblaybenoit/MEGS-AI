
import matplotlib.pyplot as plt
import sunpy as sp
import sunpy.visualization.colormaps as cm
import numpy as np
import glob
import argparse
import sys
import os
import scipy.ndimage as ndimage

from s4pi.irradiance.inference import ipredict_uncertainty
from s4pi.irradiance.utilities.data_loader import FITSDataset

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-chk_path', type=str,
            default='/mnt/training/dropout05/None/version_None/checkpoints/epoch=120-step=19359.ckpt', # TODO
            help='path to the model checkpoint.')
p.add_argument('-normalization_path', type=str,
            default='/mnt/converted_data/eve_normalization.npy',
            help='path to the EVE normalization.')
p.add_argument('-eve_wl_names', type=str,
               default='/mnt/converted_data/eve_wl_names.npy',
               help='path to the EVE norm.')

args = p.parse_args()

plot_dir = '/mnt/model_output_v4/'
effnet_img = plt.imread('/mnt/stereo-aia-calibrated/eff_net_diagram_custom.png')

normalization = np.load(args.normalization_path)

dirs = ['171', '193', '211', '304']
path = '/mnt/miniset/stereo-aia-demo/aligned/SDO/'
aia_files = [sorted(glob.glob(f'{path}{dir}/*00.fits')) for dir in dirs]
aia_files = np.array(aia_files).transpose()

dirs = ['171', '195', '284', '304']
path = '/mnt/miniset/stereo-aia-demo/aligned/STEREO_A/'
stereo_a_files = [sorted(glob.glob(f'{path}{dir}/*.fits')) for dir in dirs]
stereo_a_files = np.array(stereo_a_files).transpose()

path = '/mnt/miniset/stereo-aia-demo/aligned/STEREO_B/'
stereo_b_files = [sorted(glob.glob(f'{path}{dir}/*.fits')) for dir in dirs]
stereo_b_files = np.array(stereo_b_files).transpose()

aia_ds = FITSDataset(aia_files, aia_preprocessing=True)
sdo_generator = ipredict_uncertainty(args.chk_path, aia_ds, normalization, return_images = True)
stereo_a_ds = FITSDataset(stereo_a_files, aia_preprocessing=False)
stereo_a_generator = ipredict_uncertainty(args.chk_path, stereo_a_ds, normalization, return_images = True)
stereo_b_ds = FITSDataset(stereo_b_files, aia_preprocessing=False)
stereo_b_generator = ipredict_uncertainty(args.chk_path, stereo_b_ds, normalization, return_images = True)

sdoaia171 = plt.get_cmap('sdoaia171')
sdoaia193 = plt.get_cmap('sdoaia193')
sdoaia211 = plt.get_cmap('sdoaia211')
sdoaia304 = plt.get_cmap('sdoaia304')

wl_names = np.load(args.eve_wl_names, allow_pickle=True)

def plot_euv_images(imgs, axs):
    axs[0].imshow(imgs[0], cmap =sdoaia171, vmin = 0, vmax = 1, origin='lower')
    axs[0].text(0.01, 0.99, '171', horizontalalignment='left', verticalalignment='top', color = 'w', transform=axs[0].transAxes)
    
    axs[1].imshow(imgs[1], cmap =sdoaia193, vmin = 0, vmax = 1, origin='lower')
    axs[1].text(0.01, 0.99, '193', horizontalalignment='left', verticalalignment='top', color = 'w', transform=axs[1].transAxes)
    
    axs[2].imshow(imgs[2], cmap =sdoaia211, vmin = 0, vmax = 1, origin='lower')
    axs[2].text(0.01, 0.99, '211', horizontalalignment='left', verticalalignment='top', color = 'w', transform=axs[2].transAxes)
    
    axs[3].imshow(imgs[3], cmap =sdoaia304,vmin = 0, vmax = 1, origin='lower')
    axs[3].text(0.01, 0.99, '304', horizontalalignment='left', verticalalignment='top', color = 'w', transform=axs[3].transAxes)

    [ax.set_axis_off() for ax in axs]

def plot_eve(pred_eve, axs):
    mean, std = pred_eve
    eve_mean, eve_std = normalization
    axs.bar(np.arange(0, len(mean)), mean * eve_std, yerr=std * eve_std, width=0.5, alpha=0.5, ecolor ='red', capsize=8)
    axs.set_xticks(np.arange(0,len(mean)))
    axs.set_xticklabels(wl_names,rotation = 45)
    axs.set_ylim(0, 0.7)
    # axs.set_yticklabels(['${%d}$' % (item - offset) for item in ax.get_yticks()])

samples = len(aia_files)

os.makedirs(plot_dir, exist_ok=True)
# AIA 171


# Size definitions
dpi = 400
pxx = 600   # Horizontal size of each panel
pxy = 600   # Vertical size of each panel

nph = 6     # Number of horizontal panels
npv = 6     # Number of vertical panels 

# Padding
padv  = 0  #Vertical padding in pixels
padv2 = 50  #Vertical padding in pixels between panels
padh  = 0 #Horizontal padding in pixels at the edge of the figure
padh2 = 50  #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative units
ppxx   = pxx/fszh      # Horizontal size of each panel in relative units
ppxy   = pxy/fszv      # Vertical size of each panel in relative units
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


for i, ((pred_sdo, imgs_sdo), (pred_stereo_a, imgs_stereo_a), (pred_stereo_b, imgs_stereo_b)) in enumerate(zip(sdo_generator, stereo_a_generator, stereo_b_generator)):
    fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi, facecolor='w')

    #SDO
    ax1 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])
    ax2 = fig.add_axes([ppadh+ppxx, ppadv, ppxx, ppxy])
    ax3 = fig.add_axes([ppadh, ppadv-ppxy, ppxx, ppxy])
    ax4 = fig.add_axes([ppadh+ppxx, ppadv-ppxy, ppxx, ppxy])
    axs = [ax1, ax2, ax3, ax4]
    plot_euv_images(np.transpose(imgs_stereo_b, (0, 2, 1)), axs)
    ax2.text(0., 1, 'STEREO-B', horizontalalignment='center', verticalalignment='bottom', color = 'k', transform=ax2.transAxes,size = 20)


    #STEREO A
    ax1 = fig.add_axes([ppadh+2*ppxx+ppadh2, ppadv, ppxx, ppxy])
    ax2 = fig.add_axes([ppadh+3*ppxx+ppadh2, ppadv, ppxx, ppxy])
    ax3 = fig.add_axes([ppadh+2*ppxx+ppadh2, ppadv-ppxy, ppxx, ppxy])
    ax4 = fig.add_axes([ppadh+3*ppxx+ppadh2, ppadv-ppxy, ppxx, ppxy])

    axs = [ax1, ax2, ax3, ax4]
    plot_euv_images(np.transpose(imgs_sdo, (0, 2, 1)), axs)
    ax2.text(0., 1, 'SDO', horizontalalignment='center', verticalalignment='bottom', color = 'k', transform=ax2.transAxes,size = 20)


    #STEREO B
    ax1 = fig.add_axes([ppadh+4*ppxx+2*ppadh2, ppadv, ppxx, ppxy])
    ax2 = fig.add_axes([ppadh+5*ppxx+2*ppadh2, ppadv, ppxx, ppxy])
    ax3 = fig.add_axes([ppadh+4*ppxx+2*ppadh2, ppadv-ppxy, ppxx, ppxy])
    ax4 = fig.add_axes([ppadh+5*ppxx+2*ppadh2, ppadv-ppxy, ppxx, ppxy])

    axs = [ax1, ax2, ax3, ax4]
    plot_euv_images(np.transpose(imgs_stereo_a, (0, 2, 1)), axs)
    ax2.text(0., 1, 'STEREO-A', horizontalalignment='center', verticalalignment='bottom', color = 'k', transform=ax2.transAxes,size = 20)


    #MODEL DIAGRAM
    ax = fig.add_axes([ppadh, ppadv-3*ppxy-ppadv2, 2*ppxx, 2*ppxy])
    ax.imshow(effnet_img)
    ax.set_axis_off()
    
    ax = fig.add_axes([ppadh+2*ppxx+ppadh2, ppadv-3*ppxy-ppadv2, 2*ppxx, 2*ppxy])
    ax.imshow(effnet_img)
    ax.set_axis_off()

    ax = fig.add_axes([ppadh+4*ppxx+2*ppadh2, ppadv-3*ppxy-ppadv2, 2*ppxx, 2*ppxy])
    ax.imshow(effnet_img)
    ax.set_axis_off()

    #EVE BAR PLOTS
    ax = fig.add_axes([ppadh, ppadv-5*ppxy-2*ppadv2, 2*ppxx, 2*ppxy])
    plot_eve(pred_stereo_b,ax)
    ax.set_ylabel('Relative Irradiance log(w/m$^2$)')

    ax = fig.add_axes([ppadh+2*ppxx+ppadh2, ppadv-5*ppxy-2*ppadv2, 2*ppxx, 2*ppxy])
    plot_eve(pred_sdo,ax)
    ax.set_yticklabels([])
    
    ax = fig.add_axes([ppadh+4*ppxx+2*ppadh2, ppadv-5*ppxy-2*ppadv2, 2*ppxx, 2*ppxy])
    plot_eve(pred_stereo_a,ax)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel('Relative Irradiance log(w/m$^2$)')


    fig.savefig(os.path.join(plot_dir, f'%02d.jpg' % i), bbox_inches='tight', dpi = dpi, pad_inches=0)
    plt.close(fig)


