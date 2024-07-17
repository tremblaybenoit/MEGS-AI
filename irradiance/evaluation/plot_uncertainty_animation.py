import matplotlib.pyplot as plt
import sunpy as sp
import sunpy.visualization.colormaps as cm
import numpy as np
import glob
import argparse
import sys
import os
import scipy.ndimage as ndimage

import argparse
import glob
import os

import torch
from astropy import units as u
from matplotlib import cm
from matplotlib.colors import Normalize
from sunpy.map import Map
from tqdm import tqdm

from s4pi.data.utils import loadAIAMap
from s4pi.irradiance.inference import ipredict_uncertainty, ipredict_ensembles
from s4pi.irradiance.utilities.data_loader import FITSDataset
from s4pi.maps.reprojection import load_views
from s4pi.data.utils import str2bool

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-chk_path', type=str,
            default='/mnt/training/limb_effnetb0_dropout075_augmentations09/epoch=41-step=4536.ckpt', # TODO
            help='path to the model checkpoint.')
p.add_argument('-normalization_path', type=str,
            default='/mnt/converted_data/eve_normalization.npy',
            help='path to the EVE normalization.')
p.add_argument('-eve_wl_names', type=str,
               default='/mnt/converted_data/eve_wl_names.npy',
               help='path to the EVE norm.')
p.add_argument('-MC_dropout', dest='MC_dropout', type=str2bool, default=True,
               help="choose if uncertainty is calculated with MC dropout or ensembles")

args = p.parse_args()

if not args.MC_dropout:
    args.chk_paths = ['/mnt/training/seed-experiments/12/epoch=32-step=3564.ckpt',
                    '/mnt/training/seed-experiments/19810105/epoch=35-step=3888.ckpt',
                    '/mnt/training/seed-experiments/19910623/epoch=30-step=3348.ckpt',
                    '/mnt/training/seed-experiments/1993/epoch=58-step=6372.ckpt',
                    '/mnt/training/seed-experiments/2022/epoch=98-step=10692.ckpt',
                    '/mnt/training/seed-experiments/3110/epoch=21-step=2376.ckpt',
                    '/mnt/training/seed-experiments/808/epoch=35-step=3888.ckpt']

wl_names = np.load(args.eve_wl_names, allow_pickle=True)

plot_dir = '/home/margaritampintsi/4piuvsun/results/irradiance-uncertainty-mc_dropout/'

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

aia_wls_file, stereo_a_wls_file, stereo_b_wls_file = aia_files[0], stereo_a_files[0], stereo_b_files[0]

sdoaia171 = plt.get_cmap('sdoaia171')
sdoaia193 = plt.get_cmap('sdoaia193')
sdoaia211 = plt.get_cmap('sdoaia211')
sdoaia304 = plt.get_cmap('sdoaia304')

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
    # mean = mean * eve_std + eve_mean
    axs.bar(np.arange(0, len(mean)), mean, yerr=std, width=0.5, alpha=0.5, ecolor ='red', capsize=6)
    # axs.bar(np.arange(0, len(mean)), mean * eve_std, yerr=std * eve_std, width=0.5, alpha=0.5, ecolor ='red', capsize=6)
    axs.set_xticks(np.arange(0,len(mean)))
    axs.set_xticklabels(wl_names,rotation = 45)
    # axs.semilogy()
    # axs.set_ylim(0, 0.7)
    # axs.set_yticklabels(['${%d}$' % (item - offset) for item in ax.get_yticks()])


def _loadITIMap(path):
    return Map(path).resample((512, 512) * u.pix)


def _loadAIAMap(path):
    return loadAIAMap(path, resolution=512)

def _createViewDirections(strides = 180):
    # create view directions and save to disk
    for aia_file, stereo_a_file, stereo_b_file in zip(aia_wls_file, stereo_a_wls_file, stereo_b_wls_file):
        sdo_map = _loadAIAMap(aia_file)
        stereo_a_map = _loadITIMap(stereo_a_file)
        stereo_b_map = _loadITIMap(stereo_b_file)
        for (lat, lon), s_map in tqdm(load_views(sdo_map, stereo_a_map, stereo_b_map, strides=strides)):
            file_path = f'/mnt/converted_data/reprojected_views/{int(s_map.wavelength.value)}/{lat}_{lon}.fits'
            if os.path.exists(file_path):
                continue
            os.makedirs(f'/mnt/converted_data/reprojected_views/{int(s_map.wavelength.value)}', exist_ok=True)
            s_map.save(file_path)
# _createViewDirections(strides = 90)

# load views and estimate irradiance
dirs = ['171', '193', '211', '304']
path = '/mnt/converted_data/reprojected_views'
view_files = [sorted(glob.glob(f'{path}/{dir}/*.fits')) for dir in dirs]
view_files = np.array(view_files).transpose()
view_ds = FITSDataset(view_files, resolution=256, aia_preprocessing=False)

normalization = np.load(args.normalization_path)

if args.MC_dropout:
    prediction = [pred for pred in ipredict_uncertainty(args.chk_path, view_ds, normalization, return_images=False)]
else:
    sprediction = [pred for pred in ipredict_ensembles(args.chk_paths, view_ds, normalization, return_images=False)]

os.makedirs(plot_dir, exist_ok=True)

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


for i, img_stack in enumerate(view_ds):
    fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi, facecolor='w')

    #NORTH POLE
    ax1 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])
    ax2 = fig.add_axes([ppadh+ppxx, ppadv, ppxx, ppxy])
    ax3 = fig.add_axes([ppadh, ppadv-ppxy, ppxx, ppxy])
    ax4 = fig.add_axes([ppadh+ppxx, ppadv-ppxy, ppxx, ppxy])
    axs = [ax1, ax2, ax3, ax4]
    plot_euv_images(np.transpose(img_stack, (0, 2, 1)), axs)

    #EVE BAR PLOTS
    ax = fig.add_axes([ppadh, ppadv-3*ppxy-ppadv2, 2*ppxx, 2*ppxy])
    plot_eve(prediction[i],ax)
    ax.set_ylabel('Normalized Irradiance log(w/m$^2$)')

    fig.savefig(os.path.join(plot_dir, f'%02d.jpg' % i), bbox_inches='tight', dpi = dpi, pad_inches=0)
    plt.close(fig)








