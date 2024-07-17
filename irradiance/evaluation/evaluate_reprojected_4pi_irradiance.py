import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
from astropy import units as u
from matplotlib import cm
from matplotlib.colors import Normalize
from sunpy.map import Map
from tqdm import tqdm

from s4pi.data.utils import loadAIAMap
from s4pi.irradiance.inference import ipredict
from s4pi.irradiance.utilities.data_loader import FITSDataset
from s4pi.maps.utilities.reprojection import load_views

p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-chk_path', type=str,
               default='/mnt/model_checkpoints/runv2/final_model.ckpt',
               help='path to the model checkpoint.')
p.add_argument('-normalization_path', type=str,
               default='/mnt/converted_data/eve_normalization.npy',
               help='path to the EVE normalization.')

args = p.parse_args()

dirs = ['171', '193', '211', '304']
path = '/mnt/stereo-aia-calibrated/aligned/SDO/'
aia_files = [sorted(glob.glob(f'{path}{dir}/*00.fits')) for dir in dirs]
aia_files = np.array(aia_files).transpose()

dirs = ['171', '195', '284', '304']
path = '/mnt/stereo-aia-calibrated/aligned/STEREO_A/'
stereo_a_files = [sorted(glob.glob(f'{path}{dir}/*.fits')) for dir in dirs]
stereo_a_files = np.array(stereo_a_files).transpose()

path = '/mnt/stereo-aia-calibrated/aligned/STEREO_B/'
stereo_b_files = [sorted(glob.glob(f'{path}{dir}/*.fits')) for dir in dirs]
stereo_b_files = np.array(stereo_b_files).transpose()

aia_wls_file, stereo_a_wls_file, stereo_b_wls_file = aia_files[0], stereo_a_files[0], stereo_b_files[0]


def _loadITIMap(path):
    return Map(path).resample((512, 512) * u.pix)


def _loadAIAMap(path):
    return loadAIAMap(path, resolution=512)

def _createViewDirections(strides = 5):
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

strides = 5
# load views and estimate irradiance
dirs = ['171', '193', '211', '304']
path = '/mnt/converted_data/reprojected_views'
view_files = [sorted(glob.glob(f'{path}/{dir}/*.fits')) for dir in dirs]
view_files = np.array(view_files).transpose()
view_ds = FITSDataset(view_files, aia_preprocessing=False)

normalization = np.load(args.normalization_path)

total_irradiance = [irr for irr in ipredict(args.chk_path, view_ds, normalization, return_images=False)]
total_irradiance = torch.stack(total_irradiance).numpy()

# load coordinates from file names
coords = []
for bn in [os.path.basename(f) for f in view_files[:, 0]]:
    f = bn.split('_')
    lat = float(f[0])
    lon = float(f[1][:-5])
    coords += [(lat, lon)]

# sort and reshape arrays
target_shape = (180 // strides + 1, 360 // strides + 1, -1)
coords = np.array(coords)
ind = np.lexsort((coords[..., 0], coords[..., 1]))
reshaped_irradiance = total_irradiance[ind].reshape(target_shape)
coords = coords[ind].reshape(target_shape)

# convert to x,y,z coordinates
phi, theta = (coords[..., 0] + 90) * np.pi / 180, coords[..., 1] * np.pi / 180
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

os.makedirs('/mnt/model_output_v2/irradiance_sphere', exist_ok=True)
# plt.rcParams["figure.autolayout"] = True

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 16))
gs = fig.add_gridspec(ncols = 4, nrows= 5)
axs = []

for idx in tqdm(range(reshaped_irradiance.shape[-1])):
    facecolors = cm.plasma(Normalize()(reshaped_irradiance[..., idx]))
    # plot view directions
    ax = fig.add_subplot(gs[idx + 4], projection='3d')
    ax.plot_surface(x, y, z, facecolors=facecolors, antialiased=False, shade=False)
    ax.set_axis_off()
    axs += [ax]

sdoaia171 = plt.get_cmap('sdoaia171')
sdoaia193 = plt.get_cmap('sdoaia193')
sdoaia211 = plt.get_cmap('sdoaia211')
sdoaia304 = plt.get_cmap('sdoaia304')


view_ds_ordered = FITSDataset(view_files[ind], aia_preprocessing=False)
for i, ((lat, lon), aia_map_stack) in enumerate(zip(coords.reshape((-1, 2)), view_ds_ordered)):
    [ax.view_init(elev=lat, azim=lon) for ax in axs]
    ax = fig.add_subplot(gs[0])
    ax.imshow(aia_map_stack[0], cmap =sdoaia171, vmin = 0, vmax = 1, origin='lower')
    ax.set_axis_off()
    ax = fig.add_subplot(gs[1])
    ax.imshow(aia_map_stack[1], cmap =sdoaia193, vmin = 0, vmax = 1, origin='lower')
    ax.set_axis_off()
    ax = fig.add_subplot(gs[2])
    ax.imshow(aia_map_stack[2], cmap =sdoaia211, vmin = 0, vmax = 1, origin='lower')
    ax.set_axis_off()
    ax = fig.add_subplot(gs[3])
    ax.imshow(aia_map_stack[3], cmap =sdoaia304, vmin = 0, vmax = 1, origin='lower')
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig('/mnt/model_output_v2/irradiance_sphere/%03d.jpg' % i)


plt.close()

shutil.make_archive('/mnt/model_output_v2/irradiance_sphere', 'zip', '/mnt/model_output_v2/irradiance_sphere')
