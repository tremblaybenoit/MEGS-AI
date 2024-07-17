import glob
import os.path
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from sunpy.map import Map
from s4pi.data.utils import loadAIAMap, sdo_norms
from tqdm import tqdm

result_path = "/home/benoit_tremblay_23/iti"
iti_files = sorted(glob.glob("/mnt/disks/observational_data/EUVI/B/171/*.fits"))[::100]
aia_files = sorted(glob.glob("/mnt/disks/observational_data/AIA/171/*.fits"))[::100]



def _load_AIA(f):
    try:
        s_map = loadAIAMap(f)
    except:
        return (None, None, None)
    d = s_map.date.datetime
    m = s_map.data.mean()
    s = s_map.data.std()
    return d, m, s

def _load_ITI(f):
    s_map = Map(f)
    d = s_map.date.datetime
    m = s_map.data.mean()
    s = s_map.data.std()
    return d, m, s


with Pool(16) as p:
    aia_result = [r for r in tqdm(p.imap(_load_AIA, aia_files), total=len(aia_files))]
    aia_dates, aia_mean, aia_std = list(map(list, zip(*aia_result)))
    aia_dates, aia_mean, aia_std = np.array(aia_dates), np.array(aia_mean), np.array(aia_std)
    cond = aia_dates == None
    aia_dates, aia_mean, aia_std = aia_dates[~cond], aia_mean[~cond], aia_std[~cond]
    #
    iti_result = [r for r in tqdm(p.imap(_load_ITI, iti_files), total=len(iti_files))]
    iti_dates, iti_mean, iti_std = list(map(list, zip(*iti_result)))
    iti_dates, iti_mean, iti_std = np.array(iti_dates), np.array(iti_mean), np.array(iti_std)

# plot light curves
fig, axs = plt.subplots(2, 1, figsize=(6, 5))
ax = axs[0]
ax.set_title('MEAN')
ax.plot(aia_dates, aia_mean, label='AIA')
ax.plot(iti_dates, iti_mean, label='ITI')
ax.legend()
ax = axs[1]
ax.set_title('STD')
ax.plot(aia_dates, aia_std, label='AIA')
ax.plot(iti_dates, iti_std, label='ITI')
ax.legend()
plt.savefig(os.path.join(result_path, 'lightcurve.jpg'))
plt.close()

# plot example images
plot_dates = aia_dates[::len(aia_dates) // 10]
aia_condition = [np.argmin(np.abs(aia_dates -d)) for d in plot_dates]
iti_condition = [np.argmin(np.abs(iti_dates -d)) for d in plot_dates]
aia_plot_files = np.array(aia_files)[aia_condition]
iti_plot_files = np.array(iti_files)[iti_condition]

for d, aia_f, iti_f in zip(plot_dates, aia_plot_files, iti_plot_files):
    aia_map = loadAIAMap(aia_f)
    iti_map = Map(iti_f)
    #
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #
    ax = axs[0]
    ax.imshow(sdo_norms[171](aia_map.data), vmin=0, vmax=1, origin='lower', cmap='sdoaia171')
    ax.set_title(f'{d.isoformat("T", timespec="minutes")}')
    #
    ax = axs[1]
    ax.imshow(sdo_norms[171](iti_map.data.T), vmin=0, vmax=1, origin='lower', cmap='sdoaia171')
    ax.set_title(f'MEAN ({aia_map.data.mean():.2e}, {iti_map.data.mean():.2e}), STD ({aia_map.data.std():.2e}, {iti_map.data.std():.2e})')
    #
    plt.savefig(os.path.join(result_path, f'{d.isoformat("T", timespec="minutes")}.jpg'))
    plt.close()
