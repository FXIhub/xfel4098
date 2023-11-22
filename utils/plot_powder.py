import argparse
parser = argparse.ArgumentParser(description = 
r"""Calculate histogram over the entire 2**9 ADU range
""")
parser.add_argument('--runs', nargs='+', 
                    help='list of run numbers',
                    type=int, default=[30])
parser.add_argument('--mask', default = 'geom/badpixel_background_mask_r0037.h5',
                    type=str)
args = parser.parse_args()



import matplotlib.pyplot as plt
import extra_geom
import h5py
import numpy as np

from constants import *

from geom_0456 import get_geom

def rad_av(ar, mask, xyz):
    """
    xyz should be scaled of integer binning
    """
    #xyz = make_xyz(corner, basis, ar.shape)
    r = np.rint((xyz[0]**2 + xyz[1]**2)**0.5).astype(int)
    
    rsum = np.bincount(r.ravel(), (mask*ar).ravel())
    
    rcounts = np.bincount(r.ravel(), mask.ravel())
    
    rcounts[rcounts==0] = 1
    
    rav = rsum / rcounts
    
    # broacast
    ar_rav = np.zeros_like(ar)
    
    ar_rav[:] = rav[r.ravel()].reshape(ar.shape)

    return rav, ar_rav


geom = get_geom()


mask_fnam   = PREFIX + args.mask

# load mask
with h5py.File(mask_fnam) as f:
    mask = f['entry_1/good_pixels'][()]




ravs = []
for run in args.runs :
    powder_fnam = PREFIX + 'powder/powder_r%.4d.cxi'%run

    # load powder
    with h5py.File(powder_fnam) as f:
        powder = f['data'][()]
        N = f['Nframes'][()]

    # powder to average
    powder /= N

    # to get assembled image (2D)
    image, centre = geom.position_modules(mask * powder)

    xyz = np.transpose( geom.get_pixel_positions() , (3, 0, 1, 2))

    #powder *= mask
    
    rav, ar_rav = rad_av(powder, mask, xyz / 300e-6)

    # plot with pyqtgraph

    #import pyqtgraph as pg
    #pg.show(image)

    # plot powder directly from data using extra_geom with matplotlib
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)

    vmin = 0
    vmax = np.percentile(powder[mask], 99)
    ax = geom.plot_data(powder, axis_units='m', ax = ax, vmin=vmin, vmax=vmax, colorbar=True)

    ax.set_title(f'powder plot run {run}')
    
    plt.savefig(PREFIX + 'powder/powder_r%.4d.png'%run)
    plt.show()
    
    ravs.append(rav.copy())

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 8)

import extra_data

for run, rav in zip(args.runs, ravs):
    chamber = None
    heflow = None
    _r = extra_data.open_run(proposal=4098, run=run)
    try:
        chamber = _r['SQS_NQS_VAC/GAUGE/CHAMBER_1', 'value'].ndarray().mean()
    except extra_data.exceptions.SourceNameError:
        pass
    try:
        heflow = _r['SQS_NQS_LJET/FLOW/HE_FLOW_METER', 'measureCapacity'].ndarray().mean()
    except extra_data.exceptions.SourceNameError:
        pass
    label = f'{run} (' 
    if chamber is not None:
        label += f'{chamber:.2E}, '
    else:
        label += 'None, '
    if heflow is not None:
        label += f'{heflow:.2E}'
    else:
        label += 'None'
    label += ')'
    ax.plot(300e-6 * np.arange(rav.shape[0]), rav, label = label, alpha = 0.5, linewidth=2)

ax.set_yscale('log')
ax.set_xlabel('radius (m)')
ax.set_ylabel('azimuthal average')
ax.legend(title='RUN (Chamber [mbar], Helium [mg/min])')

plt.savefig(PREFIX + 'powder/powder_radial_plot_r%.4d-r%.4d.png'%(args.runs[0], args.runs[-1]))
plt.show()
