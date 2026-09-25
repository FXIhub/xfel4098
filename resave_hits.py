import argparse
import multiprocessing as mp
import os
import sys
import time

import dragonfly
import h5py
import numpy as np

import common
import get_thresh
from constants import ADU_PER_PHOTON, MODULE_SHAPE, PREFIX, VDS_DATASET

parser = argparse.ArgumentParser(description='Save hits to emc file')
parser.add_argument('run', help='Run number', type=int)
parser.add_argument('-d', '--dark_run', type=int, help='Dark run number', default=-1)
args = parser.parse_args()

if args.dark_run < 0:
    args.dark_run = common.get_relevant_dark_run(args.run)

# Get hit indices from events file
with h5py.File(PREFIX+'events/r%.4d_events.h5'%args.run, 'r') as f:
    hit_inds = f['entry_1/hit_indices'][:]

# Save hits for modules
def worker(module):
    wemc = dragonfly.EMCWriter(PREFIX+'emc/corr/r%.4d_m%.2d.emc' % (args.run, module), 128*512, hdf5=False)
    sys.stdout.flush()

    offset_all, noise_all = common._get_dark_data(args.dark_run)
    offset = offset_all[module]
    noise = noise_all[module]
    corr = np.zeros(MODULE_SHAPE)

    f = h5py.File(PREFIX+'vds/r%.4d.cxi' % args.run, 'r')
    cell_id = f['entry_1/cellId'][:, module]
    dset = f[VDS_DATASET]

    stime = time.time()

    for i, ind in enumerate(hit_inds):
        cid = cell_id[ind]
        common._calibrate_module3(dset[ind, module, 0], offset[cid], noise[cid], output=corr)
        phot = corr.astype('i4').ravel()
        wemc.write_frame(phot)
        if module == 0 and (i+1) % 10 == 0:
            sys.stderr.write('\rWritten frame %d/%d (%.3f Hz)' % (i+1, len(hit_inds), (i+1)/(time.time()-stime)))
            sys.stderr.flush()
    if module == 0:
        sys.stderr.write('\n')
        sys.stderr.flush()

    wemc.finish_write()
    f.close()

jobs = [mp.Process(target=worker, args=(m,)) for m in range(16)]
[j.start() for j in jobs]
[j.join() for j in jobs]
sys.stdout.flush()

# Merge modules
print('Merging modules')
det = dragonfly.Detector()
det.x, det.y = np.indices(MODULE_SHAPE)
det.x = det.x.ravel()
det.y = det.y.ravel()
emods = [dragonfly.EMCReader(PREFIX+'emc/corr/r%.4d_m%.2d.emc' % (args.run, m), det) for m in range(16)]

wemc_all = dragonfly.EMCWriter(PREFIX+'emc/corr/r%.4d.emc' % args.run, 1024**2, hdf5=False)
stime = time.time()
for i in range(emods[0].num_frames):
    phot = np.array([emods[m].get_frame(i, raw=True) for m in range(16)]).ravel()
    wemc_all.write_frame(phot)

    if (i+1) % 10 == 0:
        sys.stderr.write('\rWritten frame %d/%d (%.3f Hz)' % (i+1, emods[0].num_frames, (i+1)/(time.time()-stime)))
        sys.stderr.flush()
sys.stderr.write('\n')
sys.stderr.flush()

wemc_all.finish_write()

# Delete module-wise files
mod_fnames = [PREFIX+'emc/corr/r%.4d_m%.2d.emc' % (args.run, m) for m in range(16)]
[os.remove(fname) for fname in mod_fnames]
print('Deleted module-wise files')

print('DONE')
