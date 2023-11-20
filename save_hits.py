import sys
import os
import time
import argparse
import multiprocessing as mp

import numpy as np
import h5py
from scipy import optimize

import dragonfly

from constants import PREFIX, VDS_DATASET, MODULE_SHAPE, NCELLS, ADU_PER_PHOTON
import get_thresh
 
parser = argparse.ArgumentParser(description='Save hits to emc file')
parser.add_argument('run', help='Run number', type=int)
parser.add_argument('-t', '--thresh', help='Hitscore threshold (default: auto)', type=float, default=-1)
parser.add_argument('-N', '--norm', help='Normalize hit scores by pulse_energy', action='store_true')
args = parser.parse_args()

if args.thresh < 0:
    thresh = get_thresh.linearize(get_thresh.get_thresh(args.run, normed=args.norm))
else:
    thresh = np.ones(NCELLS)*args.thresh
hit_inds = get_thresh.get_hitinds(args.run, thresh, normed=args.norm, verbose=True)

# Write hit indices to events file
with h5py.File(PREFIX+'events/r%.4d_events.h5'%args.run, 'a') as f:
    if 'entry_1/is_hit' in f:
        del f['entry_1/is_hit']
    if 'entry_1/hit_indices' in f:
        del f['entry_1/hit_indices']
    f['entry_1/hit_indices'] = hit_inds

# Save hits for modules
def worker(module):
    wemc = dragonfly.EMCWriter(PREFIX+'emc/r%.4d_m%.2d.emc' % (args.run, module), 128*512, hdf5=False)
    sys.stdout.flush()

    f = h5py.File(PREFIX+'vds/r%.4d.cxi' % args.run, 'r')
    dset = f[VDS_DATASET]

    stime = time.time()

    for i, ind in enumerate(hit_inds):
        frame = dset[ind, module]
        phot = np.clip(np.round(frame/ADU_PER_PHOTON-0.3).astype('i4'), 0, None).ravel()
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
emods = [dragonfly.EMCReader(PREFIX+'emc/r%.4d_m%.2d.emc' % (args.run, m), det) for m in range(16)]

wemc = dragonfly.EMCWriter(PREFIX+'emc/r%.4d.emc' % args.run, 1024**2, hdf5=False)
stime = time.time()
for i in range(emods[0].num_frames):
    phot = np.array([emods[m].get_frame(i, raw=True) for m in range(16)]).ravel()
    wemc.write_frame(phot)

    if (i+1) % 10 == 0:
        sys.stderr.write('\rWritten frame %d/%d (%.3f Hz)' % (i+1, emods[0].num_frames, (i+1)/(time.time()-stime)))
        sys.stderr.flush()
sys.stderr.write('\n')
sys.stderr.flush()

wemc.finish_write()

# Delete module-wise files
mod_fnames = [PREFIX+'emc/r%.4d_m%.2d.emc' % (args.run, m) for m in range(16)]
[os.remove(fname) for fname in mod_fnames]
print('Deleted module-wise files')

print('DONE')
