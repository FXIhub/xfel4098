import os
import argparse
import glob

import numpy as np
import h5py

from constants import PREFIX, NPULSES
from constants import XGM_DATASET, XGM_DA_NUM

def set_values(run, da_num, dset_name, outf, out_dset_name, force=False):
    if out_dset_name in outf:
        if force:
            del outf[out_dset_name]
        else:
            print('Skipping', out_dset_name)
            return

    flist = sorted(glob.glob(PREFIX+'/raw/r%.4d/*DA%.2d*.h5'%(run, da_num)))

    with h5py.File(PREFIX + 'vds/proc/r%.4d_proc.cxi'%run, 'r') as f:
        vds_tid = f['entry_1/trainId'][:]
        vds_cid = f['entry_1/cellId'][:,0]

    out_dset = outf.create_dataset(out_dset_name, shape=vds_tid.shape, dtype='f8')
    da_tid_dset_name = os.path.dirname(dset_name) + '/trainId'

    num_pos = 0
    for fname in flist:
        with h5py.File(fname, 'r') as f:
            da_tid = f[da_tid_dset_name][:]
            vals = np.insert(f[dset_name][:], 0, 0, axis=1)

        for i, tid in enumerate(da_tid):
            pos = np.where(vds_tid == tid)[0]
            out_dset[pos] = vals[i, vds_cid[pos]]
            num_pos += len(pos)

    if vds_tid.shape[0] != num_pos:
        print('WARNING: Unfilled %s values: %d vs %d' % (out_dset_name, vds_tid.shape[0], num_pos))

def main():
    parser = argparse.ArgumentParser(description='Add pulse-resolved metadata to events file')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-f', '--force', help='Replace existing data if exists', action='store_true')
    args = parser.parse_args()

    outf = h5py.File(PREFIX + '/events/r%.4d_proc_events.h5' % args.run, 'a')

    set_values(args.run, XGM_DA_NUM, XGM_DATASET, outf, 'entry_1/pulse_energy_uJ', force=args.force)

    outf.close()

if __name__ == '__main__':
    main()
