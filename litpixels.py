#!/usr/bin/env python

'''Calculate lit pixels/frame for a run using the VDS files'''

import sys
import os.path as op
import time
import glob
import multiprocessing as mp
import ctypes
import subprocess

import h5py
import numpy as np

import common
from constants import PREFIX, ADU_PER_PHOTON, VDS_DATASET
from constants import MASK_FNAME, BAD_CELLIDS

class LitPixels():
    def __init__(self, vds_file, dark_run=-1, nproc=0, thresh=4., chunk_size=32, total_intens=False):
        self.vds_file = vds_file
        self.dark_run = dark_run
        self.thresh = thresh
        self.total_intens = total_intens
        self.chunk_size = chunk_size # Needs to be multiple of 32 for raw data
        if self.chunk_size % 32 != 0:
            print('WARNING: Performance is best with a multiple of 32 chunk_size')
        if nproc == 0:
            self.nproc = int(subprocess.check_output('nproc').decode().strip())
        else:
            self.nproc = nproc
        print('Using %d processes' % self.nproc)

        with h5py.File(vds_file, 'r') as f:
            self.dshape = f[VDS_DATASET].shape

    def run_module(self, module):
        sys.stdout.write('Calculating number of lit pixels in module %d for %d frames\n'%(module, self.dshape[0]))
        sys.stdout.flush()
        # Litpixels for each module and each frame
        litpix = mp.Array(ctypes.c_ulong, self.dshape[0])
        jobs = []
        for c in range(self.nproc):
            p = mp.Process(target=self._part_worker, args=(c, module, litpix))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        self.litpix = np.frombuffer(litpix.get_obj(), dtype='u8')
        return self.litpix

    def _parse_darks(self, module):
        with h5py.File(PREFIX + '/dark/r%.4d_dark.h5'%self.dark_run, 'r') as f:
            # Get dark for central 4 ASICs of module
            dark = f['data/mean'][module,:,:,:128]
            cells = f['data/cellId'][:]

        with h5py.File(MASK_FNAME, 'r') as f:
            # Get mask for central 4 ASICs of module
            mask = f['entry_1/good_pixels'][module,:,:128]
            
        return dark, cells, mask

    def _part_worker(self, p, m, litpix):
        np_litpix = np.frombuffer(litpix.get_obj(), dtype='u8')

        nframes = self.dshape[0]
        my_start = (nframes // self.nproc) * p
        my_end = min((nframes // self.nproc) * (p+1), nframes)
        num_chunks = int(np.ceil((my_end-my_start)/self.chunk_size))
        #num_chunks = 4
        if p == 0:
            print('Doing %d chunks of %d frames each' % (num_chunks, self.chunk_size))
            sys.stdout.flush()

        if self.dark_run >= 0:
            dark, cells, goodpix = self._parse_darks(m)

        stime = time.time()
        f_vds = h5py.File(self.vds_file, 'r')
        for c in range(num_chunks):
            pmin = my_start + c*self.chunk_size
            pmax = min(my_start + (c+1) * self.chunk_size, my_end)
            
            # Get central 4 ASICs of module
            if self.dark_run >= 0:
                vals = f_vds[VDS_DATASET][pmin:pmax, m, 0, :, :128]
                cids = f_vds['entry_1/cellId'][pmin:pmax, m]
                vals = np.array([vals[i] - dark[np.where(cids[i]==cells)[0][0]] for i in range(len(vals))])
            else:
                vals = f_vds[VDS_DATASET][pmin:pmax, m, 0, :, :128]

            # mask bad cells and pixels
            vals[np.isin(cids, BAD_CELLIDS)] = 0
            vals *= goodpix
            
            if self.total_intens:
                phot = np.round(vals/ADU_PER_PHOTON - 0.3).astype('i4')
                phot[phot<0] = 0
                np_litpix[pmin:pmax] = phot.sum((1,2))
            else:
                np_litpix[pmin:pmax] = (vals > self.thresh).sum((1,2))
            
            etime = time.time()
            if p == 0:
                sys.stdout.write('\r%.4d/%.4d: %.2f Hz' % (c+1, num_chunks, (c+1)*self.chunk_size/(etime-stime)*self.nproc))
                sys.stdout.flush()
        f_vds.close()
        if p == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()

def copy_ids(vds_fname, fptr):
    print('Copying IDs from VDS file')
    sys.stdout.flush()

    if 'entry_1/trainId' in fptr: del fptr['entry_1/trainId']
    if 'entry_1/cellId' in fptr: del fptr['entry_1/cellId']
    if 'entry_1/pulseId' in fptr: del fptr['entry_1/pulseId']

    with h5py.File(vds_fname, 'r') as f_vds:
        fptr['entry_1/trainId'] = f_vds['entry_1/trainId'][:]
        fptr['entry_1/cellId'] = f_vds['entry_1/cellId'][:]
        fptr['entry_1/pulseId'] = f_vds['entry_1/pulseId'][:]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Lit pixel calculator')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('-d', '--dark_run', type=int, help='Dark run number', default=-1)
    parser.add_argument('-n', '--nproc', 
                        help='Number of processes to use',
                        type=int, default=0)
    parser.add_argument('-m', '--module', nargs='+', 
                        help='Run on only these modules',
                        type=int, default=[0,8,15])
    parser.add_argument('-t', '--thresholdADU',
                        help='ADU threshold for lit pixel',
                        type=float, default=-100.)
    parser.add_argument('-o', '--out_folder', 
                        help='Path of output folder (default=%s/events/)'%PREFIX,
                        default=PREFIX+'/events/')
    parser.add_argument('-T', '--total_intens',
                        help='Whether to calculate total intensity per module rather than lit pixels',
                        action='store_true')
    args = parser.parse_args()
        
    vds_file = PREFIX+'vds/r%.4d.cxi' %args.run
    if args.dark_run < 0:
        args.dark_run = common.get_relevant_dark_run(args.run)
        
    if args.thresholdADU == -100.:
        args.thresholdADU = 0.75 * ADU_PER_PHOTON

    print('Calculating lit pixels from', vds_file)
    l = LitPixels(vds_file, args.dark_run, nproc=args.nproc,
                  thresh=args.thresholdADU, total_intens=args.total_intens)
    print('Running on the following modules:', args.module)
    
    litpixels = np.array([l.run_module(module) for module in args.module])
    
    out_fname = args.out_folder + op.splitext(op.basename(vds_file))[0] + '_events.h5'
    with h5py.File(out_fname, 'a') as outf:
        if args.total_intens:
            dset_name = 'entry_1/total_intens'
        else:
            dset_name = 'entry_1/litpixels'
        if dset_name in outf: del outf[dset_name]
        outf[dset_name] = litpixels
        if 'entry_1/modules' in outf: del outf['entry_1/modules']
        outf['entry_1/modules'] = args.module
        copy_ids(vds_file, outf)
    print('DONE')
                
if __name__ == '__main__':
    main()
