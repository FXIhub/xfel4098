import sys
import os.path as op
import time
import argparse
import multiprocessing as mp
import ctypes

import numpy as np
import h5py
from mpi4py import MPI

import common
from constants import PREFIX, CHUNK_SIZE, MODULE_SHAPE
from constants import CELLID_FNAME, GAIN_FNAME

class Integrator():
    def __init__(self, run, mask, selector,
                 dark_run=-1, testing=False,
                 num_frames=-1, num_cells=352,
                 cell_id=None, flag_file=None):
        self.run = run
        self.dark_run = dark_run
        self.testing = testing
        self.num_frames = num_frames
        self.num_cells = num_cells
        self.mask = None
        self.cell_id = cell_id
        self.gain = np.load(GAIN_FNAME)
        self.cid_order = np.load(CELLID_FNAME)

        if flag_file is not None:
            self.have_flag = True
            with h5py.File(flag_file, 'r') as f:
                self.flags = f['entry_1/do_integrate'][:].astype(np.bool)
            good_cells = np.array(selector.tolist()*(len(self.flags)//len(selector)))
            self.flags *= good_cells
        else:
            self.have_flag = False
            self.good_cells = selector
            self.num_cells = len(self.good_cells)

        if mask != '':
            with h5py.File(mask, 'r') as f:
                self.mask = np.array(f['data/data']).astype('bool')
        else:
            self.mask = np.zeros((self.num_cells, 16,) + MODULE_SHAPE, dtype='bool')

        self.do_raw = (self.dark_run > 0)

        if self.do_raw:
            self.f_vds = h5py.File(PREFIX + 'vds/r%.4d.cxi'%self.run, 'r')
        else:
            self.f_vds = h5py.File(PREFIX + 'vds/proc/r%.4d_proc.cxi'%self.run, 'r')
        self.dset_vds = self.f_vds['entry_1/instrument_1/detector_1/data']
        if self.have_flag:
            assert self.flags.shape[0] == self.dset_vds.shape[0]

        if testing:
            self.out_fname = 'runsum_r%.4d'%self.run
        elif mask != '':
            self.out_fname = PREFIX + 'powder/r%.4d_masked'%self.run
        else:
            self.out_fname = PREFIX + 'powder/r%.4d'%self.run

        if self.cell_id is not None:
            self.out_fname += '_cell_%.3d' % self.cell_id
        if not self.do_raw:
            self.out_fname += '_proc'
        if self.have_flag:
            self.out_fname += '_sel'
            
        if self.num_frames > 0:
            self.out_fname += '_%.8d' % self.num_frames
        #self.out_fname += '.h5'
        self.out_fname += '_gain.h5'

        if self.num_frames < 0:
            self.num_frames = self.dset_vds.shape[0]

    def finish(self, write=True):
        if write:
            with h5py.File(self.out_fname, 'w') as f:
                f['data/data'] = self.powder
                if not self.have_flag:
                    f['data/cells'] = np.where(self.good_cells)[0]
                f['data/counts'] = self.counts

        self.f_vds.close()

    def run_mpi(self):
        comm = MPI.COMM_WORLD
        rank = comm.rank
        nproc = comm.size
        if nproc % 16 != 0:
            raise ValueError('Need number of processes to be multiple of 16')
        if rank == 0:
            if self.have_flag:
                print('Processing %d/%d selected events' % (self.flags.sum(), self.flags.size))
            else:
                print('Processing %d/%d cells' % (self.good_cells.sum(), self.good_cells.size))
            print('Will write output to', self.out_fname)
            sys.stdout.flush()

        my_module = rank % 16
        psize = self.num_frames // (nproc // 16) + 1
        my_portion = np.arange(psize*(rank//16), min(self.num_frames, ((rank//16)+1)*psize))
        #print(rank, my_portion.min(), my_portion.max())

        if self.do_raw:
            with h5py.File(PREFIX + 'dark/r%.4d_dark.h5'%self.dark_run, 'r') as f:
                dark = f['data/mean'][my_module,:,:,:]
                relthresh = 0.5 - f['data/sigma'][my_module]**2/5**2 * (-6.907755)

        mygain = self.gain[my_module]
        num_chunks = len(my_portion) // CHUNK_SIZE + 1
        if self.testing:
            num_chunks = 10

        my_powder = np.zeros((16,)+MODULE_SHAPE)
        #my_counts = np.zeros(16, dtype='i4')
        my_counts = np.zeros((16,)+MODULE_SHAPE, dtype='i4')

        stime = time.time()

        for chunk in range(num_chunks):
            st = chunk*CHUNK_SIZE
            en = min(len(my_portion), (chunk+1)*CHUNK_SIZE)
            chunk_ind = my_portion[st:en]
            if self.have_flag:
                chunk_ind = chunk_ind[self.flags[chunk_ind]]
                if len(chunk_ind) == 0:
                    continue
            cells = self.cid_order[chunk_ind % self.num_cells]
            if not self.have_flag and self.good_cells[cells].sum() == 0:
                continue

            my_mask = self.mask[cells, my_module]
            if self.do_raw:
                fr = self.dset_vds[chunk_ind, my_module, 0, :, :]
                fr = fr.astype('f4') - dark[cells]
                if not self.have_flag:
                    fr = fr[self.good_cells[cells]]
                    my_mask = my_mask[self.good_cells[cells]]
                fr_tmp = np.copy(fr)
                fr = fr[~np.all(fr<0, axis=(1,2))]
                my_mask = my_mask[~np.all(fr_tmp<0, axis=(1,2))]
            else:
                fr = self.dset_vds[chunk_ind, my_module, :, :]

                if not self.have_flag:
                    fr = fr[self.good_cells[cells]]
                    my_mask = my_mask[self.good_cells[cells]]
                fr_tmp = np.copy(fr)
                fr = fr[~np.all(np.isnan(fr), axis=(1,2))]
                my_mask = my_mask[~np.all(np.isnan(fr_tmp), axis=(1,2))]
            if fr.shape[0] == 0:
                continue

            try:
                phot = np.ceil(fr/mygain - relthresh[cells]).astype('i4')
            except ValueError:
                print('\nInconsistent shape:', fr.shape, relthresh[cells].shape)
                continue
            phot[phot<0] = 0
            try:
                phot[my_mask!=0] = 0
            except IndexError:
                print('\nInconsistent mask shape:', phot.shape, my_mask.shape, fr.shape)

            my_powder[my_module] += phot.sum(0)
            #my_counts[my_module] += phot.shape[0]
            my_counts[my_module] += phot.shape[0] - my_mask.sum(0) 
            if rank == 4:
                sys.stderr.write('\r%d/%d (%f Hz)' % (chunk+1, num_chunks, (nproc//16)*(chunk+1)*CHUNK_SIZE/(time.time()-stime)))
                sys.stderr.flush()

        sys.stderr.write('Rank %d: Reducing\n' % rank)
        sys.stderr.flush()

        #self.counts = np.zeros(16, dtype='i4')
        #comm.Reduce(my_counts, self.counts, op=MPI.SUM, root=0)
        #if rank == 0:
        #    sys.stderr.write('Reduced counts\n')
        #    sys.stderr.flush()

        self.powder = np.zeros((16,) + MODULE_SHAPE, dtype='f8').flatten()
        self.counts = np.zeros((16,) + MODULE_SHAPE, dtype='i4').flatten()
        num_pix = np.prod(MODULE_SHAPE)
        for m in range(16):
            comm.Reduce(my_powder.flatten()[m*num_pix:(m+1)*num_pix], self.powder[m*num_pix:(m+1)*num_pix], op=MPI.SUM, root=0)
            comm.Reduce(my_counts.flatten()[m*num_pix:(m+1)*num_pix],
                        self.counts[m*num_pix:(m+1)*num_pix], op=MPI.SUM, root=0)
            if rank == 0:
                sys.stderr.write('Reduced powder %d\n' % m)
                sys.stderr.write('Reduced counts %d\n' % m)
                sys.stderr.flush()

        if rank == 0:
            self.powder = self.powder.reshape((16,)+MODULE_SHAPE)
            self.counts = self.counts.reshape((16,)+MODULE_SHAPE)
            # AGIPD-specific hack for double-wide pixels
            #for i in range(512//64):
            #    self.counts[:, i*64-1:i*64+1, :] *= 2

            #self.powder /= self.counts[:,np.newaxis,np.newaxis]
            #sys.stderr.write('The following warning is anticipated and should be neglected')
            self.powder /= self.counts
        self.finish(write=(rank==0))
        if rank == 0:
            sys.stderr.write('Wrote file\n')
            sys.stderr.flush()

def main():
    parser = argparse.ArgumentParser(description='Calculate run integral')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-d', '--dark_run', help='Process raw data with this dark run', type=int, default=-1)
    parser.add_argument('-c', '--cells', help='Cell range to integrate (default: all)', default='all')
    parser.add_argument('-f', '--flag_file', help='Path to file containing flags of which events to process')
    parser.add_argument('-t', '--testing', help='Testing mode (only 10 chunks)', action='store_true')
    parser.add_argument('-n', '--num_trains', help='Only integrate the first N trains', type=int, default=-1)
    parser.add_argument('-m', '--mask', help='Mask file, otherwise all cells are integrated', type=str, default='')
    parser.add_argument('--num_cells', help='Set if number of detector cells is not 800', type=int, default=800)
    args = parser.parse_args()

    if args.dark_run < 0:
        args.dark_run = common.get_relevant_dark_run(args.run)
        
    i = None
    if args.cells == 'all':
        cells = [0, args.num_cells, 1]
    else:
        cells = [int(n) for n in args.cells.split(',')]
        if cells[-1] != 1:
            i = cells[0]
            sys.stderr.write(str(i)+'\n')
            sys.stderr.flush()
        if len(cells) < 2:
            raise ValueError('Need at least start and end values for cell range')
        if len(cells) < 3:
            cells = cells + [1]

    good_cells = np.zeros(args.num_cells, dtype='bool')
    good_cells[cells[0]:cells[1]:cells[2]] = True

    integ = Integrator(args.run, args.mask, selector=good_cells,
                       dark_run=args.dark_run, num_frames=args.num_trains*args.num_cells,
                       testing=args.testing, num_cells=args.num_cells, cell_id=i, flag_file=args.flag_file)
    integ.run_mpi()

if __name__ == '__main__':
    main()
