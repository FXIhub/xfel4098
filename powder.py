import argparse

parser = argparse.ArgumentParser(description='Save hits in photon units to a cxi file')

parser.add_argument('run', type=int, help='Run number')

parser.add_argument('-n', '--max_frames',
                    help='maximum number of frames to sum',
                    type=int)

args = parser.parse_args()

import common
from constants import *
DARK_PATH   = 'data/mean'
DARK_cellID = 'data/cellId'
DATA_PATH   = 'entry_1/instrument_1/detector_1/data'

# get dark run
args.dark_run = common.get_relevant_dark_run(args.run)

args.dark_file        = PREFIX+'dark/r%.4d_dark.h5'%args.dark_run
args.vds_file         = PREFIX+'vds/r%.4d.cxi' %args.run
args.output_file      = PREFIX+'powder/powder_r%.4d.cxi'%args.run


import numpy as np
import h5py
#import extra_data
from tqdm import tqdm
import sys
import time
import os

import multiprocessing as mp

if os.path.exists(args.output_file):
    print('Deleting existing powder file:', args.output_file)
    os.remove(args.output_file)



# read darks
print('Reading dark from      ', args.dark_file)
sys.stdout.flush()
with h5py.File(args.dark_file) as f:
    dark        = f[DARK_PATH][()]
    cellID_dark = f[DARK_cellID][:]

# make a dictionary for easy look up
dark_dict = {}
for i in range(dark.shape[1]):
    dark_dict[cellID_dark[i]] = dark[:,i]

done = False
with h5py.File(args.vds_file) as f:
    cellID    = f['/entry_1/cellId'][:, 0]
    trainID   = f['/entry_1/trainId'][()]
    pulseID   = f['/entry_1/pulseId'][()]
            
Nevents = cellID.shape[0]

if args.max_frames :
    Nevents = min(args.max_frames, cellID.shape[0])

indices = np.arange(cellID.shape[0], -1, -1)[:Nevents]

size = min(mp.cpu_count(), 32)

# split frames over ranks
events_rank = np.linspace(0, Nevents, size+1).astype(int)

frame_shape = (16,128,512)

def worker(rank, lock):
    my_indices = indices[events_rank[rank]: events_rank[rank+1]] 
    
    print(f'rank {rank} is processing indices {events_rank[rank]} to {events_rank[rank+1]}')
    sys.stdout.flush()

    if rank == 0 :
        it = tqdm(range(len(my_indices)), desc = f'Processing data from {args.vds_file}')
    else :
        it = range(len(my_indices))

    frame      = np.empty(frame_shape, dtype = np.uint16)
    corr_frame = np.empty(frame_shape, dtype = float)
    
    powder = np.zeros(frame_shape, dtype = float)
    sum_index = 0
    done = False
    with h5py.File(args.vds_file) as g:
        data = g[DATA_PATH]
                
        for i in it:
            index = my_indices[i]
            if cellID[index] not in [0, 810] :
                frame[:] = np.squeeze(data[index]).astype(float)
                
                common.calibrate(frame, dark_dict[cellID[index]], output = corr_frame)
                
                powder += corr_frame
                sum_index += 1
            
    # take turns writing frame_buf to file 
        
    # write to file sequentially
    if rank == 0: 
        print('Writing photons to     ', args.output_file)
        sys.stdout.flush()
    
    if lock.acquire() :
        with h5py.File(args.output_file, 'a') as f:
            if 'data' in f :
                powder += f['data'][()] 
                sum_index += f['Nframes'][()] 
                f['data'][...] = powder
                f['Nframes'][...] = sum_index
            else :
                f['data'] = powder
                f['Nframes'] = sum_index
        
        print(f'rank {rank} done')
        sys.stdout.flush()
        lock.release()


lock = mp.Lock()
jobs = [mp.Process(target=worker, args=(m, lock)) for m in range(size)]
[j.start() for j in jobs]
[j.join() for j in jobs]

print('Done')
sys.stdout.flush()



