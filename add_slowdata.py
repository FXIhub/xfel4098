import glob
import logging
import argparse

import h5py
import numpy as np

from constants import PREFIX

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def read_slowdata_map(raw_data_files, value_dataset: str):
    train_mapping = dict()
    for fname in raw_data_files:
        logging.info('loading %s', fname)
        with h5py.File(fname, 'r') as fptr:
            for train_id, val in zip(fptr['/INDEX/trainId'][...], fptr[value_dataset][...]):
                if train_id in train_mapping:
                    logging.warning('Duplicate train id %d (file: %s)', train_id, fname)
                train_mapping[train_id] = val
    return train_mapping

def main():
    parser = argparse.ArgumentParser(description='Add slowdata (1 number/train) to events file')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('da_num', help='DA file number in raw data', type=int)
    parser.add_argument('value_dataset', help='Dataset name in raw DA data')
    parser.add_argument('output_name', help='Output dataset name in events file (eg. pp_delay_ps for pump-probe delay)')
    parser.add_argument('-f', '--force', help='Overwrite data if already exists', action='store_true')
    args = parser.parse_args()

    event_fname = PREFIX + '/events/r%.4d_events.h5' % args.run
    with h5py.File(event_fname, 'r') as fptr:
        if not args.force and args.output_dataset in fptr['entry_1']:
            logging.info('%s already exists. Skipping.', args.output_dataset)
            return

    glob_str = PREFIX + '/raw/r%.4d/RAW-R%.4d-DA%.2d-S*.h5' % (args.run, args.run, args.da_num)
    raw_data_files = sorted(glob.glob(glob_str))
    smap = read_slowdata_map(raw_data_files, args.value_dataset)

    logging.info('Writing to %s/entry_1/%s', event_fname, args.output_dataset)
    with h5py.File(event_fname, 'a') as fptr:
        values = np.array([smap[train] for train in fptr['/entry_1/trainId'][...]])

        if args.output_dataset in fptr['entry_1']:
            logging.warning('The output dataset exists, replacing it.')
            del fptr['entry_1'][args.output_dataset]
        fptr['entry_1'][args.output_dataset] = values

if __name__ == '__main__':
    main()
