import os
import sys
import argparse

import numpy as np
import dragonfly

from constants import PREFIX, MODULE_SHAPE

def bin_frame(fr, binning=4):
    bin_shape = tuple(np.array(MODULE_SHAPE)//binning)
    return fr.reshape((16,bin_shape[0],binning,bin_shape[1],binning)).sum((2,4)).ravel()

def main():
    parser = argparse.ArgumentParser(description='Bin emc hits')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-b', '--binning', help='Binning factor', type=int, default=4)
    args = parser.parse_args()

    det = dragonfly.Detector(PREFIX+'geom/det_4098_v1.h5')
    emc = dragonfly.EMCReader(PREFIX+'emc/r%.4d.emc'%args.run, det)

    try:
        assert MODULE_SHAPE[0] % args.binning == 0
        assert MODULE_SHAPE[1] % args.binning == 0
    except AssertionError:
        print('Binning factor %d does not divide module shape:'%args.binning, MODULE_SHAPE)
        return 1

    bin_npix = 16 * np.prod(MODULE_SHAPE) // args.binning**2
    os.makedirs(PREFIX+'emc/bin%d' % args.binning, exist_ok=True)
    wemc = dragonfly.EMCWriter(PREFIX+'emc/bin%d/r%.4d_bin%d.emc' % (args.binning, args.run, args.binning), bin_npix, hdf5=False)

    for d in range(emc.num_frames):
        mframe = emc.get_frame(d, raw=True) * det.mask
        wemc.write_frame(bin_frame(mframe))
        sys.stderr.write('\r%d/%d' % (d+1, emc.num_frames))
    sys.stderr.write('\n')
    wemc.finish_write()

if __name__ == '__main__':
    main()
