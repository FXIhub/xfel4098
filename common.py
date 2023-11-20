import os
import glob

import numpy as np

from constants import PREFIX, MODULE_SHAPE, ADU_PER_PHOTON

def get_relevant_dark_run(run_num):
    '''Get dark run corresponding to given data run

    Currently returning the latest processed dark run before the given run
    '''
    dark_files = sorted(glob.glob(PREFIX + 'dark/r????_dark.h5'))
    dark_runs = np.array([int(op.splitext(op.basename(fname))[0][1:5]) for fname in dark_files])

    # Get nearest dark run *before* given run
    return dark_runs[dark_runs < run_num][-1]

def calibrate(raw, offset, output=None):
    '''Calibrate raw data given offsets to photon counts
    
    Can calibrate individual module or whole detector
    If output is specified, that array will be updated
    '''
    assert raw.shape == offset.shape
    if raw.shape == MODULE_SHAPE:
        return _calibrate_module(raw, offset, output=output)
    elif raw.shape == (16,) + MODULE_SHAPE:
        if output is not None:
            [_calibrate_module(raw[i], offset[i], output=output[i]) for i in range(16)]
            return output
        else:
            return np.array([_calibrate_module(raw[i], offset[i], output=output[i]) for i in range(16)])
    else:
        raise ValueError('Unknown data shape: %s' % (raw.shape))

def _calibrate_module(raw, offset, output=None):
    if output:
        output[:] = raw - offset
    else:
        output = raw - offset

    # ASIC-wide common mode correction
    output[:] = (output.reshape(128, 2, 256) - np.median(output.reshape(128, 2, 256), axis=(0,2), keepdims=True)).reshape(MODULE_SHAPE)

    # Photon conversion with 0.7-photon threshold
    output[:] = np.rint(output/ADU_PER_PHOTON - 0.2)
    output[output < 0] = 0

    return output
