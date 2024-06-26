import glob
import os.path as op
from functools import lru_cache

import numpy as np
import h5py

from constants import ADU_PER_PHOTON, MODULE_SHAPE, PREFIX, GAIN_FNAME

@lru_cache(maxsize=1000)
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
    assert raw.shape == offset.shape, (raw.shape, offset.shape)
    if raw.shape == MODULE_SHAPE:
        return _calibrate_module(raw, offset, output=output)
    elif raw.shape == (16,) + MODULE_SHAPE:
        if output is not None:
            [_calibrate_module(raw[i], offset[i], output=output[i]) for i in range(16)]
            return output
        else:
            return np.array([_calibrate_module(raw[i], offset[i]) for i in range(16)])
    else:
        raise ValueError('Unknown data shape: %s' % (raw.shape))

def _calibrate_module(raw, offset, output=None):
    if output is not None:
        output[:] = raw - offset
    else:
        output = raw - offset

    # ASIC-wide common mode correction
    output[:] = (output.reshape(128, 2, 256) - np.median(output.reshape(128, 2, 256), axis=(0,2), keepdims=True)).reshape(MODULE_SHAPE)

    # Photon conversion with 0.7-photon threshold
    output[:] = np.rint(output/ADU_PER_PHOTON - 0.2)
    output[output < 0] = 0

    return output

def calibrate2(raw, offset, thresh, output=None):
    '''Calibrate raw data to photon counts, give offsets and pixel-wise thresholds

    Can calibrate individual module or whole detector
    If output is specified, that array will be updated
    '''
    assert raw.shape == offset.shape, (raw.shape, offset.shape)
    assert raw.shape == thresh.shape
    if raw.shape == MODULE_SHAPE:
        return _calibrate_module(raw, offset, output=output)
    elif raw.shape == (16,) + MODULE_SHAPE:
        if output is not None:
            [_calibrate_module2(raw[i], offset[i], thresh[i], output=output[i]) for i in range(16)]
            return output
        else:
            return np.array([_calibrate_module2(raw[i], offset[i], thresh[i]) for i in range(16)])
    else:
        raise ValueError('Unknown data shape: %s' % (raw.shape))

def _calibrate_module2(raw, offset, thresh, output=None):
    if output is not None:
        output[:] = raw - offset
    else:
        output = raw - offset

    # ASIC-wide common mode correction
    output[:] = (output.reshape(128, 2, 256) - np.median(output.reshape(128, 2, 256), axis=(0,2), keepdims=True)).reshape(MODULE_SHAPE)

    # Photon conversion with 0.7-photon threshold
    output[:] = np.floor(output/ADU_PER_PHOTON - thresh)
    output[output < 0] = 0

    return output

@lru_cache(maxsize=2)
def _get_dark_data(dark_run):
    with h5py.File(PREFIX + 'dark/r%.4d_dark.h5' % dark_run, 'r') as f:
        offset = f['data/mean'][:]
        noise = f['data/sigma'][:]
    return offset, noise

@lru_cache(maxsize=1)
def _get_gain(gain_fname):
    return np.load(gain_fname)

def _calibrate_module3(raw, offset, noise, output=None):
    if output is not None:
        output[:] = raw - offset
    else:
        output = raw - offset
    relthresh = 0.5 - noise**2/5**2 * (-6.907755)
    gain = _get_gain(GAIN_FNAME)[0]
    output[:] = np.clip(np.ceil(output/gain - relthresh), 0, np.inf)

    return output

def calibrate3(raw, cid, run_num, output=None):
    '''Calibrate raw data to photon counts for whole detector

    Uses run number and cellId to get relevant constants
    If output is specified, that array will be updated
    '''
    offset, noise = _get_dark_data(get_relevant_dark_run(run_num))
    assert raw.shape == offset[:,0].shape, (raw.shape, offset[:,0].shape)

    if output is not None:
        [_calibrate_module3(raw[m], offset[m,cid], noise[m,cid], output=output[m]) for m in range(16)]
        return output
    else:
        return np.array([_calibrate_module3(raw[m], offset[m,cid], noise[m,cid]) for m in range(16)])
