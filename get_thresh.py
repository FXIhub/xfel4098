import argparse
import os
import sys

import h5py
import numpy as np
from scipy import ndimage, optimize

from constants import NCELLS, PREFIX, NPULSES
LOCAL_SUBT = False

def _get_litpix(run, normed=False):
    # Get lit pixels
    with h5py.File(PREFIX+'events/r%.4d_events.h5'%run, 'r') as f:
        litpix = f['entry_1/litpixels'][:].sum(0)
        if normed:
            xgm = f['entry_1/pulse_energy_uJ'][:]
            # Some trains at the end of a DA file have null XGM values
            xgm[np.where(xgm==0)[0]] = xgm[np.where(xgm==0)[0] - NCELLS].copy()

            litpix = litpix / xgm
            litpix[~np.isfinite(litpix)] = 0
            litpix[xgm < 2] = 0
            if LOCAL_SUBT:
                litpix -= ndimage.median_filter(litpix.reshape(-1,NCELLS), (50,1)).ravel()

    return litpix.reshape(-1,NCELLS)

def get_thresh_all(run, return_litpix=False, normed=False, verbose=False):
    sel_litpix = _get_litpix(run, normed=normed)[:,:NPULSES]

    # Get hit indices
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x-x0)**2 / 2 / sigma**2)

    if LOCAL_SUBT:
        binvals = np.arange(-1,10,0.01) if normed else np.arange(100,10000,10)
    else:
        binvals = np.arange(0.01,10,0.01) if normed else np.arange(100,10000,10)
    hy = np.histogram(sel_litpix.ravel(), bins=binvals)[0]
    hcen = (binvals[1:] + binvals[:-1]) * 0.5
    thresh = np.ones(NCELLS) * -100.
    #xmax = np.abs(hcen).argmin() if normed else hy.argmax() + 1 # Ignoring first bin
    xmax = hy.argmax() if normed else hy.argmax() + 1 # Ignoring first bin
    try:
        p0_std = 0.01 if normed else 100
        popt, pcov = optimize.curve_fit(gaussian,
                                        hcen, hy,
                                        p0=(hy.max(), hcen[xmax], p0_std))
        thresh[:] = popt[1] + 4*np.abs(popt[2])
        if verbose:
            print('Fitted background Gaussian: %.3f +- %.3f' % (popt[1], popt[2]))
    except (RuntimeError, ValueError):
        print('Fitting failed')

    if return_litpix:
        return thresh, sel_litpix
    return thresh

def get_thresh(run, return_litpix=False, normed=False, verbose=False):
    sel_litpix = _get_litpix(run, normed=normed)

    # Get hit indices
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x-x0)**2 / 2 / sigma**2)

    binvals = np.arange(-1,1,0.001) if normed else np.arange(10,10000,10)
    chists = np.array([np.histogram(sel_litpix[:,i], bins=binvals)[0] for i in range(NCELLS)])
    hcen = (binvals[1:] + binvals[:-1]) * 0.5
    thresh = np.ones(NCELLS) * -100.

    for i in range(NCELLS):
        hy = chists[i]
        xmax = np.abs(binvals).argmin() if normed else hy[1:].argmax() + 1 # Ignoring first bin
        try:
            p0_std = 0.01 if normed else 100
            popt, pcov = optimize.curve_fit(gaussian,
                                            hcen[1:xmax], hy[1:xmax],
                                            p0=(hy.max(), hcen[xmax], p0_std))
            #thresh[i] = popt[1] + 3*np.abs(popt[2])
            thresh[i] = popt[1] + 4*np.abs(popt[2])
            if verbose:
                print('Fitted background Gaussian: %.3f +- %.3f' % (popt[1], popt[2]))
        except RuntimeError:
            print('Fitting failed for cell %d' % i)
        except ValueError:
            print('Fitting failed for cell %d' % i)

    if return_litpix:
        return thresh, sel_litpix
    return thresh

def linearize(thresh, verbose=False):
    sel = (thresh > -100)
    if sel.shape[0] == 352:
        sel[0] = False
    if verbose:
        print('%d cells used for fitting' % sel.sum())
    xvals = np.arange(NCELLS)
    fit = np.polyfit(xvals[sel], thresh[sel], 1)
    #fit = np.polyfit(xvals[sel], thresh[sel], 2)
    if verbose:
        print('Fit parameters:', fit)
    return np.polyval(fit, xvals)

def get_hitinds(run, thresh, litpix=None, normed=False, verbose=False):
    if litpix is None:
        litpix = _get_litpix(run, normed=normed)
    hit_inds = np.where((litpix > thresh).ravel())[0]
    hit_inds = hit_inds[hit_inds % NCELLS <= NPULSES]
    if verbose:
        print('%d hits using a threshold range of %.3f - %.3f (%.2f %%)' % (len(hit_inds), thresh.min(), thresh.max(), len(hit_inds) / litpix.size * 100))
    return hit_inds

def main():
    global NCELLS, NPULSES
    parser = argparse.ArgumentParser(description='Save hits to emc file')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-N', '--norm', help='Normalize by pulse energy', action='store_true')
    parser.add_argument('--ncells', help='Override constant NCELLS', type=int)
    parser.add_argument('--npulses', help='Override constant NPULSES', type=int) 
    args = parser.parse_args()

    if args.ncells is not None:
        NCELLS = args.ncells
    if args.npulses is not None:
        NPULSES = args.npulses
    thresh = get_thresh_all(args.run, normed=args.norm, verbose=True, return_litpix=False)
    linthresh = linearize(thresh, verbose=True)
    hitinds = get_hitinds(args.run, linthresh, normed=args.norm, verbose=True)
    #print(np.unique(hitinds%NCELLS, return_counts=True))

if __name__ == '__main__':
    main()
