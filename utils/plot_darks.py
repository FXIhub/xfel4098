import argparse
parser.add_argument('runs', nargs='+', 
                    help='list of run numbers',
                    type=int, default=[23, 24])

args = parser.parse_args()

runs = args.runs

import h5py 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import extra_data

from ..constants import *

means  = np.zeros((len(runs), 800), dtype = float)
sigmas = np.zeros((len(runs), 800), dtype = float)

cellIDs = []
# load darks and plot mean and std against cell id and gain mode
for i, run in tqdm(enumerate(runs)) :
    dark_fnam = PREFIX + 'dark/r%.4d_dark.h5'%run
    with h5py.File(dark_fnam) as f:
        mean    = f['data/mean'][()]
        sig     = f['data/sigma'][()]
        #cellID  = f['data/cellId'][()]
        cellIDs.append(f['data/cellId'][()])
    
    # median of mean 
    means[i, :mean.shape[1]]  = np.nanmean(mean, axis=(0, 2, 3))
    
    # median of sigma 
    sigmas[i, :sig.shape[1]] = np.nanmean(sig, axis=(0, 2, 3))

"""
# save data
import pickle
pickle.dump([means, sigmas], open('dark_temp.pickle', 'wb'))

# load data
import pickle
cellIDs = []
for run in runs :
    dark_fnam = PREFIX + 'dark/r%.4d_dark.h5'%run
    with h5py.File(dark_fnam) as f:
        cellIDs.append(f['data/cellId'][()])

means, sigmas = pickle.load(open('dark_temp.pickle', 'rb'))
"""





def _decode_fstring(s, form):
    index = 0
    index_form = 0
    out = {}
    for z in range(5):
        i = form.find('{', index_form)
        j = form.find('}', index_form)

        start = form[index_form + 1 : i]
        end   = form[j + 1 : form.find('{', j)]
         
        key = form[i + 1 : j]
        index_form = j+1

        if i == -1 :
            break

        i = s.find(start, index)
        j = s.find(end, index)
        value = s[i + len(start) : j]
    
        index = j

        out[key] = value

    return out

def _get_gain_frequency_integration_time(proposal, run_num):
    run = extra_data.open_run(proposal, run_num)
    
    # e.g. fnam = '/scratch/xctrl/karabo/var/data/maia_4098/GenConf_TG2.5_nG12_trimm_f4.5_intgr50/Q1/GenConf_TG2.5_nG12_trimm_f4.5_intgr50_epc.xml'
    # e.g. fnam = '.../GenConf_TG2.{gain}_nG{n}_trimm_f{frequency}_intgr{integration_time}_epc.xml'
    fnam = run.get_run_value('SQS_NQS_DSSC/FPGA/PPT_Q1', 'epcRegisterFilePath.value').split('/')[-1]
    form = 'GenConf_TG2.{gain}_nG{n}_trimm_f{frequency}_intgr{integration_time}_epc.xml'
    
    # in ADU per photon (this depends on photon energy)
    out = _decode_fstring(fnam, form)
    gain             = float(out['gain'])
    frequency        = 1e-6 * float(out['frequency'])
    integration_time = 1e-9 * float(out['integration_time'])

    return gain, frequency, integration_time

# get gain and frequency 
run_str = []
gains = []
for run in runs :
    try :
        g, f, t = _get_gain_frequency_integration_time(PROPOSAL, run)
        s = f'{run}, {g}, {round(1e6 * f, 1)}, {round(1e9 * t, 1)}'
        gains.append(g)
    except :
        s = f'run {run} --'
        gains.append(1)

    # ?
    if g == 0 :
        gains[-1] = 1

    run_str.append(s)


# plot
fig, axs = plt.subplots(2, 1)
fig.set_size_inches(8, 12)
#fig.set_layout_engine(layout='compressed')

for i, run in enumerate(runs):
    print(run)
    cell = cellIDs[i]
    axs[0].plot(cell, means[i, :len(cell)] , label=run_str[i], alpha=1.0, linewidth=1.)
    axs[1].plot(cell, sigmas[i, :len(cell)], label=run_str[i], alpha=1.0, linewidth=1.)
    
    axs[0].legend(fontsize=6)
    axs[1].legend(fontsize=6)
    
    axs[0].set_title('average of mean of cell value \n (run, gain (adu/keV), freq. (Mhz), integration time (ns)')
    axs[1].set_title('average of std of cell value ')
    

#plt.show()
plt.savefig('dark_plots.png')
