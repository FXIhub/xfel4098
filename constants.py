'''Constants for beamtime p4098 at SQS'''

PREFIX = '/gpfs/exfel/exp/SQS/202302/p004098/scratch/'

VDS_DATASET = '/entry_1/instrument_1/detector_1/data'

DET_NAME = 'SQS_DET_DSSC1M-1'
MODULE_SHAPE = (128, 512)
NPULSES = 400
NCELLS = 400
CHUNK_SIZE = 32
BAD_CELLIDS = list(range(0,17,2)) + list(range(484,499)) + [810]
ADU_PER_PHOTON = 5.

MASK_FNAME = PREFIX + 'geom/badpixel_mask_r0010.h5'

XGM_DA_NUM = 1
XGM_DATASET = "INSTRUMENT/SQS_DIAG1_XGMD/XGM/DOOCS:output/data/intensitySa3TD"
NBUNCHES_DA_NUM = 1
NBUNCHES_DATASET = '/CONTROL/SQS_DIAG1_XGMD/XGM/DOOCS/pulseEnergy/numberOfSa3BunchesActual/value'
DCONF_DA_NUM = 1
DCONF_DATASET = '/RUN/SQS_NQS_DSSC/FPGA/PPT_Q1/epcRegisterFilePath/value'
