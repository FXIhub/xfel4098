'''Constants for beamtime p4098 at SQS'''

PREFIX = '/gpfs/exfel/exp/SQS/202302/p004098/scratch/'
VDS_DATASET = '/entry_1/instrument_1/detector_1/data'
BAD_CELLIDS = list(range(0,17,2)) + list(range(484,499)) + [810]
ADU_PER_PHOTON = 5.
MASK_FNAME = PREFIX + 'geom/badpixel_mask_p2601.h5'
