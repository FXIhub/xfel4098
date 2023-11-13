import sys
import os
import numpy as np
import h5py
import extra_geom
import warnings


from hummingbird import plotting
from hummingbird import analysis
from hummingbird import ipc
from hummingbird.backend import add_record


from emcwrt import HitSaver

state = {}
state['Facility'] = 'EuXFEL'
state['EventIsTrain'] = True
state['EuXFEL/DataSource'] = 'tcp://exflong23-ib:55555'
state['EuXFEL/DataFormat'] = 'Calib'
state['EuXFEL/SelModule'] = None
state['EuXFEL/MaxTrainAge'] = 4e20
state['EuXFEL/FirstCell'] = 0
state['EuXFEL/LastCell'] = 799

#geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom('../../geom/motor_p4462_from_3375.geom')
quad_pos = [
    (-130, 5),
    (-130, -125),
    (5, -125),
    (5, 5),
]
geom = extra_geom.DSSC_1MGeometry.from_quad_positions(quad_pos)
adu_per_photon = 5.0

prop_dir = "/gpfs/exfel/exp/SQS/202302/p003004"

send_hits = True
send_powdersum = True

save_hits = False

if save_hits:
    hits_dir = os.path.join(prop_dir, "scratch/emc")
    hit_saver = HitSaver(0, hits_dir, adu_per_photon, 16*128*512,
                         [0,8,15], np.s_[:], np.s_[:128], 3*128*128, 500)

def onEvent(evt):
    sys.stdout.flush()
    analysis.event.printProcessingRate()

    native_evt = evt._evt

    arbiter = native_evt['SQS_NQS_DSSC/CAL/ARBITER:output']
    if 'SQS_DET_DSSC1M-1/DET/STACKED:xtdf' not in native_evt:
        return

    det = native_evt['SQS_DET_DSSC1M-1/DET/STACKED:xtdf']

    hitscores = arbiter['sphits.litpixelCount']
    hitscoreThreshold = arbiter['sphits.threshold']
    hit_mask = arbiter['sphits.hits']
    num_hits = arbiter['sphits.numberOfHits']
    num_miss = arbiter['sphits.numberOfMiss']

    # plot hitscore
    for i in range(hitscores.shape[0]):
        hitscore_pulse = add_record(evt['analysis'], 'analysis', 'hitscore', hitscores[i])
        try:
            plotting.line.plotHistory(hitscore_pulse, group='Hitfinding', hline=hitscoreThreshold, history=10000)
        except KeyError:
            print('Timestamp not found')
            return

    hitscore_mean = add_record(evt['analysis'], 'analysis', 'avg_hitscore', hitscores.mean())
    plotting.line.plotHistory(hitscore_mean, group='Hitfinding', history=10000)

    # plot hitrate
    #hit_mask = (hitscores > hitscoreThreshold)
    #analysis.hitfinding.hitrate(evt, hit_mask, history=3500)
    hitrate = add_record(evt['analysis'], 'analysis', 'hitrate', arbiter['sphits.hitrate'] * 100)

    if ipc.mpi.is_main_event_reader():
        plotting.line.plotHistory(evt['analysis']['hitrate'], label='Hit rate [%]', group='Hitfinding')

    hit_ix = np.flatnonzero(hit_mask)
    num_hits = len(hit_ix)

    if send_hits and (num_hits > 0):
        stacked = det['image.data']
        #print(np.sum(hitscores > 10))
        #print(f"Num hits {num_hits}")
        #print(f"Images shape: {stacked.shape}")

        # random hit
        rnd_i = np.random.choice(num_hits)
        assem = geom.position_modules_fast(stacked[rnd_i])[0][::-1,::-1]
        assem[np.isnan(assem)] = -1
        random_hit = add_record(evt['analysis'], 'analysis', 'Random Hit', assem)
        plotting.image.plotImage(random_hit, group='Hits', history=10)

        # brightest hit
        brt_i = hitscores[hit_ix].argmax()
        assem = geom.position_modules_fast(stacked[brt_i])[0][::-1,::-1]
        assem[np.isnan(assem)] = -1
        brightest_hit = add_record(evt['analysis'], 'analysis', 'Brightest Hit', assem)
        plotting.image.plotImage(brightest_hit, group='Hits', history=10)

    if save_hits and (num_hits > 0):
        stacked = det['image.data']
        hit_saver.write(stacked)

    if send_powdersum:
        if num_hits > 0:
            if 'aggregate.sumHits' in det:
                pw_hits = det['aggregate.sumHits']
            else:
                stacked = det['image.data']
                #stacked[stacked < 0.75 * adus_per_photon] = 0
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    pw_hits = np.nanmean(stacked, axis=0)

            assem = geom.position_modules_fast(pw_hits)[0][::-1,::-1]
            assem[np.isnan(assem)] = -1
            hits_integral = add_record(evt['analysis'], 'analysis', 'Hits integral', assem)
            plotting.image.plotImage(hits_integral, group='Hits', history=1, sum_over=True)
            train_integral = add_record(evt['analysis'], 'analysis', 'Train integral', assem)
            plotting.image.plotImage(train_integral, group='Hits')
        if num_miss > 0 and 'aggregate.sumMiss' in det:
            pw_miss = det['aggregate.sumMiss']
            assem = geom.position_modules_fast(pw_miss)[0][::-1,::-1]
            assem[np.isnan(assem)] = -1
            miss_integral = add_record(evt['analysis'], 'analysis', 'Miss integral', assem)
            plotting.image.plotImage(miss_integral, group='Hits', history=1, sum_over=True)

