#!/bin/bash

#SBATCH --array=
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH -J flag_integrate
#SBATCH -o .int-%4a-%j.out
#SBATCH -e .int-%4a-%j.out
#SBATCH --partition=upex

####SBATCH --partition=upex-beamtime
####SBATCH --reservation=upex_002995

source /etc/profile.d/modules.sh
module purge
source ../source_this
module load exfel openmpi-no-python

PREFIX=/gpfs/exfel/exp/SQS/202302/p004098/

run=`printf %.4d "${SLURM_ARRAY_TASK_ID}"`
flag_file=${PREFIX}/scratch/events/r${run}_events.h5

mpirun -mca btl_tcp_if_include ib0 python ../integrate.py $run -f $flag_file --num_cells=600
#mpirun -mca btl_tcp_if_include ib0 python ../integrate.py $run -c 1,326,1
