#!/bin/bash

#SBATCH --array=
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH -J dark
#SBATCH -o .dark-%4a-%j.out
#SBATCH -e .dark-%4a-%j.out
##SBATCH --partition=upex

#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_004098

PREFIX=/gpfs/exfel/exp/SQS/202302/p004098/

source /etc/profile.d/modules.sh
module load exfel exfel-python

python ../proc_darks.py ${SLURM_ARRAY_TASK_ID} -m
