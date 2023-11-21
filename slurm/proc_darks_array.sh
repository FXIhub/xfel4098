#!/bin/bash

#SBATCH --array=
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH -J litpix
#SBATCH -o .dark-%4a-%j.out
#SBATCH -e .dark-%4a-%j.out

#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_004098
####SBATCH --partition=upex

PREFIX=/gpfs/exfel/exp/SQS/202302/p004098/

source /etc/profile.d/modules.sh
source ${PREFIX}/scratch/ayyerkar/ana/source_this

python ../proc_darks.py ${SLURM_ARRAY_TASK_ID}
