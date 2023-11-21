#!/bin/bash

#SBATCH --array=441
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH -J powder
#SBATCH -o .powder-%A_%a.out
#SBATCH -e .powder-%A_%a.out

#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_004098
#####SBATCH --partition=upex

# Change the runs to process using the --array option on line 3


source /etc/profile.d/modules.sh
module load exfel exfel-python

MAX_FRAMES=100000000

python ../powder.py ${SLURM_ARRAY_TASK_ID} -n ${MAX_FRAMES}



