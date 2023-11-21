#!/bin/bash

#SBATCH --array=
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH -J vds
#SBATCH -o .vds-%4a-%j.out
#SBATCH -e .vds-%4a-%j.out

#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_004098
#####SBATCH --partition=upex

PREFIX=/gpfs/exfel/exp/SQS/202302/p004098/

source /etc/profile.d/modules.sh
source ${PREFIX}/scratch/ayyerkar/ana/source_this

run=$(printf %.4d "${SLURM_ARRAY_TASK_ID}")
extra-data-make-virtual-cxi ${PREFIX}/raw/r${run} -o ${PREFIX}/scratch/vds/r${run}.cxi --exc-suspect-trains
