#!/bin/bash

#SBATCH --array=
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH -J savehits
#SBATCH -o .res-%4a-%j.out
#SBATCH -e .res-%4a-%j.out
#SBATCH --partition=upex

#####SBATCH --partition=upex-beamtime
#####SBATCH --reservation=upex_004098

PREFIX=/gpfs/exfel/exp/SQS/202302/p004098/

source /etc/profile.d/modules.sh
source ${PREFIX}/scratch/ayyerkar/ana/source_this

python ../resave_hits.py ${SLURM_ARRAY_TASK_ID}
