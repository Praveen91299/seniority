#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=24:00:00
#SBATCH --job-name=VO_parallel

module load python/3.10

python main_sampling_cost_VO_parallel.py ${1} ${2}