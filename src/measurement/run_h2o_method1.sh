#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=1:00:00
#SBATCH --job-name=h2o_method1

module load intelpython3

source activate sqseenv

python main_h2o_method1.py ${1}