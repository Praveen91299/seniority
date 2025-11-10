#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=6:00:00
#SBATCH --job-name=n2_method1

module load intelpython3

source activate sqseenv

python main_n2_method1.py ${1}