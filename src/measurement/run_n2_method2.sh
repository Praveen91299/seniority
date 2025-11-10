#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name=n2_method2

module load intelpython3

source activate sqseenv

python main_n2_method2.py ${1}