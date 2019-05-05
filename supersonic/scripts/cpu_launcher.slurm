#!/bin/bash
#SBATCH -A uva-dsi-mlunder
#SBATCH -n 2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=16
#SBATCH -o agent.out
#SBATCH -t 24:00:00
#SBATCH -p standard

#load anaconda and other python packages
module purge
module load anaconda/5.2.0-py3.6
source activate cavml

#train the agent
srun python train.py
