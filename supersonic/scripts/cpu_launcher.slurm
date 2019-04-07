#!/bin/bash
#SBATCH -A uva-dsi-mlunder
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH -o agent.out
#SBATCH -t 24:00:00
#SBATCH -p standard

#load anaconda and other python packages
module purge
module load anaconda/5.2.0-py3.6
source activate cavml

#train the agent
python "$1"