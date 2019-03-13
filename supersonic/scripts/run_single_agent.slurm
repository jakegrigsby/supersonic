#!/bin/bash
#SBATCH -A uva-dsi-mlunder
#SBATCH -n 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks-per-node=1
#SBATCH -o agent.out

#load anaconda and other python packages
module load anaconda/5.2.0-py3.6
source activate cavml

#train the agent
python run_single_agent.py