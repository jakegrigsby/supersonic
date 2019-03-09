#!/bin/bash
#SBATCH -A uva-dsi-mlunder
#SBATCH -n 1
#SBATCH -p standard
#SBATCH --gres=gpu:k80:1

#load anaconda and other python packages
module load anaconda/5.2.0-py3.6
module load cmake/3.12.3
source activate cavml

#train the agent
python run_single_agent.py