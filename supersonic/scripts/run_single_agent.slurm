#!/bin/bash
#SBATCH -A uva-dsi-mlunder
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks-per-node=1
#SBATCH -o agent.out

#load anaconda and other python packages
module purge
module load singularity/2.6.1
module load anaconda/5.2.0-py3.6

containerdir=/scratch/$USER
#train the agent
singularity-gpu exec --nv $containerdir/tensorflow-1.12.0-py36.simg python run_single_agent.py