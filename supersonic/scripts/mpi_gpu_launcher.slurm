#!/bin/bash
#SBATCH -A uva-dsi-mlunder
#SBATCH -n 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:4
#SBATCH --ntasks-per-node=16
#SBATCH -o mpi_agent.out

#load anaconda and other python packages
module purge
module load singularity/2.6.1
module load anaconda/5.2.0-py3.6
module load gcc
module load openmpi

containerdir=/scratch/$USER
program_name=$1
shift
shift
#train the agent
singularity-gpu exec --nv $containerdir/tensorflow-1.12.0-py36.simg srun python "$program_name" "$@"