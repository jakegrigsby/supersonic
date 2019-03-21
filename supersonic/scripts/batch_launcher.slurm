#!/bin/bash
#SBATCH -A uva-dsi-mlunder
#SBATCH -n "$1"
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:4
#SBATCH --ntasks-per-node=1
#SBATCH -o hyp_search.out

#load anaconda and other python packages
module purge
module load singularity/2.6.1
module load anaconda/5.2.0-py3.6
module load mpi4py/3.0.0-py3.6

containerdir=/scratch/$USER
#train the agent
singularity-gpu exec --nv $containerdir/tensorflow-1.12.0-py36.simg mpiexec -n "$1" python "$2"