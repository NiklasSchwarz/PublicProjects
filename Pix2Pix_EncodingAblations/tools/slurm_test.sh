#!/usr/bin/env bash

##GENERAL -----
#SBATCH --cpus-per-task=2
##SBATCH --gres=gpu:1080:1
##SBATCH --gres=gpu:2080:1
##SBATCH --gres=gpu:3080:1
##SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:a100_80gb:1
#SBATCH --gres=gpu
#SBATCH --mem=32000M
##SBATCH --mem=10000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --job-name=TMANet
#SBATCH --output=log/%j.out

##DEBUG -----
##SBATCH --partition=debug
##SBATCH --time=00:20:00

##NORMAL -----
#SBATCH --partition=gpu
#SBATCH --time=6-00:00:00
##SBATCH --exclude=gpu[01,02,04]

#module load comp/gcc/11.2.0
#module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hoyos-env
export PYTHONPATH=/beegfs/work_fast/schwarz/hadamard-hoyos:$PYTHONPATH
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4


scontrol show job $SLURM_JOB_ID
scontrol write batch_script $SLURM_JOB_ID -

srun python -u tools/test.py "$@"
