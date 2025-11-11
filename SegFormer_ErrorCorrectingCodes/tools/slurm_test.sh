#!/usr/bin/env bash

##GENERAL -----
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
##SBATCH --gres=gpu:h100:1
#SBATCH --mem=10000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --job-name=hadamardTest
#SBATCH --output=log/%j_Debug.out

##DEBUG -----
#SBATCH --partition=gpub,gpu
##SBATCH --partition=debug
#SBATCH --time=00:20:00

##NORMAL -----
##SBATCH --partition=gpub
##SBATCH --time=1-00:00:00
##SBATCH --exclude=gpu[04,01]

##module load comp/gcc/11.2.0
##module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate experiments-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/beegfs/work_fast/schwarz/hadamard-experiments:$PYTHONPATH
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

scontrol show job $SLURM_JOB_ID
scontrol write batch_script $SLURM_JOB_ID -

port=$(comm -23 <(seq 20000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

srun python -u tools/test.py $1 $2 --launcher="slurm" --cfg-options env_cfg.dist_cfg.port=${port} "${@:3}"
