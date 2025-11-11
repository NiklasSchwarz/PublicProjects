#!/usr/bin/env bash

##GENERAL ----
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=Hadamard
#SBATCH --output=log/%j.out
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

##Buffer Jobs ----
#SBATCH --partition=gpub
#SBATCH --qos=gpub_qos

##Specific GPU ----
##SBATCH --nodelist=gpu05
##SBATCH --gres=gpu:1080:1
#SBATCH --gres=gpu:ada6000:1
##SBATCH --gres=gpu:h100:1
##SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:a100_80gb:1

#module load comp/gcc/11.2.0
#module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate experiments-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

scontrol show job $SLURM_JOB_ID
scontrol write batch_script $SLURM_JOB_ID -

port=$(comm -23 <(seq 30000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

srun python -m tools.train "$1" --launcher=slurm --resume "${@:2}" \
  --cfg-options env_cfg.dist_cfg.port="$port" env_cfg.mp_cfg.mp_start_method=forkserver

#srun python -m tools.train $1 --launcher="slurm" --resume --cfg-options env_cfg.dist_cfg.port=${port} "${@:2}" env_cfg.mp_cfg.mp_start_method=forkserver
