#! /bin/bash
# ============================
#SBATCH --job-name=ENEL645-Project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=100G
# ==================================

eval "$(~/miniforge3/bin/conda shell.bash hook)"

conda init
conda activate sam_env

python checkmodelsize.py