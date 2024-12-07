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

python main.py \
    --mode train \
    --modality text_only \
    --num_classes 4 \
    --text_model_name distilbert-base-uncased \
    --image_model_name resnet18 \
    --tokenizer distilbert-base-uncased
    
python main.py \
    --mode test \
    --modality text_only \
    --num_classes 4 \
    --text_model_name distilbert-base-uncased \
    --image_model_name resnet18 \
    --tokenizer distilbert-base-uncased \
    --checkpoint checkpoints/best_model.pt