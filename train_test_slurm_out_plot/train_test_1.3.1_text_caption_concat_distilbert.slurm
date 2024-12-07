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

source ~/software/init-conda
conda activate pytorch

python main.py \
    --mode train \
    --modality text_caption_concat \
    --num_classes 4 \
    --text_model_name distilbert-base-uncased \
    --image_model_name resnet18 \
    
python main.py \
    --mode test \
    --modality text_caption_concat \
    --num_classes 4 \
    --text_model_name distilbert-base-uncased \
    --image_model_name resnet18 \
    --checkpoint checkpoints/best_model.pt