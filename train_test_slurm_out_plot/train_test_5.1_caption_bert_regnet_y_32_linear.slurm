#! /bin/bash
# ============================
#SBATCH --job-name=ENEL645-Project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=100G
# ==================================

source ~/software/init-conda
conda activate pytorch

python main.py \
    --mode train \
    --modality caption_image \
    --num_classes 4 \
    --text_model_name bert-base-uncased \
    --image_model_name regnet_y_32_linear \

    
python main.py \
    --mode test \
    --modality caption_image \
    --num_classes 4 \
    --text_model_name bert-base-uncased \
    --image_model_name regnet_y_32_linear \
    --checkpoint checkpoints/best_model.pt