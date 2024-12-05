#!/bin/bash
#SBATCH --job-name=logs/bpe_new
#SBATCH --output=logs/bpe_new.out
#SBATCH --error=logs/bpe_new.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=aaditd@andrew.cmu.edu
#SBATCH -N 1
#SBATCH -p shire-general
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --mem=32G
#SBATCH --time=1-01:00:00

echo "LOADING THE ENVIRONMENT"
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate py31
echo "Starting"

# Your job commands go here
python epoch_default_finetune.py
echo "DONE!"
