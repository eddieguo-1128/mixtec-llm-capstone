#!/bin/bash
#SBATCH --job-name=eval_bpe_v2
#SBATCH --output=eval_bpe_v2.out
#SBATCH --error=eval_bpe_v2.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=aaditd@andrew.cmu.edu
#SBATCH -N 1
#SBATCH -p shire-general
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --mem=32G
#SBATCH --time=01:00:00

echo "LOADING THE ENVIRONMENT"
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate py31
echo "Starting"

# Your job commands go here
python eval.py
echo "DONE!"
