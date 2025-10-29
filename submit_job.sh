#!/bin/bash

#SBATCH --job-name=deeplabv3_plus                 # Job name
#SBATCH --output=output/train_out_%j.txt          # Stdout log (%j = Job ID)
#SBATCH --error=output/train_err_%j.txt           # Stderr log (%j = Job ID)
#SBATCH --partition=stampede                        # Partition/queue
#SBATCH --cpus-per-task=8                         # CPU cores
#SBATCH --time=5:00:00                           # Max runtime HH:MM:SS

source ~/miniconda3/etc/profile.d/conda.sh

conda activate crafter_env

cd ~/puzzle-segmentation

# Run training
python main.py
