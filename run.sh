#!/bin/sh

# SLURM options:
#SBATCH --partition=IPPMED-A40               # Choice of partition (mandatory)
#SBATCH --ntasks=1                            # Run a single task
#SBATCH --time=0-00:05:00                     # Duration of 5 minutes
#SBATCH --gpus=1                              # Using 2 GPUs

# Environment setup:
export fold=0
export MODEL="3DUnet"

# Activate Conda environment:
eval "$(conda shell.bash hook)"
conda activate myenv

# Run the Python script:
python3 -m src.train

