#!/bin/sh

# SLURM options:
#SBATCH --partition=IPPMED-A40               # Choice of partition (mandatory)
#SBATCH --ntasks=1                            # Run a single task
#SBATCH --time=0-02:00:00                     # Duration of 5 minutes
#SBATCH --gpus=2                              # Using 2 GPUs

# Environment setup:
export fold=0
export MODEL="3DUnet"

# Activate Conda environment:
eval "$(conda shell.bash hook)"
conda activate myenv

# Run the Python script:
python3 -m src.train

