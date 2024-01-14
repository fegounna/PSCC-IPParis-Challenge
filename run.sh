#!/bin/bash

# SLURM options:
#SBATCH --partition=IPPMED-A40               # Choice of partition (mandatory)
#SBATCH --ntasks=1                            # Run a single task
#SBATCH --time=0-02:00:00                     # Duration of 5 minutes
#SBATCH --gpus=2                              # Using 2 GPUs

# Environment setup:
export fold=0
export MODEL="3DUnet"

# Activate Conda environment:
/home/ext-6401/anaconda3/bin/conda activate myenv
/home/ext-6401/anaconda3/bin/conda install pandas -y

# Run the Python script:
python3 -m src.predict

