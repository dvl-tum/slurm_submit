#!/bin/bash
#SBATCH --job-name=slurm_submit
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=20GB
#SBATCH --output=logs/slurm-%j
#SBATCH --time=0-01:00:00

python src/train.py --output_dir logs/mnist_example