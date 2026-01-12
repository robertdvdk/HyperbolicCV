#!/bin/bash -l
#SBATCH --job-name=train_ourlinear_theirparams_warmup_eye05
#SBATCH --output=./%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --account=a132

set -e

conda activate HCNN

python code/classification/train.py -c classification/config/L-ResNet18_ourlinear_theirparams_warmup_eye05.txt