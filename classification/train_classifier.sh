#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=resnetprocessed
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=32000M

source $HOME/melanomadetseg/bin/activate

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

wandb login a183cb44e1857d72ebd9118c7ffde2ac50957d5e
#wandb agent treye-again/moley-business/n4p9vk87 # sweep command

# Execute program 
srun python train_classifier.py
