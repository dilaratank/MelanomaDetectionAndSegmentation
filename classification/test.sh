#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=sweep
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=32000M
#SBATCH --exclusive

source $HOME/melanomadetseg/bin/activate

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

#wandb login a183cb44e1857d72ebd9118c7ffde2ac50957d5e
#wandb agent treye-again/moley-business/n4p9vk87

# Execute program located in $HOME and redirect outputs to the log file
 srun python test.py --batch_size=64 --optimizer_lr=0.02 --freeze_until_layer=5 --checkpoint_folder_path=treye-again/moley-business/model-fyesp9ra:best
