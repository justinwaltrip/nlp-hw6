#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-2

module load anaconda

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

conda activate toy_classification_env # open the Python environment

pip install -r requirements.txt # install Python dependencies
pip install typing-extensions --upgrade

# runs your code
models=("distilbert-base-uncased" "BERT-base-uncased" "BERT-base-cased")
batch_sizes=("64" "32" "32")
srun python classification.py --experiment ${models[$SLURM_ARRAY_TASK_ID]} --device cuda --model ${models[$SLURM_ARRAY_TASK_ID]} --batch_size ${batch_sizes[$SLURM_ARRAY_TASK_ID]} 
