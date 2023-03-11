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

module load anaconda

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

conda activate toy_classification_env # open the Python environment

pip install -r requirements.txt # install Python dependencies
pip install typing-extensions --upgrade

# runs your code
# TODO remove small_subset
srun python classification.py --experiment "BERT-base-uncased" --device cuda --model "BERT-base-uncased" --batch_size "64" --small_subset True
srun python classification.py --experiment "BERT-large-uncased" --device cuda --model "BERT-large-uncased" --batch_size "6" --small_subset True
srun python classification.py --experiment "BERT-base-cased" --device cuda --model "BERT-base-cased" --batch_size "64" --small_subset True
srun python classification.py --experiment "BERT-large-cased" --device cuda --model "BERT-large-cased" --batch_size "6" --small_subset True
srun python classification.py --experiment "RoBERTa-base" --device cuda --model "RoBERTa-base" --batch_size "64" --small_subset True
srun python classification.py --experiment "RoBERTa-large" --device cuda --model "RoBERTa-large" --batch_size "6" --small_subset True
