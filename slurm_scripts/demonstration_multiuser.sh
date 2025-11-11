#!/bin/bash

#SBATCH --job-name=gpt20b-lbit-eval
#SBATCH --account=oz413          # Replace with your project account
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2        # Request 2 CPU cores
#SBATCH --mem=8G                 # Request 8 GB of system RAM
#SBATCH --time=6:00:00           # Set a 6-hour time limit
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --output=slurm_out/slurm-%j.out

# --- Environment Setup ---
echo "Setting up the environment..."

# Load the same modules
module purge
module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

# Activate your virtual environment
source /fred/oz413/rburke/Watermark-Project/venv/bin/activate

# Set location for downloaded huggingface model
export HF_HOME=/fred/oz413/rburke/huggingface
export HF_HUB_CACHE=$HF_HOME

# Force offline mode
export HF_HUB_OFFLINE=1

# --- Run the Python Script ---
# This script demonstrates L-bit message embedding and detection.
echo "Demonstrating multi user watermarking..."

# Generate text 
python main_multiuser.py generate "Write a story" --user-id 3 --l-bits 4 --model "gpt-oss-20b" --users-file "demonstration/DatabaseUser.csv" --entropy-threshold "4.0"

# Detect user from the generated text
# Use the generated files to find the original user
python main_multiuser.py trace "demonstration/multiuser_output.txt" --l-bits 4 --model "gpt-oss-20b" --users-file "demonstration/DatabaseUser.csv" --entropy-threshold "4.0"

echo "Job finished."

# sbatch slurm_scripts/run_gpt20b_eval.sh
# squeue -u rburke