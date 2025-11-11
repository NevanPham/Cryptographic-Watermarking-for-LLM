#!/bin/bash

#SBATCH --job-name=gpt20b-watermark-eval
#SBATCH --account=oz413               # Replace with your project account
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # Request 8 CPU cores
#SBATCH --mem=16G                     # Request 16 GB of system RAM
#SBATCH --time=06:00:00               # Set a 6-hour time limit
#SBATCH --gres=gpu:1                  # Request 1 GPU
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

# --- Run the visualise blocks ---
echo "Starting the visualise blocks job..."
cd $SLURM_SUBMIT_DIR

python "helper_scripts/visualise_lbit_blocks.py" \
 "demonstration/multiuser_output.txt" \
 --message "0011" --model gpt-oss-20b --entropy-threshold "4.0"

echo "Detection job finished."