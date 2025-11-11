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
echo "Demonstrating L-bit watermarking..."

# Generate text with the embedded L-bit watermark
python main.py generate_lbit "The future of AI is" \
    --message "01011101" \
    --l-bits 8 \
    --model gpt-oss-20b \
    --max-new-tokens 4096 \
    --output-file "demonstration/lbit_output.txt" \
    --key-file "demonstration/secret.key"

# Detect the L-bit watermark from the generated text
python main.py detect_lbit "demonstration/lbit_output.txt" \
    --l-bits 8 \
    --model gpt-oss-20b \
    --key-file "demonstration/secret.key"

echo "Job finished."

# sbatch slurm_scripts/run_gpt20b_eval.sh
# squeue -u rburke