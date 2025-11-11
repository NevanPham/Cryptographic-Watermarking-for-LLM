#!/bin/bash

#SBATCH --job-name=gpt20b-watermark-eval
#SBATCH --account=oz413               # Replace with your project account
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2            # Request 2 CPU cores
#SBATCH --mem=8G                     # Request 8 GB of system RAM
#SBATCH --time=6:00:00               # Set a 6-hour time limit
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

# --- Run the Python Script ---
# This is the main way it will be interacted with in the future
echo "Showing off basic generation capabilities..."

python main.py generate "The future of AI is" --model gpt-oss-20b -o "demonstration/output.txt" --key-file "demonstration/secret.key"
python main.py detect "demonstration/output.txt" --model gpt-oss-20b --key-file "demonstration/secret.key"

echo "Starting the evaluation job..."
# Run the full evaluation using the gpt-oss-20b model. This is for internal testing
# The script will automatically use all prompts in prompts.txt.
python main.py evaluate \
  --prompts-file "demonstration/prompts_demo.txt" \
  --model gpt-oss-20b \
  --delta "2.5" \
  --entropy-thresholds "3.0, 4.0" \
  --hashing-context "5" \
  --output-dir "evaluation_results/demonstration1" \
  --max-new-tokens 2048

echo "Job finished."

# sbatch slurm_scripts/run_gpt20b_eval.sh
# squeue -u rburke
# tail -f slurm_out/slurm-1234567.out  # Replace with your actual job ID