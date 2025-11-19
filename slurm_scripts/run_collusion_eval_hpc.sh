#!/bin/bash

#SBATCH --job-name=collusion_eval
#SBATCH --account=oz402
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/slurm-%j.out

echo "Setting up environment for collusion evaluation..."
module purge
module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

source /fred/oz402/kpham-watermark/crypto-watermark/venv/bin/activate

export HF_HOME=/fred/oz402/kpham-watermark/huggingface
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export NLTK_DATA=$HF_HOME

cd /fred/oz402/kpham-watermark/crypto-watermark

echo "Starting collusion resistance evaluation..."
echo "==============================================================="

for NUM_COLLUDERS in 2 3; do
    echo "Running collusion test with $NUM_COLLUDERS colluders..."

    python helper_scripts/compare_collusion_resistance.py \
        --prompts-file assets/prompts.txt \
        --users-file assets/users.csv \
        --model gpt2 \
        --num-colluders $NUM_COLLUDERS \
        --l-bits 10 \
        --delta 2.5 \
        --entropy-threshold 4.0 \
        --hashing-context 5 \
        --z-threshold 4.0 \
        --max-new-tokens 400 \
        --max-prompts 0 \
        --deletion-percentage 0.2 \
        --output-dir evaluation/collusion_resistance_gpt2
done

echo "All collusion evaluations completed."
echo "Results under: evaluation/collusion_resistance_gpt2_*"
