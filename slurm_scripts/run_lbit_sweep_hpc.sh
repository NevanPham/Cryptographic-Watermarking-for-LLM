#!/bin/bash

#SBATCH --job-name=lbit_sweep
#SBATCH --account=oz402
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/slurm-%j.out

echo "Setting up environment for L-bit sweep..."
module purge
module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

source /fred/oz402/kpham-watermark/crypto-watermark/venv/bin/activate
export HF_HOME=/fred/oz402/kpham-watermark/huggingface
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=0

cd /fred/oz402/kpham-watermark/crypto-watermark

echo "Starting L-bit parameter sweep..."
echo "==============================================================="

python helper_scripts/run_lbit_sweep.py \
    --prompts-file assets/prompts.txt \
    --model gpt-oss-20b \
    --l-values 6 8 10 12 14 16 18 20 22 24 26 28 30 \
    --delta 2.5 \
    --entropy-threshold 4.0 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 1024 \
    --max-prompts 50 \
    --output-dir evaluation/lbit_sweep_gpt20b

echo "L-bit sweep completed."
