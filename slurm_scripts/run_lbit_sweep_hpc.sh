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

module purge
module load gcc/14.2.0
module load python-scientific/3.10.8-foss-2022b
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

source /fred/oz402/kpham-watermark/crypto-watermark/venv/bin/activate

export TRANSFORMERS_CACHE=/fred/oz402/kpham-watermark/huggingface
export HF_HOME=/fred/oz402/kpham-watermark/huggingface
export HF_DATASETS_CACHE=/fred/oz402/kpham-watermark/huggingface
export HF_HUB_CACHE=/fred/oz402/kpham-watermark/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NLTK_DATA=$HF_HOME

cd /fred/oz402/kpham-watermark/crypto-watermark

python helper_scripts/run_lbit_sweep.py \
    --prompts-file assets/prompts.txt \
    --max-prompts 300 \
    --model /fred/oz402/kpham-watermark/huggingface/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e \
    --min-l 4 \
    --max-l 30 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/lbit_sweep
