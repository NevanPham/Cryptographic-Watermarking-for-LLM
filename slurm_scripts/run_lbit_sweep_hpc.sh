#!/bin/bash

#SBATCH --job-name=lbit_sweep
#SBATCH --account=oz411
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/slurm-%j.out

module --force purge
module load gcc/12.2.0
module load python/3.10.8
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

source /fred/oz411/kpham/crypto-watermark/venv/bin/activate

export TRANSFORMERS_CACHE=/fred/oz411/kpham/huggingface
export HF_HOME=/fred/oz411/kpham/huggingface
export HF_DATASETS_CACHE=/fred/oz411/kpham/huggingface
export HF_HUB_CACHE=/fred/oz411/kpham/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NLTK_DATA=$HF_HOME

cd /fred/oz411/kpham/crypto-watermark

python evaluation_scripts/run_lbit_sweep.py \
    --prompts-file assets/prompts.txt \
    --max-prompts 300 \
    --model gpt2 \
    --min-l 4 \
    --max-l 30 \
    --delta 3.5 \
    --entropy-threshold 2.5 \   
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/lbit_sweep
