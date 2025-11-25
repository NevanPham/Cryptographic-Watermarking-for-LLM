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

# Hierarchical configurations: G=4,U=4; G=5,U=3; G=6,U=2; G=7,U=1
# Each with L=8 (G+U=8)

echo "Running hierarchical configurations..."

# G=4, U=4 → 8 groups, 16 users per group
python evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 4 \
    --user-bits 4 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance

# G=5, U=3 → 16 groups, 8 users per group
python evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 5 \
    --user-bits 3 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance

# G=6, U=2 → 32 groups, 4 users per group
python evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 6 \
    --user-bits 2 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance

# G=7, U=1 → 64 groups, 2 users per group
python evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 7 \
    --user-bits 1 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance

echo "Running naive configuration..."

# Naive scheme with L=8
python evaluation_scripts/compare_collusion_resistance.py \
    --scheme naive \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance

echo "All evaluations complete!"
