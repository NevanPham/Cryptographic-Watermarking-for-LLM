#!/bin/bash

#SBATCH --job-name=hier_rewrite
#SBATCH --account=oz402
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/slurm-%j.out

module purge
module load gcc/14.2.0
module load python-scientific/3.10.8-foss-2022b
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

source /home/kpham/watermark-venv/bin/activate

export TRANSFORMERS_CACHE=/fred/oz402/kpham-watermark/huggingface
export HF_HOME=/fred/oz402/kpham-watermark/huggingface
export HF_DATASETS_CACHE=/fred/oz402/kpham-watermark/huggingface
export HF_HUB_CACHE=/fred/oz402/kpham-watermark/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NLTK_DATA=$HF_HOME

cd /fred/oz402/kpham-watermark/crypto-watermark

RUN_TAG=${RUN_TAG:-job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
echo "Using run tag: ${RUN_TAG}"

echo "Running rewrite attack evaluation for all 9 configurations..."
echo "L = 8 for all configurations"
echo "Deterministic rewrite using the same base model/tokenizer"
echo ""

run_eval () {
    local scheme=$1
    local group_bits=$2
    local user_bits=$3
    local label=$4

    echo ""
    echo "=========================================="
    echo "Configuration: ${label}"
    echo "=========================================="

    if [ "${scheme}" = "naive" ]; then
        python evaluation_scripts/evaluate_rewrite_attack.py \
            --scheme naive \
            --l-bits 8 \
            --prompts-file assets/prompts.txt \
            --num-prompts 200 \
            --users-file assets/users.csv \
            --model gpt2 \
            --delta 3.5 \
            --entropy-threshold 2.5 \
            --hashing-context 5 \
            --z-threshold 4.0 \
            --max-new-tokens 512 \
            --output-dir evaluation/rewrite_attack \
            --run-tag ${RUN_TAG}
    else
        python evaluation_scripts/evaluate_rewrite_attack.py \
            --scheme hierarchical \
            --group-bits "${group_bits}" \
            --user-bits "${user_bits}" \
            --l-bits 8 \
            --prompts-file assets/prompts.txt \
            --num-prompts 200 \
            --users-file assets/users.csv \
            --model gpt2 \
            --delta 3.5 \
            --entropy-threshold 2.5 \
            --hashing-context 5 \
            --z-threshold 4.0 \
            --max-new-tokens 512 \
            --output-dir evaluation/rewrite_attack \
            --run-tag ${RUN_TAG}
    fi
}

# Configuration 1: Naive (L=8, no hierarchy)
run_eval "naive" 0 0 "Naive (L=8)"

# Configuration 2: Hierarchical G=1, U=7
run_eval "hierarchical" 1 7 "Hierarchical G=1, U=7"

# Configuration 3: Hierarchical G=2, U=6
run_eval "hierarchical" 2 6 "Hierarchical G=2, U=6"

# Configuration 4: Hierarchical G=3, U=5
run_eval "hierarchical" 3 5 "Hierarchical G=3, U=5"

# Configuration 5: Hierarchical G=4, U=4
run_eval "hierarchical" 4 4 "Hierarchical G=4, U=4"

# Configuration 6: Hierarchical G=5, U=3
run_eval "hierarchical" 5 3 "Hierarchical G=5, U=3"

# Configuration 7: Hierarchical G=6, U=2
run_eval "hierarchical" 6 2 "Hierarchical G=6, U=2"

# Configuration 8: Hierarchical G=7, U=1
run_eval "hierarchical" 7 1 "Hierarchical G=7, U=1"

# Configuration 9: Group-only G=8, U=0
run_eval "hierarchical" 8 0 "Group-only G=8, U=0"

echo ""
echo "=========================================="
echo "All rewrite attack evaluations complete!"
echo "=========================================="

