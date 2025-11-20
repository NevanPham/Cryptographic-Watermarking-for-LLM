#!/bin/bash

#SBATCH --job-name=gptoss20b_demo
#SBATCH --account=oz402
#SBATCH --partition=milan-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
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

MODEL_PATH="/fred/oz402/kpham-watermark/huggingface/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

cd /fred/oz402/kpham-watermark/crypto-watermark

python main.py generate \
    "The future of AI is" \
    --model gpt-oss-20b \
    --model-path "${MODEL_PATH}" \
    --max-new-tokens 300 \
    -o demonstration/gptoss20b_demo.txt

echo "DONE"
