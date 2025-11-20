#!/bin/bash

#SBATCH --job-name=download_20b
#SBATCH --account=oz402
#SBATCH --partition=milan-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
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
export NLTK_DATA=$HF_HOME

cd /fred/oz402/kpham-watermark/crypto-watermark

python - << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/fred/oz402/kpham-watermark/huggingface")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/fred/oz402/kpham-watermark/huggingface")
print("GPT-OSS-20B downloaded successfully.")
EOF
