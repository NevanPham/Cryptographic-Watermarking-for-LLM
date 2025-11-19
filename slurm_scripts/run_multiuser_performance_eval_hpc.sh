#!/bin/bash

#SBATCH --job-name=multiuser_perf_eval
#SBATCH --account=oz402
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/slurm-%j.out

# --- Environment Setup ---
echo "Setting up environment for multi-user performance evaluation..."
module purge
module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

source /fred/oz402/kpham-watermark/crypto-watermark/venv/bin/activate

export HF_HOME=/fred/oz402/kpham-watermark/huggingface
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1

echo "Starting multi-user performance evaluation..."
echo "================================================================================"

cd /fred/oz402/kpham-watermark/crypto-watermark   # FIXED

# Run performance evaluation
python helper_scripts/evaluate_multiuser_performance.py \
    --users-file assets/users.csv \
    --model gpt-oss-20b \
    --l-bits 10 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --prompt "The future of artificial intelligence is" \
    --user-id 0 \
    --output-dir evaluation/multiuser_performance

echo ""
echo "Performance evaluation completed!"
echo "Results saved to:"
echo "  - evaluation/multiuser_performance/"

# Send completion email
mail -s "HPC Job Complete: run_multiuser_performance_eval_hpc.sh (Job ID: $SLURM_JOB_ID)" 104772183@student.swin.edu.au << EOF
Your run_multiuser_performance_eval_hpc.sh job has finished!

Job ID: $SLURM_JOB_ID
Status: COMPLETED
Results location:
- /fred/oz402/kpham-watermark/crypto-watermark/evaluation/multiuser_performance/

Check results with:
scp -r kpham@ozstar.swin.edu.au:/fred/oz402/kpham-watermark/crypto-watermark/evaluation/multiuser_performance ~/Downloads/

EOF

