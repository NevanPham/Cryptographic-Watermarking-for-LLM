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

# --- Environment Setup ---
echo "Setting up environment for collusion resistance evaluation..."
module purge
module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

source /home/kpham/crypto-watermark/venv/bin/activate

export HF_HOME=/home/kpham/huggingface
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1

echo "Starting collusion resistance evaluation..."
echo "================================================================================"

# Test different numbers of colluders
for NUM_COLLUDERS in 2 3 4; do
    echo "Testing collusion resistance with $NUM_COLLUDERS colluders..."
    
    python compare_collusion_resistance.py \
        --prompts-file assets/prompts.txt \
        --users-file assets/users.csv \
        --model gpt-oss-20b \
        --num-colluders $NUM_COLLUDERS \
        --l-bits 10 \
        --delta 2.5 \
        --entropy-threshold 4.0 \
        --max-new-tokens 800 \
        --max-prompts 20 \
        --deletion-percentage 0.2 \
        --output-dir evaluation/collusion_resistance_${NUM_COLLUDERS}users
    
    echo "Completed evaluation for $NUM_COLLUDERS colluders"
    echo ""
done

echo "All collusion evaluations completed!"
echo "Results saved to:"
echo "  - evaluation/collusion_resistance_2users/"
echo "  - evaluation/collusion_resistance_3users/"
echo "  - evaluation/collusion_resistance_4users/"

# Send completion email
mail -s "HPC Job Complete: run_collusion_eval_hpc.sh (Job ID: $SLURM_JOB_ID)" 104772183@student.swin.edu.au << EOF
Your run_collusion_eval_hpc.sh job has finished!

Job ID: $SLURM_JOB_ID
Status: COMPLETED
Results locations:
- /home/kpham/crypto-watermark/evaluation/collusion_resistance_2users/
- /home/kpham/crypto-watermark/evaluation/collusion_resistance_3users/
- /home/kpham/crypto-watermark/evaluation/collusion_resistance_4users/

Check results with:
scp -r kpham@ozstar.swin.edu.au:/home/kpham/crypto-watermark/evaluation/collusion_resistance_* ~/Downloads/

EOF