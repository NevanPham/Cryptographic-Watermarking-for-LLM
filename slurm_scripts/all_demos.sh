#!/bin/bash

#SBATCH --job-name=watermark_demos
#SBATCH --account=oz402
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/slurm-%j.out

# --- Environment Setup ---
echo "Setting up the environment..."
module purge
module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

source /home/kpham/crypto-watermark/venv/bin/activate

export HF_HOME=/home/kpham/huggingface
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1

echo "Starting comprehensive watermarking demonstrations..."
echo "================================================================================"

# Available models: gpt2, gpt-oss-20b, gpt-oss-120b
MODELS=("gpt2" "gpt-oss-20b")  # Skip 120b unless you have 80GB+ GPU
PROMPT="The future of AI is"

for MODEL in "${MODELS[@]}"; do
    echo "Testing with model: $MODEL"
    echo "------------------------------------------------------------------------"
    
    # 1. Zero-bit watermarking demo
    echo "[1/5] Zero-bit watermarking with $MODEL..."
    python main.py generate "$PROMPT" \
        --model $MODEL \
        --delta 2.5 \
        --entropy-threshold 4.0 \
        --max-new-tokens 512 \
        -o demonstration/zero_bit_${MODEL}.txt \
        --key-file demonstration/zero_bit_${MODEL}.key
    
    python main.py detect demonstration/zero_bit_${MODEL}.txt \
        --model $MODEL \
        --key-file demonstration/zero_bit_${MODEL}.key
    
    # 2. L-bit watermarking demo
    echo "[2/5] L-bit watermarking with $MODEL..."
    python main.py generate_lbit "$PROMPT" \
        --model $MODEL \
        --l-bits 8 \
        --message 01010101 \
        --delta 2.5 \
        --entropy-threshold 4.0 \
        --max-new-tokens 512 \
        -o demonstration/lbit_${MODEL}.txt \
        --key-file demonstration/lbit_${MODEL}.key
    
    python main.py detect_lbit demonstration/lbit_${MODEL}.txt \
        --model $MODEL \
        --l-bits 8 \
        --key-file demonstration/lbit_${MODEL}.key
    
    # 3. Naive multi-user (no groups)
    echo "[3/5] Naive multi-user with $MODEL..."
    python main_multiuser.py generate \
        --users-file assets/users.csv \
        --model $MODEL \
        --user-id 42 \
        --l-bits 10 \
        --min-distance 0 \
        --delta 2.5 \
        --entropy-threshold 4.0 \
        --max-new-tokens 512 \
        -o demonstration/naive_multiuser_${MODEL}.txt \
        --key-file demonstration/naive_multiuser_${MODEL}.key \
        "$PROMPT"
    
    python main_multiuser.py trace \
        --users-file assets/users.csv \
        --model $MODEL \
        --l-bits 10 \
        --min-distance 0 \
        --key-file demonstration/naive_multiuser_${MODEL}.key \
        demonstration/naive_multiuser_${MODEL}.txt
    
    # 4. Group-based multi-user (min-distance 3)
    echo "[4/5] Group-based multi-user (min-distance 3) with $MODEL..."
    python main_multiuser.py generate \
        --users-file assets/users.csv \
        --model $MODEL \
        --user-id 42 \
        --l-bits 10 \
        --min-distance 3 \
        --delta 2.5 \
        --entropy-threshold 4.0 \
        --max-new-tokens 512 \
        -o demonstration/group_multiuser_${MODEL}.txt \
        --key-file demonstration/group_multiuser_${MODEL}.key \
        "$PROMPT"
    
    python main_multiuser.py trace \
        --users-file assets/users.csv \
        --model $MODEL \
        --l-bits 10 \
        --min-distance 3 \
        --key-file demonstration/group_multiuser_${MODEL}.key \
        demonstration/group_multiuser_${MODEL}.txt
    
    # 5. Collusion scenario creation
    echo "[5/5] Creating collusion scenario with $MODEL..."
    python create_collusion_scenario.py \
        --users-file assets/users.csv \
        --model $MODEL \
        --l-bits 10 \
        --min-distance 3 \
        --num-users 3 \
        --max-new-tokens 300
    
    echo "Completed demos for $MODEL"
    echo ""
done

echo "All demonstrations completed successfully!"

# Send completion email
mail -s "HPC Job Complete: all_demos.sh (Job ID: $SLURM_JOB_ID)" 104772183@student.swin.edu.au << EOF
Your all_demos.sh job has finished!

Job ID: $SLURM_JOB_ID
Status: COMPLETED
Results location: /home/kpham/crypto-watermark/demonstration/

Check results with:
scp -r kpham@ozstar.swin.edu.au:/home/kpham/crypto-watermark/demonstration ~/Downloads/

EOF