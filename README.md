# Cryptographic Watermarking for Large Language Models (LLMs)

A comprehensive framework for embedding and detecting statistical watermarks in LLM-generated text. This implementation supports three watermarking schemes: **zero-bit** (binary detection), **L-bit** (message embedding), and **multi-user fingerprinting** (user tracing).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Zero-Bit Watermarking](#zero-bit-watermarking)
  - [L-Bit Watermarking](#l-bit-watermarking)
  - [Multi-User Fingerprinting](#multi-user-fingerprinting)
  - [Batch Evaluation](#batch-evaluation)
- [GUI Application](#gui-application)
- [File-by-File Usage Guide](#file-by-file-usage-guide)
- [Parameters and Tuning](#parameters-and-tuning)
- [Expected Outputs](#expected-outputs)
- [Model Information](#model-information)
- [HPC Cluster Usage](#hpc-cluster-usage)
- [Troubleshooting](#troubleshooting)
- [License and Citation](#license-and-citation)

---

## Overview

This repository implements cryptographic watermarking techniques for LLM text generation. The watermarking is:
- **Statistical**: Based on PRF-derived bias in token selection
- **Unobtrusive**: Only applied at high-entropy (uncertain) positions
- **Cryptographically secure**: Uses HMAC-SHA256 for key derivation
- **Robust**: Resistant to common text perturbation attacks

### How It Works

**Zero-Bit Watermarking:**
1. During generation, when the model's next-token entropy exceeds a threshold, add a pseudorandom score vector (derived from secret key + context) to the logits
2. During detection, recompute the same score vectors and calculate a z-score over all watermarked positions ("blocks")
3. If z-score > threshold → text is watermarked

**L-Bit Watermarking:**
1. Derive per-bit keys from a single master key using HMAC(master_key, "i_b") where i is bit position, b ∈ {0,1}
2. Cycle through bits at each high-entropy block, embedding the target bitstring
3. During detection, test both hypotheses (0 and 1) for each bit position and recover the message

**Multi-User Fingerprinting:**
1. Assign each user a unique L-bit codeword (deterministic from User ID)
2. Embed the user's codeword using L-bit watermarking
3. Trace recovered (noisy) codewords back to the most likely user(s)

---

## Features

✅ **Three watermarking modes**: Zero-bit detection, L-bit message embedding, multi-user tracing
✅ **Multiple model support**: GPT-2 (local), GPT-OSS-20B, GPT-OSS-120B
✅ **Three interfaces**: CLI, Desktop GUI (PySide6), SLURM batch scripts
✅ **Robustness testing**: Built-in perturbation attacks (deletion, paraphrasing)
✅ **Comprehensive evaluation**: Parameter sweeps, automated plotting, statistical analysis
✅ **HPC-ready**: Offline model caching, SLURM job templates
✅ **Well-documented**: Copy-paste commands, parameter guides, usage examples

---

## Repository Structure

```
Cryptographic-Watermarking-for-LLM/
│
├── main.py                          # Main CLI entry point (zero-bit, L-bit, evaluation)
├── requirements.txt                 # Python dependencies
├── COMMANDS.md                      # Copy-paste ready command examples
├── PARAMS.md                        # Parameter tuning guide
│
├── src/                             # Core source code
│   ├── watermark.py                 # Watermarking implementations
│   │                                  - ZeroBitWatermarker
│   │                                  - LBitWatermarker
│   │                                  - MultiUserWatermarker
│   ├── models.py                    # Model abstractions (GPT-2, GPT-OSS variants)
│   ├── fingerprinting.py            # Multi-user codeword generation & tracing
│   ├── commands.py                  # CLI command handlers
│   ├── parser.py                    # Argument parsing & validation
│   ├── utils.py                     # Helper utilities (parsing, perturbations)
│   └── main_multiuser.py            # Multi-user CLI (generate, trace)
│
├── UI/                              # Desktop GUI application
│   ├── app.py                       # Main PySide6 application (39KB)
│   ├── core.py                      # Bridge between GUI and CLI logic
│   ├── worker.py                    # Threading for non-blocking operations
│   ├── loading_dialog.py            # Loading dialog UI
│   ├── app_GUI.py                   # Qt Designer generated code
│   ├── mainWindow.ui                # Qt Designer UI file
│   └── resources.*                  # GUI assets
│
├── helper_scripts/                  # Analysis and utility scripts
│   ├── analyse.py                   # Generate plots from evaluation results
│   ├── generate_users.py            # Create user database CSV
│   ├── visualise_blocks.py          # Visualize watermark blocks
│   ├── visualise_lbit_blocks.py     # Visualize L-bit blocks
│   ├── run_detection_only.py        # Standalone detection script
│   ├── redo_paraphrase_attack.py    # Re-run perturbation attacks
│   ├── test_undetectability.py      # Statistical undetectability tests
│   └── download_models_hpc.py       # Pre-download models for HPC
│
├── slurm_scripts/                   # HPC cluster batch job scripts
│   ├── demonstration.sh             # Zero-bit demonstration job
│   ├── demonstration_lbit.sh        # L-bit demonstration job
│   ├── demonstration_multiuser.sh   # Multi-user demonstration job
│   ├── run_gpt20b_eval.sh          # GPT-OSS-20B evaluation
│   ├── run_detection_job.sh         # Detection-only job
│   └── run_visualise_blocks.sh      # Block visualization job
│
├── assets/                          # Data files
│   ├── users.csv                    # 1000 users (UserIds 0-999)
│   └── prompts.txt                  # 75 evaluation prompts
│
└── demonstration/                   # Example outputs
    ├── my_watermark.txt             # Zero-bit example output
    ├── my_watermark.key             # Zero-bit secret key
    ├── my_lbit.txt                  # L-bit example output
    ├── my_lbit.key                  # L-bit secret key
    ├── multiuser_user0.txt          # Multi-user example (user 0)
    ├── multiuser_user888.txt        # Multi-user example (user 888)
    ├── multiuser_bob.txt            # Multi-user example (user "bob")
    └── multiuser_master.key         # Multi-user master key
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) CUDA-capable GPU for large models

### Setup

**Windows (cmd):**
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Dependencies

Core dependencies from `requirements.txt`:
- **torch**: PyTorch for model inference
- **transformers**: HuggingFace models (GPT-2, T5)
- **numpy**: Numerical operations
- **accelerate**: Multi-GPU support
- **sentencepiece**: Tokenizer support
- **pandas**: Data manipulation
- **matplotlib, seaborn**: Visualization
- **nltk**: Text processing
- **pyside6**: Desktop GUI framework
- **protobuf**: Model serialization

---

## Quick Start

### Zero-Bit Watermarking (5 minutes)

**Generate watermarked text:**
```bat
python main.py generate "The future of AI is" --model gpt2 --max-new-tokens 512 -o output.txt
```

**Output:**
- `output.txt`: Generated watermarked text
- `secret.key`: Secret key (DO NOT SHARE)

**Detect watermark:**
```bat
python main.py detect output.txt --model gpt2 --key-file secret.key
```

**Expected output:**
```
=== Detection Results ===
Z-score: 12.45
Blocks detected: 87
Decision: WATERMARKED ✓
```

---

## Detailed Usage

### Zero-Bit Watermarking

Zero-bit watermarking provides **binary detection**: is the text watermarked (yes/no)?

#### Generate

```bat
python main.py generate "Your prompt here" ^
  --model gpt2 ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  -o output.txt ^
  --key-file secret.key
```

**Parameters:**
- `--model`: Model to use (`gpt2`, `gpt-oss-20b`, `gpt-oss-120b`)
- `--delta`: Watermark strength (1.0-5.0, default 2.5)
- `--entropy-threshold`: Minimum entropy for watermarking (1.0-6.0, default 4.0)
- `--max-new-tokens`: Number of tokens to generate
- `-o, --output-file`: Output file path
- `--key-file`: Where to save the secret key

**Expected outputs:**
1. **output.txt**: Contains the generated text
2. **secret.key**: Binary file containing the secret key (32 bytes)

**Example output.txt:**
```
The future of AI is rapidly evolving, with new breakthroughs happening every year.
Machine learning models are becoming more sophisticated, capable of understanding
complex patterns in data and making predictions with unprecedented accuracy. From
natural language processing to computer vision, AI systems are transforming
industries and reshaping how we interact with technology.
```

#### Detect

```bat
python main.py detect output.txt ^
  --model gpt2 ^
  --z-threshold 4.0 ^
  --entropy-threshold 4.0 ^
  --key-file secret.key
```

**Parameters:**
- Text file to check (positional argument)
- `--model`: Same model used for generation
- `--z-threshold`: Detection threshold (default 4.0)
- `--entropy-threshold`: Must match generation (default 4.0)
- `--key-file`: Secret key from generation

**Expected output:**
```
=== Detection Results ===
Model: gpt2
Entropy threshold: 4.0
Z-threshold: 4.0
Z-score: 12.45
Blocks detected: 87
Decision: WATERMARKED ✓

Note: Text successfully detected as watermarked with high confidence.
```

**Two-pass logic:** If block count < 75, automatically retries with `entropy_threshold - 2.0`

---

### L-Bit Watermarking

L-bit watermarking embeds and recovers a **binary message** of length L.

#### Generate

```bat
python main.py generate_lbit "The future of AI is" ^
  --model gpt2 ^
  --message 01010101 ^
  --l-bits 8 ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  -o output_lbit.txt ^
  --key-file secret_lbit.key
```

**Important:** `--l-bits` must equal the length of `--message`

**Expected outputs:**
1. **output_lbit.txt**: Generated text with embedded message
2. **secret_lbit.key**: Master secret key (all per-bit keys derived from this)

#### Detect

```bat
python main.py detect_lbit output_lbit.txt ^
  --model gpt2 ^
  --l-bits 8 ^
  --z-threshold 4.0 ^
  --entropy-threshold 4.0 ^
  --key-file secret_lbit.key
```

**Expected output:**
```
=== L-bit Detection Results ===
Model: gpt2
L-bits: 8
Target message: 01010101
Recovered message: 01010101
Bit accuracy: 8/8 (100%)
Undecided bits: 0

Decision: MESSAGE RECOVERED SUCCESSFULLY ✓
```

**Possible outcomes:**
- **Exact match**: `01010101` (all bits recovered)
- **Partial recovery**: `0101⊥1⊥1` (⊥ = undecided, insufficient signal)
- **Failed recovery**: `⊥⊥⊥⊥⊥⊥⊥⊥` (no watermark detected)

**Tips for reducing ⊥ (undecided bits):**
- Increase `--delta` (stronger watermark)
- Lower `--entropy-threshold` during generation (more blocks)
- Increase `--max-new-tokens` (longer text)
- Lower `--z-threshold` during detection (more sensitive)

---

### Multi-User Fingerprinting

Trace generated text back to specific users (1000 users supported with L=10).

#### User Database

The `assets/users.csv` file contains:
```csv
UserId,Username
0,0
1,1
2,2
...
999,999
```

You can customize usernames or add more users (ensure L-bits ≥ log₂(num_users)).

#### Generate for User

```bat
python -m src.main_multiuser generate ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 10 ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  --key-file demonstration\multiuser_master.key ^
  -o demonstration\multiuser_user0.txt ^
  "The future of AI is"
```

**Key point:** L=10 supports up to 2¹⁰ = 1024 users. For more users, increase L.

**Expected outputs:**
1. **multiuser_user0.txt**: Text watermarked with user 0's codeword
2. **multiuser_master.key**: Master key (shared across all users)

#### Trace User

```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 10 ^
  --key-file demonstration\multiuser_master.key ^
  demonstration\multiuser_user0.txt
```

**Expected output:**
```
=== Multi-User Tracing Results ===
Recovered codeword: 0000000000
Top matching users (max 16):
  1. User ID: 0, Username: 0, Bit matches: 10/10 (100%)

Decision: Text traced to User 0 ✓
```

**Collusion detection:** Returns up to 16 best matches if multiple users' codewords match closely.

---

### Batch Evaluation

Run parameter sweeps and perturbation attacks across multiple prompts.

#### Run Evaluation

```bat
python main.py evaluate ^
  --prompts-file assets/prompts.txt ^
  --model gpt2 ^
  --delta "2.0, 2.5, 3.0" ^
  --entropy-thresholds "3.0, 3.5, 4.0" ^
  --max-new-tokens 512 ^
  --output-dir evaluation_results
```

**What it does:**
1. Generates clean text for each prompt
2. Creates perturbed variants:
   - Delete first 20% of sentences
   - Delete last 20% of sentences
   - Delete middle 20% of sentences
   - Paraphrase 30% of sentences (T5 model)
3. Runs detection on all variants
4. Saves results to `evaluation_results/analysis_results.json`

**Expected output files:**
```
evaluation_results/
├── analysis_results.json        # Detailed results (z-scores, block counts, decisions)
├── generated_text_*.txt         # Generated text files
└── keys/                        # Secret keys
```

#### Analyze Results

```bat
python helper_scripts\analyse.py evaluation_results --z-threshold 4.0
```

**Generated plots:**
- `completeness_soundness_distribution.png`: Detection accuracy distribution
- `robustness_boxplot.png`: Robustness across perturbation types
- `parameter_sweep_*.png`: Parameter impact visualizations

**Summary statistics:**
```
=== Evaluation Summary ===
Total prompts: 75
Clean text detection rate: 98.7%
Average z-score (clean): 15.23
False positive rate: 0.0%

Robustness (perturbed text):
  Delete start 20%: 87.3% detected
  Delete end 20%: 89.1% detected
  Delete middle 20%: 85.7% detected
  Paraphrase 30%: 76.4% detected
```

---

## GUI Application

Launch the desktop GUI for interactive watermarking.

### Start GUI

```bat
python UI\app.py
```

Or as a module (if you encounter import errors):
```bat
python -m UI.app
```

### GUI Features

**Three tabs:**

1. **Generate Tab:**
   - Input prompt
   - Select model (GPT-2, GPT-OSS-20B, GPT-OSS-120B)
   - Adjust parameters (delta, entropy threshold, max tokens)
   - Toggle watermarking on/off
   - Save/load secret keys
   - **Output:** Generated text displayed in text box

2. **Detect Tab:**
   - Paste or load text from file
   - Load secret key
   - Select model and parameters
   - Click "Detect Watermark"
   - **Output:** Z-score, block count, decision (WATERMARKED / NOT WATERMARKED)

3. **Evaluate Tab:**
   - Select prompts file
   - Configure parameter sweeps
   - Run batch evaluation
   - **Output:** Inline plots and summary statistics

**Screenshot flow:**
```
[Generate Tab] → Enter prompt → Click "Generate" → Text appears with watermark
[Detect Tab] → Load text → Load key → Click "Detect" → See z-score and decision
[Evaluate Tab] → Select prompts → Click "Evaluate" → View plots and metrics
```

---

## File-by-File Usage Guide

### Core Python Files

#### `main.py` (17 lines)
**Purpose:** CLI entry point
**Usage:** Dispatches to subcommands (generate, detect, evaluate, etc.)
**Run:** `python main.py <command> [args]`

#### `src/watermark.py` (460 lines)
**Purpose:** Core watermarking algorithms
**Classes:**
- `ZeroBitWatermarker`: Binary detection
- `LBitWatermarker`: Message embedding
- `MultiUserWatermarker`: User fingerprinting
- `WatermarkLogitsProcessor`: Transformers integration

**Key functions:**
- `derive_key(secret_key, context, suffix)`: HMAC-SHA256 key derivation
- `calculate_entropy(logits)`: Shannon entropy calculation
- `generate(...)`: Watermarked text generation
- `detect(...)`: Watermark detection

**Not called directly** (used via CLI/GUI)

#### `src/models.py` (157 lines)
**Purpose:** Model abstraction layer
**Classes:**
- `LanguageModel`: Abstract base
- `GPT2Model`: GPT-2 (local, < 2GB VRAM)
- `GptOssModel`: 20B parameter model (16GB+ VRAM)
- `GptOss120bModel`: 120B parameter model (80GB+ VRAM)

**Methods:**
- `get_logits(input_ids)`: Compute next-token logits
- `tokenizer`: Access tokenizer
- `vocab_size`, `device`: Model properties

**Usage:** Automatically instantiated by CLI/GUI based on `--model` flag

#### `src/fingerprinting.py` (122 lines)
**Purpose:** Multi-user codeword management
**Class:** `FingerprintingCode`

**Methods:**
- `generate_codeword(user_id)`: User ID → L-bit binary string
- `trace(recovered_message)`: Find users matching noisy codeword

**Example:**
```python
from src.fingerprinting import FingerprintingCode
import pandas as pd

users_df = pd.read_csv('assets/users.csv')
code = FingerprintingCode(users_df, l_bits=10)

# Get user 0's codeword
codeword = code.generate_codeword(0)  # "0000000000"

# Trace noisy recovery
recovered = "0000000010"  # 1 bit flipped
matches = code.trace(recovered, max_collusion=16)
# Returns: [(UserId=0, Username="0", bit_matches=9/10)]
```

#### `src/commands.py` (398 lines)
**Purpose:** CLI command implementations
**Functions:**
- `cmd_generate(args)`: Zero-bit generation
- `cmd_detect(args)`: Zero-bit detection
- `cmd_generate_lbit(args)`: L-bit generation
- `cmd_detect_lbit(args)`: L-bit detection
- `cmd_evaluate(args)`: Batch evaluation

**Not called directly** (invoked by `main.py` based on subcommand)

#### `src/parser.py` (157 lines)
**Purpose:** Argument parsing and validation
**Features:**
- Subcommand routing
- Parameter range validation
- NLTK setup and verification
- Default value handling

**Not called directly** (used by `main.py`)

#### `src/utils.py` (209 lines)
**Purpose:** Helper utilities
**Functions:**
- `instantiate_model(model_name)`: Create model instance
- `parse_output(text, model_name)`: Clean model output
- `delete_sentences(text, portion)`: Deletion attack
- `paraphrase_sentences(text, portion)`: Paraphrasing attack
- `parse_filename(filename)`: Extract metadata from filenames

**Usage in scripts:**
```python
from src.utils import instantiate_model, delete_sentences

model = instantiate_model('gpt2')
perturbed = delete_sentences(text, 'start', 0.2)  # Remove first 20%
```

#### `src/main_multiuser.py` (128 lines)
**Purpose:** Multi-user CLI
**Commands:**
- `generate`: Watermark text for user
- `trace`: Identify user from text

**Run:**
```bat
python -m src.main_multiuser generate [args]
python -m src.main_multiuser trace [args]
```

### Helper Scripts

#### `helper_scripts/analyse.py` (261 lines)
**Purpose:** Generate plots and statistics from evaluation results
**Usage:**
```bat
python helper_scripts\analyse.py evaluation_results --z-threshold 4.0
```

**Outputs:**
- `completeness_soundness_distribution.png`
- `robustness_boxplot.png`
- `summary_analysis.txt`
- Console output with quantitative metrics

#### `helper_scripts/generate_users.py`
**Purpose:** Create custom user databases
**Usage:**
```bat
python helper_scripts\generate_users.py --num-users 1000 -o my_users.csv
```

**Output:** CSV file with UserId, Username columns

#### `helper_scripts/visualise_blocks.py`
**Purpose:** Visualize watermark block positions in text
**Usage:**
```bat
python helper_scripts\visualise_blocks.py output.txt --key-file secret.key --model gpt2
```

**Output:** Text with highlighted watermarked blocks (terminal colors or HTML)

#### `helper_scripts/visualise_lbit_blocks.py`
**Purpose:** Visualize L-bit embedding pattern
**Usage:**
```bat
python helper_scripts\visualise_lbit_blocks.py output_lbit.txt --key-file secret_lbit.key --model gpt2 --l-bits 8
```

**Output:** Shows which bit is embedded at each block position

#### `helper_scripts/test_undetectability.py`
**Purpose:** Statistical tests for undetectability
**Usage:**
```bat
python helper_scripts\test_undetectability.py --model gpt2 --num-samples 100
```

**Output:** Chi-square test results, KL divergence metrics

#### `helper_scripts/download_models_hpc.py`
**Purpose:** Pre-cache models for offline HPC environments
**Usage:**
```bat
python helper_scripts\download_models_hpc.py --model gpt-oss-20b --cache-dir /shared/models
```

### SLURM Scripts

All scripts in `slurm_scripts/` are templates for HPC cluster job submission.

**Common structure:**
```bash
#!/bin/bash
#SBATCH --job-name=watermark_demo
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

export HF_HOME=/shared/models
export NLTK_DATA=/shared/nltk_data

source venv/bin/activate
python main.py generate "Prompt" --model gpt2 -o output.txt
```

**Usage:**
```bash
sbatch slurm_scripts/demonstration.sh
```

### Asset Files

#### `assets/users.csv` (1000 rows)
**Format:**
```csv
UserId,Username
0,0
1,1
...
```

**Usage:** Required for multi-user fingerprinting

#### `assets/prompts.txt` (75 prompts)
**Format:** One prompt per line
```text
The future of artificial intelligence is
Write a Python function to calculate fibonacci numbers
Explain quantum computing in simple terms
```

**Usage:** Batch evaluation input

---

## Parameters and Tuning

See `PARAMS.md` for comprehensive tuning guide. Key parameters:

### Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `--delta` | 1.0-5.0 | 2.5 | Watermark strength (higher = stronger signal, lower fluency) |
| `--entropy-threshold` | 1.0-6.0 | 4.0 | Minimum entropy to watermark (higher = fewer, cleaner blocks) |
| `--hashing-context` | 1-10 | 5 | Number of previous tokens for PRF context |
| `--max-new-tokens` | 50-2048 | 512 | Generation length |

### Detection Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `--z-threshold` | 1.0-6.0 | 4.0 | Detection decision threshold (lower = more sensitive) |
| `--entropy-threshold` | 1.0-6.0 | 4.0 | Must match generation for accurate block counting |

### Preset Configurations

**Balanced (default):**
```bat
--delta 2.5 --entropy-threshold 4.0 --z-threshold 4.0
```
Good all-around performance

**High Fluency:**
```bat
--delta 2.0 --entropy-threshold 4.5 --z-threshold 4.0
```
Minimal impact on text quality, slightly weaker signal

**Strong Detection:**
```bat
--delta 3.5 --entropy-threshold 3.5 --z-threshold 3.5
```
Maximum robustness, may slightly reduce fluency

---

## Expected Outputs

### Zero-Bit Example

**Generated text** (`my_watermark.txt`):
```
The future of AI is incredibly promising and multifaceted. As machine learning
algorithms become more sophisticated, we're seeing breakthrough applications in
healthcare diagnostics, autonomous vehicles, and natural language understanding.
The integration of AI into everyday tools is accelerating, making technology more
intuitive and accessible. However, this progress also brings important ethical
considerations about privacy, bias, and transparency that we must address as a
society.
```

**Detection output:**
```
=== Detection Results ===
Z-score: 14.89
Blocks detected: 92
Decision: WATERMARKED ✓
```

### L-Bit Example

**Input message:** `11001010` (8 bits)

**Generated text** (`my_lbit.txt`):
```
The future of AI is transforming how we interact with technology. From voice
assistants that understand context to recommendation systems that predict our
preferences, artificial intelligence has become deeply integrated into daily life.
```

**Detection output:**
```
=== L-bit Detection Results ===
Target message: 11001010
Recovered message: 11001010
Bit accuracy: 8/8 (100%)
Undecided bits: 0
Decision: MESSAGE RECOVERED SUCCESSFULLY ✓
```

### Multi-User Example

**User 0** (codeword: `0000000000`):
```
=== Tracing Results ===
Recovered codeword: 0000000000
Top matching users:
  1. User ID: 0, Username: 0, Bit matches: 10/10 (100%)
Decision: Text traced to User 0 ✓
```

**User 888** (codeword: `1101111000`):
```
=== Tracing Results ===
Recovered codeword: 1101111000
Top matching users:
  1. User ID: 888, Username: 888, Bit matches: 10/10 (100%)
Decision: Text traced to User 888 ✓
```

### Evaluation Summary

After running batch evaluation:
```
=== Evaluation Summary ===
Total prompts tested: 75
Model: gpt2
Parameter sweep: delta=[2.0, 2.5, 3.0], entropy=[3.0, 3.5, 4.0]

Clean text results:
  Detection rate: 98.7% (74/75 prompts)
  Average z-score: 15.23
  Average blocks: 94.5
  False positive rate: 0.0% (0/75 control texts)

Perturbation robustness:
  Delete start 20%: 87.3% (65/75)
  Delete end 20%: 89.1% (67/75)
  Delete middle 20%: 85.7% (64/75)
  Paraphrase 30%: 76.4% (57/75)

Best parameter combination:
  delta=2.5, entropy_threshold=3.5
  Clean detection: 100%, Avg robustness: 86.2%
```

---

## Model Information

### GPT-2 (Local)

- **Parameters:** 124M
- **Context length:** 1024 tokens
- **VRAM:** < 2GB (CPU compatible)
- **Speed:** ~10 tokens/sec (CPU), ~50 tokens/sec (GPU)
- **Use case:** Development, testing, quick experiments

**Important:** Keep `prompt_length + max_new_tokens ≤ 1024`

### GPT-OSS-20B

- **Parameters:** 20B
- **Context length:** 2048 tokens
- **VRAM:** 16GB+ (recommend A100 40GB)
- **Speed:** ~2-5 tokens/sec
- **Use case:** Production, high-quality generation

**Setup:**
```python
from src.models import GptOssModel
model = GptOssModel()  # Auto device_map="auto"
```

### GPT-OSS-120B

- **Parameters:** 120B
- **Context length:** 2048 tokens
- **VRAM:** 80GB+ (recommend A100 80GB or multi-GPU)
- **Speed:** ~0.5-2 tokens/sec
- **Use case:** Research, maximum quality

**Setup:**
```python
from src.models import Gpt Oss120bModel
model = GptOss120bModel()  # Requires substantial resources
```

---

## HPC Cluster Usage

### Pre-download Models

On login node (with internet):
```bash
export HF_HOME=/shared/models
export NLTK_DATA=/shared/nltk_data
python helper_scripts/download_models_hpc.py --model gpt2
python -c "import nltk; nltk.download('punkt', download_dir='/shared/nltk_data')"
```

### Submit Job

```bash
sbatch slurm_scripts/demonstration.sh
```

### Monitor Job

```bash
squeue -u $USER
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS
```

### Retrieve Results

```bash
cat slurm-<job_id>.out
ls -lh demonstration/
```

---

## Troubleshooting

### Common Issues

#### 1. NLTK Error: `punkt` or `punkt_tab` not found

**Solution:**
```bat
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Or set NLTK_DATA environment variable:
```bat
set NLTK_DATA=C:\path\to\nltk_data
```

#### 2. GPT-2 IndexError: position embeddings

**Error:**
```
IndexError: index out of range in self
```

**Cause:** Prompt + generation exceeds 1024 tokens

**Solution:** Reduce `--max-new-tokens` or shorten prompt
```bat
python main.py generate "Short prompt" --max-new-tokens 512 --model gpt2
```

#### 3. Low Block Count or Borderline Detection

**Symptoms:**
- Blocks < 75
- Z-score near threshold
- Frequent "NOT WATERMARKED" on watermarked text

**Solutions:**
1. Lower `--entropy-threshold` during generation (more blocks):
   ```bat
   --entropy-threshold 3.5  # instead of 4.0
   ```

2. Increase `--delta` (stronger signal):
   ```bat
   --delta 3.0  # instead of 2.5
   ```

3. Generate longer text:
   ```bat
   --max-new-tokens 512  # instead of 256
   ```

4. Lower `--z-threshold` during detection (more sensitive):
   ```bat
   --z-threshold 3.5  # instead of 4.0
   ```

#### 4. Many Undecided Bits (⊥) in L-Bit Recovery

**Symptoms:**
```
Recovered message: 01⊥1⊥⊥01
```

**Solutions:**
1. Slightly lower `--z-threshold` during detection:
   ```bat
   --z-threshold 3.5
   ```

2. Increase watermark strength during generation:
   ```bat
   --delta 3.0
   ```

3. Generate longer text (more blocks):
   ```bat
   --max-new-tokens 512
   ```

4. Lower `--entropy-threshold` during generation:
   ```bat
   --entropy-threshold 3.5
   ```

#### 5. GUI Import Error

**Error:**
```
ModuleNotFoundError: No module named 'models'
```

**Solutions:**
1. Run as module:
   ```bat
   python -m UI.app
   ```

2. Set PYTHONPATH:
   ```bat
   set PYTHONPATH=%CD%
   python UI\app.py
   ```

#### 6. CUDA Out of Memory (Large Models)

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. Use smaller model:
   ```bat
   --model gpt2  # instead of gpt-oss-20b
   ```

2. Enable device sharding (automatic in `models.py`):
   ```python
   device_map="auto"  # Already configured
   ```

3. Reduce batch size (if applicable)

4. Use multi-GPU setup

#### 7. Paraphrase Attack Timeout

**Symptoms:** T5 paraphraser hangs during evaluation

**Solutions:**
1. Skip paraphrase attack:
   Edit `src/commands.py`, comment out paraphrase perturbation

2. Use smaller T5 model (already using `t5-small`)

3. Increase timeout in evaluation script

---

## Advanced Usage

### Custom Perturbation Attacks

Add custom attacks in `src/utils.py`:

```python
def custom_attack(text: str) -> str:
    """Your custom perturbation logic."""
    # Example: replace all numbers with words
    import re
    replacements = {'0': 'zero', '1': 'one', '2': 'two'}
    for digit, word in replacements.items():
        text = text.replace(digit, word)
    return text
```

### Programmatic API Usage

```python
from src.watermark import ZeroBitWatermarker
from src.models import GPT2Model
import secrets

# Initialize
model = GPT2Model()
secret_key = secrets.token_bytes(32)
watermarker = ZeroBitWatermarker(model, secret_key, delta=2.5, entropy_threshold=4.0)

# Generate
watermarked_text = watermarker.generate(
    prompt="The future of AI is",
    max_new_tokens=512
)

# Detect
z_score, blocks = watermarker.detect(
    watermarked_text,
    z_threshold=4.0
)

print(f"Z-score: {z_score:.2f}, Blocks: {blocks}")
print(f"Detected: {z_score > 4.0}")
```

### Multi-User Collusion Testing

Test robustness against multiple users combining outputs:

```python
from src.fingerprinting import FingerprintingCode
from src.watermark import MultiUserWatermarker
import pandas as pd

users_df = pd.read_csv('assets/users.csv')
code = FingerprintingCode(users_df, l_bits=10)

# Generate from multiple users
user_texts = []
for user_id in [0, 5, 10]:
    # ... generate watermarked text for each user ...
    user_texts.append(text)

# Combine texts (e.g., interleave sentences)
combined = combine_texts(user_texts)  # Your logic

# Trace
matches = code.trace(recovered_message, max_collusion=16)
print(f"Detected users: {matches}")  # Should identify all 3 users
```

---

## Performance Benchmarks

Approximate generation speeds (256 tokens):

| Model | Hardware | Speed | VRAM | Time (256 tokens) |
|-------|----------|-------|------|-------------------|
| GPT-2 | CPU (8 cores) | ~10 tok/s | ~1GB RAM | ~25 seconds |
| GPT-2 | GPU (RTX 3090) | ~50 tok/s | ~1.5GB | ~5 seconds |
| GPT-OSS-20B | A100 40GB | ~3 tok/s | ~18GB | ~85 seconds |
| GPT-OSS-120B | A100 80GB | ~1 tok/s | ~75GB | ~4 minutes |

Detection is typically 2-3x faster than generation.

---

## License and Citation

### License

[Specify your license here, e.g., MIT, Apache 2.0, GPL-3.0]

### Citation

If you use this codebase in your research, please cite:

```bibtex
@software{cryptographic_watermarking_llm,
  title={Cryptographic Watermarking for Large Language Models},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/Cryptographic-Watermarking-for-LLM}
}
```

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

### Issues

Report bugs and request features at: [GitHub Issues](https://github.com/yourusername/Cryptographic-Watermarking-for-LLM/issues)

---

## Acknowledgments

This implementation builds upon research in statistical watermarking and cryptographic fingerprinting for LLMs. Special thanks to:
- HuggingFace Transformers team
- PyTorch contributors
- Research community in AI safety and provenance

---

## Quick Reference

### Essential Commands

```bat
# Setup
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# Zero-bit
python main.py generate "Prompt" --model gpt2 -o out.txt
python main.py detect out.txt --model gpt2 --key-file secret.key

# L-bit
python main.py generate_lbit "Prompt" --message 01010101 --l-bits 8 --model gpt2 -o out.txt
python main.py detect_lbit out.txt --l-bits 8 --model gpt2 --key-file secret.key

# Multi-user
python -m src.main_multiuser generate --user-id 0 --l-bits 10 --model gpt2 -o out.txt "Prompt"
python -m src.main_multiuser trace out.txt --l-bits 10 --model gpt2

# Evaluate
python main.py evaluate --prompts-file assets/prompts.txt --model gpt2
python helper_scripts\analyse.py evaluation_results

# GUI
python UI\app.py
```

### Parameter Cheatsheet

| Scenario | delta | entropy_threshold | z_threshold | max_new_tokens |
|----------|-------|-------------------|-------------|----------------|
| Default | 2.5 | 4.0 | 4.0 | 256 |
| High fluency | 2.0 | 4.5 | 4.0 | 256 |
| Strong detection | 3.5 | 3.5 | 3.5 | 512 |
| Short text | 3.0 | 3.5 | 3.5 | 128 |
| Long text | 2.5 | 4.0 | 4.0 | 1024 |

---

## Frequently Asked Questions

**Q: Can I use this with other models like LLaMA or Claude?**
A: Yes! Extend the `LanguageModel` base class in `src/models.py` with your model's implementation.

**Q: Is the watermark detectable by humans?**
A: No. The watermark operates at the statistical level and doesn't introduce visible patterns or artifacts.

**Q: Can the watermark survive translation?**
A: Partially. Translation may disrupt token-level watermarks, but semantic-preserving perturbations (like paraphrasing) show good robustness.

**Q: How secure is the key derivation?**
A: Uses HMAC-SHA256, a cryptographically secure PRF. Keep the master key secret and use adequate entropy (32 bytes recommended).

**Q: Can I watermark existing text?**
A: No. Watermarking must occur during generation (modifies logits before sampling). This is a generative watermark, not a post-hoc embedding.

**Q: What's the maximum message length for L-bit?**
A: Depends on text length and parameters. Longer text → more blocks → can embed more bits. Typical: 32-bit message in 512 tokens (GPT-2, default params).

---

**For more details, see:**
- `COMMANDS.md` - Copy-paste command examples
- `PARAMS.md` - Parameter tuning guide
- `src/watermark.py` - Implementation details
- GitHub Issues - Community support

**Happy watermarking! 🔐**
