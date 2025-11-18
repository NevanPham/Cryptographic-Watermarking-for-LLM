## Local quickstart (Windows cmd)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Running the core CLI

### Zero‑bit watermarking
- Generate (saves `demonstration/watermarked_output.txt` and `demonstration/secret.key`; add `-o`/`-k` to rename):
```bat
python main.py generate "The future of AI is" --model gpt2 --max-new-tokens 512 ^
  -o demonstration/my_watermark.txt ^
  -k demonstration/my_watermark.key
```
- Detect (prints z-score, detected, block count; point to the same files you generated):
```bat
python main.py detect demonstration/my_watermark.txt --model gpt2 ^
  --key-file demonstration/my_watermark.key
```
- Plain (no watermark):
```bat
python main.py generate "The future of AI is" --model gpt2 --no-watermark --max-new-tokens 512 -o demonstration/output_plain.txt
```

### L‑bit watermarking (embed and recover a bitstring)
- Generate (message must be exactly L bits, saves to `demonstration/watermarked_lbit.txt` and `demonstration/secret_lbit.key`; append `-o`/`--key-file` to rename):
```bat
python main.py generate_lbit "The future of AI is" --model gpt2 ^
  --message 01010101 --l-bits 8 ^
  --max-new-tokens 512 ^
  -o demonstration/my_lbit.txt ^
  --key-file demonstration/my_lbit.key
```
- Detect (recovers the message; may output ⊥ for undecided bits; match the filenames you set above):
```bat
python main.py detect_lbit demonstration/my_lbit.txt --model gpt2 --l-bits 8 ^
  --key-file demonstration/my_lbit.key
```

### Full evaluation (batch over prompts)
- Sweep entropy thresholds (writes results to `evaluation/evaluation_results/`; change with `--output-dir`):
```bat
python main.py evaluate ^
  --prompts-file assets/prompts.txt ^
  --model gpt2 ^
  --entropy-thresholds "3.0, 3.5, 4.0" ^
  --max-new-tokens 512
```
- Include L‑bit in the evaluation (example L=8):
```bat
python main.py evaluate ^
  --prompts-file assets/prompts.txt ^
  --model gpt2 ^
  --l-bit-message 01010101 ^
  --l-bits 8 ^
  --entropy-thresholds "3.5, 4.0" ^
  --max-new-tokens 512 ^
  --output-dir evaluation/evaluation_results_lbit
```

> By default evaluation uses the first 100 prompts. Pass `--max-prompts 300` (or any number you need) when running on HPC to process the full file.

## Running helpers and tools

### Analyse plots and metrics (from evaluation output)
- Reads `analysis_results.json` and writes PNG plots + `summary_analysis.txt`:
```bat
python helper_scripts\analyse.py evaluation/evaluation_results --z-threshold 4.0
```

### GUI (desktop app)
```bat
python UI\app.py
```
- Tabs:
  - Generate: create watermarked/plain text; saves `secret.key` when watermarking.
  - Detect: paste text + select key folder; shows detection result.
  - Evaluate: run sweeps and display plots inline.

### Multi‑user fingerprinting (BCH-based with minimum distance)
1) The `assets/users.csv` file contains 1000 users (UserIds 0-999, with usernames matching the ID):
```text
UserId,Username
0,0
1,1
2,2
...
999,999
```
2) Generate for a user (L=10 required for 1000 users; uses grouped BCH codes with guaranteed minimum Hamming distance):
```bat
python -m src.main_multiuser generate "The future of AI is" ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 10 ^
  --scheme grouped ^
  --min-distance 3 ^
  --max-new-tokens 512 ^
  -o demonstration/multiuser_user0.txt
```

**BCH minimum distance options:**
- `--min-distance 2`: Up to 100 groups, 10 users per group (weaker collusion resistance)
- `--min-distance 3`: Up to 50 groups, 20 users per group (default, balanced)
- `--min-distance 4`: Up to 10 groups, 100 users per group (strongest collusion resistance)

**Note:** Users are assigned to groups sequentially (Users 0-19 → Group 0, Users 20-39 → Group 1, etc.). All users in the same group share the same group codeword, providing better collusion resistance than binary ID-based fingerprinting.

**Legacy (naive) scheme:** Pass `--scheme naive` to fall back to the original per-user binary codeword assignment (no grouping, no BCH codes). `--min-distance` is ignored in this mode.
```bat
python -m src.main_multiuser generate "Binary fingerprint" ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 42 ^
  --l-bits 10 ^
  --scheme naive ^
  --max-new-tokens 256
```

**Using file-based prompts (for collusion scenarios):**
- Generate with a prompt from a file (e.g., previous user's output):
```bat
python -m src.main_multiuser generate ^
  --prompt-file demonstration/multiuser_user0.txt ^
  --prompt-suffix "Rewrite the paragraph above and add more details about " ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 2 ^
  --l-bits 10 ^
  --scheme grouped ^
  --min-distance 3 ^
  --max-new-tokens 512 ^
  -o demonstration/multiuser_user2_collusion_with_user[].txt
```
- **Note:** Only the NEW text generated will be watermarked with user 2's codeword. The prompt file content (user 0's output) is NOT watermarked—it's just context. This is NOT collusion; collusion would require mixing multiple users' watermarks in the same generated text.
3) Trace back to user(s) (shows group membership):
```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 10 ^
  --scheme grouped ^
  --min-distance 3 ^
  demonstration\multiuser_user0.txt
```

**Creating true collusion scenarios:**
- Use the interactive script to generate watermarked text from multiple users and combine them:
```bat
python helper_scripts\create_collusion_scenario.py
```
- The script will prompt you for:
  - Number of users (minimum 2)
  - User ID and prompt for each user
- Output: Combined text file with all users' watermarked text concatenated
- Then trace the combined file to detect collusion (look for `*` symbols in recovered codeword)

**Visualizing group assignments and codewords:**
- View all groups, user assignments, and codewords:
```bat
python helper_scripts\visualize_groups.py ^
  --users-file assets/users.csv ^
  --l-bits 10 ^
  --min-distance 3
```
- Show detailed output with all user IDs:
```bat
python helper_scripts\visualize_groups.py ^
  --users-file assets/users.csv ^
  --l-bits 10 ^
  --min-distance 3 ^
  --detailed
```
- The script displays:
  - Group assignments (Group ID, codeword, user range)
  - Minimum distance verification between all group codewords
  - Statistics (average users per group, codeword distribution)
  - Distance violations (if any)

**Comparing collusion resistance across approaches:**
- Compare naive vs fingerprinting (min-distance-2, min-distance-3) approaches:
```bat
python helper_scripts\compare_collusion_resistance.py ^
  --prompts-file assets/prompts.txt ^
  --max-prompts 100 ^
  --model gpt2 ^
  --users-file assets/users.csv ^
  --num-colluders 2 ^
  --l-bits 10 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --max-new-tokens 256
```
- Test with 3 colluders:
```bat
python helper_scripts\compare_collusion_resistance.py ^
  --num-colluders 3 ^
  --max-prompts 100
```
- **What it does:**
  1. For each prompt, selects colluding users from different groups (ensures fair comparison)
  2. Generates watermarked text for each colluding user using three approaches:
     - Naive (binary user ID)
     - Min-distance-2 (fingerprinting with distance 2)
     - Min-distance-3 (fingerprinting with distance 3)
  3. Combines texts in two ways:
     - Normal combination (concatenation)
     - With deletion (5% of each user's text deleted before combining)
  4. Attempts to trace back to original colluding users
  5. Calculates success rates for each approach and combination method
- **Output:**
  - Comparison table printed to console
  - JSON file: `collusion_resistance_results_<N>users.json` (detailed results)
  - CSV file: `collusion_resistance_summary_<N>users.csv` (summary)
  - Organized folder structure:
    ```
    evaluation/collusion_resistance_<N>/
    ├── naive/prompt_0/, prompt_1/, ...
    ├── min-distance-2/prompt_0/, prompt_1/, ...
    ├── min-distance-3/prompt_0/, prompt_1/, ...
    └── collusion_resistance_results_<N>users.json
    ```
- **Key features:**
  - Same colluding users used across all three approaches per prompt (fair comparison)
  - Users selected from different groups (tests true collusion resistance)
  - Two combination methods test robustness
  - Default: first 100 prompts (use `--max-prompts` to change)

## Parameter tuning guide

### GPT‑2 safety
- Keep prompt_length + max_new_tokens ≤ 1024 (GPT‑2 context limit).
- Recommended: `--max-new-tokens 512` (or up to 512 for shorter prompts).

### Watermark strength and robustness
- `--delta` (bias strength): higher = easier detection, may lower fluency.
  - Typical: 2.0–3.5. Example: `--delta 3.0`.
- `--entropy-threshold` (where to watermark): higher = fewer blocks, more selective.
  - Typical: 3.5–4.5. Example: `--entropy-threshold 4.0`.
- `--hashing-context` (PRF context length): usually 5 is fine.
  - Example: `--hashing-context 5`.
- `--z-threshold` (detector decision): lower increases sensitivity, risks false positives.
  - Default 4.0; try 3.5 if borderline.

### Examples: balancing fluency vs detection
- More fluent (lighter watermark):
```bat
python main.py generate "Topic..." --model gpt2 --delta 2.0 --entropy-threshold 4.5 --hashing-context 5 --max-new-tokens 512
```
- Stronger detection (may hurt fluency):
```bat
python main.py generate "Topic..." --model gpt2 --delta 3.5 --entropy-threshold 3.5 --hashing-context 5 --max-new-tokens 512
```

### L‑bit recovery tips
- If you see ⊥ in recovered bits:
  - Increase bias or lower entropy threshold on generation:
```bat
python main.py generate_lbit "Topic..." --model gpt2 --message 010101 --l-bits 6 --delta 3.0 --entropy-threshold 3.5 --max-new-tokens 512
```
  - Lower the detector threshold a bit:
```bat
python main.py detect_lbit demonstration/watermarked_lbit.txt --model gpt2 --l-bits 6 --z-threshold 3.5
```

## Notes
- First run downloads models; ensure internet for GPT‑2 and the paraphrase model used in evaluation.
- The attention_mask warning for GPT‑2 can be ignored in these flows.
- For larger, more coherent outputs, a bigger model (e.g., gpt‑oss‑20b) is recommended but typically requires HPC.

