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
python main.py generate "The future of AI is" --model gpt2 --max-new-tokens 256 ^
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
python main.py generate "The future of AI is" --model gpt2 --no-watermark --max-new-tokens 256 -o demonstration/output_plain.txt
```

### L‑bit watermarking (embed and recover a bitstring)
- Generate (message must be exactly L bits, saves to `demonstration/watermarked_lbit.txt` and `demonstration/secret_lbit.key`; append `-o`/`--key-file` to rename):
```bat
python main.py generate_lbit "The future of AI is" --model gpt2 ^
  --message 01010101 --l-bits 8 ^
  --max-new-tokens 256 ^
  -o demonstration/my_lbit.txt ^
  --key-file demonstration/my_lbit.key
```
- Detect (recovers the message; may output ⊥ for undecided bits; match the filenames you set above):
```bat
python main.py detect_lbit demonstration/my_lbit.txt --model gpt2 --l-bits 8 ^
  --key-file demonstration/my_lbit.key
```

### Full evaluation (batch over prompts)
- Sweep entropy thresholds (writes results to `demonstration/evaluation_results/`; change with `--output-dir`):
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
  --max-new-tokens 256 ^
  --output-dir demonstration/evaluation_results_lbit
```

## Running helpers and tools

### Analyse plots and metrics (from evaluation output)
- Reads `analysis_results.json` and writes PNG plots + `summary_analysis.txt`:
```bat
python helper_scripts\analyse.py demonstration/evaluation_results --z-threshold 4.0
```

### GUI (desktop app)
```bat
python UI\app.py
```
- Tabs:
  - Generate: create watermarked/plain text; saves `secret.key` when watermarking.
  - Detect: paste text + select key folder; shows detection result.
  - Evaluate: run sweeps and display plots inline.

### Multi‑user fingerprinting (optional)
1) The `assets/users.csv` file contains 1000 users (UserIds 0-999, with usernames matching the ID):
```text
UserId,Username
0,0
1,1
2,2
...
999,999
```
2) Generate for a user (L=10 required for 1000 users; pass `-o` to change filename):
```bat
python -m src.main_multiuser generate "The future of AI is" ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 10 ^
  --max-new-tokens 256 ^
  -o demonstration/multiuser_user0.txt
```
3) Trace back to user(s):
```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 10 ^
  demonstration\multiuser_user0.txt
```

## Parameter tuning guide

### GPT‑2 safety
- Keep prompt_length + max_new_tokens ≤ 1024 (GPT‑2 context limit).
- Recommended: `--max-new-tokens 256` (or up to 512 for shorter prompts).

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
python main.py generate "Topic..." --model gpt2 --delta 2.0 --entropy-threshold 4.5 --hashing-context 5 --max-new-tokens 256
```
- Stronger detection (may hurt fluency):
```bat
python main.py generate "Topic..." --model gpt2 --delta 3.5 --entropy-threshold 3.5 --hashing-context 5 --max-new-tokens 256
```

### L‑bit recovery tips
- If you see ⊥ in recovered bits:
  - Increase bias or lower entropy threshold on generation:
```bat
python main.py generate_lbit "Topic..." --model gpt2 --message 010101 --l-bits 6 --delta 3.0 --entropy-threshold 3.5 --max-new-tokens 256
```
  - Lower the detector threshold a bit:
```bat
python main.py detect_lbit demonstration/watermarked_lbit.txt --model gpt2 --l-bits 6 --z-threshold 3.5
```

## Notes
- First run downloads models; ensure internet for GPT‑2 and the paraphrase model used in evaluation.
- The attention_mask warning for GPT‑2 can be ignored in these flows.
- For larger, more coherent outputs, a bigger model (e.g., gpt‑oss‑20b) is recommended but typically requires HPC.

