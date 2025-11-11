## Local quickstart (Windows cmd)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Running the core CLI

### Zero‑bit watermarking
- Generate (saves `output.txt` and `secret.key`):
```bat
python main.py generate "The future of AI is" --model gpt2 --max-new-tokens 256 -o output.txt
```
- Detect (prints z-score, detected, block count):
```bat
python main.py detect output.txt --model gpt2 --key-file secret.key
```
- Plain (no watermark):
```bat
python main.py generate "The future of AI is" --model gpt2 --no-watermark --max-new-tokens 256 -o output_plain.txt
```

### L‑bit watermarking (embed and recover a bitstring)
- Generate (message must be exactly L bits):
```bat
python main.py generate_lbit "The future of AI is" --model gpt2 ^
  --message 01010101010101010101010101010101 --l-bits 32 ^
  --max-new-tokens 256 --output-file output_lbit.txt --key-file secret_lbit.key
```
- Detect (recovers the message; may output ⊥ for undecided bits):
```bat
python main.py detect_lbit output_lbit.txt --model gpt2 --l-bits 32 --key-file secret_lbit.key
```

### Full evaluation (batch over prompts)
- Sweep entropy thresholds (writes results to `evaluation_results/`):
```bat
python main.py evaluate ^
  --prompts-file prompts.txt ^
  --model gpt2 ^
  --entropy-thresholds "3.0, 3.5, 4.0" ^
  --max-new-tokens 512 ^
  --output-dir evaluation_results
```
- Include L‑bit in the evaluation (example L=8):
```bat
python main.py evaluate ^
  --prompts-file prompts.txt ^
  --model gpt2 ^
  --l-bit-message 01010101 ^
  --l-bits 8 ^
  --entropy-thresholds "3.5, 4.0" ^
  --max-new-tokens 256 ^
  --output-dir evaluation_results_lbit
```

## Running helpers and tools

### Analyse plots and metrics (from evaluation output)
- Reads `analysis_results.json` and writes PNG plots + `summary_analysis.txt`:
```bat
python helper_scripts\analyse.py evaluation_results --z-threshold 4.0
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
1) Create `users.csv`:
```text
UserId,Username
0,Alice
1,Bob
2,Carol
```
2) Generate for a user (defaults to L=8 in `main_multiuser.py`):
```bat
python main_multiuser.py generate ^
  --users-file users.csv ^
  --model gpt2 ^
  --user-id 1 ^
  --l-bits 8 ^
  --max-new-tokens 256 ^
  --key-file demonstration\multiuser_master.key ^
  -o demonstration\multiuser_output.txt
```
3) Trace back to user(s):
```bat
python main_multiuser.py trace ^
  --users-file users.csv ^
  --model gpt2 ^
  --l-bits 8 ^
  --key-file demonstration\multiuser_master.key ^
  demonstration\multiuser_output.txt
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
python main.py generate "Topic..." --model gpt2 --delta 2.0 --entropy-threshold 4.5 --hashing-context 5 --max-new-tokens 256 -o output.txt
```
- Stronger detection (may hurt fluency):
```bat
python main.py generate "Topic..." --model gpt2 --delta 3.5 --entropy-threshold 3.5 --hashing-context 5 --max-new-tokens 256 -o output.txt
```

### L‑bit recovery tips
- If you see ⊥ in recovered bits:
  - Increase bias or lower entropy threshold on generation:
```bat
python main.py generate_lbit "Topic..." --model gpt2 --message 010101 --l-bits 6 --delta 3.0 --entropy-threshold 3.5 --max-new-tokens 256 --output-file output_lbit.txt --key-file secret_lbit.key
```
  - Lower the detector threshold a bit:
```bat
python main.py detect_lbit output_lbit.txt --model gpt2 --l-bits 6 --key-file secret_lbit.key --z-threshold 3.5
```

## Notes
- First run downloads models; ensure internet for GPT‑2 and the paraphrase model used in evaluation.
- The attention_mask warning for GPT‑2 can be ignored in these flows.
- For larger, more coherent outputs, a bigger model (e.g., gpt‑oss‑20b) is recommended but typically requires HPC.

