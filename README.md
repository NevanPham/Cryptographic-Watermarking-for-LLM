## LLM Watermarking Evaluation Framework

This repository implements zero‑bit, L‑bit, and multi‑user (fingerprinting) watermarking for LLM text generation, with a CLI, a desktop GUI, evaluation/plotting tools, and optional SLURM scripts for HPC.

If you want the quickest way to run commands, see `COMMANDS.md`. For parameter explanations and tuning tips, see `PARAMS.md`.

---

## Repository structure

Core (root):
- `main.py`: CLI for zero‑bit, L‑bit, and evaluation (single/multi‑prompt, perturbation, detection, analysis output).
- `main_multiuser.py`: CLI for multi‑user generation and tracing (fingerprinting over L‑bit).
- `watermark.py`: Core watermarking logic
  - `ZeroBitWatermarker`: bias logits on high‑entropy positions, detect via z‑score
  - `LBitWatermarker`: encode/decode an L‑bit message using per‑bit key derivation
  - `MultiUserWatermarker`: embed user codewords (from `fingerprinting.py`) using the L‑bit scheme
  - Key derivation (`derive_key`), entropy calc, logits processors, block counting
- `models.py`: Model wrappers for GPT‑2 and large `gpt‑oss` variants (tokenizer/device abstraction).
- `fingerprinting.py`: Deterministic user codeword generation (from `UserId` and L) and tracing (matching recovered codeword to users).
- `prompts.txt`: Example prompts for evaluation.
- `requirements.txt`: Python dependencies.

GUI:
- `UI/app.py`: PySide6 desktop app (Generate / Detect / Evaluate tabs).
- `UI/core.py`: Thin wrappers around CLI logic for the GUI.
- `UI/app_GUI.py`, `UI/mainWindow.ui`, `UI/resources.*`: Qt UI files and resources.

Helper tools:
- `helper_scripts/analyse.py`: Plot completeness/soundness and robustness from `analysis_results.json`.
- Other helpers: visualizations, re‑running attacks, model download for HPC.

HPC (optional):
- `slurm_scripts/`: SLURM job scripts to run evaluation/detection/visualization on clusters.

Documentation:
- `COMMANDS.md`: Curated commands for local runs (CLI, GUI, helpers, multi‑user).
- `PARAMS.md`: Parameter descriptions and tuning tips.

---

## How it works (high level)

- Zero‑bit watermarking: At generation time, when the model’s next‑token entropy is high enough, we add a small PRF‑derived score vector (seeded by a secret key and recent token context) to the logits, gently biasing token choice. At detection, we re‑derive the same vectors and compute a normalized sum (z‑score) over “block” positions; z > threshold ⇒ detected.

- L‑bit watermarking: Extends zero‑bit to encode a binary string of length L. For each bit position i and bit value b, derive a per‑bit key from a single master key using HMAC, biasing logits accordingly. At detection, test both hypotheses per bit (b∈{0,1}) and decide 0/1/⊥.

- Multi‑user (fingerprinting): Assign each user an L‑bit codeword deterministically from `UserId`. Embed that codeword via L‑bit using a single master key. At tracing, recover a (possibly noisy) codeword and match to one or more users.

Important:
- Only the master secret key is stored; per‑bit keys are derived on the fly.
- Entropy gating ensures we watermark only where the model is uncertain (limits perceptibility and concentrates signal).

---

## Workflows (CLI)

Below are condensed flows; see `COMMANDS.md` for complete, copy‑pastable commands.

### Local setup (Windows cmd)
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Zero‑bit (single run)
Generate (saves `output.txt` and `secret.key`):
```bat
python main.py generate "The future of AI is" --model gpt2 --max-new-tokens 256 -o output.txt
```
Detect:
```bat
python main.py detect output.txt --model gpt2 --key-file secret.key
```

### L‑bit (single run)
Generate (L must equal the message length in bits):
```bat
python main.py generate_lbit "Prompt" --model gpt2 --message 01010101 --l-bits 8 --max-new-tokens 256 -o output_lbit.txt --key-file secret_lbit.key
```
Detect:
```bat
python main.py detect_lbit output_lbit.txt --model gpt2 --l-bits 8 --key-file secret_lbit.key
```

### Evaluation (batch over prompts)
Creates clean + perturbed variants, runs detection, writes `analysis_results.json`:
```bat
python main.py evaluate --prompts-file prompts.txt --model gpt2 --entropy-thresholds "3.0, 3.5, 4.0" --max-new-tokens 512 --output-dir evaluation_results
```
Plot and summarize:
```bat
python helper_scripts\analyse.py evaluation_results --z-threshold 4.0
```

### Multi‑user (fingerprinting)
Create `users.csv`:
```text
UserId,Username
0,Alice
1,Bob
2,Carol
```
Generate for a user (saves one master key per run):
```bat
python main_multiuser.py generate --users-file users.csv --model gpt2 --user-id 1 --l-bits 8 --max-new-tokens 256 --key-file demonstration\multiuser_master.key -o demonstration\multiuser_output.txt "The future of AI is"
```
Trace back to user(s):
```bat
python main_multiuser.py trace --users-file users.csv --model gpt2 --l-bits 8 --key-file demonstration\multiuser_master.key demonstration\multiuser_output.txt
```

---

## GUI (desktop)

Launch the PySide6 app:
```bat
python UI\app.py
```
Tabs:
- Generate: Produce watermarked/plain text; save/load key when watermarking.
- Detect: Paste or load text, select model/key, view z‑score and blocks.
- Evaluate: Run parameter sweeps and view generated plots.

If you see `ModuleNotFoundError: models`, run from the repo root as a module (`python -m UI.app`) or set `PYTHONPATH=%CD%` before running.

---

## Parameters and tuning

See `PARAMS.md` for detailed descriptions and presets. Key knobs:
- Generation: `--delta`, `--entropy-threshold`, `--hashing-context`, `--max-new-tokens`
- Detection: `--z-threshold` (two‑pass logic may lower effective entropy threshold if block count is low)
- L‑bit: `--l-bits`, message length equals L; reduce undecided bits (⊥) by modestly increasing `delta`, lowering generation `entropy-threshold`, or lowering detect `z-threshold`.

GPT‑2 context note: keep `prompt_length + max_new_tokens ≤ 1024`.

---

## HPC notes (optional)

On clusters without internet on compute nodes:
- Pre‑download models and NLTK data on the login node into a shared path (`HF_HOME`, `NLTK_DATA`).
- Use scripts in `slurm_scripts/` as templates. Ensure `HF_HOME`/`NLTK_DATA` are exported and your venv is activated.

Large models (`gpt‑oss‑20b/120b`) require significant GPU memory and may need device sharding via `device_map="auto"` (already configured in `models.py`).

---

## Troubleshooting

- NLTK error (`punkt`/`punkt_tab`): run the download commands shown in setup; optionally set `NLTK_DATA`.
- GPT‑2 position embedding IndexError: reduce `--max-new-tokens` so total tokens ≤ 1024.
- Detection borderline (low blocks or z): lower detector `--entropy-threshold` and/or `--z-threshold`; generation two‑pass and detection two‑pass try this automatically in many paths.
- GUI import error from `UI/app.py`: run as module (`python -m UI.app`) or set `PYTHONPATH` to the repo root.

---

## License and citation

If you use this codebase in research, please cite appropriately. Contributions and issues are welcome.
## Crypto-Watermarking for LLMs

This repository provides a practical framework to embed and detect statistical watermarks in LLM generations, including:

- Zero‑bit watermarking (is text watermarked? yes/no)
- L‑bit watermarking (embed and recover a bitstring)
- Multi‑user fingerprinting (trace a generation back to one or more users)

It includes a modular model layer (GPT‑2 and large OSS models), a multi‑user CLI, and reference docs for parameters and commands.

## Repository structure

- `main_multiuser.py`: CLI for multi‑user watermarking. Supports `generate` and `trace` commands, wired to the watermarking stack and the fingerprinting code.
- `fingerprinting.py`: Implements `FingerprintingCode` for generating per‑user codewords from `users.csv` and tracing recovered messages to likely users.
- `models.py`: Swappable language model adapters: `GPT2Model`, `GptOssModel` (20B), `GptOss120bModel` (120B). Each exposes `get_logits`, `tokenizer`, `vocab_size`, and `device`.
- `PARAMS.md`: Parameter reference with guidance for `delta`, `entropy-threshold`, `hashing-context`, `max-new-tokens`, `z-threshold`, and L‑bit specifics.
- `COMMANDS.md`: End‑to‑end command examples for Windows cmd, including quickstart, evaluation flows, GUI entrypoint, and multi‑user recipes.
- `users.csv`: Example user metadata (`UserId,Username`) used by the fingerprinting module.
- `multiuser_output.txt`: Example output file produced by `main_multiuser.py generate`.
- `.gitignore`: Standard git ignore rules.

Note: The core zero‑bit and L‑bit watermarkers are used via `ZeroBitWatermarker`, `LBitWatermarker`, and `MultiUserWatermarker` in the CLI. These classes are expected to live in `watermark.py` within this project layout.

## Installation

Create a virtual environment and install dependencies:
```
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

If you use GPT‑2 locally, the first run will download model weights and tokenizer. Large OSS models (20B/120B) require substantial resources; see notes below.

## Quickstart (Windows cmd)

Generate and detect with GPT‑2:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

python main.py generate "The future of AI is" --model gpt2 --max-new-tokens 256 -o output.txt
python main.py detect output.txt --model gpt2 --key-file secret.key
```

L‑bit example (embed and recover a bitstring):
```
python main.py generate_lbit "The future of AI is" --model gpt2 ^
  --message 01010101010101010101010101010101 --l-bits 32 ^
  --max-new-tokens 256 --output-file output_lbit.txt --key-file secret_lbit.key

python main.py detect_lbit output_lbit.txt --model gpt2 --l-bits 32 --key-file secret_lbit.key
```

More end‑to‑end command recipes are in `COMMANDS.md`.

## Multi‑user fingerprinting workflow

1) Prepare users:
```
UserId,Username
0,Alice
1,Bob
2,Carol
```

2) Generate watermarked text for a specific user (defaults shown in CLI):
```
python main_multiuser.py generate ^
  --users-file users.csv ^
  --model gpt2 ^
  --user-id 1 ^
  --l-bits 8 ^
  --max-new-tokens 256 ^
  --key-file demonstration\multiuser_master.key ^
  -o demonstration\multiuser_output.txt
```

3) Trace a text back to likely user(s):
```
python main_multiuser.py trace ^
  --users-file users.csv ^
  --model gpt2 ^
  --l-bits 8 ^
  --key-file demonstration\multiuser_master.key ^
  demonstration\multiuser_output.txt
```

## How it works (high level)

- **Zero‑bit watermarking**: Adds a pseudorandom, key‑seeded score vector to next‑token logits at high‑entropy positions (“blocks”). This creates a weak but statistically detectable bias. Detection computes a z‑score over block statistics and compares to `--z-threshold`.
- **L‑bit watermarking**: Iteratively embeds bits of a target message across eligible blocks using derived keys. Detection recovers each bit by hypothesis testing; undecided positions may be marked as `⊥`.
- **Multi‑user fingerprinting**: Maps each user to a codeword of length L (e.g., 8 bits). The L‑bit embedder uses the user’s codeword as the message. On recovery, the codeword is matched against all known users to identify the best candidates (with support for ties and a collusion cap).

## Parameters

Tuning guidance and defaults are documented in `PARAMS.md`. Common flags:

- `--delta`: watermark strength (bias magnitude)
- `--entropy-threshold`: block selection threshold (higher → fewer, cleaner blocks)
- `--hashing-context`: PRF context length in tokens
- `--max-new-tokens`: generation length
- `--z-threshold`: detection decision threshold
- `--l-bits`: length of embedded message (for L‑bit and multi‑user)

Suggested presets and troubleshooting tips are also in `PARAMS.md`.

## Models

Adapters in `models.py` allow swapping backends without changing watermark logic:

- `GPT2Model`: lightweight local runs. Keep `prompt_length + max_new_tokens ≤ 1024`.
- `GptOssModel` (20B) and `GptOss120bModel` (120B): require significant memory; use `device_map="auto"` and bf16 when available.

## Evaluation and analysis

Batch evaluation (sweeps over prompts and thresholds) and analysis scripts are demonstrated in `COMMANDS.md`. Plots and summary metrics can be generated from evaluation outputs.

## Troubleshooting

- Low block count or borderline detection: lower detector `--entropy-threshold` or increase `--delta`; consider longer `--max-new-tokens`.
- Many `⊥` in L‑bit recovery: slightly lower `--z-threshold`; reduce generation `--entropy-threshold`; modestly raise `--delta`.
- GPT‑2 context errors: ensure `prompt_length + max_new_tokens ≤ 1024`.

## Notes

- First run will download model/tokenizer weights for selected backends.
- For best fluency on long texts, consider larger models; for experiments and unit tests, GPT‑2 is fastest.
- For HPC environments without internet, pre‑download models on a login node and set `HF_HOME` to shared storage.
