## Watermarking and Detection Parameters

This guide explains the knobs you can tune for embedding (generation) and detection, what they do, and how to pick values. Examples assume GPT‑2 locally.

---

### Core parameters (used by both Zero‑bit and L‑bit)

#### delta (bias strength)
- What: Scales the PRF‑derived score vector added to logits at eligible positions.
- Effect: Higher → stronger statistical signal → easier detection, but may hurt fluency.
- Typical: 2.0–3.5. Default 2.5–3.0.
- CLI:
```bat
--delta 3.0
```

#### entropy_threshold (where watermark applies)
- What: Only positions whose next‑token entropy ≥ threshold are considered “blocks”.
- Effect: Higher → fewer, higher‑entropy positions (cleaner but less frequent). Lower → more positions (noisier but more samples).
- Typical: 3.5–4.5. Default ~4.0.
- CLI:
```bat
--entropy-threshold 4.0
```

#### hashing_context (PRF context length)
- What: Number of previous tokens used to seed the PRF (score vector is prefix‑dependent).
- Effect: Larger context gives more variability; 5 is a good default.
- Typical: 3–8. Default 5.
- CLI:
```bat
--hashing-context 5
```

#### max_new_tokens (generation length)
- What: How many new tokens to generate.
- Effect: Longer texts usually produce more blocks → more reliable detection (up to model context limits).
- GPT‑2 note: prompt_len + max_new_tokens ≤ 1024. Safe values: 256–512.
- CLI:
```bat
--max-new-tokens 256
```

---

### Detector‑specific parameter

#### z_threshold (decision threshold)
- What: Detection when z_score > z_threshold.
- Effect: Lower increases sensitivity (may raise false positives); higher is more conservative.
- Typical: 3.5–5.0. Default 4.0.
- CLI (zero‑bit):
```bat
python main.py detect demonstration/watermarked_output.txt --model gpt2 --z-threshold 3.5
```
- CLI (L‑bit):
```bat
python main.py detect_lbit demonstration/watermarked_lbit.txt --model gpt2 --l-bits 32 --z-threshold 3.5
```

Detector behavior: two‑pass logic may retry with a lower effective entropy threshold if the initial block count is low.

---

### L‑bit‑specific parameters

#### l-bits (message length)
- What: Length of the embedded binary message.
- Constraint: `--message` must be exactly L bits of 0/1.
- Trade‑off: Larger L spreads signal across more positions; ensure your text length yields enough blocks.
- CLI:
```bat
--l-bits 32 --message 0101... (32 bits total)
```

Recovered message symbols:
- `0` or `1`: confident bit.
- `⊥`: undecided (neither hypothesis exceeded z_threshold).

To reduce `⊥`:
- Increase `--delta` moderately (e.g., 3.0).
- Lower generation `--entropy-threshold` a bit (e.g., 3.5).
- Lower detection `--z-threshold` slightly (e.g., 3.5).
- Increase `--max-new-tokens` within model limits.

---

### Practical presets

#### Balanced (good starting point)
```bat
--delta 2.5 --entropy-threshold 4.0 --hashing-context 5 --max-new-tokens 256 --z-threshold 4.0
```

#### Higher fluency (lighter watermark)
```bat
--delta 2.0 --entropy-threshold 4.5 --hashing-context 5 --max-new-tokens 256 --z-threshold 4.0
```

#### Stronger detection (may reduce fluency)
```bat
--delta 3.5 --entropy-threshold 3.5 --hashing-context 5 --max-new-tokens 256 --z-threshold 4.0
```

#### L‑bit robust recovery (small L example)
```bat
--l-bits 6 --delta 3.0 --entropy-threshold 3.5 --hashing-context 5 --max-new-tokens 256 --z-threshold 3.5
```

---

### Troubleshooting tips

- Low block count (< ~75) or non‑detection:
  - Lower detector `--entropy-threshold` by ~1.0–2.0; or raise generation `--delta` by ~0.5–1.0.
- Many `⊥` in L‑bit:
  - Slightly lower `--z-threshold` (e.g., 3.5), and/or lower generation `--entropy-threshold`.
- GPT‑2 IndexError about position embeddings:
  - Reduce `--max-new-tokens` so total ≤ 1024.

---

### Where these are defined in code
- Watermarking logic and parameters: `src/watermark.py` (`ZeroBitWatermarker`, `LBitWatermarker`).
- CLI flags: `main.py` (generate/detect/evaluate; zero‑bit and L‑bit).
- Two‑pass generation/detection logic: `src/watermark.py` and `main.py` detect/evaluate paths.

