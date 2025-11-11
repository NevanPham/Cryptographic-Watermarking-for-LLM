# visualise_lbit_blocks.py

import argparse
import json
import os
import torch
from collections import Counter

import sys

# Add the parent_dir to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

# Now we can grab the model and watermarker definitions from the parent dir
from models import GPT2Model, GptOssModel, GptOss120bModel
from watermark import _calculate_entropy

def get_model(model_name: str):
    """Factory function to instantiate the correct model."""
    if model_name == 'gpt2':
        return GPT2Model()
    elif model_name == 'gpt-oss-20b':
        return GptOssModel()
    elif model_name == 'gpt-oss-120b':
        return GptOss120bModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a single L-bit watermarked text file to visualize its block structure.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('text_file', type=str, help='Path to the watermarked .txt file.')
    parser.add_argument('--message', type=str, required=True, help='The original binary message that was embedded.')
    parser.add_argument('--model', type=str, required=True, choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                        help='The model that was used to generate the text.')
    parser.add_argument('--entropy-threshold', type=float, default=3.0, help='The entropy threshold used during generation.')
    args = parser.parse_args()

    # --- 1. Load Everything ---
    try:
        print(f"Loading model '{args.model}'...")
        model = get_model(args.model)
        tokenizer = model.tokenizer

        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()

    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e.filename}")
        return

    original_message = args.message
    L = len(original_message)

    print(f"⚙️ Analyzing with L={L} and Entropy Threshold={args.entropy_threshold}")

    # --- 2. Find Block Boundaries ---
    token_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)[0]
    
    with torch.no_grad():
        outputs = model._model(token_ids.unsqueeze(0))
    all_logits = outputs.logits.squeeze(0)

    block_indices = [i + 1 for i, logits in enumerate(all_logits[:-1]) if _calculate_entropy(logits) >= args.entropy_threshold]
    
    # --- 3. Map Blocks to Bit Positions (Round-Robin) ---
    block_to_bit_map = {token_index: i % L for i, token_index in enumerate(block_indices)}

    # --- 4. Print Summary ---
    print(f"\n✅ Found {len(block_indices)} total blocks.")
    embedded_bits_count = Counter(block_to_bit_map.values())
    print("\n--- Bit Embedding Summary ---")
    for i in range(L):
        count = embedded_bits_count.get(i, 0)
        status = "✅ Embedded" if count > 0 else "❌ MISSED"
        print(f"  Bit #{i} (value: {original_message[i]}): {status} {count} time(s)")
    print("-----------------------------\n")

    # --- 5. Reconstruct and Print Visualized Text ---
    print("--- Text with Block Visualization ---")
    output_parts = []
    for i, token_id in enumerate(token_ids):
        if i in block_to_bit_map:
            bit_pos = block_to_bit_map[i]
            output_parts.append(f"|B{bit_pos}|")
        output_parts.append(tokenizer.decode(token_id))
    
    visualized_text = "".join(output_parts)
    print(visualized_text)
    print("-" * 33)

if __name__ == "__main__":
    main()