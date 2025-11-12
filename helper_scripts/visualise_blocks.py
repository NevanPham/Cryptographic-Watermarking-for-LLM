# visualize_blocks.py

import argparse
import json
import os
import re
import torch
from tqdm import tqdm

import sys

# Add the parent_dir to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

# Now we can grab the model and watermarker definitions from the parent dir
from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import ZeroBitWatermarker

def get_model(model_name: str):
    """Factory function to instantiate the correct model for its tokenizer."""
    if model_name == 'gpt2':
        return GPT2Model()
    elif model_name == 'gpt-oss-20b':
        return GptOssModel()
    elif model_name == 'gpt-oss-120b':
        return GptOss120bModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def parse_filename(filename: str):
    """Parses the complex evaluation filename to extract parameters."""
    if "unwatermarked" in filename:
        return None

    match = re.match(
        r"(\d+)_wm_delta_([\d.]+)_hc_(\d+)_et_([\d.]+)_([\w_]+)\.txt",
        filename
    )
    if match:
        prompt_id, delta, hc, et, _ = match.groups()
        return {
            'entropy_threshold': float(et),
        }
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Analyze all watermarked texts in an evaluation directory to visualize their block structures.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('eval_dir', type=str, help='Path to the evaluation directory containing generated texts.')
    parser.add_argument('--model', type=str, required=True, choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                        help='The model that was used to generate the text.')
    args = parser.parse_args()

    if not os.path.isdir(args.eval_dir):
        print(f"❌ Error: Directory not found at '{args.eval_dir}'")
        return

    # --- 1. Load Model ---
    print(f"Loading model '{args.model}'...")
    model = get_model(args.model)
    tokenizer = model.tokenizer

    # --- 2. Find all watermarked files ---
    all_files = os.listdir(args.eval_dir)
    watermarked_files = [f for f in all_files if "_wm_" in f and f.endswith('clean.txt')]
    
    if not watermarked_files:
        print(f"⚠️ No watermarked files ('_wm_...txt') found in '{args.eval_dir}'.")
        return
        
    print(f"Found {len(watermarked_files)} watermarked text files to process.")
    
    # --- 3. Prepare Output File ---
    output_filepath = os.path.join(args.eval_dir, "_block_visualization_all.txt")
    # Clear the file to start fresh
    with open(output_filepath, 'w') as f:
        f.write(f"Block Visualization for {len(watermarked_files)} files in '{args.eval_dir}'\n")

    # --- 4. Process Each File ---
    for filename in tqdm(watermarked_files, desc="Visualizing Blocks"):
        params = parse_filename(filename)
        if not params:
            print(f"\nSkipping file with unparsable name: {filename}")
            continue
        
        entropy_threshold = params['entropy_threshold']
        
        with open(os.path.join(args.eval_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            
        token_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)[0]
        
        # Skip very short texts
        if len(token_ids) < 2:
            continue

        with torch.no_grad():
            outputs = model._model(token_ids.unsqueeze(0))
        all_logits = outputs.logits.squeeze(0)

        block_indices = set()
        for i in range(len(token_ids) - 1):
            current_logits = all_logits[i, :]
            entropy = _calculate_entropy(current_logits)
            if entropy >= entropy_threshold:
                block_indices.add(i + 1)

        # Reconstruct text with separators
        output_parts = []
        for i, token_id in enumerate(token_ids):
            decoded_token = tokenizer.decode(token_id)
            if i in block_indices:
                output_parts.append("|" + decoded_token)
            else:
                output_parts.append(decoded_token)
        
        # Using join is faster and cleaner than repeated string concatenation
        visualized_text = "".join(output_parts)
        
        # Append result to the main output file
        with open(output_filepath, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"FILE: {filename}\n")
            f.write(f"PARAMETERS: entropy_threshold={entropy_threshold}\n")
            f.write(f"BLOCKS FOUND: {len(block_indices)}\n")
            f.write("="*80 + "\n\n")
            f.write(visualized_text)
            f.write("\n")

    print(f"\n✅ Processing complete. All visualizations have been appended to '{output_filepath}'.")

if __name__ == "__main__":
    main()