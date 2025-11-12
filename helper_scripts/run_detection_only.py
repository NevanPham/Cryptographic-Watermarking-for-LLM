# run_detection_only.py

import argparse
import json
import os
import re
from tqdm import tqdm
import pandas as pd

import os
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
        try:
            prompt_id = int(filename.split('_')[0])
            return {'type': 'unwatermarked', 'prompt_id': prompt_id}
        except (ValueError, IndexError):
            return None

    match = re.match(
        r"(\d+)_wm_delta_([\d.]+)_hc_(\d+)_et_([\d.]+)_([\w_]+)\.txt",
        filename
    )
    if match:
        prompt_id, delta, hc, et, perturbation_full = match.groups()
        perturbation = perturbation_full.replace('_', ' ')
        
        return {
            'type': 'watermarked',
            'prompt_id': int(prompt_id),
            'delta': float(delta),
            'hashing_context': int(hc),
            'entropy_threshold': float(et),
            'perturbation': perturbation
        }
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Run the detection and analysis phase on an existing evaluation directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('eval_dir', type=str, help='Path to the evaluation directory containing generated texts and key_map.json.')
    parser.add_argument('--model', type=str, required=True, choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                        help='The model that was used for generation (important for the tokenizer).')
    parser.add_argument('--z-threshold', type=float, default=4.0, help='The z-score threshold for defining detection success (does not affect calculation).')
    
    args = parser.parse_args()

    print("--- Starting Standalone Detection & Analysis Job (with Two-Pass Logic) ---")
    
    # --- 1. Load Model & Key Map ---
    print(f"Loading model '{args.model}'...")
    model = get_model(args.model)
    
    key_map_path = os.path.join(args.eval_dir, 'key_map.json')
    if not os.path.exists(key_map_path):
        print(f"❌ Error: 'key_map.json' not found in '{args.eval_dir}'. Cannot run detection.")
        return
        
    with open(key_map_path, 'r') as f:
        loaded_key_map = json.load(f)
    print(f"🔑 Loaded key map from '{key_map_path}'")

    # --- 2. Run Detection on all .txt files ---
    all_files = [f for f in os.listdir(args.eval_dir) if f.endswith('.txt')]
    results = []

    print(f"\nFound {len(all_files)} text files to analyze. Starting detection...")
    for filename in tqdm(all_files, desc="Detecting Watermarks"):
        parsed_info = parse_filename(filename)
        if not parsed_info:
            continue

        with open(os.path.join(args.eval_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
        
        result_entry = {'filename': filename, **parsed_info}
        
        if parsed_info['type'] == 'unwatermarked':
            watermarker = ZeroBitWatermarker(model=model)
            random_key = watermarker.keygen()
            z_score, _, block_count = watermarker.detect(random_key, text)
            result_entry['final_entropy_threshold'] = watermarker.entropy_threshold
        
        else: # It's a watermarked file
            secret_key_hex = loaded_key_map.get(filename)
            if not secret_key_hex:
                z_score, block_count, final_et = "Error: Key not found", 0, parsed_info.get('entropy_threshold')
            else:
                secret_key = bytes.fromhex(secret_key_hex)
                
                wm_params = {k: v for k, v in parsed_info.items() if k in ['delta', 'hashing_context', 'entropy_threshold']}
                
                # Detect with parameters from the filename
                watermarker_pass1 = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **wm_params)
                z_score, _, block_count = watermarker_pass1.detect(secret_key, text)
                        
        result_entry['z_score'] = z_score
        result_entry['final_block_count'] = block_count # This now reflects the final count after a potential second pass
        results.append(result_entry)

    # --- 3. Save Final Analysis File ---
    results_path = os.path.join(args.eval_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Detection complete. Final analysis summary saved to '{results_path}'.")
    print("You can now run 'analyze.py' on this directory to generate plots.")

if __name__ == "__main__":
    main()