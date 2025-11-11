import argparse
import torch
import random
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models import GPT2Model, GptOssModel, GptOss120bModel
from watermark import *
from main import *
import re
import nltk

def generate_text(watermark: bool, prompt: str, max_new_tokens: int, key_file: str, gen_model: str, delta: float, entropy_threshold: float, hashing_context: int):
    # begin process
    print(f"Loading model '{gen_model}'...")
    model = get_model(gen_model)

    print(f"\nPrompt: '{prompt}'")

    #generate unwatermarked text
    if not watermark:
        final_text = generate_unwatermarked(model, prompt, max_new_tokens, gen_model)
        print("\n--- Generated Text ---")
        print(final_text)
        print("------------------------")
    #generate watermarked text
    else:
        watermarker = ZeroBitWatermarker(
            model=model, 
            delta=delta, 
            entropy_threshold=entropy_threshold, 
            hashing_context=hashing_context
        )
        secret_key = watermarker.keygen()
            
        print("\nEmbedding watermark...")
        # Unpack the text and the final parameters used
        raw_watermarked_text, final_params = watermarker.embed(secret_key, prompt, max_new_tokens)
            
        # Parse the raw output to get the final answer
        final_text = parse_final_output(raw_watermarked_text, gen_model)
            
        print("\n--- Final Watermarked Response ---")
        print(final_text)
        print("----------------------------------")
        print(f"(Final parameters used: {final_params})")
            
        with open(key_file, 'wb') as f:
            f.write(secret_key)
        print(f"🔑 Secret key saved to {key_file}")

    return final_text


def evaluate_model(prompts_file, max_new_tokens, eval_model, delta, entropy_threshold, hashing_context, z_threshold, output_dir, deltas, entropy_thresholds, hashing_contexts):
    print("--- Starting End-to-End Evaluation ---")
    os.makedirs(output_dir, exist_ok=True)
        
    # --- Stage 1: Setup ---
    model = get_model(eval_model)
    paraphraser_model, paraphraser_tokenizer = get_paraphraser(model.device)
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Found {len(prompts)} prompts. Using all for evaluation.")

    if deltas:
        param_to_sweep, sweep_values = 'delta', deltas
    elif hashing_contexts:
        param_to_sweep, sweep_values = 'hashing_context', hashing_contexts
    else:
        param_to_sweep, sweep_values = 'entropy_threshold', entropy_thresholds
        
    print(f"Sweeping parameter: '{param_to_sweep}' with values: {sweep_values}")
    key_map = {}

    # --- Stage 2: Generation and Perturbation ---
    for i, prompt in enumerate(tqdm(prompts, desc="Generating Texts")):
        unwatermarked_text = generate_unwatermarked(model, prompt, max_new_tokens, eval_model)
        with open(os.path.join(output_dir, f"{i}_unwatermarked.txt"), 'w', encoding='utf-8') as f:
            f.write(unwatermarked_text)
            
        for value in sweep_values:
            params = {
                'delta': float(value if param_to_sweep == 'delta' else delta),
                'hashing_context': int(value if param_to_sweep == 'hashing_context' else hashing_context),
                'entropy_threshold': float(value if param_to_sweep == 'entropy_threshold' else entropy_threshold)
            }

            watermarker = ZeroBitWatermarker(model=model, **params)
            secret_key = watermarker.keygen()
                
            # The embed function now returns the text AND the final params used
            raw_watermarked_text, final_params = watermarker.embed(secret_key, prompt, max_new_tokens)
                
            clean_final_text = parse_final_output(raw_watermarked_text, eval_model)
            perturbed_texts = perturb_text(clean_final_text, paraphraser_model, paraphraser_tokenizer, model.device)
                
            # Filename now uses the initial parameters from the embedding stage
            base_filename = f"{i}_wm_delta_{value if param_to_sweep == 'delta' else delta:.1f}_hc_{value if param_to_sweep == 'hashing_context' else hashing_context}_et_{value if param_to_sweep == 'entropy_threshold' else entropy_threshold:.1f}"

            for p_name, p_text in [('clean', clean_final_text)] + list(perturbed_texts.items()):
                filename = f"{base_filename}_{p_name.replace(' ', '_')}.txt"
                key_map[filename] = secret_key.hex()
                with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(p_text)

    key_map_path = os.path.join(output_dir, 'key_map.json')
    with open(key_map_path, 'w') as f:
        json.dump(key_map, f, indent=2)
    print(f"\n🔑 Generation complete. Key map saved to '{key_map_path}'")

    # --- Stage 3: Detection and Analysis ---
    print("\n--- Starting Detection & Analysis Stage ---")
    all_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    results = []

    for filename in tqdm(all_files, desc="Detecting Watermarks"):
        parsed_info = parse_filename(filename)
        if not parsed_info:
            print(f"\nWarning: Could not parse filename '{filename}'. Skipping.")
            continue

        with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            
        result_entry = {'filename': filename, **parsed_info}
            
        if parsed_info['type'] == 'unwatermarked':
            watermarker = ZeroBitWatermarker(model=model, z_threshold=z_threshold)
            random_key = watermarker.keygen()

            z_score, _, block_count = watermarker.detect(random_key, text)
                
            result_entry['final_entropy_threshold'] = watermarker.entropy_threshold

        else: # Watermarked file
            secret_key_hex = key_map.get(filename)
            if not secret_key_hex:
                print(f"Error: Key not found {filename}")
                z_score, block_count = 0, 0
            else:
                secret_key = bytes.fromhex(secret_key_hex)
                # --- NEW TWO-PASS DETECTION LOGIC (Mirrors 'detect' command) ---
                # 1. First pass with original parameters from filename
                wm_params_pass1 = {k: v for k, v in parsed_info.items() if k in ['delta', 'hashing_context', 'entropy_threshold']}
                watermarker_pass1 = ZeroBitWatermarker(model=model, z_threshold=z_threshold, **wm_params_pass1)
                z_score, is_detected, block_count = watermarker_pass1.detect(secret_key, text)
                    
                final_et = wm_params_pass1['entropy_threshold']

                # 2. If block count is low AND it wasn't detected, try a more aggressive pass
                if block_count < 75 and not is_detected:
                    pass_2_params = wm_params_pass1.copy()
                    pass_2_params['entropy_threshold'] -= 2.0

                    # Check if the new parameters are within the valid range
                    if pass_2_params['entropy_threshold'] >= 1.0:
                        watermarker_pass2 = ZeroBitWatermarker(model=model, z_threshold=z_threshold, **pass_2_params)
                        # Overwrite results with the second pass
                        z_score, is_detected, block_count = watermarker_pass2.detect(secret_key, text)
                        final_et = pass_2_params['entropy_threshold']
                
            result_entry['final_entropy_threshold'] = final_et
                                
        result_entry['z_score'] = z_score
        result_entry['final_block_count'] = block_count
        results.append(result_entry)

    results_path = os.path.join(output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
            
    print(f"\n✅ Evaluation complete. Final analysis summary saved to '{results_path}'.")
