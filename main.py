# main.py; Holds the main function for the program and the functions allowing it to work as a CLI

import argparse
import torch
import random
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models import GPT2Model, GptOssModel, GptOss120bModel
#from watermark import ZeroBitWatermarker
#from watermark import LBitWatermarker
from watermark import *
import re
import nltk

# --- Argparse Type Checkers for Parameter Constraints ---
def check_delta_range(value):
    """Checks if a single delta value is between 1.0 and 5.0."""
    try:
        f_value = float(value)
        if 1.0 <= f_value <= 5.0:
            return f_value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"'{value}' is an invalid delta. Must be a float between 1.0 and 5.0.")

def check_entropy_range(value):
    """Checks if a single entropy_threshold is between 1.0 and 6.0."""
    try:
        f_value = float(value)
        if 1.0 <= f_value <= 6.0:
            return f_value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"'{value}' is an invalid entropy threshold. Must be a float between 1.0 and 6.0.")

def check_list_of(checker_func):
    """Factory to create a checker for comma-separated lists."""
    def _check_list(value_str):
        try:
            # Split the string by comma and apply the single value checker to each part
            return [checker_func(v.strip()) for v in value_str.split(',')]
        except argparse.ArgumentTypeError as e:
            # Re-raise the error from the checker function for a clear message
            raise argparse.ArgumentTypeError(f"Invalid value in list '{value_str}': {e}")
        except Exception:
            raise argparse.ArgumentTypeError(f"Could not parse '{value_str}' as a comma-separated list of numbers.")

    return _check_list

def setup_nltk():
    """Checks for the NLTK 'punkt' tokenizer model"""
    nltk_data_path = "/fred/oz413/rburke/huggingface"
    
    # Add your custom path to NLTK's search paths
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    # Now, verify that the package can be found without triggering a download
    try:
        nltk.data.find('tokenizers/punkt')
        
    except LookupError:
        print("❌ ERROR: NLTK 'punkt' package not found in the specified path.")
        print(f"Please run 'download_models_hpc.py' on a login node first.")
        # Exit if the required data is not found
        exit(1)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
        
    except LookupError:
        print("❌ ERROR: NLTK 'punkt_tab' package not found in the specified path.")
        print(f"Please run 'download_models_hpc.py' on a login node first.")
        # Exit if the required data is not found
        exit(1)

# --- New Helper Function for Parsing ---
def parse_final_output(raw_text: str, model_name: str) -> str:
    """
    Extracts the content from the 'final' channel of a gpt-oss output.
    If the model is not gpt-oss or the separator isn't found, it returns the original text.
    """
    if 'gpt-oss' in model_name:
        separator = "assistantfinal"
        if separator in raw_text:
            # Return the part of the string that comes after the last occurrence of the separator
            return raw_text.split(separator)[-1].strip()
    
    # For non-gpt-oss models or if separator is missing, return the full text
    return raw_text

# --- Helper Functions for the 'evaluate' command ---
def get_paraphraser(device):
    """Loads a T5 model for the substitution/paraphrasing attack."""
    model_id = "Vamsi/T5_Paraphrase_Paws"
    print(f"Loading paraphrasing model ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    return model, tokenizer

def perturb_text(text: str, paraphraser_model, paraphraser_tokenizer, device) -> dict:
    """
    Applies stronger, more realistic perturbations to a text to test robustness.
    This version is more robust and guarantees all attack types are returned.
    """
    perturbed = {}
    
    # Use NLTK for robust sentence splitting
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        print(f"Warning: NLTK sentence tokenization failed. Error: {e}. Falling back to simple split.")
        sentences = text.split('.') # Fallback
        
    num_sentences = len(sentences)

    # --- 1. Deletion Attacks ---
    num_to_delete = max(1, int(num_sentences * 0.20))
    
    # a) Delete from the start
    if num_sentences > num_to_delete:
        perturbed['delete_start_20_percent'] = " ".join(sentences[num_to_delete:])
    else:
        perturbed['delete_start_20_percent'] = text # Not enough sentences to delete, return original

    # b) Delete from the end
    if num_sentences > num_to_delete:
        perturbed['delete_end_20_percent'] = " ".join(sentences[:-num_to_delete])
    else:
        perturbed['delete_end_20_percent'] = text

    # c) Delete from the middle
    if num_sentences > num_to_delete * 2: # Need enough sentences to see a middle
        mid_index = num_sentences // 2
        start_del = max(0, mid_index - num_to_delete // 2)
        end_del = start_del + num_to_delete
        middle_deleted_sentences = sentences[:start_del] + sentences[end_del:]
        perturbed['delete_middle_20_percent'] = " ".join(middle_deleted_sentences)
    else:
        perturbed['delete_middle_20_percent'] = text

    # --- 2. Substitution Attack ---
    if num_sentences > 0:
        num_to_paraphrase = max(1, int(num_sentences * 0.30))
        indices_to_paraphrase = random.sample(range(num_sentences), min(num_to_paraphrase, num_sentences))
        
        substituted_sentences = list(sentences)
        for i in indices_to_paraphrase:
            sentence_to_paraphrase = sentences[i].strip()
            if not sentence_to_paraphrase: continue
            
            if len(sentence_to_paraphrase.split()) > 400:
                print(f"\nWarning: Skipping paraphrase for a sentence with >400 words.")
                continue
            
            try:
                # Use the format required by the fine-tuned paraphrasing model
                input_text = f"paraphrase: {sentence_to_paraphrase} </s>"
                input_ids = paraphraser_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

                with torch.no_grad():
                    outputs = paraphraser_model.generate(
                        input_ids, 
                        max_length=int(len(input_ids[0]) * 2.0), # Allow longer paraphrases
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=1.2, # Increase creativity
                        num_return_sequences=3 # Generate 3 different options
                    )
                
                decoded_outputs = [paraphraser_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                paraphrased_sentence = random.choice(decoded_outputs)
                substituted_sentences[i] = paraphrased_sentence

            except Exception as e:
                print(f"\nWarning: Paraphrasing sentence failed. Skipping. Error: {e}")
                
        perturbed['substitute_30_percent'] = " ".join(substituted_sentences)
    else:
        print("Warning: No sentences found")
        perturbed['substitute_30_percent'] = text

    return perturbed

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
        # The perturbation name might have multiple parts, join them back
        perturbation = perturbation_full.replace('_', ' ')
        
        return {
            'type': 'watermarked',
            'prompt_id': int(prompt_id),
            'delta': float(delta),
            'hashing_context': int(hc),
            'entropy_threshold': float(et),
            'perturbation': perturbation
        }
    match = re.match(
        r"(\d+)_lbit_delta_([\d.]+)_hc_(\d+)_et_([\d.]+)_([\w_]+)\.txt",
        filename
    )
    if match:
        prompt_id, delta, hc, et, perturbation_full = match.groups()
        perturbation = perturbation_full.replace('_', ' ')
        return {
            'type': 'lbit',
            'prompt_id': int(prompt_id),
            'delta': float(delta),
            'hashing_context': int(hc),
            'entropy_threshold': float(et),
            'perturbation': perturbation
        }
    
    return None

def get_model(model_name: str):
    """Function to instantiate the correct model based on its name."""
    if model_name == 'gpt2':
        return GPT2Model()
    elif model_name == 'gpt-oss-20b':
        return GptOssModel()
    elif model_name == 'gpt-oss-120b':
        return GptOss120bModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def generate_unwatermarked(model_wrapper, prompt: str, max_new_tokens: int, model_name: str):
    """Helper to generate text without a watermark for comparison."""
    tokenizer = model_wrapper.tokenizer
    device = model_wrapper.device
    
    # Conditional text formatting based on base vs chat model
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
        ).to(device)
    else:
        token_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output_ids = model_wrapper._model.generate(
            token_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_k=50, top_p=0.95, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Parse the output before returning
    return parse_final_output(raw_output, model_name)

def main():
    setup_nltk()
    parser = argparse.ArgumentParser(description="Zero-Bit Watermarking Tool for LLMs.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Generate Command ---
    parser_gen = subparsers.add_parser('generate', help='Generate and watermark text.')
    parser_gen.add_argument('prompt', type=str, help='The prompt to send to the model.')
    parser_gen.add_argument('--model', type=str, default='gpt2',
                            choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                            help='The model to use for generation.')
    parser_gen.add_argument('--max-new-tokens', type=int, default=2048, help='Maximum number of new tokens to generate.')
    parser_gen.add_argument('--output-file', '-o', type=str, default='watermarked_output.txt',
                            help='File to save the generated text.')
    parser_gen.add_argument('--key-file', '-k', type=str, default='secret.key',
                            help='File to save the secret key.')
    parser_gen.add_argument('--no-watermark', action='store_true', help='Generate plain text without a watermark.')
    parser_gen.add_argument('--hashing-context', type=int, default=5, help='The number of previous tokens to use for the PRF context.')
    parser_gen.add_argument('--delta', type=check_delta_range, default=2.5, help='Watermark bias strength (1.0 to 5.0).')
    parser_gen.add_argument('--entropy-threshold', type=check_entropy_range, default=4.0, help='Entropy threshold to define a block (1.0 to 6.0).')

    # --- Detect Command ---
    parser_detect = subparsers.add_parser('detect', help='Detect a watermark in a text file.')
    parser_detect.add_argument('input_file', type=str, help='Path to the text file to check.')
    parser_detect.add_argument('--model', type=str, required=True,
                                choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                                help='The model that was used to generate the text (for tokenizer matching).')
    parser_detect.add_argument('--key-file', '-k', type=str, default='secret.key',
                                help='Path to the secret key file.')
    parser_detect.add_argument('--z-threshold', type=float, default=4.0,
                                help='The z-score threshold for detection.')
    parser_detect.add_argument('--entropy-threshold', type=check_entropy_range, default=4.0, help='Entropy threshold used during generation (1.0 to 6.0).')
    parser_detect.add_argument('--hashing-context', type=int, default=5, help='The hashing context used during generation.')
    
    # --- Evaluate Command ---
    parser_eval = subparsers.add_parser('evaluate', help='Run a full evaluation of the watermarking framework.')
    parser_eval.add_argument('--prompts-file', type=str, required=True, help='Path to a .txt file with one prompt per line.')
    parser_eval.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'], help='Model to use for generation.')
    parser_eval.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save all generated texts and results.')
    parser_eval.add_argument('--max-new-tokens', type=int, default=2048, help='Tokens to generate for each prompt.')
    parser_eval.add_argument('--z-threshold', type=float, default=4.0, help='The z-score threshold for detection.')
    parser_eval.add_argument('--l-bit-message', type=str, help='L-bit binary message to embed (e.g., "0101...").')
    parser_eval.add_argument('--l-bits', type=int, default=32, help='Number of bits for the L-bit message.')

    # --- Generate L-Bit Command ---
    parser_lgen = subparsers.add_parser('generate_lbit', help='Generate and watermark with L-bit message.')
    parser_lgen.add_argument('prompt', type=str, help='The prompt to send to the model.')
    parser_lgen.add_argument('--message', type=str, required=True, help='L-bit binary message (e.g., "0101...").')
    parser_lgen.add_argument('--l-bits', type=int, default=32, help='Number of bits for the message.')
    parser_lgen.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                            help='The model to use for generation.')
    parser_lgen.add_argument('--max-new-tokens', type=int, default=256, help='Maximum number of new tokens to generate.')
    parser_lgen.add_argument('--output-file', type=str, default='watermarked_lbit.txt', help='File to save the generated text.')
    parser_lgen.add_argument('--key-file', type=str, default='secret.key', help='File to save the secret key.')
    parser_lgen.add_argument('--delta', type=check_delta_range, default=2.5, help='Watermark bias strength (1.0 to 5.0).')
    parser_lgen.add_argument('--entropy-threshold', type=check_entropy_range, default=3.5, help='Entropy threshold to define a block (1.0 to 6.0).')
    parser_lgen.add_argument('--hashing-context', type=int, default=5, help='The number of previous tokens to use for the PRF context.')
    parser_lgen.add_argument('--z-threshold', type=float, default=4.0, help='The z-score threshold for detection.')

    # --- Detect L-Bit Command ---
    parser_ldet = subparsers.add_parser('detect_lbit', help='Detect L-bit message in text.')
    parser_ldet.add_argument('input_file', type=str, help='Path to the text file to check.')
    parser_ldet.add_argument('--l-bits', type=int, default=32, help='Number of bits for the message.')
    parser_ldet.add_argument('--model', type=str, required=True, choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                            help='The model that was used to generate the text (for tokenizer matching).')
    parser_ldet.add_argument('--key-file', type=str, default='secret.key', help='Path to the secret key file.')
    parser_ldet.add_argument('--delta', type=check_delta_range, default=2.5, help='Watermark bias strength used during generation (1.0 to 5.0).')
    parser_ldet.add_argument('--entropy-threshold', type=check_entropy_range, default=3.5, help='Entropy threshold used during generation (1.0 to 6.0).')
    parser_ldet.add_argument('--hashing-context', type=int, default=5, help='The hashing context used during generation.')
    parser_ldet.add_argument('--z-threshold', type=float, default=4.0, help='The z-score threshold for detection.')

    # Arguments for parameter sweeps
    parser_eval.add_argument('--delta', type=check_delta_range, default=2.5, help='A single delta value (1.0 to 5.0).')
    parser_eval.add_argument('--entropy-threshold', type=check_entropy_range, default=4.0, help='A single entropy threshold value (1.0 to 6.0).')
    parser_eval.add_argument('--hashing-context', type=int, default=5, help='A single hashing context value.')

    # Mutually exclusive group for list-style arguments
    sweep_group = parser_eval.add_mutually_exclusive_group(required=True)
    sweep_group.add_argument('--deltas', type=check_list_of(check_delta_range), help='Comma-separated list of delta values to sweep (each between 1.0-5.0).')
    sweep_group.add_argument('--hashing-contexts', type=lambda s: [int(v) for v in s.split(',')], help='Comma-separated list of hashing contexts to sweep.')
    sweep_group.add_argument('--entropy-thresholds', type=check_list_of(check_entropy_range), help='Comma-separated list of entropy thresholds to sweep (each between 1.0-6.0).')

    args = parser.parse_args()

    # --- Command Logic ---
    if args.command == 'generate':
        print(f"Loading model '{args.model}'...")
        model = get_model(args.model)
        
        print(f"\nPrompt: '{args.prompt}'")
        
        if args.no_watermark:
            final_text = generate_unwatermarked(model, args.prompt, args.max_new_tokens, args.model)
            print("\n--- Generated Text ---")
            print(final_text)
            print("------------------------")
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(final_text)
            print(f"\n✅ Output saved to {args.output_file}")
            
        else:
            watermarker = ZeroBitWatermarker(
                model=model, 
                delta=args.delta, 
                entropy_threshold=args.entropy_threshold, 
                hashing_context=args.hashing_context
            )
            secret_key = watermarker.keygen()
            
            print("\nEmbedding watermark...")
            # Unpack the text and the final parameters used
            raw_watermarked_text, final_params = watermarker.embed(secret_key, args.prompt, args.max_new_tokens)
            
            # Parse the raw output to get the final answer
            final_text = parse_final_output(raw_watermarked_text, args.model)
            
            print("\n--- Final Watermarked Response ---")
            print(final_text)
            print("----------------------------------")
            print(f"(Final parameters used: {final_params})")
            
            # Save the parsed output and key
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(final_text)
            print(f"\n✅ Watermarked text saved to {args.output_file}")
            
            with open(args.key_file, 'wb') as f:
                f.write(secret_key)
            print(f"🔑 Secret key saved to {args.key_file}")
            
    elif args.command == 'detect':
        print(f"Loading model '{args.model}' for tokenizer...")
        model = get_model(args.model)
        
        try:
            with open(args.key_file, 'rb') as f:
                secret_key = f.read()
            print(f"🔑 Loaded secret key from {args.key_file}")

            with open(args.input_file, 'r', encoding='utf-8') as f:
                text_to_check = f.read()
            print(f"📄 Loaded text from {args.input_file}")

        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file {e.filename}")
            return
            
        print("\nRunning detection algorithm...")

        pass_1_params = {
            'delta': 3.0,
            'hashing_context': args.hashing_context,
            'entropy_threshold': args.entropy_threshold
        }
        watermarker_pass1 = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **pass_1_params)
        z_score, is_detected, block_count = watermarker_pass1.detect(secret_key, text_to_check)
        
        final_params = pass_1_params

        if block_count < 75 and not is_detected:
            print(f"⚠️ Initial block count ({block_count}) is low. Running a more aggressive second pass...")
            
            pass_2_params = pass_1_params.copy()
            pass_2_params['entropy_threshold'] -= 2.0

            # Check if the new parameters are within the valid ranges
            if pass_2_params['entropy_threshold'] >= 1.0:
                watermarker_pass2 = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **pass_2_params)
                # Overwrite the results with the second pass
                z_score, is_detected, block_count = watermarker_pass2.detect(secret_key, text_to_check)
                final_params = pass_2_params

        print("\n--- Detection Results ---")
        print(f"  Z-Score: {z_score:.4f}")
        print(f"  Threshold: {args.z_threshold}")
        print(f"  Detected: {'✅ Yes' if is_detected else '❌ No'}")
        print(f"  Blocks Found: {block_count}")
        print(f"  Final Params Used: {final_params}")
        print("-------------------------")

    elif args.command == 'generate_lbit':
        model = get_model(args.model)

        zbw = ZeroBitWatermarker(
            model=model,
            delta=args.delta,
            entropy_threshold=args.entropy_threshold,
            hashing_context=args.hashing_context,
            z_threshold=args.z_threshold
        )
        lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zbw, L=args.l_bits)

        print(f"\nGenerating key for an {args.l_bits}-bit message...")
        master_secret_key = lbit_watermarker.keygen()  # Single key
        print(f"Message to embed: {args.message}")

        print(f"\nPrompt: '{args.prompt}'")
        print("\nEmbedding message into generated text...")
        watermarked_text = lbit_watermarker.embed(master_secret_key, args.message, args.prompt, args.max_new_tokens)

        print("\n--- Watermarked Text ---")
        print(watermarked_text)
        print("------------------------")

        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(watermarked_text)

        with open(args.key_file, 'w') as f:
            json.dump(master_secret_key.hex(), f)  # Save single key as hex
        print(f"Watermarked L-bit text saved to {args.output_file}")
        print(f"🔑 Secret key saved to {args.key_file}")

    elif args.command == 'detect_lbit':
        model = get_model(args.model)
        zbw = ZeroBitWatermarker(
            model=model,
            delta=args.delta,
            entropy_threshold=args.entropy_threshold,
            hashing_context=args.hashing_context,
            z_threshold=args.z_threshold
        )
        lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zbw, L=args.l_bits)

        try:
            with open(args.key_file, 'r') as f:
                secret_key_hex = json.load(f)
            print(f"Loaded secret key from {args.key_file}")

            master_secret_key = bytes.fromhex(secret_key_hex)  # Single key

            with open(args.input_file, 'r', encoding='utf-8') as f:
                text_to_check = f.read()
            print(f"Loaded text from {args.input_file}")

        except FileNotFoundError as e:
            print(f"Error: Could not find file {e.filename}")
            return
        except (json.JSONDecodeError, ValueError):
            print(f"Error: Could not parse the key file '{args.key_file}'. It may be corrupted or in the wrong format.")
            return

        print(f"\nExtracting {args.l_bits}-bit message...")
        recovered_message = lbit_watermarker.detect(master_secret_key, text_to_check)

        print("\n--- Extraction Results ---")
        print(f"  Recovered L-bit message: {recovered_message}")
        print("--------------------------")

    elif args.command == 'evaluate':
        print("--- Starting End-to-End Evaluation ---")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # --- Stage 1: Setup ---
        model = get_model(args.model)
        paraphraser_model, paraphraser_tokenizer = get_paraphraser(model.device)
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Found {len(prompts)} prompts. Using all for evaluation.")

        if args.deltas:
            param_to_sweep, sweep_values = 'delta', args.deltas
        elif args.hashing_contexts:
            param_to_sweep, sweep_values = 'hashing_context', args.hashing_contexts
        else:
            param_to_sweep, sweep_values = 'entropy_threshold', args.entropy_thresholds
        
        print(f"Sweeping parameter: '{param_to_sweep}' with values: {sweep_values}")
        key_map = {}

        # --- Stage 2: Generation and Perturbation ---
        for i, prompt in enumerate(tqdm(prompts, desc="Generating Texts")):
            unwatermarked_text = generate_unwatermarked(model, prompt, args.max_new_tokens, args.model)
            with open(os.path.join(args.output_dir, f"{i}_unwatermarked.txt"), 'w', encoding='utf-8') as f:
                f.write(unwatermarked_text)
            
            for value in sweep_values:
                params = {
                    'delta': value if param_to_sweep == 'delta' else args.delta,
                    'hashing_context': value if param_to_sweep == 'hashing_context' else args.hashing_context,
                    'entropy_threshold': value if param_to_sweep == 'entropy_threshold' else args.entropy_threshold
                }
                watermarker = ZeroBitWatermarker(model=model, **params)
                secret_key = watermarker.keygen()
                
                # The embed function now returns the text AND the final params used
                raw_watermarked_text, final_params = watermarker.embed(secret_key, prompt, args.max_new_tokens)
                
                clean_final_text = parse_final_output(raw_watermarked_text, args.model)
                perturbed_texts = perturb_text(clean_final_text, paraphraser_model, paraphraser_tokenizer, model.device)
                
                # Filename now uses the initial parameters from the embedding stage
                base_filename = f"{i}_wm_delta_{value if param_to_sweep == 'delta' else args.delta:.1f}_hc_{value if param_to_sweep == 'hashing_context' else args.hashing_context}_et_{value if param_to_sweep == 'entropy_threshold' else args.entropy_threshold:.1f}"

                for p_name, p_text in [('clean', clean_final_text)] + list(perturbed_texts.items()):
                    filename = f"{base_filename}_{p_name.replace(' ', '_')}.txt"
                    key_map[filename] = secret_key.hex()
                    with open(os.path.join(args.output_dir, filename), 'w', encoding='utf-8') as f:
                        f.write(p_text)
                
                if args.l_bit_message:
                    if len(args.l_bit_message) != args.l_bits or any(c not in {'0', '1'} for c in args.l_bit_message):
                        print(f"Error: L-bit message must be a {args.l_bits}-bit string")
                        continue
                    lbit_watermarker = LBitWatermarker(zero_bit_watermarker=watermarker, L=args.l_bits)
                    master_secret_key = lbit_watermarker.keygen()
                    lbit_text = lbit_watermarker.embed(master_secret_key, args.l_bit_message, prompt, args.max_new_tokens)
                    lbit_clean_text = parse_final_output(lbit_text, args.model)
                    lbit_perturbed_texts = perturb_text(lbit_clean_text, paraphraser_model, paraphraser_tokenizer, model.device)
                    lbit_base_filename = f"{i}_lbit_delta_{params['delta']:.1f}_hc_{params['hashing_context']}_et_{params['entropy_threshold']:.1f}"
                    for p_name, p_text in [('clean', lbit_clean_text)] + list(lbit_perturbed_texts.items()):
                        filename = f"{lbit_base_filename}_{p_name.replace(' ', '_')}.txt"
                        key_map[filename] = master_secret_key.hex()
                        with open(os.path.join(args.output_dir, filename), 'w', encoding='utf-8') as f:
                            f.write(p_text)

        key_map_path = os.path.join(args.output_dir, 'key_map.json')
        with open(key_map_path, 'w') as f:
            json.dump(key_map, f, indent=2)
        print(f"\n🔑 Generation complete. Key map saved to '{key_map_path}'")

        # --- Stage 3: Detection and Analysis ---
        print("\n--- Starting Detection & Analysis Stage ---")
        all_files = [f for f in os.listdir(args.output_dir) if f.endswith('.txt')]
        results = []

        for filename in tqdm(all_files, desc="Detecting Watermarks"):
            parsed_info = parse_filename(filename)
            if not parsed_info:
                print(f"\nWarning: Could not parse filename '{filename}'. Skipping.")
                continue

            with open(os.path.join(args.output_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            
            result_entry = {'filename': filename, **parsed_info}
            
            if parsed_info['type'] == 'unwatermarked':
                watermarker = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold)
                random_key = watermarker.keygen()

                z_score, _, block_count = watermarker.detect(random_key, text)
                
                result_entry['final_entropy_threshold'] = watermarker.entropy_threshold
                results.append(result_entry) ############################################

            elif parsed_info['type'] == 'watermarked': # Watermarked file
                secret_key_hex = key_map.get(filename)
                if not secret_key_hex:
                    print(f"Error: Key not found {filename}")
                    z_score, block_count = 0, 0
                else:
                    secret_key = bytes.fromhex(secret_key_hex)
                    # --- NEW TWO-PASS DETECTION LOGIC (Mirrors 'detect' command) ---
                    # 1. First pass with original parameters from filename
                    wm_params_pass1 = {k: v for k, v in parsed_info.items() if k in ['delta', 'hashing_context', 'entropy_threshold']}
                    watermarker_pass1 = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **wm_params_pass1)
                    z_score, is_detected, block_count = watermarker_pass1.detect(secret_key, text)
                    
                    final_et = wm_params_pass1['entropy_threshold']

                    # 2. If block count is low AND it wasn't detected, try a more aggressive pass
                    if block_count < 75 and not is_detected:
                        pass_2_params = wm_params_pass1.copy()
                        pass_2_params['entropy_threshold'] -= 2.0

                        # Check if the new parameters are within the valid range
                        if pass_2_params['entropy_threshold'] >= 1.0:
                            watermarker_pass2 = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **pass_2_params)
                            # Overwrite results with the second pass
                            z_score, is_detected, block_count = watermarker_pass2.detect(secret_key, text)
                            final_et = pass_2_params['entropy_threshold']
                
                result_entry['final_entropy_threshold'] = final_et
            elif parsed_info['type'] == 'lbit':
                secret_key_hex = key_map.get(filename)
                if not secret_key_hex:
                    print(f"Error: Key not found for {filename}")
                    result_entry['recovered_message'] = ""
                    result_entry['recovery_accuracy'] = 0.0
                    result_entry['z_score'] = None
                    result_entry['final_block_count'] = None
                    result_entry['final_entropy_threshold'] = parsed_info['entropy_threshold']
                else:
                    master_secret_key = bytes.fromhex(secret_key_hex)
                    lbit_params = {k: v for k, v in parsed_info.items() if k in ['delta', 'hashing_context', 'entropy_threshold']}
                    watermarker = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **lbit_params)
                    lbit_watermarker = LBitWatermarker(model=model, L=args.l_bits, z_threshold=args.z_threshold, **lbit_params)
                    recovered_message = lbit_watermarker.detect(master_secret_key, text)
                    result_entry['recovered_message'] = recovered_message
                    result_entry['final_entropy_threshold'] = lbit_params['entropy_threshold']
                    z_scores = []
                    block_counts = []
                    for i in range(1, args.l_bits + 1):
                        z_i0, _, bc0 = lbit_watermarker.zero_bit.detect(derive_key(master_secret_key, i, 0), text)
                        z_i1, _, bc1 = lbit_watermarker.zero_bit.detect(derive_key(master_secret_key, i, 1), text)
                        z_scores.append((z_i0, z_i1))
                        block_counts.append(max(bc0, bc1))
                    if block_counts and min(block_counts) < 75:
                        pass_2_params = lbit_params.copy()
                        pass_2_params['entropy_threshold'] -= 2.0
                        if pass_2_params['entropy_threshold'] >= 1.0:
                            lbit_watermarker_pass2 = LBitWatermarker(model=model, L=args.l_bits, z_threshold=args.z_threshold, **pass_2_params)
                            recovered_message = lbit_watermarker_pass2.detect(master_secret_key, text)
                            result_entry['recovered_message'] = recovered_message
                            result_entry['final_entropy_threshold'] = pass_2_params['entropy_threshold']
                            z_scores = []
                            block_counts = []
                            for i in range(1, args.l_bits + 1):
                                z_i0, _, bc0 = lbit_watermarker_pass2.zero_bit.detect(derive_key(master_secret_key, i, 0), text)
                                z_i1, _, bc1 = lbit_watermarker_pass2.zero_bit.detect(derive_key(master_secret_key, i, 1), text)
                                z_scores.append((z_i0, z_i1))
                                block_counts.append(max(bc0, bc1))
                    result_entry['z_score'] = z_scores
                    result_entry['final_block_count'] = block_counts
                    if args.l_bit_message:
                        correct_bits = sum(a == b for a, b in zip(args.l_bit_message, recovered_message) if b not in {str(None), '?'})
                        total_bits = len(args.l_bit_message)
                        result_entry['recovery_accuracy'] = (correct_bits / total_bits) * 100 if total_bits > 0 else 0.0
                    else:
                        result_entry['recovery_accuracy'] = 0.0

        results_path = os.path.join(args.output_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Evaluation complete. Final analysis summary saved to '{results_path}'.")
        print("You can now run 'analyse.py' on this directory to generate plots.")

if __name__ == "__main__":
    main()