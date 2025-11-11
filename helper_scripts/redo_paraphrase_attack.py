# redo_paraphrase_attack.py

import argparse
import os
import random
import nltk
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

def get_paraphraser(device):
    """Loads the fine-tuned T5 model for paraphrasing."""
    model_id = "Vamsi/T5_Paraphrase_Paws"
    print(f"Loading paraphrasing model ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    return model, tokenizer

def perform_substitution_attack(text: str, paraphraser_model, paraphraser_tokenizer, device) -> str:
    """
    Performs the substitution attack on a single text.
    """
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        print(f"Warning: NLTK sentence tokenization failed. Error: {e}. Falling back to simple split.")
        sentences = text.split('.')
        
    num_sentences = len(sentences)

    if num_sentences < 1:
        return text # Return original text if there's nothing to paraphrase

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
            input_text = f"paraphrase: {sentence_to_paraphrase} </s>"
            input_ids = paraphraser_tokenizer.encode(
                input_text, return_tensors="pt", max_length=512, truncation=True
            ).to(device)
            
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
            
    return " ".join(substituted_sentences)

def main():
    parser = argparse.ArgumentParser(description="Re-run the paraphrasing attack on all clean watermarked texts in an evaluation folder.")
    parser.add_argument('eval_dir', type=str, help='Path to the evaluation directory containing the generated text files.')
    args = parser.parse_args()

    if not os.path.isdir(args.eval_dir):
        print(f"❌ Error: Directory not found at '{args.eval_dir}'")
        return

    # --- 1. Setup ---
    setup_nltk()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    paraphraser_model, paraphraser_tokenizer = get_paraphraser(device)

    # --- 2. Find Target Files ---
    all_files = os.listdir(args.eval_dir)
    clean_files = [f for f in all_files if "_wm_" in f and f.endswith("_clean.txt")]
    
    if not clean_files:
        print(f"⚠️ No clean watermarked files ('..._clean.txt') found in '{args.eval_dir}'.")
        return
        
    print(f"\nFound {len(clean_files)} clean watermarked files to attack.")

    # --- 3. Process Each File ---
    for filename in tqdm(clean_files, desc="Applying Paraphrase Attack"):
        # Read the clean text
        clean_filepath = os.path.join(args.eval_dir, filename)
        with open(clean_filepath, 'r', encoding='utf-8') as f:
            clean_text = f.read()

        # Perform the new substitution attack
        substituted_text = perform_substitution_attack(clean_text, paraphraser_model, paraphraser_tokenizer, device)

        # Construct the output filename and overwrite the old file
        output_filename = filename.replace("_clean.txt", "_substitute_30_percent.txt")
        output_filepath = os.path.join(args.eval_dir, output_filename)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(substituted_text)
    
    print(f"\n✅ Processing complete. All {len(clean_files)} substitution attack files have been regenerated in '{args.eval_dir}'.")
    print("You can now re-run the detection job.")

if __name__ == "__main__":
    main()