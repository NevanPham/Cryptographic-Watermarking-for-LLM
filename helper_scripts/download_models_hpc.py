# download_models.py
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import nltk

def setup_nltk():
    """Checks for and downloads the NLTK 'punkt' tokenizer models if needed."""
    nltk.download('punkt', download_dir='/fred/oz413/rburke/huggingface')
    nltk.download('punkt_tab', download_dir='/fred/oz413/rburke/huggingface')

# Get the Hugging Face cache directory from the environment variable
cache_dir = os.environ.get("HF_HOME")
if not cache_dir:
    raise ValueError("Please set the HF_HOME environment variable to your desired cache directory.")

print(f"Using Hugging Face cache directory: {cache_dir}")

# --- List of models to download ---
models_to_download = {
    #"gpt-oss-20b": ("openai/gpt-oss-20b", AutoModelForCausalLM, AutoTokenizer),
    #"Vamsi/T5_Paraphrase_Paws": ("Vamsi/T5_Paraphrase_Paws", AutoModelForSeq2SeqLM, AutoTokenizer),
    #"gpt2": ("gpt2", GPT2LMHeadModel, GPT2Tokenizer)
}

# --- Download loop ---
for name, (model_id, model_class, tokenizer_class) in models_to_download.items():
    print(f"\nDownloading {name} ({model_id})...")
    try:
        # Download and cache the model and tokenizer
        model_class.from_pretrained(model_id, cache_dir=cache_dir)
        tokenizer_class.from_pretrained(model_id, cache_dir=cache_dir)
        print(f"Successfully downloaded {name}.")
    except Exception as e:
        print(f"Failed to download {name}. Error: {e}")

setup_nltk()

print("\nAll model downloads attempted.")