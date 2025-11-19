import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import nltk

# Use HF_HOME as the NLTK download directory
cache_dir = os.environ.get("HF_HOME")
if not cache_dir:
    raise ValueError("Please set HF_HOME before running this script.")

print(f"Using Hugging Face cache directory: {cache_dir}")

def setup_nltk():
    """Download NLTK tokenizer data into HF_HOME."""
    print("Downloading NLTK 'punkt'...")
    nltk.download('punkt', download_dir=cache_dir)
    nltk.download('punkt_tab', download_dir=cache_dir)
    print("NLTK data downloaded successfully.")

# Models to download (light, safe for HPC)
models_to_download = {
    "gpt2": ("gpt2", GPT2LMHeadModel, GPT2Tokenizer),
    "T5_Paraphrase": ("Vamsi/T5_Paraphrase_Paws", AutoModelForSeq2SeqLM, AutoTokenizer)
}

# Large models like gpt-oss-20b cannot load on P100, but we can still cache tokenizer only:
print("\nCaching tokenizer for gpt-oss-20b...")
try:
    AutoTokenizer.from_pretrained("openai/gpt-oss-20b", cache_dir=cache_dir)
    print("Tokenizer cached successfully.")
except Exception as e:
    print(f"Failed to cache tokenizer: {e}")

# Download normal models
for name, (model_id, model_class, token_class) in models_to_download.items():
    print(f"\nDownloading {name} ({model_id})...")
    try:
        model_class.from_pretrained(model_id, cache_dir=cache_dir)
        token_class.from_pretrained(model_id, cache_dir=cache_dir)
        print(f"Successfully downloaded {name}.")
    except Exception as e:
        print(f"Failed to download {name}. Error: {e}")

setup_nltk()
print("\nAll downloads completed.")
