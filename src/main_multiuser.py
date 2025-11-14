# main_multiuser.py

import argparse
import json
import os
import sys

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import ZeroBitWatermarker, LBitWatermarker, MultiUserWatermarker

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

def main():
    parser = argparse.ArgumentParser(description="Multi-User Watermarking Tool using Fingerprinting Codes.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Base arguments for both commands ---
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--users-file', type=str, default='assets/users.csv', help="Path to the user metadata CSV file.")
    base_parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'])
    base_parser.add_argument('--key-file', '-k', type=str, default='demonstration/multiuser_master.key', help="Path to the master secret key.")
    # L-bit parameters
    base_parser.add_argument('--delta', type=float, default=3.5)
    base_parser.add_argument('--entropy-threshold', type=float, default=2.5)
    base_parser.add_argument('--hashing-context', type=int, default=5)
    base_parser.add_argument('--z-threshold', type=float, default=4.0)
    base_parser.add_argument('--l-bits', type=int, default=10)

    # --- Generate Command ---
    gen_parser = subparsers.add_parser('generate', help='Generate text watermarked for a specific user.', parents=[base_parser])
    gen_parser.add_argument('prompt', type=str)
    gen_parser.add_argument('--user-id', type=int, required=True, help="Integer ID of the user to watermark for (e.g., 2).")
    gen_parser.add_argument('--max-new-tokens', type=int, default=2048)
    gen_parser.add_argument('--output-file', '-o', type=str, default='demonstration/multiuser_output.txt')

    # --- Trace Command ---
    trace_parser = subparsers.add_parser('trace', help='Trace a watermarked text back to a user ID.', parents=[base_parser])
    trace_parser.add_argument('input_file', type=str)
    
    args = parser.parse_args()

    # --- Setup Watermarking Stack ---
    print(f"Loading model '{args.model}'...")
    model = get_model(args.model)

    zbw = ZeroBitWatermarker(
        model=model, 
        delta=args.delta, 
        entropy_threshold=args.entropy_threshold, 
        z_threshold=args.z_threshold,
        hashing_context=args.hashing_context
    )
    lbw = LBitWatermarker(zero_bit_watermarker=zbw, L=args.l_bits)
    muw = MultiUserWatermarker(lbit_watermarker=lbw)

    # --- Command Logic ---
    try:
        # Load user data and generate codes for both commands
        muw.fingerprinter.gen(users_file=args.users_file)
        print(f"📖 Loaded {muw.fingerprinter.N} users and codes from {args.users_file}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"❌ Error initializing fingerprinting: {e}")
        return

    if args.command == 'generate':
        master_key = muw.keygen()
        
        raw_text = muw.embed(master_key, args.user_id, args.prompt, max_new_tokens=args.max_new_tokens)
        final_text = parse_final_output(raw_text, args.model.__class__.__name__.lower())
        
        print("\n--- Final Watermarked Response ---")
        print(final_text)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)
        with open(args.key_file, 'w') as f:
            f.write(master_key.hex())
            
        print(f"\n✅ Output for User ID {args.user_id} saved to {args.output_file}")
        print(f"🔑 Master key saved to {args.key_file}")

    elif args.command == 'trace':
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text_to_trace = f.read()
        with open(args.key_file, 'r') as f:
            master_key_hex = f.read()
        master_key = bytes.fromhex(master_key_hex)
        
        accused_users = muw.trace(master_key, text_to_trace)
        
        print("\n--- Trace Results ---")
        if accused_users:
            print(f"  ✅ Text traced back to user(s):")
            for user in accused_users:
                print(f"     - User ID: {user['user_id']}, Username: {user['username']}, Match: {user['match_score_percent']:.2f}%")
        else:
            print("  ❌ Could not confidently trace text to any user.")
        print("---------------------")

if __name__ == "__main__":
    main()