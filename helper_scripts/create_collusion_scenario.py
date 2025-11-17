# create_collusion_scenario.py
# Interactive script to create collusion scenarios by generating watermarked text
# for multiple users and combining them into one document

import os
import sys
from datetime import datetime

# Add the parent directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import ZeroBitWatermarker, LBitWatermarker, GroupedMultiUserWatermarker


def parse_final_output(raw_text: str, model_name: str) -> str:
    """
    Extracts the content from the 'final' channel of a gpt-oss output.
    If the model is not gpt-oss or the separator isn't found, it returns the original text.
    """
    if 'gpt-oss' in model_name:
        separator = "assistantfinal"
        if separator in raw_text:
            return raw_text.split(separator)[-1].strip()
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
    print("=" * 60)
    print("Collusion Scenario Generator")
    print("=" * 60)
    print()
    
    # Default settings (matching COMMANDS.md example, but reduced max_new_tokens for collusion)
    users_file = 'assets/users.csv'
    model_name = 'gpt2'
    l_bits = 10
    max_new_tokens = 400  # Reduced to allow multiple users' texts to fit within GPT-2's 1024 token limit
    delta = 4.0
    entropy_threshold = 2.0
    hashing_context = 5
    z_threshold = 4.0
    key_file = 'demonstration/multiuser_master.key'
    
    # Get number of users
    while True:
        try:
            num_users = int(input("Enter number of users for collusion scenario: "))
            if num_users < 2:
                print("Error: Need at least 2 users for collusion. Please try again.")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid number.")
    
    print()
    print(f"Using settings:")
    print(f"  - Model: {model_name}")
    print(f"  - L-bits: {l_bits}")
    print(f"  - Max new tokens: {max_new_tokens}")
    print(f"  - Users file: {users_file}")
    print()
    
    # Setup watermarking stack
    print(f"Loading model '{model_name}'...")
    model = get_model(model_name)
    
    zbw = ZeroBitWatermarker(
        model=model,
        delta=delta,
        entropy_threshold=entropy_threshold,
        z_threshold=z_threshold,
        hashing_context=hashing_context
    )
    lbw = LBitWatermarker(zero_bit_watermarker=zbw, L=l_bits)
    muw = GroupedMultiUserWatermarker(lbit_watermarker=lbw)
    
    # Load user data and generate codes
    try:
        muw.load_users(users_file=users_file)
        print(f"Loaded {muw.N} users and grouped codes from {users_file}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error initializing fingerprinting: {e}")
        return
    
    # Load or generate master key
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            master_key_hex = f.read().strip()
            master_key = bytes.fromhex(master_key_hex)
        print(f"Loaded existing master key from {key_file}")
    else:
        master_key = muw.keygen()
        with open(key_file, 'w') as f:
            f.write(master_key.hex())
        print(f"Generated new master key and saved to {key_file}")
    
    print()
    print("-" * 60)
    print("Enter user information:")
    print("-" * 60)
    
    # Collect user information
    user_data = []
    for i in range(num_users):
        print()
        print(f"User {i+1}/{num_users}:")
        
        # Get user ID
        while True:
            try:
                user_id = int(input(f"  Enter User ID (0-{muw.fingerprinter.N-1}): "))
                if user_id < 0 or user_id >= muw.fingerprinter.N:
                    print(f"    Error: User ID must be between 0 and {muw.fingerprinter.N-1}")
                    continue
                break
            except ValueError:
                print("    Error: Please enter a valid number.")
        
        # Get prompt
        prompt = input(f"  Enter prompt for User {user_id}: ").strip()
        if not prompt:
            print("    Warning: Empty prompt, using default...")
            prompt = "The future of AI is"
        
        user_data.append({
            'user_id': user_id,
            'prompt': prompt
        })
    
    print()
    print("-" * 60)
    print("Generating watermarked text for each user...")
    print("-" * 60)
    
    # Generate text for each user
    generated_texts = []
    for i, user_info in enumerate(user_data):
        user_id = user_info['user_id']
        prompt = user_info['prompt']
        
        print()
        print(f"[{i+1}/{num_users}] Generating for User ID {user_id}...")
        print(f"  Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        
        try:
            raw_text = muw.embed(master_key, user_id, prompt, max_new_tokens=max_new_tokens)
            final_text = parse_final_output(raw_text, model_name.lower())
            generated_texts.append({
                'user_id': user_id,
                'text': final_text,
                'prompt': prompt
            })
            print(f"  Generated {len(final_text)} characters")
        except Exception as e:
            print(f"  Error generating text: {e}")
            return
    
    # Combine all texts
    print()
    print("-" * 60)
    print("Combining texts...")
    print("-" * 60)
    
    combined_parts = []
    for gen_data in generated_texts:
        text = gen_data['text']
        combined_parts.append(text)
    
    combined_text = "\n\n".join(combined_parts)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_ids_str = "_".join(str(d['user_id']) for d in generated_texts)
    output_file = f"demonstration/collusion_users_{user_ids_str}_{timestamp}.txt"
    
    # Ensure demonstration directory exists
    os.makedirs('demonstration', exist_ok=True)
    
    # Save combined text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    print()
    print("=" * 60)
    print("Collusion scenario created successfully!")
    print("=" * 60)
    print(f"Combined text saved to: {output_file}")
    print(f"Master key location: {key_file}")
    print()
    print("To trace this collusion scenario, run:")
    print(f'  python -m src.main_multiuser trace ^')
    print(f'    --users-file {users_file} ^')
    print(f'    --model {model_name} ^')
    print(f'    --l-bits {l_bits} ^')
    print(f'    {output_file}')
    print()


if __name__ == "__main__":
    main()

