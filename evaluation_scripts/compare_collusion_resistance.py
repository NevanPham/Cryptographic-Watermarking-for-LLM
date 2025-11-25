# compare_collusion_resistance.py
# Script to compare naive vs fingerprinting multi-user watermarking approaches for collusion resistance
# Tests with 2-3 users across different groups using three approaches: naive, min-distance-2, and min-distance-3
# Multiple combination methods: normal combination and combinations with deletion (5%, 10%, 15%) using different types (random, start, end)

import argparse
import json
import os
import random
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk

# Add the parent directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import ZeroBitWatermarker, LBitWatermarker, NaiveMultiUserWatermarker, GroupedMultiUserWatermarker
from src.utils import get_model, parse_final_output


def setup_nltk():
    """Setup NLTK data path if needed."""
    try:
        # Try to find punkt tokenizer
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # If not found, try to download (may fail in some environments)
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass  # Will fall back to simple split in delete_percentage_text


def delete_percentage_text(text: str, percentage: float = 0.05, deletion_type: str = 'random') -> str:
    """
    Delete a percentage of sentences from the text.
    
    Args:
        text: Input text
        percentage: Percentage of sentences to delete (default: 0.05 for 5%)
        deletion_type: Type of deletion - 'random', 'start', or 'end' (default: 'random')
    
    Returns:
        Text with percentage of sentences deleted
    """
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        # Fallback to simple split if NLTK fails
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if len(sentences) == 0:
        return text
    
    num_to_delete = max(1, int(len(sentences) * percentage))
    if num_to_delete >= len(sentences):
        # If we need to delete all or more, keep at least one sentence
        num_to_delete = len(sentences) - 1
    
    # Select sentences to delete based on deletion type
    if deletion_type == 'start':
        # Delete from the beginning
        remaining_sentences = sentences[num_to_delete:]
    elif deletion_type == 'end':
        # Delete from the end
        remaining_sentences = sentences[:-num_to_delete] if num_to_delete > 0 else sentences
    else:  # 'random' (default)
        # Randomly select sentences to delete
        indices_to_delete = random.sample(range(len(sentences)), num_to_delete)
        remaining_sentences = [sentences[i] for i in range(len(sentences)) if i not in indices_to_delete]
    
    return " ".join(remaining_sentences)


def combine_texts_normal(texts: list[str]) -> str:
    """Combine texts normally (just concatenate with newlines)."""
    return "\n\n".join(texts)


def combine_texts_with_deletion(texts: list[str], deletion_percentage: float = 0.05, deletion_type: str = 'random') -> str:
    """Combine texts after deleting a percentage from each user's text."""
    deleted_texts = [delete_percentage_text(text, deletion_percentage, deletion_type) for text in texts]
    return "\n\n".join(deleted_texts)


def json_default_encoder(obj):
    """Convert NumPy/Pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def select_colluding_users(num_users: int, total_users: int, min_distance: int = None) -> list[int]:
    """
    Select colluding users. For grouped schemes, ALWAYS select from different groups.
    For naive scheme, just select random users.
    
    Args:
        num_users: Number of colluding users (2 or 3)
        total_users: Total number of users available
        min_distance: Minimum distance for grouped schemes (None for naive)
    
    Returns:
        List of user IDs
    """
    if min_distance is None:
        # Naive scheme: just select random users
        return sorted(random.sample(range(total_users), num_users))
    
    # For grouped schemes, ALWAYS select users from different groups
    # Based on DISTANCE_CONFIG in fingerprinting.py:
    # min_distance=2: 10 users per group
    # min_distance=3: 20 users per group
    
    users_per_group = {2: 10, 3: 20}.get(min_distance, 10)
    num_groups = (total_users + users_per_group - 1) // users_per_group
    
    # Check if we have enough groups
    if num_groups < num_users:
        print(f"  ⚠ Warning: Only {num_groups} groups available but need {num_users} colluders. "
              f"Some users may be from the same group.")
    
    # Select one user from each of num_users different groups
    selected_groups = random.sample(range(num_groups), min(num_users, num_groups))
    selected_users = []
    
    for group_id in selected_groups:
        # Get users in this group
        group_start = group_id * users_per_group
        group_end = min((group_id + 1) * users_per_group, total_users)
        group_users = list(range(group_start, group_end))
        
        if group_users:
            # Randomly select one user from this group
            selected_users.append(random.choice(group_users))
    
    # If we need more users but ran out of groups, randomly select from remaining users
    while len(selected_users) < num_users:
        remaining = [u for u in range(total_users) if u not in selected_users]
        if remaining:
            selected_users.append(random.choice(remaining))
        else:
            break
    
    return sorted(selected_users)


def trace_collusion(muw, master_key: bytes, combined_text: str, original_user_ids: list[int]) -> dict:
    """
    Try to trace back to original colluding users.
    
    Returns:
        Dictionary with tracing results including success status
    """
    try:
        accused_users = muw.trace(master_key, combined_text)
        
        if not accused_users:
            return {
                'success': False,
                'accused_user_ids': [],
                'original_user_ids': original_user_ids,
                'reason': 'No users accused'
            }
        
        accused_ids = [accused['user_id'] for accused in accused_users]
        
        # Check if any of the accused users match the original colluding users
        matches = set(accused_ids) & set(original_user_ids)
        
        # Success if at least one original user is correctly identified
        success = len(matches) > 0
        
        return {
            'success': success,
            'accused_user_ids': accused_ids,
            'original_user_ids': original_user_ids,
            'matches': list(matches),
            'num_matches': len(matches),
            'accused_details': accused_users
        }
    except Exception as e:
        return {
            'success': False,
            'accused_user_ids': [],
            'original_user_ids': original_user_ids,
            'reason': f'Error during tracing: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compare naive vs fingerprinting multi-user watermarking approaches for collusion resistance",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--prompts-file',
        type=str,
        default='assets/prompts.txt',
        help='Path to prompts file (default: assets/prompts.txt)'
    )
    parser.add_argument(
        '--max-prompts',
        type=int,
        default=100,
        help='Limit on number of prompts to process (default: 100; set <=0 for all prompts)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
        help='Model to use for generation and detection'
    )
    parser.add_argument(
        '--users-file',
        type=str,
        default='assets/users.csv',
        help='Path to users CSV file (default: assets/users.csv)'
    )
    parser.add_argument(
        '--l-bits',
        type=int,
        default=10,
        help='Number of L-bits for watermarking (default: 10)'
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=3.5,
        help='Watermark strength (default: 3.5)'
    )
    parser.add_argument(
        '--entropy-threshold',
        type=float,
        default=2.5,
        help='Entropy threshold for watermarking (default: 2.5)'
    )
    parser.add_argument(
        '--hashing-context',
        type=int,
        default=5,
        help='Hashing context window (default: 5)'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=4.0,
        help='Z-score threshold for detection (default: 4.0)'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=400,
        help='Maximum number of tokens to generate per user (default: 400)'
    )
    parser.add_argument(
        '--num-colluders',
        type=int,
        default=2,
        choices=[2, 3],
        help='Number of colluding users (default: 2)'
    )
    parser.add_argument(
        '--deletion-percentages',
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.15],
        help='Percentages of text to delete per user in deletion scenarios (default: 0.05 0.10 0.15 for 5%%, 10%%, 15%%)'
    )
    parser.add_argument(
        '--deletion-types',
        type=str,
        nargs='+',
        default=['random', 'start', 'end'],
        choices=['random', 'start', 'end'],
        help='Types of deletion to test: random, start, end (default: all three)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation/collusion_resistance',
        help='Output directory for results (default: evaluation/collusion_resistance). Will append _<num_colluders> automatically.'
    )
    parser.add_argument(
        '--csv-only',
        action='store_true',
        help='Skip generation and only build JSON/CSV summaries from existing prompt-level results.'
    )
    
    args = parser.parse_args()
    
    # Create output directory with number of colluders appended
    base_output_dir = args.output_dir
    # Append _<num_colluders> to the directory name
    if not base_output_dir.endswith(f'_{args.num_colluders}'):
        base_output_dir = f'{base_output_dir}_{args.num_colluders}'
    
    if not os.path.isabs(base_output_dir):
        output_dir = os.path.join(parent_dir, base_output_dir)
    else:
        output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Print header
    print("\n" + "="*80)
    print(" " * 20 + "COLLUSION RESISTANCE EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  • Number of colluders: {args.num_colluders}")
    print(f"  • Model: {args.model}")
    print(f"  • L-bits: {args.l_bits}")
    print(f"  • Delta: {args.delta}")
    print(f"  • Entropy threshold: {args.entropy_threshold}")
    print(f"  • Hashing context: {args.hashing_context}")
    print(f"  • Z-threshold: {args.z_threshold}")
    print(f"  • Max new tokens: {args.max_new_tokens}")
    print(f"  • Deletion percentages: {[f'{p*100:.1f}%' for p in args.deletion_percentages]}")
    print(f"  • Deletion types: {args.deletion_types}")
    print(f"  • Output directory: {output_dir}")
    print("="*80)
    
    prompt_results_dir = os.path.join(output_dir, 'prompt_results')
    
    if args.csv_only:
        print("\n[CSV-ONLY MODE] Rebuilding summaries from existing prompt results...\n")
        if not os.path.isdir(prompt_results_dir):
            print(f"  ❌ Error: Prompt results directory not found: {prompt_results_dir}")
            print("  Please run the full comparison first.")
            return
        all_results = load_prompt_results(prompt_results_dir)
        if not all_results:
            print(f"  ❌ Error: No prompt result files found in {prompt_results_dir}")
            print("  Please run the full comparison first.")
            return
        total_prompts_processed = len({result['prompt_id'] for result in all_results})
        generate_reports(all_results, args, output_dir, total_prompts_processed)
        return
    
    # Setup NLTK (only needed when generating new data)
    setup_nltk()
    
    # Load prompts
    print(f"\n[1/4] Loading prompts...")
    prompts_path = os.path.join(parent_dir, args.prompts_file)
    if not os.path.exists(prompts_path):
        print(f"  ❌ Error: Prompts file not found: {prompts_path}")
        return
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    total_prompts = len(prompts)
    if args.max_prompts and args.max_prompts > 0 and args.max_prompts < total_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"  ✓ Loaded {total_prompts} prompts, using first {len(prompts)} for evaluation")
    else:
        print(f"  ✓ Loaded {total_prompts} prompts")
    
    # Load model
    print(f"\n[2/4] Loading model and initializing watermarkers...")
    print(f"  → Loading model '{args.model}'...")
    model = get_model(args.model)
    print(f"  ✓ Model loaded successfully")
    
    # Load users file to get total number of users
    users_path = os.path.join(parent_dir, args.users_file)
    if not os.path.exists(users_path):
        print(f"  ❌ Error: Users file not found: {users_path}")
        return
    
    users_df = pd.read_csv(users_path)
    total_users = len(users_df)
    print(f"  ✓ Loaded {total_users} users from users file")
    
    # Initialize watermarkers for each approach
    print(f"\n  → Initializing watermarkers...")
    zero_bit = ZeroBitWatermarker(
        model=model,
        delta=args.delta,
        entropy_threshold=args.entropy_threshold,
        hashing_context=args.hashing_context,
        z_threshold=args.z_threshold
    )
    lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zero_bit, L=args.l_bits)
    
    # Create watermarkers for each approach
    naive_muw = NaiveMultiUserWatermarker(lbit_watermarker=lbit_watermarker)
    grouped_muw_d2 = GroupedMultiUserWatermarker(lbit_watermarker=lbit_watermarker, min_distance=2)
    grouped_muw_d3 = GroupedMultiUserWatermarker(lbit_watermarker=lbit_watermarker, min_distance=3)
    
    # Load users for each watermarker
    print(f"    • Loading users for naive scheme...")
    naive_muw.load_users(users_path)
    print(f"    • Loading users for min-distance-2 scheme...")
    grouped_muw_d2.load_users(users_path)
    print(f"    • Loading users for min-distance-3 scheme...")
    grouped_muw_d3.load_users(users_path)
    print(f"  ✓ All watermarkers initialized")
    
    # Results storage
    all_results = []
    
    # Process each prompt
    print(f"\n[3/4] Processing {len(prompts)} prompts with {args.num_colluders} colluders...")
    print(f"  → Testing 3 approaches: naive, min-distance-2, min-distance-3")
    print(f"  → Each prompt uses same colluding users across all approaches")
    num_deletion_tests = len(args.deletion_percentages) * len(args.deletion_types)
    print(f"  → Combination methods: normal + {num_deletion_tests} deletion scenarios")
    print(f"    (Percentages: {[f'{p*100:.1f}%' for p in args.deletion_percentages]}, Types: {args.deletion_types})\n")
    
    os.makedirs(prompt_results_dir, exist_ok=True)
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts", unit="prompt")):
        prompt_specific_results = []
        # Select colluding users ONCE per prompt - same users for all approaches
        # This ensures fair comparison across approaches using the same collusion scenario
        # For grouped schemes, try to select from different groups if possible
        colluding_users = select_colluding_users(args.num_colluders, total_users, min_distance=3)
        
        # Use the same users for all three approaches
        naive_users = colluding_users
        d2_users = colluding_users
        d3_users = colluding_users
        
        # Generate master keys for each approach
        naive_key = naive_muw.keygen()
        d2_key = grouped_muw_d2.keygen()
        d3_key = grouped_muw_d3.keygen()
        
        # Test each approach
        approaches = [
            ('naive', naive_muw, naive_key, naive_users),
            ('min-distance-2', grouped_muw_d2, d2_key, d2_users),
            ('min-distance-3', grouped_muw_d3, d3_key, d3_users)
        ]
        
        for approach_name, muw, master_key, user_ids in approaches:
            try:
                # Create subdirectory structure: {approach}/prompt_{idx}
                approach_base_dir = os.path.join(output_dir, approach_name)
                os.makedirs(approach_base_dir, exist_ok=True)
                prompt_dir = os.path.join(approach_base_dir, f'prompt_{prompt_idx}')
                os.makedirs(prompt_dir, exist_ok=True)
                
                # Save master key for this approach
                key_filename = f'master_key.key'
                key_path = os.path.join(prompt_dir, key_filename)
                with open(key_path, 'w', encoding='utf-8') as f:
                    f.write(master_key.hex())
                
                # Generate watermarked text for each colluding user
                user_texts = []
                user_text_files = []
                for idx, user_id in enumerate(user_ids):
                    raw_text = muw.embed(master_key, user_id, prompt, max_new_tokens=args.max_new_tokens)
                    final_text = parse_final_output(raw_text, args.model)
                    user_texts.append(final_text)
                    
                    # Save individual user text file
                    user_text_filename = f'user_{user_id}_text.txt'
                    user_text_path = os.path.join(prompt_dir, user_text_filename)
                    with open(user_text_path, 'w', encoding='utf-8') as f:
                        f.write(final_text)
                    user_text_files.append(user_text_filename)
                
                # Test combination methods: normal + all deletion scenarios
                combination_methods = [('normal', None, None)]
                
                # Add all deletion scenarios
                for deletion_percentage in args.deletion_percentages:
                    for deletion_type in args.deletion_types:
                        combination_methods.append(('with_deletion', deletion_percentage, deletion_type))
                
                for combination_method, deletion_percentage, deletion_type in combination_methods:
                    if combination_method == 'normal':
                        combined_text = combine_texts_normal(user_texts)
                        combined_filename = f'combined_normal.txt'
                    else:
                        combined_text = combine_texts_with_deletion(user_texts, deletion_percentage, deletion_type)
                        # Format: combined_deletion_5.0%_random.txt
                        pct_str = f"{deletion_percentage*100:.1f}%".replace('.', '_')
                        combined_filename = f'combined_deletion_{pct_str}_{deletion_type}.txt'
                    
                    # Save combined text file
                    combined_path = os.path.join(prompt_dir, combined_filename)
                    with open(combined_path, 'w', encoding='utf-8') as f:
                        f.write(combined_text)
                    
                    # Try to trace back
                    trace_result = trace_collusion(muw, master_key, combined_text, user_ids)
                    
                    # Store result
                    result = {
                        'prompt_id': prompt_idx,
                        'prompt': prompt,
                        'approach': approach_name,
                        'combination_method': combination_method,
                        'deletion_percentage': deletion_percentage if deletion_percentage is not None else 0.0,
                        'deletion_type': deletion_type if deletion_type is not None else 'none',
                        'num_colluders': args.num_colluders,
                        'original_user_ids': user_ids,
                        'trace_result': trace_result,
                        'success': trace_result['success'],
                        'files': {
                            'master_key': key_filename,
                            'user_texts': user_text_files,
                            'combined_text': combined_filename
                        },
                        'parameters': {
                            'l_bits': args.l_bits,
                            'delta': args.delta,
                            'entropy_threshold': args.entropy_threshold,
                            'hashing_context': args.hashing_context,
                            'z_threshold': args.z_threshold,
                            'max_new_tokens': args.max_new_tokens
                        }
                    }
                    all_results.append(result)
                    prompt_specific_results.append(result)
        
            except Exception as e:
                print(f"\n  ⚠ Warning: Error processing prompt {prompt_idx} with approach '{approach_name}': {e}")
                # Store error result for all combination methods
                combination_methods = [('normal', None, None)]
                for deletion_percentage in args.deletion_percentages:
                    for deletion_type in args.deletion_types:
                        combination_methods.append(('with_deletion', deletion_percentage, deletion_type))
                
                for combination_method, deletion_percentage, deletion_type in combination_methods:
                    result = {
                        'prompt_id': prompt_idx,
                        'prompt': prompt,
                        'approach': approach_name,
                        'combination_method': combination_method,
                        'deletion_percentage': deletion_percentage if deletion_percentage is not None else 0.0,
                        'deletion_type': deletion_type if deletion_type is not None else 'none',
                        'num_colluders': args.num_colluders,
                        'original_user_ids': user_ids,
                        'trace_result': {'success': False, 'reason': str(e)},
                        'success': False,
                        'error': str(e)
                    }
                    all_results.append(result)
                    prompt_specific_results.append(result)
        
        # Save per-prompt results for later csv-only summary generation
        prompt_json_path = os.path.join(prompt_results_dir, f'prompt_{prompt_idx}_results.json')
        with open(prompt_json_path, 'w', encoding='utf-8') as prompt_file:
            json.dump(prompt_specific_results, prompt_file, indent=2, default=json_default_encoder)
    
    total_prompts_processed = len({result['prompt_id'] for result in all_results})
    generate_reports(all_results, args, output_dir, total_prompts_processed)


def load_prompt_results(prompt_results_dir: str) -> list[dict]:
    """Load all per-prompt result JSON files from disk."""
    all_results: list[dict] = []
    if not os.path.isdir(prompt_results_dir):
        return all_results
    prompt_files = sorted(
        filename for filename in os.listdir(prompt_results_dir)
        if filename.startswith('prompt_') and filename.endswith('_results.json')
    )
    for filename in prompt_files:
        path = os.path.join(prompt_results_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                prompt_results = json.load(f)
                if isinstance(prompt_results, list):
                    all_results.extend(prompt_results)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ⚠ Warning: Could not load {filename}: {exc}")
            continue
    return all_results


def generate_reports(all_results: list[dict], args, output_dir: str, total_prompts_processed: int):
    """Calculate success rates and write summary artifacts."""
    if not all_results:
        print("No results available to summarize.")
        return
    
    print(f"\n[4/4] Calculating success rates and generating reports...")
    print("="*80)
    
    success_rates = {}
    
    # Get unique deletion configurations from results
    deletion_configs = set()
    for result in all_results:
        if result['combination_method'] == 'with_deletion':
            config = (result['deletion_percentage'], result['deletion_type'])
            deletion_configs.add(config)
    deletion_configs = sorted(deletion_configs)
    
    for approach in ['naive', 'min-distance-2', 'min-distance-3']:
        # Normal combination
        normal_results = [r for r in all_results if r['approach'] == approach and r['combination_method'] == 'normal']
        if normal_results:
            successful = sum(1 for r in normal_results if r.get('success', False))
            total = len(normal_results)
            success_rate = (successful / total) * 100.0 if total > 0 else 0.0
            success_rates[f"{approach}_normal"] = {
                'successful': successful,
                'total': total,
                'success_rate': success_rate
            }
        
        # Deletion combinations
        for deletion_percentage, deletion_type in deletion_configs:
            key = f"{approach}_deletion_{deletion_percentage}_{deletion_type}"
            approach_results = [
                r for r in all_results 
                if r['approach'] == approach 
                and r['combination_method'] == 'with_deletion'
                and r['deletion_percentage'] == deletion_percentage
                and r['deletion_type'] == deletion_type
            ]
            
            if approach_results:
                successful = sum(1 for r in approach_results if r.get('success', False))
                total = len(approach_results)
                success_rate = (successful / total) * 100.0 if total > 0 else 0.0
                success_rates[key] = {
                    'successful': successful,
                    'total': total,
                    'success_rate': success_rate,
                    'deletion_percentage': deletion_percentage,
                    'deletion_type': deletion_type
                }
    
    # Print comparison table
    print("\n" + "="*80)
    print(" " * 25 + "RESULTS SUMMARY")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  • Number of colluders: {args.num_colluders}")
    print(f"  • Total prompts tested: {total_prompts_processed}")
    print(f"  • Deletion percentages: {[f'{p*100:.1f}%' for p in args.deletion_percentages]}")
    print(f"  • Deletion types: {args.deletion_types}")
    print(f"\nSuccess Rates by Approach:")
    print("-"*80)
    
    # Create a comprehensive table
    table_data = []
    for approach in ['naive', 'min-distance-2', 'min-distance-3']:
        row = {'Approach': approach}
        
        # Normal combination
        normal_key = f"{approach}_normal"
        if normal_key in success_rates:
            sr = success_rates[normal_key]
            row['Normal'] = f"{sr['success_rate']:.2f}% ({sr['successful']}/{sr['total']})"
        else:
            row['Normal'] = "N/A"
        
        # Deletion combinations
        for deletion_percentage, deletion_type in deletion_configs:
            key = f"{approach}_deletion_{deletion_percentage}_{deletion_type}"
            col_name = f"{deletion_percentage*100:.0f}% {deletion_type}"
            if key in success_rates:
                sr = success_rates[key]
                row[col_name] = f"{sr['success_rate']:.2f}% ({sr['successful']}/{sr['total']})"
            else:
                row[col_name] = "N/A"
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    print()
    
    # Save detailed results to JSON
    json_path = os.path.join(output_dir, f'collusion_resistance_results_{args.num_colluders}users.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'num_prompts': total_prompts_processed,
                'num_colluders': args.num_colluders,
                'deletion_percentages': args.deletion_percentages,
                'deletion_types': args.deletion_types,
                'parameters': {
                    'l_bits': args.l_bits,
                    'delta': args.delta,
                    'entropy_threshold': args.entropy_threshold,
                    'hashing_context': args.hashing_context,
                    'z_threshold': args.z_threshold,
                    'max_new_tokens': args.max_new_tokens
                }
            },
            'success_rates': success_rates,
            'detailed_results': all_results
        }, f, indent=2, default=json_default_encoder)
    
    print(f"\n" + "="*80)
    print(" " * 25 + "FILES SAVED")
    print("="*80)
    print(f"\n✓ Detailed results JSON: {json_path}")
    
    # Save summary CSV
    summary_data = []
    for result in all_results:
        summary_data.append({
            'prompt_id': result['prompt_id'],
            'approach': result['approach'],
            'combination_method': result['combination_method'],
            'deletion_percentage': result.get('deletion_percentage', 0.0),
            'deletion_type': result.get('deletion_type', 'none'),
            'num_colluders': result['num_colluders'],
            'original_user_ids': str(result['original_user_ids']),
            'success': result.get('success', False),
            'num_matches': result.get('trace_result', {}).get('num_matches', 0)
        })
    
    csv_path = os.path.join(output_dir, f'collusion_resistance_summary_{args.num_colluders}users.csv')
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_path, index=False)
    
    print(f"✓ Summary CSV: {csv_path}")
    print(f"\nIntermediate files structure:")
    print(f"  {output_dir}/")
    print(f"    ├── naive/")
    print(f"    │   ├── prompt_0/")
    print(f"    │   │   ├── master_key.key")
    print(f"    │   │   ├── user_<ID>_text.txt")
    print(f"    │   │   ├── combined_normal.txt")
    print(f"    │   │   └── combined_with_deletion.txt")
    print(f"    │   └── prompt_1/, prompt_2/, ...")
    print(f"    ├── min-distance-2/")
    print(f"    │   └── prompt_0/, prompt_1/, ...")
    print(f"    └── min-distance-3/")
    print(f"        └── prompt_0/, prompt_1/, ...")
    
    print("\n" + "="*80)
    print(" " * 30 + "✓ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()

