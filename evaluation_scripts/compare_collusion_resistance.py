# compare_collusion_resistance.py
# Script to evaluate naive and hierarchical watermarking schemes under controlled collusion
# Tests with 2-3 colluders in different configurations: same group, cross group, mixed

import argparse
import json
import os
import random
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add the parent directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import (
    ZeroBitWatermarker, 
    LBitWatermarker, 
    NaiveMultiUserWatermarker, 
    HierarchicalMultiUserWatermarker
)
from src.utils import get_model, parse_final_output


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


def merge_codewords_bitwise_majority(codewords: list[str]) -> str:
    """
    Merge multiple L-bit codewords using bitwise majority voting.
    Handles uncertain bits ('⊥') and conflict bits ('*') properly.
    
    Args:
        codewords: List of codeword strings (each should be L bits)
    
    Returns:
        Merged codeword string of length L
    """
    if not codewords:
        raise ValueError("Cannot merge empty list of codewords")
    
    L = len(codewords[0])
    if not all(len(c) == L for c in codewords):
        raise ValueError(f"All codewords must have the same length (L={L})")
    
    merged = []
    for bit_pos in range(L):
        # Collect all bits at this position
        bits_at_pos = [c[bit_pos] for c in codewords]
        
        # Filter out uncertain ('⊥') and conflict ('*') bits, keep only valid bits ('0' or '1')
        valid_bits = [b for b in bits_at_pos if b in ('0', '1')]
        
        if not valid_bits:
            # All codewords are uncertain at this position, result is uncertain
            merged.append('⊥')
        else:
            # Do majority voting on valid bits only
            valid_ints = [int(b) for b in valid_bits]
            # Majority vote: if more 1s than 0s, result is 1, else 0
            # In case of tie, default to 0
            majority_bit = 1 if sum(valid_ints) > len(valid_ints) / 2 else 0
            merged.append(str(majority_bit))
    
    return "".join(merged)


def sample_same_group(muw, k: int) -> list[int]:
    """
    Sample k users from the same group (for hierarchical scheme).
    For naive scheme, just sample k random users.
    
    Args:
        muw: Multi-user watermarker instance
        k: Number of users to sample
    
    Returns:
        List of user IDs
    """
    if not hasattr(muw, 'user_to_group') or muw.user_to_group is None:
        # Naive scheme: all users are independent
        return sorted(random.sample(range(muw.N), k))
    
    # Hierarchical scheme: pick a random group and sample k users from it
    # Get all groups and their users
    group_to_users = {}
    for user_id, group_id in muw.user_to_group.items():
        if group_id not in group_to_users:
            group_to_users[group_id] = []
        group_to_users[group_id].append(user_id)
    
    # Find groups with at least k users
    valid_groups = [gid for gid, users in group_to_users.items() if len(users) >= k]
    
    if not valid_groups:
        raise ValueError(f"No group has at least {k} users")
    
    # Pick a random group
    selected_group = random.choice(valid_groups)
    group_users = group_to_users[selected_group]
    
    # Sample k users from this group
    return sorted(random.sample(group_users, k))


def sample_different_groups(muw, k: int) -> list[int]:
    """
    Sample k users from k different groups (for hierarchical scheme).
    For naive scheme, just sample k random users (since all are independent).
    
    Args:
        muw: Multi-user watermarker instance
        k: Number of users to sample
    
    Returns:
        List of user IDs
    """
    if not hasattr(muw, 'user_to_group') or muw.user_to_group is None:
        # Naive scheme: all users are independent
        return sorted(random.sample(range(muw.N), k))
    
    # Hierarchical scheme: pick k different groups
    # Get all groups
    group_to_users = {}
    for user_id, group_id in muw.user_to_group.items():
        if group_id not in group_to_users:
            group_to_users[group_id] = []
        group_to_users[group_id].append(user_id)
    
    available_groups = list(group_to_users.keys())
    
    if len(available_groups) < k:
        raise ValueError(f"Only {len(available_groups)} groups available, need {k} different groups")
    
    # Pick k different groups
    selected_groups = random.sample(available_groups, k)
    
    # Pick one user from each group
    selected_users = []
    for group_id in selected_groups:
        group_users = group_to_users[group_id]
        selected_users.append(random.choice(group_users))
    
    return sorted(selected_users)


def sample_2_same_1_diff(muw) -> list[int]:
    """
    Sample 2 users from the same group and 1 user from a different group.
    For naive scheme, just sample 3 random users (since all are independent).
    
    Args:
        muw: Multi-user watermarker instance
    
    Returns:
        List of 3 user IDs
    """
    if not hasattr(muw, 'user_to_group') or muw.user_to_group is None:
        # Naive scheme: all users are independent
        return sorted(random.sample(range(muw.N), 3))
    
    # Hierarchical scheme: pick 2 from one group, 1 from another
    # Get all groups
    group_to_users = {}
    for user_id, group_id in muw.user_to_group.items():
        if group_id not in group_to_users:
            group_to_users[group_id] = []
        group_to_users[group_id].append(user_id)
    
    # Find groups with at least 2 users
    valid_groups = [gid for gid, users in group_to_users.items() if len(users) >= 2]
    
    if len(valid_groups) < 2:
        raise ValueError("Need at least 2 groups, with at least one having 2+ users")
    
    # Pick a group for the 2 users
    group_with_2 = random.choice(valid_groups)
    users_from_group = random.sample(group_to_users[group_with_2], 2)
    
    # Pick a different group for the 1 user
    other_groups = [gid for gid in group_to_users.keys() if gid != group_with_2]
    other_group = random.choice(other_groups)
    user_from_other = random.choice(group_to_users[other_group])
    
    return sorted(users_from_group + [user_from_other])


def trace_collusion(muw, master_key: bytes, merged_codeword: str, original_user_ids: list[int]) -> dict:
    """
    Try to trace back to original colluding users using the merged codeword.
    Uses direct codeword matching with Hamming distance.
    
    Args:
        muw: Multi-user watermarker instance
        master_key: Master secret key
        merged_codeword: The merged L-bit codeword (from bitwise majority)
        original_user_ids: List of original colluding user IDs
    
    Returns:
        Dictionary with tracing results including success status
    """
    try:
        # Direct codeword matching: compare merged codeword with each user's codeword
        matches = []
        hamming_distances = {}
        
        for user_id in original_user_ids:
            try:
                user_codeword = muw.get_codeword_for_user(user_id)
                # Calculate Hamming distance, but only on positions where merged codeword has valid bits
                # Skip positions with '⊥' (uncertain) or '*' (conflict)
                valid_positions = [
                    i for i, c in enumerate(merged_codeword) 
                    if c in ('0', '1')
                ]
                
                if not valid_positions:
                    # All positions are uncertain, can't match
                    hamming_dist = float('inf')
                else:
                    # Compare only at valid positions
                    hamming_dist = sum(
                        merged_codeword[i] != user_codeword[i] 
                        for i in valid_positions
                    )
                
                hamming_distances[user_id] = hamming_dist
                
                # Allow small errors (threshold of 2 bits)
                # Also need to check that we have enough valid positions to make a meaningful comparison
                if len(valid_positions) >= 3 and hamming_dist <= 2:
                    matches.append(user_id)
            except Exception as e:
                # If we can't get codeword for a user, skip it
                continue
        
        return {
            'success': len(matches) > 0,
            'accused_user_ids': matches,
            'original_user_ids': original_user_ids,
            'matches': matches,
            'num_matches': len(matches),
            'merged_codeword': merged_codeword,
            'hamming_distances': hamming_distances,
            'method': 'direct_codeword_match'
        }
    except Exception as e:
        return {
            'success': False,
            'accused_user_ids': [],
            'original_user_ids': original_user_ids,
            'reason': f'Error during tracing: {str(e)}',
            'merged_codeword': merged_codeword
        }


def evaluate_collusion_case(
    muw, master_key: bytes, prompt: str, colluder_ids: list[int],
    case_name: str, max_new_tokens: int, model_name: str
) -> dict:
    """
    Evaluate a single collusion case: embed for each colluder, recover codewords, merge, trace.
    
    Args:
        muw: Multi-user watermarker instance
        master_key: Master secret key
        prompt: The prompt to use
        colluder_ids: List of colluding user IDs
        case_name: Name of the collusion case (e.g., 'same_group_2')
        max_new_tokens: Maximum tokens to generate
        model_name: Model name for parsing output
    
    Returns:
        Dictionary with evaluation results
    """
    # Embed for each colluder
    user_texts = []
    recovered_codewords = []
    
    for user_id in colluder_ids:
        # Embed watermark
        raw_text = muw.embed(master_key, user_id, prompt, max_new_tokens=max_new_tokens)
        final_text = parse_final_output(raw_text, model_name)
        user_texts.append(final_text)
        
        # Recover L-bit codeword
        recovered_codeword = muw.lbw.detect(master_key, final_text)
        recovered_codewords.append(recovered_codeword)
    
    # Merge codewords via bitwise majority
    merged_codeword = merge_codewords_bitwise_majority(recovered_codewords)
    
    # Trace merged pattern
    trace_result = trace_collusion(muw, master_key, merged_codeword, colluder_ids)
    
    return {
        'case_name': case_name,
        'colluder_ids': colluder_ids,
        'recovered_codewords': recovered_codewords,
        'merged_codeword': merged_codeword,
        'trace_result': trace_result,
        'success': trace_result['success']
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate naive and hierarchical watermarking schemes under controlled collusion",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--scheme',
        type=str,
        required=True,
        choices=['naive', 'hierarchical'],
        help='Watermarking scheme to use: naive or hierarchical'
    )
    parser.add_argument(
        '--group-bits',
        type=int,
        default=None,
        help='Number of bits for group codewords (required for hierarchical scheme)'
    )
    parser.add_argument(
        '--user-bits',
        type=int,
        default=None,
        help='Number of bits for user fingerprints (required for hierarchical scheme)'
    )
    parser.add_argument(
        '--l-bits',
        type=int,
        default=8,
        help='Total number of L-bits for watermarking (default: 8)'
    )
    parser.add_argument(
        '--prompts-file',
        type=str,
        required=True,
        help='Path to prompts file'
    )
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=300,
        help='Number of prompts to use (default: 300)'
    )
    parser.add_argument(
        '--users-file',
        type=str,
        default='assets/users.csv',
        help='Path to users CSV file (default: assets/users.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
        help='Model to use for generation and detection'
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
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scheme == 'hierarchical':
        if args.group_bits is None or args.user_bits is None:
            parser.error("--group-bits and --user-bits are required for hierarchical scheme")
        if args.group_bits + args.user_bits != args.l_bits:
            parser.error(
                f"--group-bits ({args.group_bits}) + --user-bits ({args.user_bits}) "
                f"must equal --l-bits ({args.l_bits})"
            )
    
    # Create output directory structure
    if args.scheme == 'hierarchical':
        scheme_dir = f"hierarchical_G{args.group_bits}_U{args.user_bits}"
    else:
        scheme_dir = f"naive_L{args.l_bits}"
    
    base_output_dir = args.output_dir
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(parent_dir, base_output_dir)
    
    scheme_output_dir = os.path.join(base_output_dir, scheme_dir)
    os.makedirs(scheme_output_dir, exist_ok=True)
    
    # Create subdirectories for 2 and 3 colluders
    output_2_dir = os.path.join(scheme_output_dir, '2_colluders')
    output_3_dir = os.path.join(scheme_output_dir, '3_colluders')
    os.makedirs(output_2_dir, exist_ok=True)
    os.makedirs(output_3_dir, exist_ok=True)
    
    # Print header
    print("\n" + "="*80)
    print(" " * 20 + "COLLUSION RESISTANCE EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  • Scheme: {args.scheme}")
    if args.scheme == 'hierarchical':
        print(f"  • Group bits: {args.group_bits}")
        print(f"  • User bits: {args.user_bits}")
    print(f"  • L-bits: {args.l_bits}")
    print(f"  • Model: {args.model}")
    print(f"  • Number of prompts: {args.num_prompts}")
    print(f"  • Output directory: {scheme_output_dir}")
    print("="*80)
    
    # Load prompts
    print(f"\n[1/4] Loading prompts...")
    prompts_path = os.path.join(parent_dir, args.prompts_file)
    if not os.path.exists(prompts_path):
        print(f"  ❌ Error: Prompts file not found: {prompts_path}")
        return
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        all_prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(all_prompts) < args.num_prompts:
        print(f"  ⚠ Warning: Only {len(all_prompts)} prompts available, using all of them")
        prompts = all_prompts
    else:
        prompts = all_prompts[:args.num_prompts]
    
    print(f"  ✓ Loaded {len(prompts)} prompts")
    
    # Load model
    print(f"\n[2/4] Loading model and initializing watermarker...")
    print(f"  → Loading model '{args.model}'...")
    model = get_model(args.model)
    print(f"  ✓ Model loaded successfully")
    
    # Initialize watermarker
    print(f"\n  → Initializing watermarker...")
    zero_bit = ZeroBitWatermarker(
        model=model,
        delta=args.delta,
        entropy_threshold=args.entropy_threshold,
        hashing_context=args.hashing_context,
        z_threshold=args.z_threshold
    )
    lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zero_bit, L=args.l_bits)
    
    if args.scheme == 'hierarchical':
        muw = HierarchicalMultiUserWatermarker(
            lbit_watermarker=lbit_watermarker,
            group_bits=args.group_bits,
            user_bits=args.user_bits,
            min_distance=2
        )
    else:
        muw = NaiveMultiUserWatermarker(lbit_watermarker=lbit_watermarker)
    
    # Load users
    users_path = os.path.join(parent_dir, args.users_file)
    if not os.path.exists(users_path):
        print(f"  ❌ Error: Users file not found: {users_path}")
        return
    
    muw.load_users(users_path)
    print(f"  ✓ Loaded {muw.N} users")
    
    # Generate master key
    master_key = muw.keygen()
    
    # Process each prompt
    print(f"\n[3/4] Processing {len(prompts)} prompts...")
    
    all_results = []
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts", unit="prompt")):
        prompt_results = {
            'prompt_id': prompt_idx,
            'prompt': prompt,
            'scheme': args.scheme,
            'config': {
                'l_bits': args.l_bits,
                'group_bits': args.group_bits if args.scheme == 'hierarchical' else None,
                'user_bits': args.user_bits if args.scheme == 'hierarchical' else None,
            },
            'results': {}
        }
        
        # 2 colluders cases
        try:
            # same_group_2
            colluders_same_2 = sample_same_group(muw, 2)
            result_same_2 = evaluate_collusion_case(
                muw, master_key, prompt, colluders_same_2,
                'same_group_2', args.max_new_tokens, args.model
            )
            prompt_results['results']['same_group_2'] = result_same_2
            
            # cross_group_2
            colluders_cross_2 = sample_different_groups(muw, 2)
            result_cross_2 = evaluate_collusion_case(
                muw, master_key, prompt, colluders_cross_2,
                'cross_group_2', args.max_new_tokens, args.model
            )
            prompt_results['results']['cross_group_2'] = result_cross_2
        except Exception as e:
            print(f"\n  ⚠ Warning: Error processing 2-colluder cases for prompt {prompt_idx}: {e}")
            prompt_results['results']['same_group_2'] = {'error': str(e)}
            prompt_results['results']['cross_group_2'] = {'error': str(e)}
        
        # 3 colluders cases
        try:
            # same_group_3
            colluders_same_3 = sample_same_group(muw, 3)
            result_same_3 = evaluate_collusion_case(
                muw, master_key, prompt, colluders_same_3,
                'same_group_3', args.max_new_tokens, args.model
            )
            prompt_results['results']['same_group_3'] = result_same_3
            
            # cross_group_3
            colluders_cross_3 = sample_different_groups(muw, 3)
            result_cross_3 = evaluate_collusion_case(
                muw, master_key, prompt, colluders_cross_3,
                'cross_group_3', args.max_new_tokens, args.model
            )
            prompt_results['results']['cross_group_3'] = result_cross_3
            
            # mixed_2same_1diff
            colluders_mixed = sample_2_same_1_diff(muw)
            result_mixed = evaluate_collusion_case(
                muw, master_key, prompt, colluders_mixed,
                'mixed_2same_1diff', args.max_new_tokens, args.model
            )
            prompt_results['results']['mixed_2same_1diff'] = result_mixed
        except Exception as e:
            print(f"\n  ⚠ Warning: Error processing 3-colluder cases for prompt {prompt_idx}: {e}")
            prompt_results['results']['same_group_3'] = {'error': str(e)}
            prompt_results['results']['cross_group_3'] = {'error': str(e)}
            prompt_results['results']['mixed_2same_1diff'] = {'error': str(e)}
        
        # Save prompt-level results in appropriate subdirectories
        # Save 2-colluder results
        prompt_json_2_path = os.path.join(output_2_dir, f'prompt_{prompt_idx}_results.json')
        prompt_results_2 = {
            'prompt_id': prompt_idx,
            'prompt': prompt,
            'scheme': args.scheme,
            'config': prompt_results['config'],
            'results': {
                'same_group_2': prompt_results['results'].get('same_group_2', {}),
                'cross_group_2': prompt_results['results'].get('cross_group_2', {})
            }
        }
        with open(prompt_json_2_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_results_2, f, indent=2, default=json_default_encoder)
        
        # Save 3-colluder results
        prompt_json_3_path = os.path.join(output_3_dir, f'prompt_{prompt_idx}_results.json')
        prompt_results_3 = {
            'prompt_id': prompt_idx,
            'prompt': prompt,
            'scheme': args.scheme,
            'config': prompt_results['config'],
            'results': {
                'same_group_3': prompt_results['results'].get('same_group_3', {}),
                'cross_group_3': prompt_results['results'].get('cross_group_3', {}),
                'mixed_2same_1diff': prompt_results['results'].get('mixed_2same_1diff', {})
            }
        }
        with open(prompt_json_3_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_results_3, f, indent=2, default=json_default_encoder)
        
        all_results.append(prompt_results)
    
    # Generate summary reports
    print(f"\n[4/4] Generating summary reports...")
    
    # Calculate success rates for 2-colluder cases
    case_types_2 = ['same_group_2', 'cross_group_2']
    success_rates_2 = {}
    
    for case_type in case_types_2:
        case_results = [
            r['results'].get(case_type, {}) 
            for r in all_results 
            if case_type in r['results'] and 'error' not in r['results'][case_type]
        ]
        if case_results:
            successful = sum(1 for r in case_results if r.get('success', False))
            total = len(case_results)
            success_rates_2[case_type] = {
                'successful': successful,
                'total': total,
                'success_rate': (successful / total) * 100.0 if total > 0 else 0.0
            }
    
    # Calculate success rates for 3-colluder cases
    case_types_3 = ['same_group_3', 'cross_group_3', 'mixed_2same_1diff']
    success_rates_3 = {}
    
    for case_type in case_types_3:
        case_results = [
            r['results'].get(case_type, {}) 
            for r in all_results 
            if case_type in r['results'] and 'error' not in r['results'][case_type]
        ]
        if case_results:
            successful = sum(1 for r in case_results if r.get('success', False))
            total = len(case_results)
            success_rates_3[case_type] = {
                'successful': successful,
                'total': total,
                'success_rate': (successful / total) * 100.0 if total > 0 else 0.0
            }
    
    # Save summary JSON for 2 colluders
    summary_2 = {
        'scheme': args.scheme,
        'config': {
            'l_bits': args.l_bits,
            'group_bits': args.group_bits if args.scheme == 'hierarchical' else None,
            'user_bits': args.user_bits if args.scheme == 'hierarchical' else None,
        },
        'num_prompts': len(prompts),
        'num_colluders': 2,
        'success_rates': success_rates_2
    }
    
    summary_json_2_path = os.path.join(output_2_dir, 'summary.json')
    with open(summary_json_2_path, 'w', encoding='utf-8') as f:
        json.dump(summary_2, f, indent=2, default=json_default_encoder)
    
    # Save summary JSON for 3 colluders
    summary_3 = {
        'scheme': args.scheme,
        'config': {
            'l_bits': args.l_bits,
            'group_bits': args.group_bits if args.scheme == 'hierarchical' else None,
            'user_bits': args.user_bits if args.scheme == 'hierarchical' else None,
        },
        'num_prompts': len(prompts),
        'num_colluders': 3,
        'success_rates': success_rates_3
    }
    
    summary_json_3_path = os.path.join(output_3_dir, 'summary.json')
    with open(summary_json_3_path, 'w', encoding='utf-8') as f:
        json.dump(summary_3, f, indent=2, default=json_default_encoder)
    
    # Print summary
    print("\n" + "="*80)
    print(" " * 25 + "RESULTS SUMMARY")
    print("="*80)
    print(f"\n2 Colluders - Success Rates:")
    print("-"*80)
    for case_type, stats in success_rates_2.items():
        print(f"  {case_type:20s}: {stats['success_rate']:6.2f}% ({stats['successful']}/{stats['total']})")
    
    print(f"\n3 Colluders - Success Rates:")
    print("-"*80)
    for case_type, stats in success_rates_3.items():
        print(f"  {case_type:20s}: {stats['success_rate']:6.2f}% ({stats['successful']}/{stats['total']})")
    
    print(f"\n✓ 2-colluder summary saved to: {summary_json_2_path}")
    print(f"✓ 3-colluder summary saved to: {summary_json_3_path}")
    print(f"✓ Prompt-level results saved to: {output_2_dir}/ and {output_3_dir}/")
    print("\n" + "="*80)
    print(" " * 30 + "✓ EVALUATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
