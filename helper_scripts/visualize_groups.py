# visualize_groups.py
# Visualizes BCH-based fingerprinting groups, user assignments, and codewords

import os
import sys
import argparse
from collections import defaultdict

# Add the parent directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.fingerprinting import FingerprintingCode


def format_codeword(codeword_array):
    """Format codeword array as a binary string."""
    return "".join(map(str, codeword_array))


def calculate_hamming_distance(codeword1, codeword2):
    """Calculate Hamming distance between two codewords."""
    return sum(c1 != c2 for c1, c2 in zip(codeword1, codeword2))


def visualize_groups(fingerprinter, output_format='text'):
    """
    Visualize group assignments, user IDs, and codewords.
    
    Args:
        fingerprinter: FingerprintingCode instance with generated codewords
        output_format: 'text' for console output, 'detailed' for more info
    """
    if fingerprinter.codewords is None:
        print("Error: Codewords not generated. Call .gen() first.")
        return
    
    if fingerprinter.user_to_group is None:
        print("Error: User-to-group mapping not available.")
        return
    
    # Group users by group_id
    groups = defaultdict(list)
    for user_id, group_id in fingerprinter.user_to_group.items():
        groups[group_id].append(user_id)
    
    # Sort groups
    sorted_groups = sorted(groups.items())
    
    print("\n" + "=" * 80)
    print("BCH-Based Fingerprinting Group Visualization")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Codeword length (L): {fingerprinter.L}")
    print(f"  Minimum distance: {fingerprinter.min_distance}")
    print(f"  Total users: {fingerprinter.N}")
    print(f"  Total groups: {len(sorted_groups)}")
    print(f"  Users per group: {fingerprinter.users_per_group}")
    print(f"  Max groups allowed: {fingerprinter.max_groups}")
    
    print("\n" + "-" * 80)
    print("Group Assignments")
    print("-" * 80)
    
    # Display each group
    for group_id, user_ids in sorted_groups:
        user_ids.sort()
        codeword = fingerprinter.group_codewords[group_id]
        codeword_str = format_codeword(codeword)
        
        # Calculate user range
        start_user = min(user_ids)
        end_user = max(user_ids)
        
        print(f"\nGroup {group_id}:")
        print(f"  Codeword: {codeword_str}")
        print(f"  User range: {start_user}-{end_user} ({len(user_ids)} users)")
        
        if output_format == 'detailed':
            # Show all user IDs in this group
            if len(user_ids) <= 20:
                print(f"  User IDs: {', '.join(map(str, user_ids))}")
            else:
                # Show first 10 and last 10
                first_10 = user_ids[:10]
                last_10 = user_ids[-10:]
                print(f"  User IDs: {', '.join(map(str, first_10))} ... {', '.join(map(str, last_10))}")
                print(f"  (Total: {len(user_ids)} users)")
    
    # Verify minimum distance between groups
    print("\n" + "-" * 80)
    print("Minimum Distance Verification")
    print("-" * 80)
    
    group_ids = sorted(fingerprinter.group_codewords.keys())
    min_dist_found = float('inf')
    violations = []
    
    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            gid1, gid2 = group_ids[i], group_ids[j]
            codeword1 = fingerprinter.group_codewords[gid1]
            codeword2 = fingerprinter.group_codewords[gid2]
            dist = calculate_hamming_distance(codeword1, codeword2)
            
            min_dist_found = min(min_dist_found, dist)
            
            if dist < fingerprinter.min_distance:
                violations.append((gid1, gid2, dist))
    
    if violations:
        print(f"  ⚠️  WARNING: {len(violations)} distance violations found!")
        for gid1, gid2, dist in violations[:5]:  # Show first 5
            print(f"    Groups {gid1} and {gid2}: distance = {dist} (required: {fingerprinter.min_distance})")
        if len(violations) > 5:
            print(f"    ... and {len(violations) - 5} more violations")
    else:
        print(f"  ✓ All group codewords satisfy minimum distance requirement")
        print(f"  Minimum distance found: {min_dist_found} (required: {fingerprinter.min_distance})")
    
    # Statistics
    print("\n" + "-" * 80)
    print("Statistics")
    print("-" * 80)
    
    group_sizes = [len(user_ids) for _, user_ids in sorted_groups]
    print(f"  Average users per group: {sum(group_sizes) / len(group_sizes):.1f}")
    print(f"  Min users in a group: {min(group_sizes)}")
    print(f"  Max users in a group: {max(group_sizes)}")
    
    # Show codeword distribution
    print(f"\n  Codeword distribution:")
    codeword_counts = defaultdict(int)
    for group_id in sorted_groups:
        codeword_str = format_codeword(fingerprinter.group_codewords[group_id[0]])
        codeword_counts[codeword_str] += 1
    
    print(f"    Unique codewords: {len(codeword_counts)}")
    print(f"    Total groups: {len(sorted_groups)}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BCH-based fingerprinting groups, user assignments, and codewords."
    )
    parser.add_argument(
        '--users-file',
        type=str,
        default='assets/users.csv',
        help="Path to the user metadata CSV file (default: assets/users.csv)"
    )
    parser.add_argument(
        '--l-bits',
        type=int,
        default=10,
        help="Codeword length L (default: 10)"
    )
    parser.add_argument(
        '--min-distance',
        type=int,
        default=2,
        choices=[2, 3],
        help="Minimum Hamming distance between codewords (default: 2)"
    )
    parser.add_argument(
        '--max-groups',
        type=int,
        default=None,
        help="Maximum number of groups allowed (default: auto-calculated based on min-distance)"
    )
    parser.add_argument(
        '--users-per-group',
        type=int,
        default=None,
        help="Number of users per group (default: auto-calculated based on min-distance)"
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help="Show detailed output including all user IDs in each group"
    )
    
    args = parser.parse_args()
    
    # Initialize fingerprinting system
    print("Initializing fingerprinting system...")
    fingerprinter = FingerprintingCode(
        L=args.l_bits, 
        min_distance=args.min_distance,
        max_groups=args.max_groups,
        users_per_group=args.users_per_group
    )
    
    try:
        # Load users and generate codewords
        print(f"Loading users from {args.users_file}...")
        fingerprinter.gen(users_file=args.users_file)
        
        # Visualize
        output_format = 'detailed' if args.detailed else 'text'
        visualize_groups(fingerprinter, output_format=output_format)
        
    except FileNotFoundError:
        print(f"Error: Users file '{args.users_file}' not found.")
        print("Please ensure the file exists or specify a different path with --users-file")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

