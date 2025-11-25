#!/usr/bin/env python3
"""
compute_code_capacity.py

Calculates theoretical maximum number of unique binary codewords (A2(L,d))
for a given codeword length L and minimum Hamming distance d.

Uses:
- Hamming bound (sphere packing upper bound)
- Gilbert-Varshamov lower bound
"""

import math


# Known optimal values A2(L,d) for small L (L <= 10)
# A2(L,d) = maximum number of binary codewords of length L with minimum distance d
KNOWN_OPTIMAL = {
    (1, 1): 2,
    (2, 1): 4, (2, 2): 2,
    (3, 1): 8, (3, 2): 4, (3, 3): 2,
    (4, 1): 16, (4, 2): 8, (4, 3): 4, (4, 4): 2,
    (5, 1): 32, (5, 2): 16, (5, 3): 4, (5, 4): 4, (5, 5): 2,
    (6, 1): 64, (6, 2): 32, (6, 3): 8, (6, 4): 4, (6, 5): 2, (6, 6): 2,
    (7, 1): 128, (7, 2): 64, (7, 3): 16, (7, 4): 8, (7, 5): 2, (7, 6): 2, (7, 7): 2,
    (8, 1): 256, (8, 2): 128, (8, 3): 20, (8, 4): 16, (8, 5): 4, (8, 6): 2, (8, 7): 2, (8, 8): 2,
    (9, 1): 512, (9, 2): 256, (9, 3): 40, (9, 4): 20, (9, 5): 4, (9, 6): 4, (9, 7): 2, (9, 8): 2, (9, 9): 2,
    (10, 1): 1024, (10, 2): 512, (10, 3): 72, (10, 4): 40, (10, 5): 6, (10, 6): 4, (10, 7): 2, (10, 8): 2, (10, 9): 2, (10, 10): 2,
}


def hamming_bound(L: int, d: int) -> float:
    """
    Compute the Hamming bound (sphere packing upper bound) for A2(L,d).
    
    The Hamming bound states that:
    A2(L,d) <= 2^L / V(L, t)
    where t = floor((d-1)/2) and V(L, t) is the volume of a Hamming sphere of radius t.
    
    Args:
        L: Codeword length
        d: Minimum Hamming distance
    
    Returns:
        Upper bound on the number of codewords
    """
    if d <= 0:
        return float('inf')
    
    # t = floor((d-1)/2) is the error-correcting capability
    t = (d - 1) // 2
    
    # Volume of Hamming sphere of radius t
    volume = sum(math.comb(L, i) for i in range(t + 1))
    
    if volume == 0:
        return float('inf')
    
    # Hamming bound: A2(L,d) <= 2^L / V(L, t)
    upper_bound = (2 ** L) / volume
    
    return upper_bound


def gilbert_varshamov_bound(L: int, d: int) -> float:
    """
    Compute the Gilbert-Varshamov lower bound for A2(L,d).
    
    The Gilbert-Varshamov bound states that:
    A2(L,d) >= 2^L / sum_{i=0}^{d-1} C(L, i)
    where the sum is the volume of a Hamming sphere of radius d-1.
    
    Args:
        L: Codeword length
        d: Minimum Hamming distance
    
    Returns:
        Lower bound on the number of codewords
    """
    if d <= 0:
        return 0.0
    if d == 1:
        return 2 ** L
    if d > L:
        return 1.0
    
    # Volume of Hamming sphere of radius d-1
    # Sum of binomial coefficients from i=0 to d-1
    sum_binomials = sum(math.comb(L, i) for i in range(d))
    
    if sum_binomials == 0:
        return float('inf')
    
    # Gilbert-Varshamov bound: A2(L,d) >= 2^L / V(L, d-1)
    lower_bound = (2 ** L) / sum_binomials
    
    return lower_bound


def main():
    """Interactive main function."""
    print("=" * 70)
    print("Binary Code Capacity Calculator")
    print("Computes theoretical bounds for A2(L,d)")
    print("=" * 70)
    print()
    
    # Get user input
    try:
        L = int(input("Enter codeword length L: "))
        if L <= 0:
            print("Error: L must be positive.")
            return
        
        d = int(input("Enter minimum distance d: "))
        if d <= 0:
            print("Error: d must be positive.")
            return
        
        if d > L:
            print(f"Warning: d ({d}) > L ({L}). Minimum distance cannot exceed codeword length.")
            print("Setting d = L for calculation purposes.")
            d = L
        
        if d > 2 * L:
            print("Error: d cannot exceed 2*L for binary codes.")
            return
        
    except ValueError:
        print("Error: Please enter valid integers.")
        return
    except KeyboardInterrupt:
        print("\n\nExiting...")
        return
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    # Calculate values
    total_possible = 2 ** L
    hamming_upper = hamming_bound(L, d)
    gv_lower = gilbert_varshamov_bound(L, d)
    
    # Print input parameters
    print(f"Codeword length (L):           {L}")
    print(f"Minimum distance (d):          {d}")
    print()
    
    # Print total possible
    print(f"Total possible bitstrings:      {total_possible:,}")
    print()
    
    # Print bounds
    print("Theoretical Bounds:")
    print("-" * 70)
    
    if hamming_upper == float('inf'):
        print(f"Hamming bound (upper):         ∞ (infinite)")
    else:
        print(f"Hamming bound (upper):         {hamming_upper:.6f}")
        print(f"  → Rounded integer:           {int(round(hamming_upper)):,}")
    
    print()
    
    if gv_lower == float('inf'):
        print(f"Gilbert-Varshamov (lower):     ∞ (infinite)")
    else:
        print(f"Gilbert-Varshamov (lower):     {gv_lower:.6f}")
        print(f"  → Rounded integer:           {int(round(gv_lower)):,}")
    
    print()
    
    # Print known optimal value if available
    if (L, d) in KNOWN_OPTIMAL:
        optimal = KNOWN_OPTIMAL[(L, d)]
        print(f"Known optimal value A2({L},{d}):  {optimal:,}")
        
        # Check if bounds are tight
        if hamming_upper != float('inf'):
            hamming_int = int(round(hamming_upper))
            if optimal == hamming_int:
                print("  ✓ Hamming bound is tight (matches optimal)")
            elif optimal < hamming_int:
                print(f"  → Hamming bound is loose (optimal is {hamming_int - optimal} less)")
        
        if gv_lower != float('inf'):
            gv_int = int(round(gv_lower))
            if optimal == gv_int:
                print("  ✓ Gilbert-Varshamov bound is tight (matches optimal)")
            elif optimal > gv_int:
                print(f"  → Gilbert-Varshamov bound is loose (optimal is {optimal - gv_int} more)")
        
        print()
    
    # Print summary
    print("-" * 70)
    print("Summary:")
    print(f"  A2({L},{d}) is between {int(round(gv_lower)):,} and {int(round(hamming_upper)):,} codewords")
    if (L, d) in KNOWN_OPTIMAL:
        print(f"  Actual optimal: {KNOWN_OPTIMAL[(L, d)]:,} codewords")
    print("=" * 70)


if __name__ == "__main__":
    main()

