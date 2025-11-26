import numpy as np
import pandas as pd
import os

class FingerprintingCode:
    """
    Implements a fingerprinting code for multi-user watermarking using BCH error-correcting codes
    that guarantee minimum Hamming distance between codewords for better collusion resistance.
    Integrates with a CSV database for user metadata.
    """
    def __init__(self, L: int = 10, c: int = 16, delta: float = 0.1, min_distance: int = 2, 
                 max_groups: int = None, users_per_group: int = None):
        """
        Initializes the fingerprinting code.

        Args:
            L (int): Codeword length (default: 10 for BCH codes).
            c (int): Maximum number of colluders.
            delta (float): Erasure probability.
            min_distance (int): Minimum Hamming distance between codewords (default: 2).
            max_groups (int): Maximum number of groups allowed. If None, will be calculated based on min_distance.
            users_per_group (int): Number of users per group. If None, will be calculated based on min_distance.
        """
        self.N = None
        self.L = L
        self.c = c
        self.delta = delta
        self.min_distance = min_distance
        self.codewords = None
        self.user_metadata = None
        self.group_codewords = None  # Maps group_id to codeword
        self.user_to_group = None  # Maps user_id to group_id
        
        # Set group configuration - use provided values or defaults based on min_distance
        if max_groups is None or users_per_group is None:
            # Default configurations for backward compatibility
            if min_distance == 2:
                default_max_groups = 100
                default_users_per_group = 10
            elif min_distance == 3:
                default_max_groups = 50
                default_users_per_group = 20
            else:
                # For other min_distance values, use conservative defaults
                default_max_groups = 100
                default_users_per_group = 10
            
            self.max_groups = max_groups if max_groups is not None else default_max_groups
            self.users_per_group = users_per_group if users_per_group is not None else default_users_per_group
        else:
            self.max_groups = max_groups
            self.users_per_group = users_per_group

    def _generate_bch_codewords(self, num_groups: int) -> dict:
        """
        Generates codewords with guaranteed minimum Hamming distance using max-min greedy algorithm.
        
        This implementation uses a two-stage approach:
        1. Precompute all 2^L possible codewords
        2. Select codewords using maximum minimum distance greedy selection
        
        Args:
            num_groups (int): Number of group codewords needed.
            
        Returns:
            dict: Maps group_id to codeword (as numpy array of L bits).
        """
        # Stage 1: Precompute the full search space
        # Generate all integers from 0 to 2^L - 1 and convert to L-bit representations
        total_possible = 2 ** self.L
        
        # Use vectorized approach for better performance
        all_codes = np.zeros((total_possible, self.L), dtype=int)
        for i in range(total_possible):
            # Convert integer to binary representation
            binary_str = format(i, f'0{self.L}b')
            all_codes[i] = np.array([int(bit) for bit in binary_str])
        
        # Stage 2: Maximum minimum distance greedy selection
        # Special case for d=2: use optimal deterministic construction (all even-parity codewords)
        if self.min_distance == 2:
            # Optimal construction: all even-parity codewords
            # This gives exactly 2^(L-1) codewords, which is the theoretical maximum A(L,2)
            # All even-parity codewords have minimum distance ≥ 2 (any two differ in at least 2 positions)
            L = self.L
            max_codewords = 2 ** (L - 1)
            codewords = []
            
            # Deterministic enumeration: iterate through all 2^L possible codewords
            for i in range(2 ** L):
                # Convert integer to bit list
                bits = [(i >> b) & 1 for b in range(L)]
                # Check even parity (sum of bits is even)
                if sum(bits) % 2 == 0:
                    codewords.append(np.array(bits, dtype=int))
                    # Stop when we have enough codewords
                    if len(codewords) == num_groups:
                        break
            
            # Build result dictionary with exactly min(num_groups, 2^(L-1)) codewords
            group_codewords = {idx: cw for idx, cw in enumerate(codewords)}
            
            num_generated = len(group_codewords)
            print(f"Using optimal d=2 construction: {num_generated} codewords (theoretical max: {max_codewords}, requested: {num_groups})")
            
            # Validation: Check all pairwise distances (should all be ≥ 2)
            all_valid = True
            for i in range(num_generated):
                for j in range(i + 1, num_generated):
                    dist = self._hamming_distance(group_codewords[i], group_codewords[j])
                    if dist < self.min_distance:
                        all_valid = False
                        print(f"ERROR: Distance between codewords {i} and {j} is {dist} < {self.min_distance}")
            
            if all_valid:
                print(f"  All pairs have distance >= {self.min_distance}.")
            else:
                raise ValueError("Validation failed: Some codeword pairs do not meet minimum distance requirement.")
            
            return group_codewords
        else:
            # For d > 2, use max-min greedy algorithm
            selected_indices = []
            remaining_indices = set(range(total_possible))
            
            # Select the all-zero codeword as the first element
            first_codeword_idx = 0
            selected_indices.append(first_codeword_idx)
            remaining_indices.remove(first_codeword_idx)
            
            # For each subsequent codeword, select the one with maximum minimum distance
            for group_id in range(1, num_groups):
                best_candidate_idx = None
                best_min_distance = -1
                
                # For each candidate in the remaining pool
                for candidate_idx in remaining_indices:
                    candidate = all_codes[candidate_idx]
                    
                    # Compute minimum distance to all already chosen codewords
                    min_dist_to_selected = float('inf')
                    for selected_idx in selected_indices:
                        selected_codeword = all_codes[selected_idx]
                        dist = self._hamming_distance(candidate, selected_codeword)
                        if dist < min_dist_to_selected:
                            min_dist_to_selected = dist
                    
                    # Select candidate with maximum minimum distance
                    if min_dist_to_selected > best_min_distance:
                        best_min_distance = min_dist_to_selected
                        best_candidate_idx = candidate_idx
                
                # Check if we can still find a valid codeword
                if best_candidate_idx is None or best_min_distance < self.min_distance:
                    # No valid candidate remains
                    print(f"Warning: Could only generate {len(selected_indices)} codewords out of requested {num_groups}.")
                    print(f"  Maximum achievable with L={self.L}, d={self.min_distance} is approximately {len(selected_indices)} codewords.")
                    break
                
                # Accept the candidate
                selected_indices.append(best_candidate_idx)
                remaining_indices.remove(best_candidate_idx)
        
        # Build the result dictionary
        group_codewords = {}
        for group_id, code_idx in enumerate(selected_indices):
            group_codewords[group_id] = all_codes[code_idx].copy()
        
        # Validation: Check all pairwise distances
        num_generated = len(group_codewords)
        all_valid = True
        for i in range(num_generated):
            for j in range(i + 1, num_generated):
                dist = self._hamming_distance(group_codewords[i], group_codewords[j])
                if dist < self.min_distance:
                    all_valid = False
                    print(f"ERROR: Distance between codewords {i} and {j} is {dist} < {self.min_distance}")
        
        if all_valid:
            print(f"Generated {num_generated} codewords out of requested {num_groups}.")
            print(f"  All pairs have distance >= {self.min_distance}.")
        else:
            raise ValueError("Validation failed: Some codeword pairs do not meet minimum distance requirement.")
        
        return group_codewords
    
    def _find_valid_codeword(self, group_id: int, existing_codewords: dict, num_groups: int) -> np.ndarray:
        """
        Legacy method kept for backward compatibility.
        The new _generate_bch_codewords() method no longer uses this.
        """
        # This method is deprecated but kept for API compatibility
        # The new max-min greedy algorithm handles codeword selection internally
        raise NotImplementedError(
            "This method is deprecated. The new _generate_bch_codewords() uses max-min greedy selection."
        )
    
    def _hamming_distance(self, codeword1: np.ndarray, codeword2: np.ndarray) -> int:
        """Calculate Hamming distance between two codewords."""
        return np.sum(codeword1 != codeword2)
    
    def gen(self, users_file: str = "users.csv") -> np.ndarray:
        """
        Loads a user CSV and generates an N x L binary codeword matrix using
        BCH codes with guaranteed minimum Hamming distance.

        Args:
            users_file (str): Path to CSV file for user metadata.

        Returns:
            np.ndarray: N x L binary matrix (codewords for each user).
        """
        # 1. Load the user metadata from the CSV. This also sets self.N.
        self.load_metadata(users_file)
        original_N = self.N
        
        # 2. Calculate maximum users that can fit within max_groups constraint
        max_users_allowed = self.max_groups * self.users_per_group
        
        # 3. Limit to only the users that fit within the constraints
        if self.N > max_users_allowed:
            print(f"Warning: CSV contains {self.N} users, but max_groups={self.max_groups} and "
                  f"users_per_group={self.users_per_group} only allows {max_users_allowed} users. "
                  f"Using only the first {max_users_allowed} users.")
            # Truncate user_metadata to only the first max_users_allowed users
            self.user_metadata = self.user_metadata.head(max_users_allowed)
            self.N = max_users_allowed
            # Reinitialize codeword array with new N
            self.codewords = np.empty((self.N, self.L), dtype=int)
        
        # 4. Calculate number of groups needed (now guaranteed to fit)
        num_groups = (self.N + self.users_per_group - 1) // self.users_per_group
        
        # 5. Generate BCH codewords for groups
        print(f"Generating codewords with minimum distance {self.min_distance} for {num_groups} groups...")
        self.group_codewords = self._generate_bch_codewords(num_groups)
        
        # 6. Assign users to groups and assign codewords
        self.user_to_group = {}
        for index, row in self.user_metadata.iterrows():
            user_id = int(row["UserId"])
            if user_id >= self.N:
                raise ValueError(f"UserId {user_id} is out of bounds for N={self.N} users (must be 0 to {self.N-1}).")
            
            # Sequential group assignment
            group_id = user_id // self.users_per_group
            self.user_to_group[user_id] = group_id
            
            # Assign the group's codeword to this user
            if group_id not in self.group_codewords:
                raise ValueError(f"Group {group_id} not found. Maximum group ID should be {num_groups - 1}")
            
            self.codewords[user_id] = self.group_codewords[group_id].copy()
        
        # Print summary with group assignment details
        if original_N > self.N:
            print(f"Successfully loaded {self.N} users (out of {original_N} in file) and generated codes with minimum distance {self.min_distance}.")
        else:
            print(f"Successfully loaded {self.N} users and generated codes with minimum distance {self.min_distance}.")
        print(f"Users assigned to {num_groups} groups ({self.users_per_group} users per group).")
        
        # Show group ranges for clarity
        if num_groups <= 10:  # Only show if not too many groups
            print("Group assignments:")
            for gid in range(num_groups):
                start_user = gid * self.users_per_group
                end_user = min((gid + 1) * self.users_per_group - 1, self.N - 1)
                codeword_str = "".join(map(str, self.group_codewords[gid]))
                print(f"  Group {gid}: Users {start_user}-{end_user} (codeword: {codeword_str})")
        
        return self.codewords

    def trace(self, recovered_message: str) -> list:
        """
        Traces a recovered message by finding the user(s) with the highest
        number of matching bits. Shows group membership information.

        Args:
            recovered_message (str): Recovered L-bit string with possible ⊥, *, or ? symbols.

        Returns:
            list: List of dictionaries with accused user IDs and metadata, including group information.
        """
        if len(recovered_message) != self.L:
            raise ValueError(f"Recovered message must be {self.L} bits")
        if self.codewords is None:
            raise ValueError("Codewords not generated. Call .gen() first.")
        if self.user_metadata is None:
            raise ValueError("User metadata not loaded. Call .gen() first.")

        accused = []
        
        # NEW: Detect collusion
        collusion_detected = '*' in recovered_message
        collusion_positions = [i for i, bit in enumerate(recovered_message) if bit == '*']
        
        valid_bits = [i for i, bit in enumerate(recovered_message) if bit not in ['⊥', '*', '?']]
        if not valid_bits:
            return accused

        max_matches = -1
        potential_matches = []
        for user_idx in range(self.N):
            # Get user metadata by their ID (index)
            user_data = self.user_metadata.iloc[user_idx]
            
            matches = sum(
                int(recovered_message[i]) == self.codewords[user_idx, i]
                for i in valid_bits
            )
            
            if matches > max_matches:
                max_matches = matches
                potential_matches = [(user_idx, user_data['Username'], matches)]
            elif matches == max_matches:
                potential_matches.append((user_idx, user_data['Username'], matches))

        # Return all users tied for the highest match score (no artificial cap)
        for user_id, username, matches in potential_matches:
            group_id = self.user_to_group.get(user_id, None)
            
            accused.append({
                "user_id": user_id,
                "username": username,
                "match_score_percent": (matches / len(valid_bits)) * 100,
                "group_id": group_id,
                # NEW: Add collusion info
                "collusion_detected": collusion_detected,
                "collusion_positions": collusion_positions if collusion_detected else []
            })
        
        return accused

    def load_metadata(self, users_file: str):
        """
        Loads user metadata from a CSV file, sets self.N, and validates it.
        """
        if not os.path.exists(users_file):
            raise FileNotFoundError(f"User metadata file {users_file} not found")
            
        self.user_metadata = pd.read_csv(users_file)
        self.N = len(self.user_metadata) # Set N from the file
        
        if "UserId" not in self.user_metadata.columns:
            raise ValueError("users_file must contain a 'UserId' column.")
            
        if self.N > 2**self.L:
            raise ValueError(f"Number of users in file ({self.N}) exceeds maximum allowed by L={self.L} (max {2**self.L})")
            
        # Initialize codeword array now that N is known
        self.codewords = np.empty((self.N, self.L), dtype=int)

def generate_user_fingerprint(user_index_in_group: int, user_bits: int) -> str:
    """
    Generates a simple user fingerprint within a group.
    Uses binary representation of user index within the group, padded to user_bits.
    
    Args:
        user_index_in_group (int): The index of the user within their group (0-based).
        user_bits (int): The number of bits for the user fingerprint.
    
    Returns:
        str: Binary string of length user_bits representing the user fingerprint.
    """
    if user_index_in_group < 0:
        raise ValueError(f"user_index_in_group must be non-negative, got {user_index_in_group}")
    if user_bits <= 0:
        raise ValueError(f"user_bits must be positive, got {user_bits}")
    
    max_users = 2 ** user_bits
    if user_index_in_group >= max_users:
        raise ValueError(
            f"user_index_in_group {user_index_in_group} exceeds maximum for {user_bits} bits (max {max_users - 1})"
        )
    
    return format(user_index_in_group, f'0{user_bits}b')