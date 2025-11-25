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
        Generates BCH codewords with guaranteed minimum Hamming distance.
        
        Args:
            num_groups (int): Number of group codewords needed.
            
        Returns:
            dict: Maps group_id to codeword (as numpy array of L bits).
        """
        # Use a greedy approach to generate codewords with guaranteed minimum distance
        group_codewords = {}
        
        # Start with the first codeword (all zeros or a base pattern)
        first_codeword = np.zeros(self.L, dtype=int)
        group_codewords[0] = first_codeword
        
        # Generate remaining codewords ensuring minimum distance
        for group_id in range(1, num_groups):
            codeword = self._find_valid_codeword(group_id, group_codewords, num_groups)
            group_codewords[group_id] = codeword
        
        return group_codewords
    
    def _find_valid_codeword(self, group_id: int, existing_codewords: dict, num_groups: int) -> np.ndarray:
        """
        Finds a valid codeword for a group that maintains minimum distance from all existing codewords.
        Uses a greedy search approach to find valid codewords.
        """
        max_encoded = (2 ** self.L) - 1
        
        # Start with a candidate based on group_id
        # Try different starting points to find a valid codeword quickly
        start_candidates = []
        
        # Strategy 1: Use group_id with spacing
        if self.min_distance == 2:
            start_candidates.append(group_id * 2)
        elif self.min_distance == 3:
            start_candidates.append(group_id * 4)
        else:
            start_candidates.append(group_id * (2 ** (self.min_distance - 1)))
        
        # Strategy 2: Use group_id directly (might work for some cases)
        start_candidates.append(group_id)
        
        # Strategy 3: Use group_id with different multipliers
        for multiplier in [1, 2, 3, 5, 7]:
            candidate_val = group_id * multiplier
            if candidate_val <= max_encoded:
                start_candidates.append(candidate_val)
        
        # Try starting candidates first
        for start_val in start_candidates:
            if start_val > max_encoded:
                continue
            
            binary = format(start_val, f'0{self.L}b')
            if len(binary) > self.L:
                binary = binary[-self.L:]
            candidate = np.array([int(bit) for bit in binary])
            
            # Check if this candidate has minimum distance from all existing codewords
            valid = True
            for existing_id, existing_codeword in existing_codewords.items():
                dist = self._hamming_distance(candidate, existing_codeword)
                if dist < self.min_distance:
                    valid = False
                    break
            
            if valid:
                return candidate
        
        # If starting candidates don't work, use exhaustive search
        return self._find_valid_codeword_fallback(group_id, existing_codewords, num_groups)
    
    def _find_valid_codeword_fallback(self, group_id: int, existing_codewords: dict, num_groups: int) -> np.ndarray:
        """
        Fallback method to find a valid codeword by exhaustive search if the primary method fails.
        """
        max_encoded = (2 ** self.L) - 1
        
        # Try all possible codewords
        for candidate_val in range(max_encoded + 1):
            binary = format(candidate_val, f'0{self.L}b')
            candidate = np.array([int(bit) for bit in binary])
            
            # Check if this candidate has minimum distance from all existing codewords
            valid = True
            for existing_id, existing_codeword in existing_codewords.items():
                dist = self._hamming_distance(candidate, existing_codeword)
                if dist < self.min_distance:
                    valid = False
                    break
            
            if valid:
                return candidate
        
        # If still no valid codeword found, raise an error
        raise ValueError(
            f"Cannot generate codeword for group {group_id} with minimum distance {self.min_distance}. "
            f"Too many groups requested ({num_groups}) for codeword length {self.L}."
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