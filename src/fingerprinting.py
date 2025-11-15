import numpy as np
import pandas as pd
import os

class FingerprintingCode:
    """
    Implements a fingerprinting code for multi-user watermarking with unique 8-bit codewords derived from user IDs.
    Integrates with a CSV database for user metadata.
    """
    def __init__(self, L: int = 8, c: int = 16, delta: float = 0.1):
        """
        Initializes the fingerprinting code.

        Args:
            L (int): Codeword length (fixed to 8 for user ID encoding).
            c (int): Maximum number of colluders.
            delta (float): Erasure probability.
        """
        self.N = None
        self.L = L
        self.c = c
        self.delta = delta
        self.codewords = None
        self.user_metadata = None

    def gen(self, users_file: str = "users.csv") -> np.ndarray:
        """
        Loads a user CSV and generates an N x L binary codeword matrix by
        converting user IDs to L-bit binary.

        Args:
            users_file (str): Path to CSV file for user metadata.

        Returns:
            np.ndarray: N x L binary matrix (codewords for each user).
        """
        # 1. Load the user metadata from the CSV. This also sets self.N.
        self.load_metadata(users_file)
        
        # 2. Generate codewords deterministically from UserIds
        for index, row in self.user_metadata.iterrows():
            user_id = int(row["UserId"])
            if user_id >= self.N:
                 raise ValueError(f"UserId {user_id} is out of bounds for N={self.N} users (must be 0 to {self.N-1}).")
            
            # Convert user_id to L-bit binary
            binary = format(user_id, f'0{self.L}b')
            # The codeword for a user is their ID in binary
            self.codewords[user_id] = np.array([int(bit) for bit in binary])
        
        print(f"Successfully loaded {self.N} users and generated deterministic codes.")
        return self.codewords

    def trace(self, recovered_message: str) -> list:
        """
        Traces a recovered message by finding the user(s) with the highest
        number of matching bits.

        Args:
            recovered_message (str): Recovered 8-bit string with possible ⊥, *, or ? symbols.

        Returns:
            list: List of dictionaries with accused user IDs and metadata.
        """
        if len(recovered_message) != self.L:
            raise ValueError(f"Recovered message must be {self.L} bits")
        if self.codewords is None:
            raise ValueError("Codewords not generated. Call .gen_from_file() first.")
        if self.user_metadata is None:
            raise ValueError("User metadata not loaded. Call .gen_from_file() first.")

        accused = []
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

        # Return all users tied for the highest match score, up to collusion limit
        for user_id, username, matches in potential_matches:
            if len(accused) < self.c:
                accused.append({
                    "user_id": user_id,
                    "username": username,
                    "match_score_percent": (matches / len(valid_bits)) * 100
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