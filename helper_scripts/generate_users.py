"""
Script to generate users.csv file with 1000 users.
Each user has a UserId starting from 0, and Username is the same as the UserId.
UserIds are 0-indexed (0-999) to match the fingerprinting code requirements.
"""

import csv
import os

def generate_users_csv(output_path='../assets/users.csv', num_users=1000):
    """
    Generate a CSV file with user data.
    
    Args:
        output_path: Path to the output CSV file
        num_users: Number of users to generate (default: 1000)
    """
    # Get the absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(script_dir, output_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    
    # Write CSV file
    with open(full_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['UserId', 'Username'])
        
        # Write user data starting from user ID 0 (0-indexed)
        for user_id in range(0, num_users):
            writer.writerow([user_id, str(user_id)])
    
    print(f"Successfully generated {num_users} users (IDs 0-{num_users-1}) in {full_output_path}")

if __name__ == '__main__':
    generate_users_csv()

