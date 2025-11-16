# test_undetectability.py

import argparse
import os
import random
import re

def prepare_test_samples(eval_dir, delta, prompts):
    """
    Prepares a fair, non-repeating set of test samples.
    """
    print(f"Preparing test session for delta={delta} from '{eval_dir}'...")
    
    try:
        all_files = os.listdir(eval_dir)
        prompt_ids = sorted(list({int(f.split('_')[0]) for f in all_files if f.split('_')[0].isdigit()}))
        
        test_plan = []
        for pid in prompt_ids:
            # Check if both watermarked (for the specific delta) and unwatermarked versions exist
            unwatermarked_file = f"{pid}_unwatermarked.txt"
            wm_pattern = re.compile(rf"^{pid}_wm_delta_{delta}_hc_.*?_et_.*?_clean\.txt$")
            watermarked_matches = [f for f in all_files if wm_pattern.match(f)]
            
            if unwatermarked_file in all_files and watermarked_matches:
                # Randomly choose whether to show the watermarked or unwatermarked version
                show_watermarked = random.choice([True, False])
                watermarked_file = watermarked_matches[0]
                
                if show_watermarked:
                    chosen_file = watermarked_file
                else:
                    chosen_file = unwatermarked_file
                
                test_plan.append({
                    'path': os.path.join(eval_dir, chosen_file),
                    'is_watermarked': show_watermarked,
                    'prompt': prompts[pid]
                })

        if not test_plan:
            print("Error: No valid pairs of watermarked/unwatermarked texts found for the given delta.")
            return None

        # Shuffle the final test plan so the order is random
        random.shuffle(test_plan)
        print(f"  - Prepared {len(test_plan)} unique, non-repeating test samples.")
        return test_plan

    except FileNotFoundError:
        print(f"Error: Evaluation directory not found at '{eval_dir}'")
        return None

def main():
    """Runs the command-line interface for the undetectability test."""
    parser = argparse.ArgumentParser(description="Run a CLI-based test for human undetectability.")
    parser.add_argument('eval_dir', type=str, help='Path to the evaluation results directory.')
    parser.add_argument('--prompts-file', type=str, required=True, help='Path to the .txt file containing the original prompts.')
    parser.add_argument('--delta', type=float, required=True, help='The specific delta value of watermarked texts to test against.')
    parser.add_argument('--num-tests', type=int, default=20, help='The number of samples to test in this session.')
    args = parser.parse_args()

    # Load the prompts from the specified file
    try:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: Prompts file not found at '{args.prompts_file}'")
        return

    test_samples = prepare_test_samples(args.eval_dir, args.delta, prompts)
    if not test_samples:
        return
        
    # Ensure we don't request more tests than available samples
    num_tests = min(args.num_tests, len(test_samples))
    if num_tests < args.num_tests:
        print(f"Warning: Requested {args.num_tests} tests, but only {num_tests} unique samples are available. Running {num_tests} tests.")

    score = 0
    
    print(f"\n--- Starting Undetectability Test ({num_tests} trials) ---")
    print("You will be shown a prompt and a generated response.")
    print("For each, decide if the RESPONSE is watermarked. Each prompt is unique.")
    print("-" * 55)

    for i in range(num_tests):
        os.system('cls' if os.name == 'nt' else 'clear')

        accuracy = (score / i) * 100 if i > 0 else 0
        print(f"Trial: {i + 1}/{num_tests} | Score: {score}/{i} ({accuracy:.1f}%) | Testing Delta: {args.delta}")
        print("-" * 60)

        sample = test_samples[i]
        with open(sample['path'], 'r', encoding='utf-8') as f:
            generated_text = f.read()
        
        # Display the prompt first for context
        print("--- PROMPT ---")
        print(sample['prompt'])
        print("\n--- GENERATED RESPONSE ---")
        print(generated_text)
        
        user_guess_str = ""
        while user_guess_str not in ['y', 'n']:
            user_guess_str = input("\nIs this RESPONSE watermarked? (y/n): ").lower().strip()
        
        user_guess_bool = (user_guess_str == 'y')
        is_correct = (user_guess_bool == sample['is_watermarked'])
        
        if is_correct:
            score += 1
            print("Correct!")
        else:
            print("Incorrect!")
        
        print(f"The response was {'watermarked' if sample['is_watermarked'] else 'not watermarked'}.")
        if i < num_tests - 1:
            input("\nPress Enter to continue to the next trial...")

    # Final results
    total_tested = num_tests
    print("\n\n--- Test Finished ---")
    accuracy = (score / total_tested) * 100 if total_tested > 0 else 0
    print(f"Final Score: {score}/{total_tested} correct ({accuracy:.1f}%)")
    if accuracy > 65:
        print(f"Result: The watermark may be human-detectable at delta={args.delta}.")
    elif accuracy < 35:
        print(f"Result: The watermark appears to be detectable at delta={args.delta} (you might be consistently guessing the opposite).")
    else:
        print(f"Result: Your score is close to random chance (50%). The watermark appears to be undetectable at delta={args.delta}.")
    print("-" * 21)

if __name__ == "__main__":
    main()