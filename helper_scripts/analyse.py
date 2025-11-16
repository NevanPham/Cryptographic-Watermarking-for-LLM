# analyse.py

import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_robustness_static(df, output_dir, params, z_threshold):
    """Creates a box plot for a single, static parameter run."""
    plt.figure(figsize=(12, 7))
    title_suffix = f"(Delta={params['delta']}, HC={params['hashing_context']}, ET={params['entropy_threshold']})"

    robustness_data = df[(df['type'] == 'watermarked')]
    sns.boxplot(x='perturbation', y='z_score', data=robustness_data)
    
    plt.title(f'Robustness to Perturbations {title_suffix}')
    plt.xlabel('Perturbation Type')
    plt.ylabel('Z-Score')
    plt.axhline(y=z_threshold, color='red', linestyle='--', label=f'Threshold (z={z_threshold})')
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'robustness_boxplot{title_suffix.replace(" ", "")}.png')
    plt.savefig(plot_path)
    print(f"📊 Static robustness plot saved.")
    plt.close()

def plot_completeness_vs_soundness(df, output_dir, params, z_threshold):
    """
    Plots the full distribution and a zoomed-in view of z-scores for a specific parameter set.
    """
    title_suffix = f"(Delta={params['delta']}, HC={params['hashing_context']}, ET={params['entropy_threshold']})"
    file_suffix = f"_delta_{params['delta']}_hc_{params['hashing_context']}_et_{params['entropy_threshold']}"

    # Filter data for the specific plot
    wm_scores = df[
        (df['type'] == 'watermarked') & 
        (df['perturbation'] == 'clean') & 
        (df['delta'] == params['delta']) &
        (df['hashing_context'] == params['hashing_context']) &
        (df['entropy_threshold'] == params['entropy_threshold'])
    ]['z_score']
    unwm_scores = df[df['type'] == 'unwatermarked']['z_score']

    if wm_scores.empty:
        print(f"Warning: No 'clean' watermarked data found for parameters: {params}. Skipping this completeness plot.")
        return
    if unwm_scores.empty:
        print("Warning: No unwatermarked data found. Skipping all completeness plots.")
        return


    # --- 1. Generate the Full Distribution Plot (Improved) ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.histplot(wm_scores, color="red", label='Watermarked (Completeness)', stat="count", bins=100, ax=ax1, alpha=0.7)
    ax1_twin = ax1.twinx()
    sns.kdeplot(unwm_scores, color="blue", label='Unwatermarked (Soundness)', ax=ax1_twin, linewidth=2.5)
    
    ax1.set_title(f'Full Z-Score Distribution {title_suffix}')
    ax1.set_xlabel('Z-Score')
    ax1.set_ylabel('Watermarked Count')
    ax1_twin.set_ylabel('Unwatermarked Density')
    ax1.axvline(x=z_threshold, color='black', linestyle='--', label=f'Threshold (z={z_threshold})')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines + lines2, labels + labels2, loc='upper right')
    
    x_min = min(unwm_scores.min(), wm_scores.min()) - 2
    x_max = max(wm_scores.quantile(0.99), unwm_scores.max()) + 5
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    full_plot_path = os.path.join(output_dir, f'completeness_dist{file_suffix}.png')
    fig1.savefig(full_plot_path)
    plt.close(fig1)

    # --- 2. Generate the Zoomed-In Plot ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    x_zoom_range = (z_threshold - 5, z_threshold + 5)
    bins_for_zoom = max(10, int((x_zoom_range[1] - x_zoom_range[0]) * 4))

    sns.histplot(unwm_scores, color="blue", label='Unwatermarked (Soundness)', stat="count", 
                 bins=bins_for_zoom, ax=ax2, alpha=0.7, edgecolor='black', binrange=x_zoom_range, kde=False)
    sns.histplot(wm_scores, color="red", label=f'Watermarked (Completeness)', stat="count", 
                 bins=bins_for_zoom, ax=ax2, alpha=0.7, edgecolor='black', binrange=x_zoom_range, kde=False)

    ax2.set_title(f'Zoomed View @ Threshold {title_suffix}')
    ax2.set_xlabel('Z-Score')
    ax2.set_ylabel('Count')
    ax2.axvline(x=z_threshold, color='gray', linestyle='--', label=f'Threshold (z={z_threshold})')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_xlim(*x_zoom_range)
    
    zoomed_plot_path = os.path.join(output_dir, f'completeness_dist{file_suffix}_ZOOMED.png')
    fig2.savefig(zoomed_plot_path)
    print(f"📊 Completeness/Soundness plots saved for {title_suffix}")
    plt.close(fig2)

def plot_robustness_sweep(df, output_dir, sweep_var, z_threshold):
    """
    Plots the detection rate on PERTURBED watermarked text vs. the sweep parameter
    using a grouped bar chart for clearer comparison, with improved spacing and grid.
    """
    plt.figure(figsize=(14, 8))
    
    # Exclude 'clean' texts to focus only on robustness against attacks
    robustness_df = df[(df['type'] == 'watermarked') & (df['perturbation'] != 'clean')].copy()
    
    if robustness_df.empty:
        print("No perturbed text data found to generate robustness plot.")
        plt.close()
        return

    # Calculate detection success rate as a percentage
    robustness_df['Detection Rate (%)'] = (robustness_df['z_score'] > z_threshold) * 100
    
    # --- Create the Grouped Bar Chart with improved aesthetics ---
    sns.barplot(
        data=robustness_df,
        x=sweep_var,
        y='Detection Rate (%)',
        hue='perturbation',
        errorbar='sd',
        capsize=.08,  # Slightly larger caps for error bars
        palette='deep', # Use a different palette for distinct colors
        edgecolor='black', # Add black edges for better bar definition
        linewidth=0.5 # Line width for the edges
    )
    
    plt.title(f'Robustness vs. {sweep_var.replace("_", " ").title()}')
    plt.xlabel(f'{sweep_var.replace("_", " ").title()}')
    plt.ylabel(f'Detection Rate (%) @ Z > {z_threshold}')
    plt.ylim(0, 105)
    
    # --- Adjust legend position for better fit ---
    plt.legend(title='Attack Type', loc='lower left', bbox_to_anchor=(0.01, 0.01))
    
    # --- Add both X and Y grid lines ---
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7) # Added X-axis grid
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'robustness_sweep_vs_{sweep_var}.png')
    plt.savefig(plot_path)
    print(f"📊 Robustness sweep plot saved to {plot_path}")
    plt.close()

def main():
    """Main function to parse results and generate plots."""
    parser = argparse.ArgumentParser(description="Parse evaluation results and generate plots for the watermarking framework.")
    parser.add_argument('eval_dir', type=str, help='Path to the evaluation results directory.')
    parser.add_argument('--z-threshold', type=float, default=4.0, help='The z-score threshold to use for calculating detection rates.')
    args = parser.parse_args()

    results_path = os.path.join(args.eval_dir, 'analysis_results.json')
    if not os.path.exists(results_path):
        print(f"Error: Could not find 'analysis_results.json' in '{args.eval_dir}'")
        return

    print(f"Loading results from {results_path}")
    df = pd.read_json(results_path)
    
    df = df[df['z_score'].apply(lambda x: isinstance(x, (int, float)))]
    p_categories = ['clean', 'substitute 30 percent', 'delete start 20 percent', 'delete middle 20 percent', 'delete end 20 percent']
    df['perturbation'] = pd.Categorical(df['perturbation'], categories=[c for c in p_categories if c in df['perturbation'].unique()], ordered=True)

    # --- Identify the swept parameter ---
    sweep_var = None
    param_cols = [col for col in ['delta', 'hashing_context', 'entropy_threshold'] if col in df.columns]
    unique_counts = {col: df[col].nunique() for col in param_cols}
    for var, count in unique_counts.items():
        if count > 1:
            sweep_var = var
            break
    
    summary_lines = []
    
    # --- Branch logic based on whether a sweep was detected ---
    if sweep_var:
        print(f"\n--- Detected sweep of parameter: '{sweep_var}' ---")
        sweep_values = sorted(df[df[sweep_var].notna()][sweep_var].unique())
        
        # --- Generate Plots for the Sweep ---
        print(f"Generating plots for each value: {sweep_values}")
        for value in sweep_values:
            # Construct a params dict for the current value in the sweep
            static_params = {
                'delta': df['delta'].iloc[0] if 'delta' != sweep_var else value,
                'hashing_context': df['hashing_context'].iloc[0] if 'hashing_context' != sweep_var else value,
                'entropy_threshold': df['entropy_threshold'].iloc[0] if 'entropy_threshold' != sweep_var else value
            }
            plot_completeness_vs_soundness(df, args.eval_dir, static_params, args.z_threshold)
        
        plot_robustness_sweep(df, args.eval_dir, sweep_var, args.z_threshold)

        # --- Quantitative Analysis for the Sweep ---
        summary_lines.append("--- Quantitative Analysis Summary (Sweep) ---")
        unwatermarked_df = df[df['type'] == 'unwatermarked']
        false_positives = len(unwatermarked_df[unwatermarked_df['z_score'] > args.z_threshold])
        fp_rate = (false_positives / len(unwatermarked_df)) * 100 if len(unwatermarked_df) > 0 else 0
        summary_lines.append(f"\nSoundness @ Z-Threshold={args.z_threshold}:")
        summary_lines.append(f"  - False Positive Rate: {fp_rate:.2f}% ({false_positives}/{len(unwatermarked_df)})")

        for value in sweep_values:
            summary_lines.append(f"\nMetrics for {sweep_var.replace('_', ' ')} = {value} @ Z-Threshold={args.z_threshold}:")
            for p_type in df['perturbation'].cat.categories:
                subset = df[(df[sweep_var] == value) & (df['perturbation'] == p_type)]
                if subset.empty: continue
                total, detected = len(subset), len(subset[subset['z_score'] > args.z_threshold])
                detection_rate = (detected / total) * 100 if total > 0 else 0
                metric_name = "Completeness" if p_type == 'clean' else f"Robustness ({p_type.replace('_', ' ')})"
                summary_lines.append(f"  - {metric_name}: {detection_rate:.2f}% detected ({detected}/{total})")

    else: # Handle the static parameter case
        print("\n--- Detected a single parameter run ---")
        
        if not df[df['type'] == 'watermarked'].empty:
            # Extract the single set of parameters from the first watermarked entry
            static_params = df[df['type'] == 'watermarked'].iloc[0][param_cols].to_dict()
            
            # --- Generate Plots for the Static Run ---
            plot_completeness_vs_soundness(df, args.eval_dir, static_params, args.z_threshold)
            plot_robustness_static(df, args.eval_dir, static_params, args.z_threshold) # Use the box plot for static runs

            # --- Quantitative Analysis for the Static Run ---
            summary_lines.append("--- Quantitative Analysis Summary (Static) ---")
            unwatermarked_df = df[df['type'] == 'unwatermarked']
            false_positives = len(unwatermarked_df[unwatermarked_df['z_score'] > args.z_threshold])
            fp_rate = (false_positives / len(unwatermarked_df)) * 100 if len(unwatermarked_df) > 0 else 0
            summary_lines.append(f"\nSoundness @ Z-Threshold={args.z_threshold}:")
            summary_lines.append(f"  - False Positive Rate: {fp_rate:.2f}% ({false_positives}/{len(unwatermarked_df)})")

            summary_lines.append(f"\nMetrics for {static_params} @ Z-Threshold={args.z_threshold}:")
            for p_type in df['perturbation'].cat.categories:
                subset = df[df['perturbation'] == p_type]
                if subset.empty: continue
                total, detected = len(subset), len(subset[subset['z_score'] > args.z_threshold])
                detection_rate = (detected / total) * 100 if total > 0 else 0
                metric_name = "Completeness" if p_type == 'clean' else f"Robustness ({p_type.replace('_', ' ')})"
                summary_lines.append(f"  - {metric_name}: {detection_rate:.2f}% detected ({detected}/{total})")
        else:
            print("No watermarked data found to generate plots or summary.")

    # --- Print to console and save to file ---
    if summary_lines:
        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)
        summary_path = os.path.join(args.eval_dir, 'summary_analysis.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        print(f"\n📝 Quantitative summary saved to {summary_path}")

if __name__ == "__main__":
    main()