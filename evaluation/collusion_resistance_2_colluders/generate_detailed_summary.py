import json
import csv
import os
from collections import defaultdict

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the results JSON file
json_path = os.path.join(script_dir, 'collusion_resistance_results_2users.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Group results by approach, combination_method, and deletion_type
results_by_config = defaultdict(list)

for result in data['detailed_results']:
    approach = result['approach']
    combination_method = result['combination_method']
    deletion_type = result.get('deletion_type', 'none')
    deletion_percentage = result.get('deletion_percentage', 0.0)
    key = f"{approach}_{combination_method}_{deletion_type}_{deletion_percentage}"
    results_by_config[key].append(result)

# Calculate detailed metrics for each configuration
detailed_summary = []

for key in sorted(results_by_config.keys()):
    results = results_by_config[key]
    # Get values from the first result (all results in this group have the same values)
    approach = results[0]['approach']
    combination_method = results[0]['combination_method']
    deletion_type = results[0].get('deletion_type', 'none')
    deletion_percentage = results[0].get('deletion_percentage', 0.0)
    
    # Calculate metrics
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r.get('trace_result', {}).get('success', False))
    success_rate = (successful_cases / total_cases * 100) if total_cases > 0 else 0
    
    # Average number of matches (correct identifications)
    avg_matches = sum(r['trace_result'].get('num_matches', 0) for r in results) / total_cases if total_cases > 0 else 0
    
    # Determine if this is a grouped approach
    is_grouped = approach.startswith('min-distance')
    
    # Group configuration for calculating group IDs
    users_per_group = None
    if approach == 'min-distance-2':
        users_per_group = 10
    elif approach == 'min-distance-3':
        users_per_group = 20
    
    # Calculate false positives (user level for naive, group level for grouped)
    false_positives_list = []
    match_scores_list = []
    
    for r in results:
        original_ids = set(r.get('original_user_ids', []))
        trace_result = r.get('trace_result', {})
        accused_details = trace_result.get('accused_details', [])
        
        if is_grouped and users_per_group:
            # For grouped approaches: count false positives at GROUP level
            # Get original groups
            original_groups = {user_id // users_per_group for user_id in original_ids}
            
            # Get accused groups (from accused_details)
            accused_groups = set()
            for detail in accused_details:
                group_id = detail.get('group_id')
                if group_id is not None:
                    accused_groups.add(group_id)
            
            # False positive groups = accused groups not in original groups
            false_positive_groups = accused_groups - original_groups
            false_positives_list.append(len(false_positive_groups))
        else:
            # For naive approach: count false positives at USER level
            accused_ids = set(trace_result.get('accused_user_ids', []))
            false_positives = accused_ids - original_ids
            false_positives_list.append(len(false_positives))
        
        # Collect match scores
        for detail in accused_details:
            match_scores_list.append(detail.get('match_score_percent', 0))
    
    avg_false_positives = sum(false_positives_list) / total_cases if total_cases > 0 else 0
    avg_match_score = sum(match_scores_list) / len(match_scores_list) if match_scores_list else 0
    
    # Calculate precision: correct matches / total accused
    # For grouped: correct groups / total accused groups
    # For naive: correct users / total accused users
    precision_list = []
    for r in results:
        original_ids = set(r.get('original_user_ids', []))
        trace_result = r.get('trace_result', {})
        matches = trace_result.get('num_matches', 0)
        
        if is_grouped and users_per_group:
            # For grouped: precision = correct groups / total accused groups
            original_groups = {user_id // users_per_group for user_id in original_ids}
            accused_details = trace_result.get('accused_details', [])
            accused_groups = {detail.get('group_id') for detail in accused_details if detail.get('group_id') is not None}
            
            # Count how many original groups were correctly identified
            # A group is correctly identified if at least one user from that group was matched
            matched_user_ids = set(trace_result.get('matches', []))
            matched_groups = {user_id // users_per_group for user_id in matched_user_ids}
            correct_groups = len(matched_groups & original_groups)
            total_accused_groups = len(accused_groups)
            
            if total_accused_groups > 0:
                precision = correct_groups / total_accused_groups
                precision_list.append(precision)
        else:
            # For naive: precision = correct users / total accused users
            accused_ids = set(trace_result.get('accused_user_ids', []))
            total_accused = len(accused_ids)
            if total_accused > 0:
                precision = matches / total_accused
                precision_list.append(precision)
    
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    
    # Calculate recall: correct matches / total original colluders
    # For grouped: correct groups / total original groups
    # For naive: correct users / total original users
    recall_list = []
    for r in results:
        original_ids = set(r.get('original_user_ids', []))
        trace_result = r.get('trace_result', {})
        matches = trace_result.get('num_matches', 0)
        
        if is_grouped and users_per_group:
            # For grouped: recall = correct groups / total original groups
            original_groups = {user_id // users_per_group for user_id in original_ids}
            matched_user_ids = set(trace_result.get('matches', []))
            matched_groups = {user_id // users_per_group for user_id in matched_user_ids}
            correct_groups = len(matched_groups & original_groups)
            total_original_groups = len(original_groups)
            
            if total_original_groups > 0:
                recall = correct_groups / total_original_groups
                recall_list.append(recall)
        else:
            # For naive: recall = correct users / total original users
            total_original = len(original_ids)
            if total_original > 0:
                recall = matches / total_original
                recall_list.append(recall)
    
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    
    # Calculate F1 score
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0
    
    detailed_summary.append({
        'approach': approach,
        'combination_method': combination_method,
        'deletion_type': deletion_type,
        'deletion_percentage': deletion_percentage,
        'total_cases': total_cases,
        'successful_cases': successful_cases,
        'success_rate_percent': round(success_rate, 2),
        'avg_num_matches': round(avg_matches, 3),
        'avg_false_positives': round(avg_false_positives, 3),
        'avg_match_score_percent': round(avg_match_score, 2),
        'avg_precision': round(avg_precision, 3),
        'avg_recall': round(avg_recall, 3),
        'avg_f1_score': round(f1_score, 3)
    })

# Write to CSV
output_file = os.path.join(script_dir, 'collusion_resistance_detailed_summary.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = [
        'approach', 'combination_method', 'deletion_type', 'deletion_percentage',
        'total_cases', 'successful_cases', 'success_rate_percent', 'avg_num_matches',
        'avg_false_positives', 'avg_match_score_percent', 'avg_precision', 'avg_recall', 'avg_f1_score'
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(detailed_summary)

print(f"Detailed summary written to {output_file}")
print(f"\nSummary by approach and deletion type:")
for row in detailed_summary:
    deletion_info = f"{row['deletion_type']}({row['deletion_percentage']})" if row['deletion_type'] != 'none' else 'none'
    print(f"{row['approach']:20} {row['combination_method']:20} {deletion_info:20} Success: {row['success_rate_percent']:6.2f}% "
          f"Precision: {row['avg_precision']:.3f} Recall: {row['avg_recall']:.3f} F1: {row['avg_f1_score']:.3f}")

