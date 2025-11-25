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

# Group results by approach (aggregating both combination methods)
results_by_approach = defaultdict(list)

for result in data['detailed_results']:
    approach = result['approach']
    results_by_approach[approach].append(result)

# Calculate detailed metrics for each approach
approach_summary = []

for approach in sorted(results_by_approach.keys()):
    results = results_by_approach[approach]
    
    # Calculate metrics
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r.get('success', False))
    success_rate = (successful_cases / total_cases * 100) if total_cases > 0 else 0
    
    # Average number of matches (correct identifications)
    avg_matches = sum(r.get('trace_result', {}).get('num_matches', 0) for r in results) / total_cases if total_cases > 0 else 0
    
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
    precision_list = []
    recall_list = []
    
    for r in results:
        original_ids = set(r.get('original_user_ids', []))
        trace_result = r.get('trace_result', {})
        accused_details = trace_result.get('accused_details', [])
        matches = trace_result.get('num_matches', 0)
        
        if is_grouped and users_per_group:
            # For grouped approaches: count false positives at GROUP level
            original_groups = {user_id // users_per_group for user_id in original_ids}
            accused_groups = {detail.get('group_id') for detail in accused_details if detail.get('group_id') is not None}
            false_positive_groups = accused_groups - original_groups
            false_positives_list.append(len(false_positive_groups))
            
            # Precision: correct groups / total accused groups
            matched_user_ids = set(trace_result.get('matches', []))
            matched_groups = {user_id // users_per_group for user_id in matched_user_ids}
            correct_groups = len(matched_groups & original_groups)
            total_accused_groups = len(accused_groups)
            
            if total_accused_groups > 0:
                precision = correct_groups / total_accused_groups
                precision_list.append(precision)
            
            # Recall: correct groups / total original groups
            total_original_groups = len(original_groups)
            if total_original_groups > 0:
                recall = correct_groups / total_original_groups
                recall_list.append(recall)
        else:
            # For naive approach: count false positives at USER level
            accused_ids = set(trace_result.get('accused_user_ids', []))
            false_positives = accused_ids - original_ids
            false_positives_list.append(len(false_positives))
            
            # Precision: correct users / total accused users
            total_accused = len(accused_ids)
            if total_accused > 0:
                precision = matches / total_accused
                precision_list.append(precision)
            
            # Recall: correct users / total original users
            total_original = len(original_ids)
            if total_original > 0:
                recall = matches / total_original
                recall_list.append(recall)
        
        # Collect match scores
        for detail in accused_details:
            match_scores_list.append(detail.get('match_score_percent', 0))
    
    avg_false_positives = sum(false_positives_list) / total_cases if total_cases > 0 else 0
    avg_match_score = sum(match_scores_list) / len(match_scores_list) if match_scores_list else 0
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    
    # Calculate F1 score
    if avg_precision + avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0
    
    # Separate metrics for normal and with_deletion
    normal_results = [r for r in results if r.get('combination_method') == 'normal']
    deletion_results = [r for r in results if r.get('combination_method') == 'with_deletion']
    
    def calc_metrics_for_subset(subset_results):
        if not subset_results:
            return {
                'success_rate': 0,
                'avg_matches': 0,
                'avg_false_positives': 0,
                'avg_precision': 0,
                'avg_recall': 0,
                'avg_f1': 0
            }
        
        total = len(subset_results)
        successful = sum(1 for r in subset_results if r.get('success', False))
        success_rate = (successful / total * 100) if total > 0 else 0
        
        avg_matches = sum(r.get('trace_result', {}).get('num_matches', 0) for r in subset_results) / total if total > 0 else 0
        
        false_pos_list = []
        prec_list = []
        rec_list = []
        
        for r in subset_results:
            original_ids = set(r.get('original_user_ids', []))
            trace_result = r.get('trace_result', {})
            accused_details = trace_result.get('accused_details', [])
            matches = trace_result.get('num_matches', 0)
            
            if is_grouped and users_per_group:
                # For grouped: count false positives at GROUP level
                original_groups = {user_id // users_per_group for user_id in original_ids}
                accused_groups = {detail.get('group_id') for detail in accused_details if detail.get('group_id') is not None}
                false_positive_groups = accused_groups - original_groups
                false_pos_list.append(len(false_positive_groups))
                
                # Precision: correct groups / total accused groups
                matched_user_ids = set(trace_result.get('matches', []))
                matched_groups = {user_id // users_per_group for user_id in matched_user_ids}
                correct_groups = len(matched_groups & original_groups)
                total_accused_groups = len(accused_groups)
                
                if total_accused_groups > 0:
                    prec_list.append(correct_groups / total_accused_groups)
                
                # Recall: correct groups / total original groups
                total_original_groups = len(original_groups)
                if total_original_groups > 0:
                    rec_list.append(correct_groups / total_original_groups)
            else:
                # For naive: count false positives at USER level
                accused_ids = set(trace_result.get('accused_user_ids', []))
                false_pos = accused_ids - original_ids
                false_pos_list.append(len(false_pos))
                
                total_accused = len(accused_ids)
                total_original = len(original_ids)
                
                if total_accused > 0:
                    prec_list.append(matches / total_accused)
                if total_original > 0:
                    rec_list.append(matches / total_original)
        
        avg_fp = sum(false_pos_list) / total if total > 0 else 0
        avg_prec = sum(prec_list) / len(prec_list) if prec_list else 0
        avg_rec = sum(rec_list) / len(rec_list) if rec_list else 0
        avg_f1 = 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec) if (avg_prec + avg_rec) > 0 else 0
        
        return {
            'success_rate': success_rate,
            'avg_matches': avg_matches,
            'avg_false_positives': avg_fp,
            'avg_precision': avg_prec,
            'avg_recall': avg_rec,
            'avg_f1': avg_f1
        }
    
    normal_metrics = calc_metrics_for_subset(normal_results)
    deletion_metrics = calc_metrics_for_subset(deletion_results)
    
    approach_summary.append({
        'approach': approach,
        'total_cases': total_cases,
        'successful_cases': successful_cases,
        'overall_success_rate_percent': round(success_rate, 2),
        'overall_avg_num_matches': round(avg_matches, 3),
        'overall_avg_false_positives': round(avg_false_positives, 3),
        'overall_avg_precision': round(avg_precision, 3),
        'overall_avg_recall': round(avg_recall, 3),
        'overall_avg_f1_score': round(f1_score, 3),
        'normal_success_rate': round(normal_metrics['success_rate'], 2),
        'normal_avg_matches': round(normal_metrics['avg_matches'], 3),
        'normal_avg_false_positives': round(normal_metrics['avg_false_positives'], 3),
        'normal_precision': round(normal_metrics['avg_precision'], 3),
        'normal_recall': round(normal_metrics['avg_recall'], 3),
        'normal_f1': round(normal_metrics['avg_f1'], 3),
        'with_deletion_success_rate': round(deletion_metrics['success_rate'], 2),
        'with_deletion_avg_matches': round(deletion_metrics['avg_matches'], 3),
        'with_deletion_avg_false_positives': round(deletion_metrics['avg_false_positives'], 3),
        'with_deletion_precision': round(deletion_metrics['avg_precision'], 3),
        'with_deletion_recall': round(deletion_metrics['avg_recall'], 3),
        'with_deletion_f1': round(deletion_metrics['avg_f1'], 3)
    })

# Write to CSV
output_file = os.path.join(script_dir, 'collusion_resistance_approach_summary.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = [
        'approach', 'total_cases', 'successful_cases',
        'overall_success_rate_percent', 'overall_avg_num_matches', 'overall_avg_false_positives',
        'overall_avg_precision', 'overall_avg_recall', 'overall_avg_f1_score',
        'normal_success_rate', 'normal_avg_matches', 'normal_avg_false_positives',
        'normal_precision', 'normal_recall', 'normal_f1',
        'with_deletion_success_rate', 'with_deletion_avg_matches', 'with_deletion_avg_false_positives',
        'with_deletion_precision', 'with_deletion_recall', 'with_deletion_f1'
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(approach_summary)

print(f"Approach-focused summary written to {output_file}")
print(f"\nSummary by approach:")
for row in approach_summary:
    print(f"\n{row['approach']}:")
    print(f"  Overall Success Rate: {row['overall_success_rate_percent']:.2f}%")
    print(f"  Overall Precision: {row['overall_avg_precision']:.3f}, Recall: {row['overall_avg_recall']:.3f}, F1: {row['overall_avg_f1_score']:.3f}")
    print(f"  Normal - Success: {row['normal_success_rate']:.2f}%, Precision: {row['normal_precision']:.3f}, Recall: {row['normal_recall']:.3f}")
    print(f"  With Deletion - Success: {row['with_deletion_success_rate']:.2f}%, Precision: {row['with_deletion_precision']:.3f}, Recall: {row['with_deletion_recall']:.3f}")

