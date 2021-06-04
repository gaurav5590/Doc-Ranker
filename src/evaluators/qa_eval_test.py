import sys
sys.path.append('.')

from src.evaluators.eval_utils.ms_marco_eval_qa import compute_metrics_from_files

gt_file = 'data/sample_references1.json'
out_file = 'data/sample_candidates1.json'

score = compute_metrics_from_files(gt_file, out_file, 4)

print(score)