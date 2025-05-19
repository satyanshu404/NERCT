import subprocess
import numpy as np
from scipy.stats import ttest_rel
import os

# Configuration
TEST_TYPE = ["NLS_RM3_RETRIEVAL_T2022", "SPLADE_CT2022", "WholeQ_RM3_RETRIEVAL_T2022", "WholeQ_RETRIEVAL_T2022"]
MODEL_DIRS = ["QA_NN_NEW", "FIRST_STAGE", "SIMPLE_BERT", "CT_MLM_BERT", "ZERO_SHOT"]  # "QA_NN_NEW" must at index 0
MODEL_NAMES = ["a", "b", "c", "d", "e"]
DISPLAY_NAMES = {
    "FIRST_STAGE": "FIRST_STAGE",
    "SIMPLE_BERT": "bmone (BERT)",
    "CT_MLM_BERT": "bmtwo (CT_BERT)",
    "ZERO_SHOT": "bmthree (ZS)",
    "QA_NN_NEW": "mname (NERCT)",
}
QRELS_FILE = "data/2022/ct_2022_qrels_mapped.txt"
RUNS_DIR = "runs"
OUTPUT_DIR = "t_test_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metric Config
METRICS = {
    "map": "MAP",
    "recip_rank": "MRR",
    "P.10": "P@10",    
    "recall.10": "Recall@10",
    "ndcg_cut.10": "NDCG@10",
    "P.20": "P@20",
    "recall.20": "Recall@20",
    "ndcg_cut.20": "NDCG@20"
}
ORDER = list(METRICS.keys())

# Run trec_eval for a file+metric
def run_trec_eval(run_file, metric):
    cmd = ["trec_eval", "-q", "-m", metric, QRELS_FILE, run_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    scores = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 3 and parts[1] != "all":
            scores[parts[1]] = float(parts[2])
    return scores

# Align queries across models
def align_scores(scores1, scores2):
    a, b = [], []
    for qid in scores1:
        if qid in scores2:
            a.append(scores1[qid])
            b.append(scores2[qid])
    return a, b

# Superscript helpers
def superscript(indices):
    mapping = {0: 'ᵃ', 1: 'ᵇ', 2: 'ᶜ', 3: 'ᵈ', 4: 'ᵉ'}
    return ''.join([mapping[i] for i in sorted(indices)])

# Evaluate all pairwise comparisons
def evaluate_test_type(test_type):
    num_models = len(MODEL_DIRS)
    all_model_scores = {metric: [] for metric in ORDER}

    # 1. Gather all scores for each model & metric
    for model_dir in MODEL_DIRS:
        run_file = os.path.join(RUNS_DIR, model_dir, f"{test_type}.txt")
        for metric in ORDER:
            scores = run_trec_eval(run_file, metric)
            all_model_scores[metric].append(scores)

    # 2. Compute means + significance superscripts
    table_rows = []
    for i in range(num_models):
        row = [MODEL_NAMES[i], DISPLAY_NAMES[MODEL_DIRS[i]]]
        for metric in ORDER:
            this_scores = all_model_scores[metric][i]
            this_mean = np.mean(list(this_scores.values()))
            significant = []

            # Only QA_NN_NEW (index 0) should perform t-tests against others
            if i == 0:
                for j in range(1, num_models):
                    other_scores = all_model_scores[metric][j]
                    a, b = align_scores(this_scores, other_scores)
                    if len(a) > 0:
                        t_stat, p_val = ttest_rel(a, b)
                        if p_val < 0.05 and np.mean(a) > np.mean(b):
                            significant.append(j)
                sup = superscript(significant)
            else:
                sup = ""

            row.append(f"{this_mean:.5f}{sup}")
        table_rows.append(row)

    # 3. Format table output
    # Determine column widths
    col_width = 17
    headers = ["#", "Model"] + [METRICS[m] for m in ORDER]

    # Format header
    header_line = "".join(f"{h:<{col_width}}" for h in headers)
    separator_line = "-" * len(header_line)

    # Format rows
    table = [header_line, separator_line]
    for row in table_rows:
        table.append("".join(f"{str(col):<{col_width}}" for col in row))


    output_path = os.path.join(OUTPUT_DIR, f"{test_type}.txt")
    with open(output_path, "w") as f:
        f.write(f"# Evaluation Results: {test_type}\n\n")
        f.write("\n".join(table))
        f.write("\n")

# Main Loop
for test in TEST_TYPE:
    evaluate_test_type(test)

print(f"All formatted result tables saved in '{OUTPUT_DIR}'")