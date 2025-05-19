import os
import glob
from ranx import Qrels, Run, compare

# 🔧 Configuration
TEST_TYPES = [
    "NLS_RM3_RETRIEVAL_T2022",
    "SPLADE_CT2022",
    "WholeQ_RM3_RETRIEVAL_T2022",
    "WholeQ_RETRIEVAL_T2022"
]

BENCHMARK_DIRS = [
    "FIRST_STAGE",
    "ZERO_SHOT",
    "SIMPLE_BERT",
    "CT_MLM_BERT"
]

BASE_RUN_DIR = "QA_NN_NEW"
RUNS_DIR = "runs"
QRELS_FILE = "data/2022/ct_2022_qrels_mapped.txt"
OUTPUT_DIR = "t_test_results_ranx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📊 Metric List for Comparison
METRICS = ["map", "mrr", "ndcg@10", "precision@10", "recall@10"]

# 🧠 Evaluation Function using ranx
def evaluate_test_type_with_ranx(test_type):
    base_run_file = f"{RUNS_DIR}/{BASE_RUN_DIR}/{test_type}.txt"
    benchmark_run_files = [
        f"{RUNS_DIR}/{dir}/{test_type}.txt" for dir in BENCHMARK_DIRS
    ]

    run_files = [base_run_file] + benchmark_run_files
    qrel = Qrels.from_file(QRELS_FILE)
    runs = []

    for run_file in run_files:
        run = Run.from_file(run_file)
        # Use the folder name (strategy name) as the run label
        run.name = run_file.split("/")[-2]
        runs.append(run)

    report = compare(
        qrel,
        runs=runs,
        metrics=METRICS,
        max_p=0.05,
        stat_test="student",
        rounding_digits=5,
    )

    output_path_txt = os.path.join(OUTPUT_DIR, f"{test_type}.txt")
    output_path_latex = os.path.join(OUTPUT_DIR, f"{test_type}.tex")

    with open(output_path_txt, "w") as f:
        f.write(f"===== Evaluation Report for {test_type} =====\n\n")
        f.write(report.to_table())

    with open(output_path_latex, "w") as f:
        f.write(report.to_latex())

    print(f"✅ Report for '{test_type}' saved to {output_path_txt} and {output_path_latex}")

# 🚀 Main Loop
if __name__ == "__main__":
    for test_type in TEST_TYPES:
        evaluate_test_type_with_ranx(test_type)

    print(f"\n📁 All reports saved in '{OUTPUT_DIR}' directory.")
