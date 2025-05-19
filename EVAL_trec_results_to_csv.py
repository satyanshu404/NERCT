import os
import csv

# Global configuration
result_base_dir = "results"
dir_names = ["FIRST_STAGE", "ZERO_SHOT", "QA_NN_NEW", "SIMPLE_BERT", "CT_MLM_BERT"]
parent_save_dir = "csv_results"
PRINT_METRIC_NAME = False  # Set to True to include metric names in the CSV

# Metrics mapping: raw TREC name → desired metric name
metrics_to_extract = {
    "map": "map",
    # "map_cut_10": "map@10",
    # "map_cut_20": "map@20",
    # "map_cut_100": "map@100",
    "P_10": "P@10",
    "P_20": "P@20",
    "P_100": "P@100",
    "recall_10": "Recall@10",
    "recall_20": "Recall@20",
    "recall_100": "Recall@100",
    "ndcg_cut_10": "NDCG@10",
    "ndcg_cut_20": "NDCG@20",
    "ndcg_cut_100": "NDCG@100",
}

# Desired metric order
desired_order = [
    "map",
    "map@10",
    "map@20",
    "map@100",
    "P@10",
    "P@20",
    "P@100",
    "Recall@10",
    "Recall@20",
    "Recall@100",
    "NDCG@10",
    "NDCG@20",
    "NDCG@100",
]

# Create parent output directory if it doesn't exist
os.makedirs(parent_save_dir, exist_ok=True)

# Process each directory in dir_names
for dir_name in dir_names:
    input_dir = os.path.join(result_base_dir, dir_name)
    save_dir = os.path.join(parent_save_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_dir, file_name)
            csv_file_path = os.path.join(save_dir, file_name.replace(".txt", ".csv"))

            results = {}
            # Read the file and extract the desired metrics (only rows with scope "all")
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        raw_metric, q_type, score = parts
                        if q_type == "all" and raw_metric in metrics_to_extract:
                            results[metrics_to_extract[raw_metric]] = float(score)

            # Sort and write results in the desired order
            with open(csv_file_path, "w", newline="") as csvfile:
                if PRINT_METRIC_NAME:
                    writer = csv.DictWriter(csvfile, fieldnames=["metric", "score"])
                    writer.writeheader()
                    for metric in desired_order:
                        if metric in results:
                            writer.writerow({"metric": metric, "score": results[metric]})
                else:
                    writer = csv.writer(csvfile)
                    writer.writerow(["score"])
                    for metric in desired_order:
                        if metric in results:
                            writer.writerow([results[metric]])

            print(f"Saved: {csv_file_path}")
