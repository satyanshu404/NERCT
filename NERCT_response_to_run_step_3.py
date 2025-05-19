import json
import os
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# -------------------------------
# Global Configuration
# -------------------------------

# File name list
FILE_NAMES = ["SPLADE_CT2022", "WholeQ_RETRIEVAL_T2022", "NLS_RM3_RETRIEVAL_T2022", "WholeQ_RM3_RETRIEVAL_T2022"]

# Result file paths
RESULT_FILES = [f"data/2022/{fname}_llm_responses_sanitized.jsonl" for fname in FILE_NAMES]

# Output directory
OUTPUT_DIR = "runs/QA_NN_NEW/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model checkpoint
NN_MODEL_PATH = "models/NN/best_ground_truth_model_deepseek.pt"

# Response mapping
RESPONSE_MAP = {"NO": 0, "NA": 0.5, "YES": 1}

# default score
DEFAULT_SCORE = -100.0


# -------------------------------
# Model Definition
# -------------------------------
class RegressionNN(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------
# Load NN Model
# -------------------------------
def load_nn_model():
    print("Loading NN model...")
    nn_model = RegressionNN()
    nn_model.load_state_dict(torch.load(NN_MODEL_PATH, map_location=torch.device("cpu")))
    nn_model.eval()
    return nn_model


# -------------------------------
# Extract Scores Using NN Model
# -------------------------------
def extract_nn_scores(results_file, model):
    scores_dict = defaultdict(list)
    idx = 0

    for line in open(results_file, "r", encoding="utf-8"):
        idx += 1
        try:
            entry = json.loads(line)
            qid, docid = entry["qid"], entry["docid"]
            responses = entry.get("cleaned_output")

            if responses:
                answers = np.array([
                    RESPONSE_MAP.get(responses.get(str(i), {}).get("response", "NO").upper(), 0)
                    for i in range(1, 11)
                ])
                input_tensor = torch.tensor(answers, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    score = model(input_tensor).item()
            else:
                raise ValueError("Missing cleaned_output")

        except Exception as e:
            print(f"[Warning] idx {idx}: Error, Setting score to {DEFAULT_SCORE} Error: {e}")
            score = DEFAULT_SCORE
            qid = entry.get("qid", f"unk_{idx}")
            docid = entry.get("docid", f"unk_doc_{idx}")

        scores_dict[qid].append((docid, score))

    return scores_dict


# -------------------------------
# Save Ranked Output (TREC Style)
# -------------------------------
def save_ranked_output(scores_dict, run_file_path):
    with open(run_file_path, "w", encoding="utf-8") as f:
        for qid, doc_scores in scores_dict.items():
            ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked_docs, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} Deepseek-Qwen-32B-NN\n")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    print("Running NN scoring for all files...")
    nn_model = load_nn_model()

    for file_name, result_file in zip(FILE_NAMES, RESULT_FILES):
        print(f"Processing file: {result_file}")
        scores_dict = extract_nn_scores(result_file, nn_model)

        run_file_path = os.path.join(OUTPUT_DIR, f"{file_name}_NN.txt")
        save_ranked_output(scores_dict, run_file_path)

        print(f"Saved results to: {run_file_path}")

    print("All files processed successfully.")

# import json
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from collections import defaultdict

# # -------------------------------
# # Config
# # -------------------------------
# RESULT_FILE = "data/2022/WholeQ_RETRIEVAL_T2022_llm_responses_sanitized.jsonl"
# RESPONSE_MAP = {"NO": 0, "NA": 0.5, "YES": 1}
# NUM_FEATURES = 10
# DEFAULT_SCORE = -100.0
# GROUND_MODEL_PATH = "models/NN/best_ground_truth_model_deepseek.pt"
# FEATURE_MODELS_DIR = "models/NN_features"
# OUTPUT_DIR = "runs/FEATURE_IMPORTANCE"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -------------------------------
# # Model Definition
# # -------------------------------
# class RegressionNN(nn.Module):
#     def __init__(self, input_dim=10):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )

#     def forward(self, x):
#         return self.model(x)

# # -------------------------------
# # Utilities
# # -------------------------------
# def load_model(model_path):
#     model = RegressionNN()
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     model.eval()
#     return model

# def extract_nn_scores(results_file, model):
#     scores_dict = defaultdict(list)
#     with open(results_file, "r", encoding="utf-8") as f:
#         for idx, line in enumerate(f):
#             try:
#                 entry = json.loads(line)
#                 qid, docid = entry["qid"], entry["docid"]
#                 responses = entry.get("cleaned_output")
#                 if responses:
#                     answers = np.array([
#                         RESPONSE_MAP.get(responses.get(str(i), {}).get("response", "NO").upper(), 0)
#                         for i in range(1, NUM_FEATURES + 1)
#                     ])
#                     input_tensor = torch.tensor(answers, dtype=torch.float32).unsqueeze(0)
#                     with torch.no_grad():
#                         score = model(input_tensor).item()
#                 else:
#                     raise ValueError("Missing cleaned_output")
#             except Exception as e:
#                 print(f"[Warning] idx {idx}: Malformed entry. Setting default score. Error: {e}")
#                 qid = entry.get("qid", f"unk_{idx}")
#                 docid = entry.get("docid", f"unk_doc_{idx}")
#                 score = DEFAULT_SCORE
#             scores_dict[qid].append((docid, score))
#     return scores_dict

# def save_ranked_output(scores_dict, run_file_path):
#     with open(run_file_path, "w", encoding="utf-8") as f:
#         for qid, doc_scores in scores_dict.items():
#             ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
#             for rank, (docid, score) in enumerate(ranked_docs, start=1):
#                 f.write(f"{qid} Q0 {docid} {rank} {score:.6f} Deepseek-Qwen-32B-NN\n")

# # -------------------------------
# # Main Execution
# # -------------------------------
# if __name__ == "__main__":
#     print("=== Running feature importance scoring ===")

#     # 1. Ground truth model
#     print("Scoring with ground truth model...")
#     model = load_model(GROUND_MODEL_PATH)
#     scores_dict = extract_nn_scores(RESULT_FILE, model)
#     run_path = os.path.join(OUTPUT_DIR, "GT_NN.txt")
#     save_ranked_output(scores_dict, run_path)
#     print(f"Saved ground truth run file to {run_path}")

#     # 2. Feature-ablated models
#     for i in range(1, NUM_FEATURES + 1):
#         model_path = os.path.join(FEATURE_MODELS_DIR, f"{i}_model_deepseek.pt")
#         if not os.path.exists(model_path):
#             print(f"[Warning] Model for feature {i} not found at {model_path}. Skipping...")
#             continue

#         print(f"Scoring with model ablated for feature {i}...")
#         ablated_model = load_model(model_path)
#         ablated_scores = extract_nn_scores(RESULT_FILE, ablated_model)
#         ablated_run_path = os.path.join(OUTPUT_DIR, f"NN_F{i}.txt")
#         save_ranked_output(ablated_scores, ablated_run_path)
#         print(f"Saved ablated run file to {ablated_run_path}")

#     print("✅ All run files generated.")

