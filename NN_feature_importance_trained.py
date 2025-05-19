import os
import json
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# -------------------------------
# Config and Constants
# -------------------------------
NN_MODEL_PATH = "models/NN/best_ground_truth_model_deepseek.pt"
FEATURE_MODELS_PATH = "models/NN_features"
RESULTS_FILE = "data/2022/SPLADE_CT2022_llm_responses_sanitized.jsonl"
RESPONSE_MAP = {"NO": 0, "NA": 0.5, "YES": 1}
NUM_FEATURES = 10
DEFAULT_SCORE = -100.0

os.makedirs("runs/FEATURE_IMPORTANCE", exist_ok=True)
print(f"[INFO] Created directory for run files.")
# -------------------------------
# Model Definition
# -------------------------------
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
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
# Scoring and Saving Functions
# -------------------------------
def extract_nn_scores(results_file, model):
    scores_dict = defaultdict(list)
    with open(results_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                entry = json.loads(line)
                qid, docid = entry["qid"], entry["docid"]
                responses = entry.get("cleaned_output")
                if responses:
                    answers = np.array([
                        RESPONSE_MAP.get(responses.get(str(i), {}).get("response", "NO").upper(), 0)
                        for i in range(1, NUM_FEATURES + 1)
                    ])
                    input_tensor = torch.tensor(answers, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        score = model(input_tensor).item()
                else:
                    raise ValueError("Missing cleaned_output")
            except Exception as e:
                print(f"[Warning] idx {idx}: Malformed entry. Setting default score. Error: {e}")
                qid = entry.get("qid", f"unk_{idx}")
                docid = entry.get("docid", f"unk_doc_{idx}")
                score = DEFAULT_SCORE
            scores_dict[qid].append((docid, score))
    return scores_dict

def save_ranked_output(scores_dict, run_file_path):
    with open(run_file_path, "w", encoding="utf-8") as f:
        for qid, doc_scores in scores_dict.items():
            ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked_docs, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} Deepseek-Qwen-32B-NN\n")

# -------------------------------
# Main Execution
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baseline model (no feature ablation)
print("[INFO] Scoring with baseline model")
baseline_model = RegressionNN(NUM_FEATURES).to(device)
baseline_model.load_state_dict(torch.load(NN_MODEL_PATH, map_location=device))
baseline_model.eval()

baseline_scores = extract_nn_scores(RESULTS_FILE, baseline_model)
save_ranked_output(baseline_scores, "runs/FEATURE_IMPORTANCE/baseline.txt")

# Feature-ablated models
for i in range(NUM_FEATURES):
    feature_model_path = os.path.join(FEATURE_MODELS_PATH, f"{i+1}_model_deepseek.pt")
    if os.path.exists(feature_model_path):
        print(f"[INFO] Scoring with ablated feature model: F{i+1}")
        model = RegressionNN(NUM_FEATURES).to(device)
        model.load_state_dict(torch.load(feature_model_path, map_location=device))
        model.eval()

        scores = extract_nn_scores(RESULTS_FILE, model)
        run_file_path = f"runs/FEATURE_IMPORTANCE/feature_ablated_{i+1}.txt"
        save_ranked_output(scores, run_file_path)
    else:
        print(f"[WARNING] Model for feature {i+1} not found. Skipping...")

print("[INFO] All run files (baseline + ablated) saved successfully.")


# import json
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# os.makedirs("plots/NN_feature_importance", exist_ok=True)

# # -------------------------------
# # Config and Constants
# # -------------------------------
# NN_MODEL_PATH = "models/NN/best_ground_truth_model_deepseek.pt"
# FEATURE_MODELS_PATH = "models/NN_features"
# RESULTS_FILE = "data/2022/WholeQ_RETRIEVAL_T2022_llm_responses_sanitized.jsonl"
# RESPONSE_MAP = {"NO": 0, "NA": 0.5, "YES": 1}
# NUM_FEATURES = 10
# # in save path only consider the file name, not the full path, and in file path only till T2022
# IMAGE_NAME = "_".join(os.path.basename(RESULTS_FILE).split(".")[0].split("_")[:-3])
# SAVE_IMAGE_PATH = f"plots/NN_feature_importance/{IMAGE_NAME}_feature_importance.png"

# # -------------------------------
# # Model Definition
# # -------------------------------
# class RegressionNN(nn.Module):
#     def __init__(self, input_dim):
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
# # Load Model
# # -------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = RegressionNN(NUM_FEATURES).to(device)
# model.load_state_dict(torch.load(NN_MODEL_PATH, map_location=device))
# model.eval()

# # -------------------------------
# # Load Data and Extract Scores
# # -------------------------------
# X = []

# with open(RESULTS_FILE, "r", encoding="utf-8") as f:
#     for idx, line in enumerate(f):
#         entry = json.loads(line)
#         responses = entry.get("cleaned_output")

#         if responses:
#             try:
#                 answers = np.array([RESPONSE_MAP.get(responses[str(i)]["response"].upper(), 0) for i in range(1, 11)])
#                 X.append(answers)
#             except (KeyError, TypeError, ValueError):
#                 print(f"Skipping entry at index {idx} due to malformed data.")
#         else:
#             print(f"Skipping entry at index {idx} due to missing cleaned_output.")

# X = np.array(X)
# X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# # -------------------------------
# # Ground Truth Prediction
# # -------------------------------
# with torch.no_grad():
#     baseline_preds = model(X_tensor).squeeze().cpu().numpy()

# # -------------------------------
# # Feature Importance Calculation
# # -------------------------------
# importances = []

# # Loop over each feature and compute its importance
# for i in range(NUM_FEATURES):
#     feature_model_path = os.path.join(FEATURE_MODELS_PATH, f'{i+1}_model_deepseek.pt')
    
#     if os.path.exists(feature_model_path):  # If the model with the ith feature turned off exists
#         feature_model = RegressionNN(NUM_FEATURES).to(device)
#         feature_model.load_state_dict(torch.load(feature_model_path, map_location=device))
#         feature_model.eval()

#         with torch.no_grad():
#             ablated_preds = feature_model(X_tensor).squeeze().cpu().numpy()

#         delta = abs(baseline_preds - ablated_preds)
#         avg_delta = np.mean(delta)
#         importances.append(avg_delta)
#     else:
#         print(f"Model for feature {i} is missing. Skipping...")

# # Normalize importances
# importances = np.array(importances)
# normalized_importance = importances / np.sum(importances)

# # -------------------------------
# # Plot and Save Feature Importances
# # -------------------------------

# plt.figure(figsize=(10, 6))
# plt.bar(range(NUM_FEATURES), normalized_importance, color='skyblue')
# plt.xlabel("Feature Index")
# plt.ylabel("Normalized Importance")
# plt.title(f"Feature Importance of Neural Network Model with \n{IMAGE_NAME}")
# plt.xticks(range(NUM_FEATURES), [f"F{i+1}" for i in range(NUM_FEATURES)])
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.savefig(SAVE_IMAGE_PATH, dpi=1200)
# plt.close()