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
FEATURE_MODELS_PATH = "models/NN_features"  # Not used anymore
RESULTS_FILE = "data/2022/SPLADE_CT2022_llm_responses_sanitized.jsonl"
RESPONSE_MAP = {"NO": 0, "NA": 0.5, "YES": 1}
NUM_FEATURES = 10
DEFAULT_SCORE = -100.0
DROP_FEATURES = [0, 7, 8]  # 0-based indices of features to drop (zero out)

os.makedirs(f"runs/FEATURE_IMPORTANCE", exist_ok=True)

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
def extract_nn_scores(results_file, model, drop_features=None):
    scores_dict = defaultdict(list)
    drop_features = set(drop_features or [])
    
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
                    # Zero-out selected features
                    for i in drop_features:
                        if 0 <= i < NUM_FEATURES:
                            answers[i] = 0.0

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

# Load model
print(f"[INFO] Scoring with features dropped at indices: {DROP_FEATURES}")
model = RegressionNN(NUM_FEATURES).to(device)
model.load_state_dict(torch.load(NN_MODEL_PATH, map_location=device))
model.eval()

# Score and save
scores = extract_nn_scores(RESULTS_FILE, model, drop_features=DROP_FEATURES)
drop_tag = "_".join(str(i + 1) for i in DROP_FEATURES)
run_file_path = f"runs/FEATURE_IMPORTANCE/feature_ablated_{drop_tag}.txt"
save_ranked_output(scores, run_file_path)

print(f"[INFO] Run file saved to: {run_file_path}")
