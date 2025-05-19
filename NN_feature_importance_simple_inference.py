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
RESULTS_FILE = "data/2022/SPLADE_CT2022_llm_responses_sanitized.jsonl"
RESPONSE_MAP = {"NO": 0, "NA": 0.5, "YES": 1}
NUM_FEATURES = 10
DEFAULT_SCORE = -100.0

ABLATION_OUTPUT_DIR = "runs/FEATURE_IMPORTANCE_INFERENCE"
os.makedirs(ABLATION_OUTPUT_DIR, exist_ok=True)

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
def extract_nn_scores(results_file, model, ablate_index=None):
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
                    if ablate_index is not None:
                        answers[ablate_index] = 0.0  # Mask this feature
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

# Load baseline model
print("[INFO] Loading baseline model")
model = RegressionNN(NUM_FEATURES).to(device)
model.load_state_dict(torch.load(NN_MODEL_PATH, map_location=device))
model.eval()

# Save baseline run
print("[INFO] Scoring with baseline input (no ablation)")
baseline_scores = extract_nn_scores(RESULTS_FILE, model, ablate_index=None)
save_ranked_output(baseline_scores, os.path.join(ABLATION_OUTPUT_DIR, "baseline.txt"))

# Save runs for each ablated feature
for i in range(NUM_FEATURES):
    print(f"[INFO] Scoring with feature {i+1} ablated (set to 0)")
    ablated_scores = extract_nn_scores(RESULTS_FILE, model, ablate_index=i)
    run_file_path = os.path.join(ABLATION_OUTPUT_DIR, f"feature_ablated_{i+1}.txt")
    save_ranked_output(ablated_scores, run_file_path)

print(f"[INFO] All run files (baseline + ablated) saved to: {ABLATION_OUTPUT_DIR}")