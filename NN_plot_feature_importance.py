import os
import re
import matplotlib.pyplot as plt

# ----------------------------------------
# Config
# ----------------------------------------
results_dir = "results/FEATURE_IMPORTANCE_INFERENCE"
output_dir = os.path.join("plots", os.path.basename(results_dir))
os.makedirs(output_dir, exist_ok=True)

# Metrics to visualize
target_metrics = ["P_10", "map", "recall_10", "ndcg_cut_10"]

baseline_metrics = {}
feature_deltas = {metric: [] for metric in target_metrics}

# ----------------------------------------
# Extract all metric results using regex
# ----------------------------------------
def extract_all_metrics(filepath):
    metrics = {}
    metric_pattern = re.compile(r"^([\w_.@]+)\s+all\s+([\d.]+)", re.IGNORECASE)
    with open(filepath, "r") as f:
        for line in f:
            match = metric_pattern.match(line.strip())
            if match:
                metric_name, score_str = match.groups()
                try:
                    metrics[metric_name.lower()] = float(score_str)
                except ValueError:
                    pass
    return metrics

# ----------------------------------------
# Read baseline
# ----------------------------------------
baseline_file = os.path.join(results_dir, "baseline.txt")
baseline_metrics_all = extract_all_metrics(baseline_file)
baseline_metrics = {m: baseline_metrics_all.get(m.lower(), 0.0) for m in map(str.lower, target_metrics)}
print("[INFO] Baseline Metrics:", baseline_metrics)

# ----------------------------------------
# Extract feature file info
# ----------------------------------------
feature_info = []  # list of (XX, filepath)

for fname in os.listdir(results_dir):
    if fname == "baseline.txt" or not fname.startswith("feature_ablated_") or not fname.endswith(".txt"):
        continue
    xx_part = fname[len("feature_ablated_"):-len(".txt")]
    fpath = os.path.join(results_dir, fname)
    feature_info.append((xx_part, fpath))

# Sort: numbers first, then mixed
def sort_key(xx):
    try:
        return (0, int(xx))
    except ValueError:
        return (1, xx)

feature_info.sort(key=lambda x: sort_key(x[0]))

feature_labels = [f"F_{xx}" for xx, _ in feature_info]

# ----------------------------------------
# Compare each feature with baseline
# ----------------------------------------
for xx, fpath in feature_info:
    feature_metrics_all = extract_all_metrics(fpath)
    for metric in target_metrics:
        base_score = baseline_metrics.get(metric.lower(), 0.0)
        feat_score = feature_metrics_all.get(metric.lower(), 0.0)
        delta = abs(base_score - feat_score)
        feature_deltas[metric].append(delta)

# ----------------------------------------
# Plotting
# ----------------------------------------
def plot_deltas(metric_name, deltas, labels, save_path):
    plt.figure(figsize=(18, 10))
    bars = plt.bar(range(len(deltas)), deltas, color="skyblue")
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"Feature importance Δ(Baseline, Ablated Feature)", fontsize=28)
    plt.xlabel("Feature", fontsize=28)
    plt.ylabel(f"Δ {metric_name.upper()}", fontsize=28)
    plt.xticks(range(len(labels)), labels=labels, fontsize=20, rotation=45, ha="right")
    plt.yticks(fontsize=24)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar, delta in zip(bars, deltas):
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y,
                 f"{delta:.4f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()
    print(f"[SAVED] {save_path}")

# ----------------------------------------
# Generate plots
# ----------------------------------------
for metric, deltas in feature_deltas.items():
    save_path = os.path.join(output_dir, f"feature_importance_{metric}.png")
    plot_deltas(metric, deltas, feature_labels, save_path)

print("[INFO] All plots saved successfully.")
