import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import os

# ------------------------------ VARIABLES ------------------------------

qrels_path = 'data/2022/ct_2022_qrels_mapped.txt'

run_files = {
    'SPLADE': 'data/2022/SPLADE_CT2022.txt',
    'SPLADE+Zero-Shot LLM': 'runs/SIMPLE_SCORING/SPLADE_CT2022_score_deepseek.txt',
    'SPLADE+Our Method': 'runs/QA_NN_NEW/SPLADE_CT2022_NN.txt'
}

metrics = ['P_10', 'recall_10', 'ndcg_cut_10', 'map']

eval_output_dir = 'tmp_eval_outputs'
os.makedirs(eval_output_dir, exist_ok=True)

plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

trec_eval_path = 'trec_eval'

# ------------------------------ FUNCTIONS ------------------------------

def run_trec_eval(qrels, run_file, output_file):
    cmd = [
        trec_eval_path,
        '-q',  # keep -q if you want query-wise outputs
        '-m', 'P.10',
        '-m', 'recall.10',
        '-m', 'ndcg_cut.10',
        '-m', 'map',
        qrels,
        run_file
    ]
    print(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)

    # Save stdout to file
    with open(output_file, 'w') as f:
        f.write(result.stdout)

    print(f"Saved evaluation output to {output_file}")

def read_eval_output(file_path, model_name):
    df = pd.read_csv(file_path, sep='\s+', names=['metric', 'query', 'score'])
    df = df[df['query'] != 'all']
    df['model'] = model_name
    return df

def plot_metrics(df_all, metric):
    plt.figure(figsize=(14, 7))
    metric_df = df_all[df_all['metric'] == metric].copy()
    
    # Convert query to int for sorting
    metric_df['query'] = metric_df['query'].astype(int)
    metric_df = metric_df.sort_values(by='query')

    sns.lineplot(data=metric_df, x='query', y='score', hue='model', marker='o')
    plt.title(f'Query-wise {metric} Comparison')
    plt.xlabel('Query ID')
    plt.ylabel(metric)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(plot_dir, f'LLM_{metric}_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

# ------------------------------ MAIN ------------------------------

for model_name, run_file in run_files.items():
    output_path = os.path.join(eval_output_dir, f'{model_name}_eval.txt')
    run_trec_eval(qrels_path, run_file, output_path)

dfs = []
for model_name in run_files.keys():
    eval_path = os.path.join(eval_output_dir, f'{model_name}_eval.txt')
    df = read_eval_output(eval_path, model_name)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

for metric in metrics:
    plot_metrics(df_all, metric)
