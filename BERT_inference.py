import os
import pandas as pd
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# === Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Input: List of Run Files ===
RUN_FILE_NAMES = ["SPLADE_CT2022.txt",
                  "NLS_RM3_RETRIEVAL_T2022.txt",
                  "WholeQ_RETRIEVAL_T2022.txt",
                  "WholeQ_RM3_RETRIEVAL_T2022.txt"
                  ] 

RUN_NAME = "CT_MLM_BERT"
output_dir = "runs/CT_MLM_BERT"
os.makedirs(output_dir, exist_ok=True)

checkpoint = "bert_regression_epoch3"
MODEL_BASE_CHECKPOINT = "ielabgroup/PubMedBERT-CT-MLM"


corpus_path = "data/clinicaltrials/2023/corpus.jsonl"
query_path = "data/2022/ct_queries.tsv"
MODEL_SAVE_PATH = f"models/CT_MLM_BERT/{checkpoint}.pt"

# === Tokenization Config ===
MAX_QUERY_LEN = 179
MAX_DOC_LEN = 330
MAX_LEN = MAX_QUERY_LEN + MAX_DOC_LEN + 3

# === Load Tokenizer ===
tokenizer = BertTokenizer.from_pretrained(MODEL_BASE_CHECKPOINT)

# === Model Definition ===
class BERTRegression(nn.Module):
    def __init__(self):
        super(BERTRegression, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_BASE_CHECKPOINT)
        self.regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output).squeeze()

# === Load Model ===
model = BERTRegression().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_SAVE_PATH}")

# === Tokenize input ===
def tokenize_input(query, doc):
    query_tokens = tokenizer.tokenize(query)[:MAX_QUERY_LEN]
    doc_tokens = tokenizer.tokenize(doc)[:MAX_DOC_LEN]

    tokens = tokenizer.convert_tokens_to_ids(['[CLS]'] + query_tokens + ['[SEP]'] + doc_tokens + ['[SEP]'])
    segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_tokens) + 1)
    attention_mask = [1] * len(tokens)

    padding_length = MAX_LEN - len(tokens)
    tokens += [0] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    return {
        'input_ids': torch.tensor(tokens).unsqueeze(0).to(device),
        'attention_mask': torch.tensor(attention_mask).unsqueeze(0).to(device),
        'token_type_ids': torch.tensor(segment_ids).unsqueeze(0).to(device)
    }

# === Load Corpus ===
def load_corpus(path):
    corpus_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trial = json.loads(line.strip())
            corpus_dict[trial["id"]] = trial.get("contents")
    return corpus_dict

corpus = load_corpus(corpus_path)
print(f"Loaded {len(corpus)} documents from corpus.")

# === Load Queries ===
df_queries = pd.read_csv(query_path, sep="\t", header=None, names=["id", "text"])
query_dict = dict(zip(df_queries["id"].astype(str), df_queries["text"]))

# === Scoring Function ===
def score(query, doc):
    model_input = tokenize_input(query, doc)
    with torch.no_grad():
        output = model(**model_input)
    return output.item()

# === Process Each Run File ===
for RUN_FILE_NAME in RUN_FILE_NAMES:
    print(f"\nProcessing {RUN_FILE_NAME}...")
    retrieved_trials_file = os.path.join("runs/FIRST_STAGE", RUN_FILE_NAME)
    output_file_name = f"{os.path.splitext(RUN_FILE_NAME)[0]}_{RUN_NAME.lower()}.txt"
    retrieval_txt_path = os.path.join(output_dir, output_file_name)

    # Load First-Stage Retrieval Results
    bm25_results = {}
    with open(retrieved_trials_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            topic_no, trial_id = parts[0], parts[2]
            bm25_results.setdefault(topic_no, []).append(trial_id)

    # Rerank and Save
    text_lines = []
    for topic_no, trials in tqdm(bm25_results.items(), desc=f"Reranking {RUN_FILE_NAME}"):
        query = query_dict.get(topic_no)
        if not query:
            continue

        scored_trials = []
        for trial_id in trials:
            doc_text = corpus.get(trial_id)
            if not doc_text:
                continue
            score_val = score(query, doc_text)
            scored_trials.append((trial_id, score_val))

        scored_trials.sort(key=lambda x: x[1], reverse=True)

        for rank, (trial_id, score_val) in enumerate(scored_trials[:100], start=1):
            line = f"{topic_no} Q0 {trial_id} {rank} {score_val:.5f} {RUN_NAME}"
            text_lines.append(line)

    # Write to TREC Format File
    with open(retrieval_txt_path, "w") as fout:
        fout.write("\n".join(text_lines) + "\n")

    print(f"Saved reranked run to {retrieval_txt_path}")
