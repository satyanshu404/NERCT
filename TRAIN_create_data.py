import os
import json
import csv
import random
from tqdm import tqdm
from dataclasses import dataclass

random.seed(929)

# ==== GLOBAL VARIABLES ====
HN_FILE = "data/clinicaltrials/train_run/generated_train_v2_with_gold.bm25.k1=0.82.b=0.68.tsv"
QRELS_FILE = "data/clinicaltrials/generated_train_v2_with_gold_qrels.txt"
QUERIES_FILE = "data/clinicaltrials/generated_train_v2_with_gold_queries.tsv"
COLLECTION_FILE = "data/clinicaltrials/2023/corpus.jsonl"
CT_2021_FILE = "data/clinicaltrials/ct_2021_queries.tsv"
CT_2022_FILE = "data/clinicaltrials/ct_2022_queries.tsv"
SAVE_PATH = "data/train"
OUTPUT_FILE = os.path.join(SAVE_PATH, "triplet.jsonl")

NUM_TRAIN_QUERIES = 1200
JUDGED_PERCENTAGE = 0.5
DEPTH = 200
JUDGED_CUTOFF_LINE = 20000

# ========================

def read_queries(file):
    queries = {}
    with open(file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            qid, qtext = line.strip().split('\t')
            queries[qid] = (qtext, i)
    return queries

def read_collection(file):
    collection = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            collection[doc['id']] = doc['contents'].strip()
    return collection

def read_qrel(file):
    qrels = {}
    with open(file, encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [qid, _, docid, rel] in tsvreader:
            if rel == "1":
                qrels.setdefault(qid, []).append(docid)
    return qrels

def read_filter_queries(*files):
    blocked = set()
    for file in files:
        with open(file, encoding='utf-8') as f:
            for line in f:
                qid, _ = line.strip().split('\t')
                blocked.add(qid)
    return blocked

def load_hard_negatives(file, qrels, depth=200):
    hn_dict = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            qid, pid, _ = line.strip().split()
            if qid not in qrels:
                continue
            if pid not in qrels[qid]:
                hn_dict.setdefault(qid, []).append(pid)

    for qid in hn_dict:
        hn_dict[qid] = hn_dict[qid][:depth]
    return hn_dict

def sample_queries(queries, blocked_qids, judged_cutoff=20000):
    judged = []
    synthetic = []
    for qid, (qtext, idx) in queries.items():
        if qid in blocked_qids:
            continue
        if idx >= judged_cutoff:
            judged.append((qid, qtext))
        else:
            synthetic.append((qid, qtext))

    num_judged = int(NUM_TRAIN_QUERIES * JUDGED_PERCENTAGE)
    num_synthetic = NUM_TRAIN_QUERIES - num_judged

    print(f"Sampling {num_judged} judged and {num_synthetic} synthetic queries...")

    sampled_judged = random.sample(judged, num_judged)
    sampled_synthetic = random.sample(synthetic, num_synthetic)

    return dict(sampled_judged + sampled_synthetic)

def main():
    os.makedirs(SAVE_PATH, exist_ok=True)

    print("🔹 Loading queries...")
    queries = read_queries(QUERIES_FILE)

    print("🔹 Loading collection...")
    collection = read_collection(COLLECTION_FILE)

    print("🔹 Loading qrels...")
    qrels = read_qrel(QRELS_FILE)

    print("🔹 Loading blocked queries (2021 + 2022)...")
    blocked_qids = read_filter_queries(CT_2021_FILE, CT_2022_FILE)

    print("🔹 Loading hard negatives...")
    hn_dict = load_hard_negatives(HN_FILE, qrels, depth=DEPTH)

    print("🔹 Sampling and filtering queries...")
    sampled_queries = sample_queries(queries, blocked_qids, judged_cutoff=JUDGED_CUTOFF_LINE)

    print("🔹 Mining triplets and writing output...")
    seen = set()
    count = 0
    max_needed = NUM_TRAIN_QUERIES
    buffer_multiplier = 2  # Try with 2x more queries

    extended_queries = list(sampled_queries.items())
    if len(extended_queries) < max_needed * buffer_multiplier:
        print("⚠️ Warning: Not enough queries to sample extra buffer, using available only.")
    else:
        extended_queries = random.sample(extended_queries, max_needed * buffer_multiplier)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for qid, query in tqdm(extended_queries):
            if count >= max_needed:
                break

            if qid not in qrels or qid not in hn_dict:
                continue

            pos_ids = qrels[qid]
            neg_ids = hn_dict[qid]
            random.shuffle(pos_ids)
            random.shuffle(neg_ids)

            for pos_id in pos_ids:
                for neg_id in neg_ids:
                    triplet_id = (qid, pos_id, neg_id)
                    if (
                        pos_id in collection and 
                        neg_id in collection and 
                        triplet_id not in seen
                    ):
                        seen.add(triplet_id)
                        triplet = {
                            "qid": qid,
                            "query": query,
                            "positive_docid": pos_id,
                            "positive_doc": collection[pos_id],
                            "negative_docid": neg_id,
                            "negative_doc": collection[neg_id]
                        }
                        f_out.write(json.dumps(triplet) + '\n')
                        count += 1
                        break  # move to next query
                if count >= max_needed:
                    break

    print(f"\n✅ Finished! Wrote {count} unique triplets to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
