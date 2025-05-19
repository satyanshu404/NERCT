import argparse
import os
from collections import defaultdict

class DenseSearchResult:
    def __init__(self, docid, score):
        self.docid = docid
        self.score = score

def _hybrid_results(dense_results, sparse_results, alpha, k, normalization=False, weight_on_dense=False):
    dense_hits = {hit.docid: hit.score for hit in dense_results}
    sparse_hits = {hit.docid: hit.score for hit in sparse_results}
    hybrid_result = []

    min_dense_score = min(dense_hits.values()) if len(dense_hits) > 0 else 0
    max_dense_score = max(dense_hits.values()) if len(dense_hits) > 0 else 1
    min_sparse_score = min(sparse_hits.values()) if len(sparse_hits) > 0 else 0
    max_sparse_score = max(sparse_hits.values()) if len(sparse_hits) > 0 else 1

    for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
        if doc not in dense_hits:
            sparse_score = sparse_hits[doc]
            dense_score = min_dense_score
        elif doc not in sparse_hits:
            sparse_score = min_sparse_score
            dense_score = dense_hits[doc]
        else:
            sparse_score = sparse_hits[doc]
            dense_score = dense_hits[doc]
        if normalization:
            sparse_score = (sparse_score - (min_sparse_score + max_sparse_score) / 2) / (max_sparse_score - min_sparse_score)
            dense_score = (dense_score - (min_dense_score + max_dense_score) / 2) / (max_dense_score - min_dense_score)
        # score = alpha * sparse_score + dense_score if not weight_on_dense else sparse_score + alpha * dense_score
        score = alpha * sparse_score + (1 - alpha) * dense_score
        hybrid_result.append(DenseSearchResult(doc, score))

    return sorted(hybrid_result, key=lambda x: x.score, reverse=True)[:k]

def load_run_file(run_path):
    runs = defaultdict(list)
    with open(run_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            docid = parts[2]
            score = float(parts[4])
            runs[qid].append(DenseSearchResult(docid, score))
    return runs

def save_run_file(hybrid_runs, output_path):
    with open(output_path, 'w') as f:
        for qid in sorted(hybrid_runs.keys(), key=lambda x: int(x)):
            results = hybrid_runs[qid]
            for rank, result in enumerate(results):
                f.write(f"{qid} Q0 {result.docid} {rank+1} {result.score} hybrid\n")

def main():
    parser = argparse.ArgumentParser(description="Hybrid search run combiner (sparse + dense).")
    parser.add_argument('--dense_run_file', type=str, required=True, help="Path to the dense run file (.tsv or TREC format)")
    parser.add_argument('--sparse_run_file', type=str, required=True, help="Path to the sparse run file (TREC format)")
    parser.add_argument('--alpha', type=float, required=True, help="Alpha value for interpolation")
    parser.add_argument('--k', type=int, required=True, help="Number of top documents to keep per query")
    parser.add_argument('--normalize', type=str, choices=['true', 'false'], required=True, help="Whether to normalize scores (true/false)")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the hybrid run file")
    parser.add_argument('--weight_on_dense', action='store_true', help="If set, alpha weights the dense scores instead of sparse")

    args = parser.parse_args()

    normalize = True if args.normalize.lower() == 'true' else False

    print(f"Loading sparse run from {args.sparse_run_file}...")
    sparse_runs = load_run_file(args.sparse_run_file)
    print(f"Loading dense run from {args.dense_run_file}...")
    dense_runs = load_run_file(args.dense_run_file)

    hybrid_runs = {}
    all_qids = set(dense_runs.keys()) | set(sparse_runs.keys())
    for qid in all_qids:
        dense_results = dense_runs.get(qid, [])
        sparse_results = sparse_runs.get(qid, [])
        hybrid_runs[qid] = _hybrid_results(
            dense_results,
            sparse_results,
            args.alpha,
            args.k,
            normalization=normalize,
            weight_on_dense=args.weight_on_dense
        )

    save_run_file(hybrid_runs, args.output_file)
    print(f"Hybrid run file saved at: {args.output_file}")

if __name__ == "__main__":
    main()