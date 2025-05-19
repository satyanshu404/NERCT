import pyterrier as pt
import pyt_splade
import pandas as pd
import json
import os
import argparse
from transformers import AutoTokenizer

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# ------------------------------
# Argument Parser
# ------------------------------
parser = argparse.ArgumentParser(description="SPLADE Retrieval Pipeline for Clinical Trials")

parser.add_argument("--model", type=str, required=True, help="Path to the SPLADE model checkpoint")
parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer checkpoint")
parser.add_argument("--corpus", type=str, required=True, help="Path to the corpus JSONL file")
parser.add_argument("--metadata_field", type=str, default="contents", help="Field name for document text")
parser.add_argument("--docid_field", type=str, default="id", help="Field name for document ID")
parser.add_argument("--topics", type=str, required=True, help="Path to the topics TSV file")
parser.add_argument("--index_path", type=str, required=True, help="Directory path to store or load the index")
parser.add_argument("--run_output", type=str, required=True, help="Path to save the TREC run file")
parser.add_argument("--top_k", type=int, default=100, help="Number of top results to retrieve")
parser.add_argument("--max_length", type=int, default=512, help="Max token length for tokenizer")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for indexing and encoding")

args = parser.parse_args()

# ------------------------------
# Init PyTerrier
# ------------------------------
if not pt.started():
    pt.init()

# ------------------------------
# Load SPLADE Model
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
splade = pyt_splade.Splade(model=args.model, tokenizer=tokenizer, max_length=args.max_length)

# ------------------------------
# Load or Create Index
# ------------------------------
if os.path.exists(os.path.join(args.index_path, "data.properties")):
    print("Index already exists. Loading from disk.")
    index_ref = pt.IndexFactory.of(args.index_path)
else:
    print("Index not found. Building index...")

    def load_jsonl_corpus(path):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]

    corpus_data = load_jsonl_corpus(args.corpus)
    corpus_iter = [
        {"docno": doc[args.docid_field], "text": doc[args.metadata_field]}
        for doc in corpus_data
    ]

    indexer = pt.IterDictIndexer(args.index_path, pretokenised=True)
    indxr_pipe = splade.doc_encoder() >> indexer
    index_ref = indxr_pipe.index(corpus_iter, batch_size=args.batch_size)
    print(f"Indexing complete. Saved to: {args.index_path}")

# ------------------------------
# Load Topics (Queries)
# ------------------------------
queries_df = pd.read_csv(args.topics, sep='\t', header=None, names=['qid', 'query'])

# ------------------------------
# Run Retrieval
# ------------------------------
retrieval_pipeline = splade.query_encoder() >> pt.terrier.Retriever(index_ref, wmodel='Tf')
results = retrieval_pipeline.transform(queries_df)
results = results.sort_values(['qid', 'score'], ascending=[True, False]).groupby('qid').head(args.top_k)
results['rank'] = results.groupby('qid').cumcount() + 1
# ------------------------------
# Save TREC Run File
# ------------------------------
os.makedirs(os.path.dirname(args.run_output), exist_ok=True)
pt.io.write_results(results, args.run_output, format='trec', run_name='splade_ct2022')
print(f"Retrieval complete. Run file saved at: {args.run_output}")
