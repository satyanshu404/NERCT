import os
import json
import pyterrier as pt
from tqdm import tqdm

# Initialize PyTerrier (still initializing to keep code structure consistent)
pt.java.init()

# Define global paths
collection_path = "/home/satyanshu/Documents/Clinical_Trial/Dataset/trec_collection"  # Path where TREC files are stored
output_folder = "data/clinicaltrials"  # Folder to save corpus.jsonl
corpus_file_path = os.path.join(output_folder, "corpus.jsonl")

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Step 1: Gather TREC Files for Processing
trec_files = [os.path.join(collection_path, f) for f in os.listdir(collection_path) if f.endswith(".trectext")]

print(f"Found {len(trec_files)} TREC files for processing...")

# Step 2: Parse TREC Files and Build Corpus
corpus = []

def parse_trectext_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        current_doc = {}
        capture_text = False
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '<DOC>':
                current_doc = {}
            elif line.startswith('<DOCNO>') and line.endswith('</DOCNO>'):
                docno = line.replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
                current_doc['docno'] = docno
            elif line == '<TEXT>':
                capture_text = True
                current_doc['text'] = ''
            elif line == '</TEXT>':
                capture_text = False
            elif line == '</DOC>':
                if 'docno' in current_doc and 'text' in current_doc:
                    corpus.append({"id": current_doc['docno'], "contents": current_doc['text'].strip()})
            elif capture_text:
                current_doc['text'] += line + ' '

print(f"Parsing and writing corpus to {corpus_file_path}...")

# Process each file
with tqdm(total=len(trec_files), desc="Processing Files") as pbar:
    for file_path in trec_files:
        parse_trectext_file(file_path)
        pbar.update(1)

# Step 3: Save the Corpus to JSONL
with open(corpus_file_path, 'w', encoding='utf8') as fout:
    for doc in corpus:
        fout.write(json.dumps(doc) + "\n")

print(f"Corpus successfully written to {corpus_file_path} with {len(corpus)} documents.")
