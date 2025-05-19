import os
import pyterrier as pt
from tqdm import tqdm

# Initialize PyTerrier
pt.java.init()


# Define paths
os.makedirs("indexes", exist_ok=True)
collection_path = "/home/satyanshu/Documents/Clinical_Trial/Dataset/trec_collection"  # Path where TREC files are stored
index_path = "./indexes/ct2023-splade-index"  # Path to store the PyTerrier index

# Ensure index directory exists
os.makedirs(index_path, exist_ok=True)

# Step 1: Gather TREC Files for Indexing
trec_files = [os.path.join(collection_path, f) for f in os.listdir(collection_path) if f.endswith(".trectext")]

print(f"Found {len(trec_files)} TREC files for indexing...")

# Step 2: Indexing TREC Files with PyTerrier
indexer = pt.TRECCollectionIndexer(index_path, overwrite=True)  # Safe overwriting

print(f"Indexing {len(trec_files)} TREC files with PyTerrier...")
with tqdm(total=len(trec_files), desc="Indexing Progress") as pbar:
    index_ref = indexer.index(trec_files)  # PyTerrier manages threading safely
    pbar.update(len(trec_files))

# Step 3: Load and Verify the Index
index = pt.IndexFactory.of(index_path)
print(f"PyTerrier Index Created at {index_path}")