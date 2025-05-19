#!/bin/bash
set -e

GPU=1
CORPUS_PATH=data/clinicaltrials/corpus.jsonl
QUERY_PATH=data/2022/ct_queries.tsv
TOKENIZER_NAME=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

echo "======================== 1. Retrieving with SPLADE ========================"
# SPLADE
# CUDA_VISIBLE_DEVICES=$GPU python IELAB_splade.py \
#   --model ielabgroup/trec-ct-2023-spladev2 \
#   --tokenizer microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
#   --corpus $CORPUS_PATH \
#   --metadata_field contents \
#   --docid_field id \
#   --topics $QUERY_PATH \
#   --index_path ./indexes/ct2023-splade-index-1 \
#   --run_output runs/IELAB/SPLADE/SPLADE_CT2022.txt \
#   --top_k 100 \
#   --max_length 512 \
#   --batch_size 128

echo "======================== 2. Retrieving with DR ========================"
## tokenize the queries
echo "======================== a. Tokenizing the corpus ========================"
# CUDA_VISIBLE_DEVICES=$GPU python IELAB_tokenize_passages.py \
#     --tokenizer_name $TOKENIZER_NAME \
#     --truncate 512 \
#     --file $CORPUS_PATH \
#     --n_splits 1 \
#     --save_to tokenized/clinicaltrials/

## tokenize the queries
echo "======================== b. Tokenizing the queries ========================"
# CUDA_VISIBLE_DEVICES=$GPU python IELAB_tokenize_queries.py \
#     --tokenizer_name $TOKENIZER_NAME \
#     --truncate 256 \
#     --query_file $QUERY_PATH \
#     --save_to tokenized/queries/queries_ct_2022.json

echo "======================== c. DR Retrieval ========================"
## DR
# python -m asyncval \
#     --query_file tokenized/queries/queries_ct_2022.json \
#     --candidate_dir tokenized/clinicaltrials/ \
#     --ckpts_dir models/Dense \
#     --tokenizer_name_or_path ${TOKENIZER_NAME} \
#     --qrel_file data/2022/ct_2022_qrels_mapped.txt \
#     --metrics 'RR(rel=2)' 'nDCG@10' 'P(rel=2)@10' 'Rprec(rel=2)' 'R(rel=2)@1000' \
#     --output_dir runs/IELAB/DR \
#     --depth 100 \
#     --per_device_eval_batch_size 256 \
#     --q_max_len 256 \
#     --p_max_len 512 \
#     --write_run trec \
#     --write_embeddings True \
#     --fp16 \
#     --cache_dir cache \
#     --max_num_valid 10

echo "======================== 3. Hybrid Retrieval (DR + SPLADE) ========================"

DR_CHECKPOINT=2000
dr_run=runs/IELAB/DR/set_0_checkpoint-${DR_CHECKPOINT}.tsv
splade_run=runs/IELAB/SPLADE/SPLADE_CT2022.txt
hybrid_dir=runs/IELAB/HYBRID
result_file=results/HYBRID/ct2022.hybrid.results.txt

mkdir -p $hybrid_dir
mkdir -p results/HYBRID

# Clear previous result file if it exists
if [ -f "$result_file" ]; then
    echo "Previous evaluation file found. Deleting..."
    rm $result_file
fi

idx=1
for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  num=$(echo "1 - $alpha" | bc)
  output_run=${hybrid_dir}/IELAB_HYBRID_alpha=${alpha}.txt

  echo "Fusing DR and SPLADE with alpha=${alpha} ..."
  python IELAB_hybrid.py \
    --dense_run_file $dr_run \
    --sparse_run_file $splade_run \
    --alpha $alpha \
    --k 100 \
    --normalize true \
    --output_file $output_run

  echo "Evaluating fusion output for alpha=${alpha} ..."
  echo "${idx}. Evaluation for alpha ${alpha}" >> $result_file
  trec_eval -m ndcg_cut -m P -m recall -m map data/2022/ct_2022_qrels_mapped.txt $output_run >> $result_file
  echo "=====================================" >> $result_file

  idx=$((idx+1))
done

echo "======================== 4. Evaluation ========================"
# TREC Evaluation for SPLADE
echo "======================== a. SPLADE results ========================"
if [ ! -d "results/SPLADE" ]; then
  mkdir -p results/SPLADE
fi
trec_eval -m ndcg_cut -m P -m recall -m map data/2022/ct_2022_qrels_mapped.txt runs/IELAB/SPLADE/SPLADE_CT2022.txt > results/SPLADE/ct2022.splade.results.txt

# TREC Evaluation for DR
echo "======================== b. DR results ========================"
if [ ! -d "results/DR" ]; then
  mkdir -p results/DR
fi
trec_eval -m ndcg_cut -m P -m recall -m map data/2022/ct_2022_qrels_mapped.txt ${dr_run} > results/DR/ct2022.dr.results.txt

# TREC Evaluation for Hybrid
# echo "======================== c. Hybrid results ========================"
# if [ ! -d "results/HYBRID" ]; then
#   mkdir -p results/HYBRID
# fi
# trec_eval -m ndcg_cut -m P -m recall -m map data/2022/ct_2022_qrels_mapped.txt runs/IELAB/HYBRID/HYBRID.txt > results/HYBRID/ct2022.hybrid.results1.txt
