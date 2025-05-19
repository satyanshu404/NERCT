import json
import csv

# Global variables
INPUT_JSONL_FILE = "data/train/triplet_answers_mistral_corrected.jsonl"  # path to the JSONL file
OUTPUT_CSV_FILE = f"{INPUT_JSONL_FILE.split('.')[0]}_balanced.csv"  # output CSV file
QREL_FILE = "data/clinicaltrials/generated_train_v2_with_gold_qrels.txt"

RESPONSE_MAP = {"NO": 0, "NA": 0.5, "YES": 1}


def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def load_qrel_pairs(filename):
    """Loads qrel file and returns a set of (qid, docid) pairs."""
    qrel_set = set()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid, _, docid = parts[:3]
                qrel_set.add((qid, docid))
    return qrel_set


def process_data(data, qrel_set):
    processed = []

    for item in data:
        qid = item.get("qid")
        docid = item.get("docid")
        cleaned = item.get("cleaned_output", {})

        # Normalize keys to strings
        cleaned = {str(k): v for k, v in cleaned.items()}

        # Ensure all 10 keys are present
        if not all(str(i) in cleaned for i in range(1, 11)):
            print(f"Skipping qid={qid}, docid={docid} due to missing keys.")
            continue

        try:
            features = [RESPONSE_MAP.get(cleaned[str(i)]["response"], 0.5) for i in range(1, 11)]
        except Exception as e:
            print(f"Error processing qid={qid}, docid={docid}, skipping. Reason: {e}")
            continue

        label = 1 if (qid, docid) in qrel_set else 0
        processed.append(features + [label])

    return processed


def write_csv(data, filename):
    header = [f"q{i}" for i in range(1, 11)] + ["label"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def main():
    data = load_jsonl(INPUT_JSONL_FILE)
    qrel_set = load_qrel_pairs(QREL_FILE)
    processed_data = process_data(data, qrel_set)
    write_csv(processed_data, OUTPUT_CSV_FILE)
    print(f"Final dataset written to {OUTPUT_CSV_FILE} with {len(processed_data)} entries.")


if __name__ == "__main__":
    main()