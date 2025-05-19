import json

# ======= Global Variables =======
INPUT_JSONL = "data/train/triplet_answers_mistral.jsonl"
OUTPUT_JSONL = f"{INPUT_JSONL.split('.')[0]}_corrected.jsonl"
# ================================

def clean_generated_text(text):
    """Extract JSON format from generated text."""
    try:
        text = text.split('*Output:*')[-1]  # </think> for deepseek
        text = '{' + '{'.join(text.split('{')[1:])
        text = '}'.join(text.split('}')[:11]) + '}'
        parsed = json.loads(text)

        # Check if parsed has exactly 10 keys
        if isinstance(parsed, dict) and len(parsed) == 10:
            return parsed
        else:
            print("===== Skipping due to insufficient keys in cleaned_output")
            return None
    except Exception as e:
        print(f"{'='*15} Skipping due to error: {e}")
        return None

def process_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                cleaned = clean_generated_text(data.get("result"))
                if cleaned is not None:
                    out_data = {
                        "qid": data.get("qid"),
                        "docid": data.get("docid"),
                        "type": data.get("type"),
                        "cleaned_output": cleaned
                    }
                    outfile.write(json.dumps(out_data) + '\n')
            except Exception as e:
                print(f"Error processing line: {e}")

if __name__ == "__main__":
    process_jsonl(INPUT_JSONL, OUTPUT_JSONL)
