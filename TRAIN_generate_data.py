import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# ==== Global Constants ====
TRIPLET_FILE = "data/train/triplet.jsonl"
RESULTS_FILE = "data/train/triplet_answers_mistral.jsonl"
TRACK_FILE = "data/track_triplet.json"
PROMPTS_FILE = "prompt_mistral.txt"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# ==== Parameters ====
TEMPERATURE = 0.5
DO_SAMPLE = True
START_FROM_ZERO = False  # <<-- Set this to True to start fresh

TRACKING_KEY = "DEEPSEEK_TRACKING_KEY"   # DEEPSEEK_TRACKING_KEY

# ==== Model Loading ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=torch.float16,
)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config,
    # torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded successfully in 16-bit using bitsandbytes!")

# ==== Prompt Loading ====
def load_prompt_template(file_path=PROMPTS_FILE):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ==== Tracking ====
def read_track_file(track_file=TRACK_FILE):
    try:
        with open(track_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(TRACKING_KEY, 0)
    except FileNotFoundError:
        return 0

def update_track_file(index, track_file=TRACK_FILE):
    with open(track_file, "w", encoding="utf-8") as f:
        json.dump({TRACKING_KEY: index}, f)

def reset_track_file(track_file=TRACK_FILE):
    with open(track_file, "w", encoding="utf-8") as f:
        json.dump({TRACKING_KEY: 0}, f)

# ==== Data Loading ====
def load_triplet_data(file_path=TRIPLET_FILE):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ==== LLM Call ====
def generate_response(query, doc, max_new_tokens=2048):
    prompt = TEMPLATE.replace("{0}", query).replace("{1}", doc)
    input_ids = TOKENIZER(prompt, return_tensors="pt").input_ids.to(MODEL.device)
    with torch.no_grad():
        gen_tokens = MODEL.generate(input_ids,
                                    max_new_tokens=max_new_tokens,
                                    # temperature=TEMPERATURE,
                                    # do_sample=DO_SAMPLE
                                    )
    return TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)[0]

def clean_generated_text(text):
    try:
        text = text.split('*Output:*')[-1]   # </think> for deepseek
        text = '{' + '{'.join(text.split('{')[1:])
        text = '}'.join(text.split('}')[:11]) + '}'
        return json.loads(text)
    except Exception as e:
        print(f"{'='*15}Skipping due to error: {e}")
        return None

# ==== Main Processing ====
def process_qd_triplets(triplet_data, output_file=RESULTS_FILE, track_file=TRACK_FILE):
    if START_FROM_ZERO:
        reset_track_file(track_file)
    start_index = read_track_file(track_file)

    total_pairs = len(triplet_data) * 2
    print(f"Resuming from pair index: {start_index} of {total_pairs}")

    with tqdm(total=total_pairs, initial=start_index, desc="Processing", unit="pair") as pbar:
        for i in range(start_index, total_pairs):
            triplet_idx = i // 2
            is_positive = (i % 2 == 0)
            entry = triplet_data[triplet_idx]

            qid = entry["qid"]
            query = entry["query"]
            docid = entry["positive_docid"] if is_positive else entry["negative_docid"]
            doc = entry["positive_doc"] if is_positive else entry["negative_doc"]
            doc_type = "positive" if is_positive else "negative"

            print(f"Generating for Triplet {triplet_idx + 1}, Type: {doc_type}")
            raw_output = generate_response(query, doc)
            cleaned_output = clean_generated_text(raw_output)

            result_entry = {
                "qid": qid,
                "docid": docid,
                "type": doc_type,
                "result": raw_output,
                "cleaned_output": cleaned_output
            }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry) + "\n")

            update_track_file(i + 1, track_file)
            pbar.update(1)

# ==== Entrypoint ====
if __name__ == "__main__":
    TEMPLATE = load_prompt_template()
    triplet_data = load_triplet_data()
    process_qd_triplets(triplet_data)
    print("Processing completed!")