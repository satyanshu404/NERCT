import xml.etree.ElementTree as ET
import pandas as pd
import re

# Global Variables
xml_path = "/home/satyanshu/Documents/Clinical_Trial/trec-ct-2023/data/queries/ct_2022_queries.xml"  # UPDATE if needed
tsv_output_path = "data/2022/ct_queries.tsv"

# List to collect invalid topics
invalid_topics = []

# Load and parse the XML
def load_xml():
    global tree, root
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Failed to load XML: {e}")
        return False
    return True

# Extract topics
def extract_topics():
    global data, invalid_topics
    data = []
    
    for topic in root.findall('topic'):
        topic_number = topic.get('number')
        if topic_number is None:
            print(f"[WARNING] Invalid topic found: {ET.tostring(topic, encoding='unicode').strip()}")
            invalid_topics.append(topic)
            continue
        
        text = topic.text.strip() if topic.text else ''
        data.append({'id': topic_number, 'text': text})

# Clean query text
def clean_query_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

# Save to TSV (no column names)
def save_to_tsv():
    global data
    df = pd.DataFrame(data)

    # Clean text
    df['text'] = df['text'].apply(clean_query_text)

    try:
        df.to_csv(tsv_output_path, sep='\t', index=False, header=False)
        print(f"[INFO] Queries saved to TSV at: {tsv_output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save TSV: {e}")

# Main function
def main():
    if load_xml():
        extract_topics()
        save_to_tsv()
    else:
        print("[FATAL] XML loading failed. Stopping execution.")

# Execute
if __name__ == "__main__":
    main()
