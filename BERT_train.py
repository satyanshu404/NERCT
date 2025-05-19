import os
import json
import torch
import random
import warnings
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import nn


# GLOBAL VARIABLES
MODEL_CHECKPOINT = 'ielabgroup/PubMedBERT-CT-MLM'
DATA_PATH = 'data/train/triplet.jsonl'
CSV_PATH = 'data/train/train_dataset.csv'
MODEL_SAVE_PATH = 'models/CT_MLM_BERT'
SEED = 42
BATCH_SIZE = 8
EPOCHS = 5
MAX_QUERY_LEN = 179
MAX_DOC_LEN = 330
MAX_LEN = MAX_QUERY_LEN + MAX_DOC_LEN + 3  # [CLS] + query + [SEP] + doc + [SEP]

# SETUP
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
warnings.filterwarnings('ignore')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# PREPROCESSING
if not os.path.exists(CSV_PATH):
    print("Preprocessing data...")
    data = []

    with open(DATA_PATH, 'r') as f:
        for line in f:
            obj = json.loads(line)
            query = obj['query']
            pos_doc = obj['positive_doc']
            neg_doc = obj['negative_doc']

            data.append([query, pos_doc, 1.0])
            data.append([query, neg_doc, 0.0])

    df = pd.DataFrame(data, columns=['query', 'doc', 'label'])
    df.to_csv(CSV_PATH, index=False)
else:
    print("Loading preprocessed CSV...")
    df = pd.read_csv(CSV_PATH)

# Ensure class balance in train-test split
df_pos = df[df['label'] == 1.0]
df_neg = df[df['label'] == 0.0]

df_pos_train, df_pos_test = train_test_split(df_pos, test_size=0.2, random_state=SEED)
df_neg_train, df_neg_test = train_test_split(df_neg, test_size=0.2, random_state=SEED)

train_df = pd.concat([df_pos_train, df_neg_train])
test_df = pd.concat([df_pos_test, df_neg_test])

train_df = shuffle(train_df, random_state=SEED)
test_df = shuffle(test_df, random_state=SEED)

# TOKENIZER
tokenizer = BertTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(example):
    query = tokenizer.tokenize(example['query'])[:MAX_QUERY_LEN]
    doc = tokenizer.tokenize(example['doc'])[:MAX_DOC_LEN]
    tokens = tokenizer.convert_tokens_to_ids(['[CLS]'] + query + ['[SEP]'] + doc + ['[SEP]'])
    segment_ids = [0] * (len(query) + 2) + [1] * (len(doc) + 1)

    padding_length = MAX_LEN - len(tokens)
    tokens += [0] * padding_length
    segment_ids += [0] * padding_length
    attention_mask = [1 if i < len(tokens) - padding_length else 0 for i in range(MAX_LEN)]

    return {
        'input_ids': torch.tensor(tokens),
        'attention_mask': torch.tensor(attention_mask),
        'token_type_ids': torch.tensor(segment_ids)
    }

# DATASET
class RelevanceDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = {
            'query': self.data.loc[idx, 'query'],
            'doc': self.data.loc[idx, 'doc']
        }
        features = tokenize_function(example)
        label = torch.tensor(self.data.loc[idx, 'label'], dtype=torch.float)
        return features['input_ids'], features['attention_mask'], features['token_type_ids'], label

train_dataset = RelevanceDataset(train_df)
test_dataset = RelevanceDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# MODEL
class BERTRegression(nn.Module):
    def __init__(self):
        super(BERTRegression, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_CHECKPOINT)
        self.regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.regressor(cls_output).squeeze()

model = BERTRegression().to(device)
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# TRAIN LOOP
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, (input_ids, attn_mask, token_type_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attn_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}")

    # EVALUATION
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for input_ids, attn_mask, token_type_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attn_mask, token_type_ids)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    val_loss = total_val_loss / len(test_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save checkpoint
    save_path = os.path.join(MODEL_SAVE_PATH, f"bert_regression_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to {save_path}")