import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
from sklearn.model_selection import train_test_split

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

#podaci
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

X_text = []
y_tag = []

for intent in intents["intents"]:
    tag = intent["tag"]
    for p in intent["patterns"]:
        X_text.append(p)
        y_tag.append(tag)

# split
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_text, y_tag, test_size=0.30, random_state=42, stratify=y_tag
)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

print(f"Split -> train={len(X_train)}, dev={len(X_dev)}, test={len(X_test)}")

# vokab iz train
all_words = []
tags = sorted(list(set(y_tag)))

xy_train = []
for text, tag in zip(X_train, y_train):
    w = tokenize(text)
    all_words.extend(w)
    xy_train.append((w, tag))

ignore = ["?", "!", ".", ",", ":", ";"]
all_words = [stem(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words))

# train u bow
X_train_bow = []
y_train_idx = []

for tokens, tag in xy_train:
    X_train_bow.append(bag_of_words(tokens, all_words))
    y_train_idx.append(tags.index(tag))

X_train_bow = np.array(X_train_bow, dtype=np.float32)
y_train_idx = np.array(y_train_idx, dtype=np.int64)

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

loader = DataLoader(ChatDataset(X_train_bow, y_train_idx),
                    batch_size=8, shuffle=True)

#model
model = NeuralNet(len(all_words), 8, len(tags))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def predict(text_list):
    model.eval()
    preds = []
    with torch.no_grad():
        for text in text_list:
            bow = bag_of_words(tokenize(text), all_words)
            out = model(torch.from_numpy(bow).unsqueeze(0))
            pred = torch.argmax(out, dim=1).item()
            preds.append(tags[pred])
    model.train()
    return preds

def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

#trening
for epoch in range(1, 31):
    for xb, yb in loader:
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# spremi
torch.save({
    "model_state": model.state_dict(),
    "input_size": len(all_words),
    "hidden_size": 8,
    "output_size": len(tags),
    "all_words": all_words,
    "tags": tags
}, "data.pth")

print("Saved: data.pth")
