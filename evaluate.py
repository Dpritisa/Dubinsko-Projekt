import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from model import NeuralNet
from nltk_utils import tokenize, bag_of_words


SEED = 42




def load_intents(path="intents.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_xy(intents_json) -> Tuple[List[str], List[str]]:
    X, y = [], []
    for intent in intents_json["intents"]:
        for p in intent["patterns"]:
            X.append(p)
            y.append(intent["tag"])
    return X, y


def make_split_70_15_15(X: List[str], y: List[str]) -> dict:

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    X_dev, X_test, y_dev, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )

    print(f"Split -> train={len(X_train)}, dev={len(X_dev)}, test={len(X_test)}")

    return {
        "train": (X_train, y_train),
        "dev": (X_dev, y_dev),
        "test": (X_test, y_test),
    }




def evaluate_svm(model_path: str, splits_dict: dict, labels_sorted: List[str]):
    svm = joblib.load(model_path)
    results = {}

    for split_name, (X_split, y_split) in splits_dict.items():
        preds = svm.predict(X_split)

        acc = accuracy_score(y_split, preds)
        mf1 = f1_score(y_split, preds, average="macro")

        cm = confusion_matrix(y_split, preds, labels=labels_sorted)

        results[split_name] = (acc, mf1, cm)

    return results



def load_mlp(data_path="data.pth"):
    data = torch.load(data_path, map_location="cpu")
    model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, data["all_words"], data["tags"]


def mlp_predict(model, all_words, tags, texts):
    preds = []
    with torch.no_grad():
        for text in texts:
            bow = bag_of_words(tokenize(text), all_words)
            logits = model(torch.from_numpy(bow).unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            preds.append(tags[pred_idx])
    return preds


def evaluate_mlp(model_path: str, splits_dict: dict, labels_sorted: List[str]):
    model, all_words, tags = load_mlp(model_path)
    results = {}

    for split_name, (X_split, y_split) in splits_dict.items():
        preds = mlp_predict(model, all_words, tags, X_split)

        acc = accuracy_score(y_split, preds)
        mf1 = f1_score(y_split, preds, average="macro")

        cm = confusion_matrix(y_split, preds, labels=labels_sorted)

        results[split_name] = (acc, mf1, cm)

    return results



def print_table(model_name, results):
    for split in ["train", "dev", "test"]:
        acc, mf1, cm = results[split]
        print(f"{model_name:6} | {split:5} | {acc:.4f} | {mf1:.4f}")
        print("Confusion matrix:")
        print(cm)
        print()


def main():
    intents = load_intents()
    X, y = build_xy(intents)

    labels_sorted = sorted(list(set(y)))

    splits_dict = make_split_70_15_15(X, y)

    print("\nModel  | Split | Accuracy | Macro-F1")
    print("--------------------------------------")

    if Path("svm_model.joblib").exists():
        svm_res = evaluate_svm("svm_model.joblib", splits_dict, labels_sorted)
        print_table("SVM", svm_res)
    else:
        print("SVM model ne postoji: svm_model.joblib")

    if Path("data.pth").exists():
        mlp_res = evaluate_mlp("data.pth", splits_dict, labels_sorted)
        print_table("MLP", mlp_res)
    else:
        print("MLP model ne postoji: data.pth")


if __name__ == "__main__":
    main()
