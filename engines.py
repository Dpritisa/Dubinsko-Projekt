import json
import random
import joblib
import torch
import torch.nn.functional as F
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#odgovori za kada nije sig
FALLBACKS = [
    "Ne razumijem baš pitanje. Možeš li ga preformulirati?",
    "Nisam siguran što točno pitaš. Pokušaj drugačije.",
    "Možeš li pojasniti pitanje (npr. prijava ispita, rokovi, upis, raspored...)?",
    "Oprosti, nisam to skužio. Možeš li napisati u jednoj kratkoj rečenici?"
]

class MlpEngine:
    def __init__(self, min_prob=0.45):
        self.min_prob = float(min_prob)

        with open("intents.json", "r", encoding="utf-8") as f:
            self.intents = json.load(f)

        data = torch.load("data.pth", map_location="cpu")
        self.all_words = data["all_words"]
        self.tags = data["tags"]

        self.model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

    def reply(self, text):
        tokens = tokenize(text)
        X = bag_of_words(tokens, self.all_words)
        X = torch.from_numpy(X).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=1)
            conf, predicted = torch.max(probs, dim=1)

        conf = float(conf.item())
        tag = self.tags[int(predicted.item())]

        # low-confidence fallback
        if conf < self.min_prob:
            return random.choice(FALLBACKS), {"intent": None, "confidence": conf}

        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"]), {"intent": tag, "confidence": conf}

        return random.choice(FALLBACKS), {"intent": tag, "confidence": conf}


class SvmEngine:
    def __init__(self, min_margin=0.20):
        self.min_margin = float(min_margin)
        self.model = joblib.load("svm_model.joblib")

        with open("intents.json", "r", encoding="utf-8") as f:
            self.intents = json.load(f)

    def reply(self, text):
        # LinearSVC nema predict_proba, ali ima decision_function
        try:
            scores = self.model.decision_function([text])[0]
            # margin = razlika između top1 i top2 score
            scores = list(scores)
            best = max(scores)
            scores_sorted = sorted(scores, reverse=True)
            margin = float(scores_sorted[0] - scores_sorted[1]) if len(scores_sorted) > 1 else float(scores_sorted[0])
        except Exception:
            margin = None

        tag = self.model.predict([text])[0]

        # low-confidence fallback po margini (ako možemo)
        if margin is not None and margin < self.min_margin:
            return random.choice(FALLBACKS), {"intent": None, "margin": margin}

        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"]), {"intent": tag, "margin": margin}

        return random.choice(FALLBACKS), {"intent": tag, "margin": margin}
