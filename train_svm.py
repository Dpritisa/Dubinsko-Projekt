import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

SEED = 42

#ucitaj podatke
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

X = []
y = []

for intent in intents["intents"]:
    tag = intent["tag"]
    for p in intent["patterns"]:
        X.append(p)
        y.append(tag)

#split
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
)

print(f"Split -> train={len(X_train)}, dev={len(X_dev)}, test={len(X_test)}")

#model s regulacijom
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 1), lowercase=True)),
    ("clf", LinearSVC(C=0.15, random_state=SEED))
])

model.fit(X_train, y_train)

joblib.dump(model, "svm_model.joblib")
print("\nSaved: svm_model.joblib")
