"""
Fake / deceptive review detector — training script.

DATA: Download the Deceptive Opinion Spam Corpus (Ott et al., Cornell) from
Kaggle: https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus
It's a real, published, gold-standard academic dataset — 1,600 hotel
reviews, balanced 50/50 deceptive/truthful, 50/50 positive/negative.
Save the CSV as deceptive-opinion.csv next to this script. Its columns are:
    deceptive (truthful/deceptive), hotel, polarity, source, text

HONEST LIMITATION: this dataset is hotel reviews only (Chicago hotels,
English). There is no equivalent public gold-standard dataset for
"transport" fake reviews as of writing this — see the note at the bottom
of this file for what to do about that.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import joblib

# ---------------- 1. Load data ----------------
df = pd.read_csv("deceptive-opinion.csv")
df["label"] = (df["deceptive"] == "deceptive").astype(int)  # 1 = fake, 0 = genuine

print(df["label"].value_counts())  # sanity check: should be ~800/800

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- 2. Feature extraction ----------------
# Word n-grams (1-2) is what the original Ott et al. paper found most
# effective. Character n-grams can help catch stylistic tics too, but start
# simple — you can add char n-grams later if accuracy needs a boost.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words="english",
    sublinear_tf=True,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- 3. Train ----------------
# Logistic Regression: fast, interpretable (you can inspect which words/
# phrases push a review toward "fake" — great for a hackathon demo slide).
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_vec, y_train)

# ---------------- 4. Evaluate ----------------
y_pred = model.predict(X_test_vec)
print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred, target_names=["genuine", "fake"]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: try a linear SVM too and keep whichever scores higher on YOUR
# split — different runs/papers have found either one can edge out the
# other slightly on this dataset.
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
y_pred_svm = svm_model.predict(X_test_vec)
print("\n=== Linear SVM ===")
print(classification_report(y_test, y_pred_svm, target_names=["genuine", "fake"]))

# ---------------- 5. Inspect what the model actually learned ----------------
# Good for your PPT: show the top words that push toward "fake" vs "genuine".
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_fake_idx = coefs.argsort()[-15:][::-1]
top_genuine_idx = coefs.argsort()[:15]
print("\nTop words/phrases pushing toward FAKE:")
print([feature_names[i] for i in top_fake_idx])
print("\nTop words/phrases pushing toward GENUINE:")
print([feature_names[i] for i in top_genuine_idx])

# ---------------- 6. Save (same joblib pattern your project already uses) ----------------
joblib.dump(model, "fake_review_model.pkl")
joblib.dump(vectorizer, "fake_review_vectorizer.pkl")
print("\nSaved fake_review_model.pkl and fake_review_vectorizer.pkl")

# ============================================================================
# NOTE ON TRANSPORT REVIEWS:
# This model is trained ONLY on hotel review text. Deceptive-language
# patterns (generic superlatives, lack of specific sensory/spatial detail,
# unusual pronoun use) partially transfer across domains, but published
# research (Mukherjee et al.) found accuracy drops significantly when a
# hotel-trained model is applied to a different domain (Yelp) without
# retraining — don't claim this model is validated for transport reviews
# without testing it on actual transport-review examples first, and be
# upfront about that gap if asked in your demo.
# ============================================================================
