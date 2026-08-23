"""
Train the Tourist Safety Risk Random Forest Model
SIH 2026 - AI Tourist Safety & Emergency Response System
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/home/claude/tourist_safety/data/tourist_risk_dataset.csv")

# Encode categorical features (Low/Medium/High -> 0/1/2)
level_order = ["Low", "Medium", "High"]
encoders = {}
for col in ["crowd_level", "weather_risk", "tourist_density", "zone_risk"]:
    le = LabelEncoder()
    le.fit(level_order)
    df[col + "_enc"] = le.transform(df[col])
    encoders[col] = le

feature_cols = [
    "time_hour", "previous_incidents",
    "crowd_level_enc", "weather_risk_enc", "tourist_density_enc", "zone_risk_enc",
]

X = df[feature_cols]
y_class = df["risk_label"]
y_score = df["risk_score"]

X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
)

# Classifier: predicts Low/Medium/High
clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
clf.fit(X_train, y_class_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_class_test, y_pred)
print(f"Classification Accuracy: {acc:.3f}\n")
print(classification_report(y_class_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_class_test, y_pred, labels=level_order))

# Regressor: predicts the 0-100 continuous risk score (used for the dashboard/app score display)
reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
reg.fit(X_train, y_score_train)
score_preds = reg.predict(X_test)
mae = np.mean(np.abs(score_preds - y_score_test))
print(f"\nRisk Score Regressor MAE: {mae:.2f} points (out of 100)")

# Feature importance (useful for your PPT/report)
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# Save everything the API will need
joblib.dump(clf, "/home/claude/tourist_safety/model/risk_classifier.pkl")
joblib.dump(reg, "/home/claude/tourist_safety/model/risk_regressor.pkl")
joblib.dump(encoders, "/home/claude/tourist_safety/model/encoders.pkl")
joblib.dump(feature_cols, "/home/claude/tourist_safety/model/feature_cols.pkl")

print("\nModel, regressor, and encoders saved.")
