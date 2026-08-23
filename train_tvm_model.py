"""
Train the Tourist Safety Risk Model on Tiruvannamalai-specific data
SIH 2026 - AI Tourist Safety & Emergency Response System
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\20rdn\Downloads\files\safarsafe_full_project\safarsafe\data\tiruvannamalai_tourist_risk_dataset.csv")

level_order = ["Low", "Medium", "High"]
encoders = {}
for col in ["crowd_level", "weather_risk", "tourist_density", "zone_risk"]:
    le = LabelEncoder()
    le.fit(level_order)
    df[col + "_enc"] = le.transform(df[col])
    encoders[col] = le

# Zone name encoder (needed so the model can learn zone-specific patterns directly)
zone_le = LabelEncoder()
df["zone_name_enc"] = zone_le.fit_transform(df["zone_name"])
encoders["zone_name"] = zone_le

feature_cols = [
    "time_hour", "is_festival_period", "is_monsoon_heavy_rain", "previous_incidents",
    "crowd_level_enc", "weather_risk_enc", "tourist_density_enc",
    "zone_risk_enc", "zone_name_enc",
]

X = df[feature_cols]
y_class = df["risk_label"]
y_score = df["risk_score"]

X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
)

clf = RandomForestClassifier(n_estimators=250, max_depth=9, random_state=42)
clf.fit(X_train, y_class_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_class_test, y_pred)
print(f"Classification Accuracy: {acc:.3f}\n")
print(classification_report(y_class_test, y_pred))
print("Confusion Matrix (Low/Medium/High):")
print(confusion_matrix(y_class_test, y_pred, labels=level_order))

reg = RandomForestRegressor(n_estimators=250, max_depth=9, random_state=42)
reg.fit(X_train, y_score_train)
score_preds = reg.predict(X_test)
mae = np.mean(np.abs(score_preds - y_score_test))
print(f"\nRisk Score Regressor MAE: {mae:.2f} points (out of 100)")

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

joblib.dump(clf,r"C:/Users/20rdn/Downloads/files/safarsafe_full_project/safarsafe/model/tvm_risk_classifier.pkl")
joblib.dump(reg, r"C:/Users/20rdn/Downloads/files/safarsafe_full_project/safarsafe/model/tvm_risk_regressor.pkl")
joblib.dump(encoders, r"C:/Users/20rdn/Downloads/files/safarsafe_full_project/safarsafe/model/tvm_encoders.pkl")
joblib.dump(feature_cols, r"C:/Users/20rdn/Downloads/files/safarsafe_full_project/safarsafe/model/tvm_feature_cols.pkl")

# Save zone reference table (name, lat, lng, base risk) for the backend to use directly
zones_ref = df.groupby("zone_name").agg(
    latitude=("latitude", "mean"),
    longitude=("longitude", "mean"),
    zone_risk=("zone_risk", lambda x: x.mode()[0]),
).reset_index()
zones_ref.to_csv(r"C:/Users/20rdn/Downloads/files/safarsafe_full_project/safarsafe/model/tvm_zones_reference.csv", index=False)
print("\nSaved Tiruvannamalai model, encoders, and zone reference table.")
print(zones_ref)
