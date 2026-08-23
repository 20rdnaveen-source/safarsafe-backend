"""
Synthetic Tourist Safety Risk Dataset Generator
SIH 2026 - AI Tourist Safety & Emergency Response System

Generates realistic (rule-based, not purely random) training data for the
Random Forest risk classifier. Clearly labeled as SYNTHETIC/DEMO data --
do not present as real-world incident data to judges.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_ROWS = 4000

# ---- Feature generation ----
time_hour = np.random.randint(0, 24, N_ROWS)
previous_incidents = np.random.poisson(lam=2, size=N_ROWS)  # skewed, mostly low
crowd_level = np.random.choice(["Low", "Medium", "High"], N_ROWS, p=[0.4, 0.35, 0.25])
weather_risk = np.random.choice(["Low", "Medium", "High"], N_ROWS, p=[0.5, 0.35, 0.15])
tourist_density = np.random.choice(["Low", "Medium", "High"], N_ROWS, p=[0.35, 0.4, 0.25])
zone_risk = np.random.choice(["Low", "Medium", "High"], N_ROWS, p=[0.5, 0.3, 0.2])

level_map = {"Low": 0, "Medium": 1, "High": 2}


def compute_risk_score(row):
    """Rule-based scoring used ONLY to generate labels for synthetic data.
    Real deployment would learn this relationship from actual incident data."""
    score = 0

    # Time of day: late night / very early morning raises risk
    hour = row["time_hour"]
    if hour >= 22 or hour <= 4:
        score += 25
    elif hour >= 19 or hour <= 6:
        score += 12

    # Previous incidents in the zone
    score += min(row["previous_incidents"] * 6, 30)

    # Zone risk (base danger of the location itself) - strongest factor
    score += level_map[row["zone_risk"]] * 15

    # Weather risk
    score += level_map[row["weather_risk"]] * 8

    # Crowd level: LOW crowd at a risky time/place is actually riskier (isolation)
    if row["crowd_level"] == "Low":
        score += 10
    elif row["crowd_level"] == "High":
        score -= 5  # safety in numbers, slightly lowers risk

    # Tourist density: very high density can increase pickpocketing/crowd-crush risk
    score += level_map[row["tourist_density"]] * 4

    # Add small noise so the model isn't trivially perfect (more realistic)
    score += np.random.normal(0, 6)

    return max(0, min(100, score))


df = pd.DataFrame({
    "time_hour": time_hour,
    "previous_incidents": previous_incidents,
    "crowd_level": crowd_level,
    "weather_risk": weather_risk,
    "tourist_density": tourist_density,
    "zone_risk": zone_risk,
})

df["risk_score"] = df.apply(compute_risk_score, axis=1)


def bucket(score):
    if score <= 30:
        return "Low"
    elif score <= 70:
        return "Medium"
    else:
        return "High"


df["risk_label"] = df["risk_score"].apply(bucket)

df.to_csv("/home/claude/tourist_safety/data/tourist_risk_dataset.csv", index=False)

print(f"Generated {len(df)} rows")
print(df["risk_label"].value_counts())
print(df.head())
