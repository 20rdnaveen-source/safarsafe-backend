"""
Trains the auto-dispatch suitability model used by /api/sos to pick the
best free ambulance/police unit to auto-assign to a new SOS.

HONESTY NOTE (read before presenting this in the pitch): SafarSafe has zero
real historical dispatch records right now — no real incident has ever been
auto-assigned, so there is nothing true to train on. Rather than fake that
away, this script generates REALISTIC SIMULATED dispatch scenarios (distance,
current workload, severity, time of day) and trains a real scikit-learn
model on them. Every prediction at runtime uses REAL inputs (real distance
via haversine, real current org workload from incidents_db, real severity,
real time) — only the TRAINING data is simulated. This is a standard,
legitimate way to bootstrap a dispatch/logistics model before real usage
data exists (this is exactly how many real-world dispatch systems start).
Once SafarSafe has real incident outcomes, retrain this on real data instead
— see the "Retraining on real data" note at the bottom of this file.

Nationwide, not city-specific: every feature here is relative (distance_km,
active_load, severity, hour, org type) rather than tied to any one city's
geography, so the trained model generalizes to any location in India where
SafarSafe operates — unlike the old Tiruvannamalai-only pilot risk model.

Run with: python train_dispatch_model.py   (from inside model/)
Produces: dispatch_model.pkl, dispatch_feature_cols.pkl
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

RNG = np.random.default_rng(42)
N_SAMPLES = 20000

FEATURE_COLS = ["distance_km", "active_load", "severity_num", "is_hospital_org", "hour_of_day"]


def generate_scenarios(n):
    # Distance: mostly close (dense urban India), long tail out to rural/no-coverage cases.
    distance_km = RNG.exponential(scale=6.0, size=n)
    distance_km = np.clip(distance_km, 0.05, 80.0)

    # How many incidents this org is already actively handling right now.
    # Mostly free (0), occasionally already stretched thin.
    active_load = RNG.choice([0, 1, 2, 3], size=n, p=[0.72, 0.18, 0.07, 0.03])

    # 1=Low, 2=Medium, 3=High, 4=Critical
    severity_num = RNG.choice([1, 2, 3, 4], size=n, p=[0.15, 0.30, 0.40, 0.15])

    # Whether the candidate is a hospital's own ambulance (vs a dedicated
    # ambulance/TNHSP org, or a police station — this feature only matters
    # for the ambulance category, but including it for all is harmless).
    is_hospital_org = RNG.integers(0, 2, size=n)

    hour_of_day = RNG.integers(0, 24, size=n)

    # Ground-truth suitability score (0-100, higher = better pick):
    # - distance matters more as severity rises (a critical case can't wait for a far unit)
    # - existing workload is a bigger penalty than a bit of extra distance
    # - hospital-run ambulances lose a small amount vs dedicated ambulance units
    #   (shared with in-hospital duties) — mild, real-world-plausible assumption
    # - late night (00:00-05:00) adds a small realistic delay penalty (staffing/traffic)
    night_penalty = np.where((hour_of_day >= 0) & (hour_of_day <= 5), 4.0, 0.0)
    distance_penalty = distance_km * (2.0 + severity_num * 0.35)
    score = (
        100
        - distance_penalty
        - active_load * 16.0
        - is_hospital_org * 3.0
        - night_penalty
        + RNG.normal(0, 4.0, size=n)  # real-world noise
    )
    score = np.clip(score, 0, 100)

    X = np.column_stack([distance_km, active_load, severity_num, is_hospital_org, hour_of_day])
    return X, score


def main():
    X, y = generate_scenarios(N_SAMPLES)
    model = RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X, y)

    joblib.dump(model, "dispatch_model.pkl")
    joblib.dump(FEATURE_COLS, "dispatch_feature_cols.pkl")
    print(f"Trained on {N_SAMPLES} simulated scenarios. Feature importances:")
    for name, imp in zip(FEATURE_COLS, model.feature_importances_):
        print(f"  {name}: {imp:.3f}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Retraining on real data (do this once you have real incident history):
# Replace generate_scenarios() with a function that pulls real rows from
# your incidents_db / assignment history — same FEATURE_COLS, but with a
# real label such as "did this responder actually arrive fastest / handle it
# well" instead of the simulated `score` formula above. The rest of this
# script (model type, training call, saving) stays the same.
# ---------------------------------------------------------------------------
