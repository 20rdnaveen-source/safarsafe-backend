"""
Tiruvannamalai Tourist Safety Risk Dataset
SIH 2026 - AI Tourist Safety & Emergency Response System

IMPORTANT - DATA HONESTY NOTE (read before using in your report/pitch):
------------------------------------------------------------------------
- Zone NAMES, LOCATIONS, and general risk CHARACTERISTICS below are real
  and drawn from public sources (Girivalam path structure, Arunachala
  hill climb route, forest-department restrictions on the inner path,
  Karthigai Deepam festival crowding, and the documented Dec 2024
  landslide at Arunachala Hill's base in VOC Nagar during Cyclone Fengal
  which killed 7 people).
- There is NO public, zone-level tourist-incident database for
  Tiruvannamalai. The exact incident COUNTS, crowd numbers, and risk
  SCORES in this file are SYNTHETIC - generated using rules that reflect
  each zone's real, documented characteristics (isolation, terrain,
  festival crowding, monsoon landslide history).
- Present this honestly to judges: "real zones and real risk factors,
  synthetic incident data" - do not describe the numbers as official
  records.
- Coordinates are approximate (town-level accuracy), not survey-grade.
"""

import numpy as np
import pandas as pd

np.random.seed(7)

# ---- Real Tiruvannamalai zones with DOCUMENTED risk characteristics ----
ZONES = [
    {
        "zone_name": "Annamalaiyar Temple (Base)",
        "lat": 12.2260, "lng": 79.0678,
        "base_zone_risk": "Low",
        "note": "Largest Shiva temple complex; heavy footfall, well-lit, well-patrolled year-round.",
    },
    {
        "zone_name": "Girivalam Path - Outer Loop",
        "lat": 12.2200, "lng": 79.0800,
        "base_zone_risk": "Medium",
        "note": "14km circumambulation path around the hill; safe in daytime crowds, isolated stretches at night.",
    },
    {
        "zone_name": "Sri Ramana Ashram",
        "lat": 12.2318, "lng": 79.0672,
        "base_zone_risk": "Low",
        "note": "Well-managed meditation center, controlled entry, high international visitor footfall.",
    },
    {
        "zone_name": "Hill Climb Route (via Ramanashram to Skandashram/Virupaksha Cave)",
        "lat": 12.2340, "lng": 79.0690,
        "base_zone_risk": "Medium",
        "note": "Steep rocky sections, 45-90 min climb; heat exhaustion and slip/fall risk documented in pilgrim guides.",
    },
    {
        "zone_name": "Inner Girivalam Path / Forest Tract (Restricted)",
        "lat": 12.2400, "lng": 79.0850,
        "base_zone_risk": "High",
        "note": "Forest department has restricted pilgrim movement on this inner path; isolated, dense forest, no regular patrol.",
    },
    {
        "zone_name": "VOC Nagar (Hill Base, Landslide-Prone Zone)",
        "lat": 12.2280, "lng": 79.0640,
        "base_zone_risk": "Medium",
        "note": "Site of the Dec 1, 2024 landslide (Cyclone Fengal) that killed 7 people at the hill base; risk spikes sharply in heavy monsoon rain.",
    },
    {
        "zone_name": "Hilltop Beacon Ground (Karthigai Deepam site)",
        "lat": 12.2380, "lng": 79.0710,
        "base_zone_risk": "Low",
        "note": "Normally low-traffic hilltop area; becomes an extreme crowd-crush risk zone only during the Karthigai Deepam festival (Nov/Dec).",
    },
    {
        "zone_name": "Girivalam Path - Lingam Shrine Cluster (Near Kaama Kaadu forest patch)",
        "lat": 12.2150, "lng": 79.0900,
        "base_zone_risk": "Medium",
        "note": "Path runs adjacent to a small dense forest patch; low lighting, fewer shops/patrols than the main loop.",
    },
]

level_val = {"Low": 0, "Medium": 1, "High": 2}
level_order = ["Low", "Medium", "High"]

N_PER_ZONE = 500
rows = []

for zone in ZONES:
    for _ in range(N_PER_ZONE):
        hour = np.random.randint(0, 24)
        # Is this a Karthigai Deepam festival window? (~5% of samples, Nov/Dec full moon period)
        is_festival = np.random.rand() < 0.05
        # Is this a heavy monsoon day? (Oct-Dec cyclone season, ~15% of samples)
        is_monsoon_heavy = np.random.rand() < 0.15

        crowd_level = np.random.choice(["Low", "Medium", "High"], p=[0.35, 0.4, 0.25])
        weather_risk = np.random.choice(["Low", "Medium", "High"], p=[0.55, 0.3, 0.15])
        tourist_density = np.random.choice(["Low", "Medium", "High"], p=[0.3, 0.4, 0.3])

        zone_risk = zone["base_zone_risk"]

        # Festival override: hilltop beacon ground becomes extreme risk during Karthigai Deepam
        if zone["zone_name"].startswith("Hilltop Beacon") and is_festival:
            zone_risk = "High"
            crowd_level = "High"
            tourist_density = "High"

        # Monsoon override: VOC Nagar landslide zone becomes High risk in heavy rain
        # (grounded in the real Dec 2024 landslide event)
        if zone["zone_name"].startswith("VOC Nagar") and is_monsoon_heavy:
            zone_risk = "High"
            weather_risk = "High"

        # Night discount for well-lit, high-footfall zones; night penalty for isolated ones
        night = hour >= 22 or hour <= 4

        # previous_incidents: synthetic, but skewed by zone risk level (higher base for riskier zones)
        base_lambda = {"Low": 0.8, "Medium": 2.0, "High": 4.0}[zone_risk]
        previous_incidents = np.random.poisson(base_lambda)

        # ---- Composite score (same rule logic as the general model, zone-aware) ----
        score = 0
        if night:
            score += 22 if zone_risk != "Low" else 8
        score += min(previous_incidents * 6, 30)
        score += level_val[zone_risk] * 16
        score += level_val[weather_risk] * 8
        if crowd_level == "Low" and zone_risk != "Low":
            score += 10  # isolation penalty in already-risky zones
        elif crowd_level == "High" and zone["zone_name"].startswith("Hilltop Beacon") and is_festival:
            score += 15  # crowd-crush risk specifically for festival over-crowding
        elif crowd_level == "High":
            score -= 4
        score += level_val[tourist_density] * 4
        score += np.random.normal(0, 5)
        score = max(0, min(100, score))

        label = "Low" if score <= 30 else ("Medium" if score <= 70 else "High")

        rows.append({
            "zone_name": zone["zone_name"],
            "latitude": zone["lat"] + np.random.uniform(-0.002, 0.002),
            "longitude": zone["lng"] + np.random.uniform(-0.002, 0.002),
            "time_hour": hour,
            "is_festival_period": int(is_festival),
            "is_monsoon_heavy_rain": int(is_monsoon_heavy),
            "previous_incidents": previous_incidents,
            "crowd_level": crowd_level,
            "weather_risk": weather_risk,
            "tourist_density": tourist_density,
            "zone_risk": zone_risk,
            "risk_score": round(score, 1),
            "risk_label": label,
        })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/tourist_safety/data/tiruvannamalai_tourist_risk_dataset.csv", index=False)

print(f"Generated {len(df)} rows across {len(ZONES)} real Tiruvannamalai zones\n")
print(df["risk_label"].value_counts(), "\n")
print(df.groupby("zone_name")["risk_score"].mean().sort_values(ascending=False))
