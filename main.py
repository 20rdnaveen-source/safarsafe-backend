"""
AI Tourist Safety & Emergency Response System - Backend API
SIH 2026 - Arunai Engineering College

Run with: uvicorn main:app --reload --port 8000
Docs auto-generated at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import joblib
import uuid
import os
import math

app = FastAPI(title="Tourist Safety & Emergency Response API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for demo; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
clf = joblib.load(os.path.join(MODEL_DIR, "tvm_risk_classifier.pkl"))
reg = joblib.load(os.path.join(MODEL_DIR, "tvm_risk_regressor.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "tvm_encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "tvm_feature_cols.pkl"))

# ---- In-memory "database" (swap for PostgreSQL in production) ----
users_db = {}
locations_db = []
incidents_db = []
predictions_db = []

# Real Tiruvannamalai zones (see data/generate_tiruvannamalai_dataset.py for sourcing notes).
# radius_km is an approximate demo catchment, not a surveyed boundary.
DEMO_ZONES = [
    {"name": "Annamalaiyar Temple (Base)", "lat": 12.2260, "lng": 79.0678, "radius_km": 0.6, "zone_risk": "Low"},
    {"name": "Girivalam Path - Outer Loop", "lat": 12.2200, "lng": 79.0800, "radius_km": 1.5, "zone_risk": "Medium"},
    {"name": "Sri Ramana Ashram", "lat": 12.2318, "lng": 79.0672, "radius_km": 0.4, "zone_risk": "Low"},
    {"name": "Hill Climb Route (via Ramanashram to Skandashram/Virupaksha Cave)", "lat": 12.2340, "lng": 79.0690, "radius_km": 0.7, "zone_risk": "Medium"},
    {"name": "Inner Girivalam Path / Forest Tract (Restricted)", "lat": 12.2400, "lng": 79.0850, "radius_km": 1.0, "zone_risk": "High"},
    {"name": "VOC Nagar (Hill Base, Landslide-Prone Zone)", "lat": 12.2280, "lng": 79.0640, "radius_km": 0.5, "zone_risk": "Medium"},
    {"name": "Hilltop Beacon Ground (Karthigai Deepam site)", "lat": 12.2380, "lng": 79.0710, "radius_km": 0.4, "zone_risk": "Low"},
    {"name": "Girivalam Path - Lingam Shrine Cluster (Near Kaama Kaadu forest patch)", "lat": 12.2150, "lng": 79.0900, "radius_km": 0.8, "zone_risk": "Medium"},
]


# ---------------- Schemas ----------------
class RegisterRequest(BaseModel):
    name: str
    phone: str
    language: str = "en"
    photo: Optional[str] = None  # base64 data URL, optional profile photo
    blood_group: Optional[str] = None  # e.g. "O+", filled in only if the user chooses to add it


class LoginRequest(BaseModel):
    phone: str


class LocationRequest(BaseModel):
    user_id: str
    latitude: float
    longitude: float


class RiskContext(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    time_hour: Optional[int] = None
    previous_incidents: Optional[int] = 2
    crowd_level: Optional[str] = "Medium"
    weather_risk: Optional[str] = "Low"
    tourist_density: Optional[str] = "Medium"
    is_festival_period: Optional[bool] = False
    is_monsoon_heavy_rain: Optional[bool] = False


class SOSRequest(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    incident_type: str = "SOS"
    severity: str = "High"
    name: Optional[str] = None    # filled in if the tourist is logged in
    phone: Optional[str] = None   # filled in if the tourist is logged in
    photo: Optional[str] = None   # filled in if the tourist added a photo
    blood_group: Optional[str] = None  # filled in if the tourist added it to their profile


class IncidentUpdate(BaseModel):
    status: str
    responder: Optional[str] = None
    action: Optional[str] = None


# ---------------- Helper: nearest real Tiruvannamalai zone lookup ----------------
def _haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Risk levels ranked so we can "step down" one level for the nearby-buffer case.
_RISK_ORDER = ["Low", "Medium", "High"]


def _step_down(risk_level):
    idx = _RISK_ORDER.index(risk_level)
    return _RISK_ORDER[max(0, idx - 1)]


def nearest_zone(lat, lng):
    inside = []
    for z in DEMO_ZONES:
        d = _haversine_km(lat, lng, z["lat"], z["lng"])
        if d <= z["radius_km"]:
            inside.append((d, z))
    if inside:
        # Genuinely inside a zone's boundary — use its exact risk level.
        inside.sort(key=lambda t: t[0])
        return {"name": inside[0][1]["name"], "zone_risk": inside[0][1]["zone_risk"], "distance_km": round(inside[0][0], 3)}

    # Not inside any zone boundary — find the truly closest one and how far
    # past its edge we are, instead of flattening everything to "Low".
    nearest = min(DEMO_ZONES, key=lambda z: _haversine_km(lat, lng, z["lat"], z["lng"]))
    dist_to_center = _haversine_km(lat, lng, nearest["lat"], nearest["lng"])
    dist_past_edge = dist_to_center - nearest["radius_km"]

    if dist_past_edge <= 0.3:
        # Within 300m of a zone's edge: still meaningfully close to that risk,
        # so use one level below the zone's own risk rather than jumping to Low.
        return {"name": nearest["name"], "zone_risk": _step_down(nearest["zone_risk"]), "distance_km": round(dist_to_center, 3)}

    # Genuinely far from every known zone: baseline Low.
    return {"name": None, "zone_risk": "Low", "distance_km": round(dist_to_center, 3)}


# ---------------- Auth ----------------
@app.post("/api/auth/register")
def register(req: RegisterRequest):
    user_id = str(uuid.uuid4())[:8]
    users_db[user_id] = {"user_id": user_id, "name": req.name, "phone": req.phone,
                          "language": req.language, "photo": req.photo,
                          "blood_group": req.blood_group,
                          "created_at": datetime.utcnow().isoformat()}
    return {"user_id": user_id, "message": "registered"}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    for u in users_db.values():
        if u["phone"] == req.phone:
            return u
    raise HTTPException(status_code=404, detail="User not found")


# ---------------- Location ----------------
@app.post("/api/location")
def update_location(req: LocationRequest):
    entry = {"user_id": req.user_id, "latitude": req.latitude, "longitude": req.longitude,
              "timestamp": datetime.utcnow().isoformat()}
    locations_db.append(entry)
    return {"message": "location updated", "entry": entry}


@app.get("/api/location/{user_id}")
def get_latest_location(user_id: str):
    user_locations = [l for l in locations_db if l["user_id"] == user_id]
    if not user_locations:
        return {"user_id": user_id, "latitude": None, "longitude": None, "timestamp": None}
    return user_locations[-1]


# ---------------- Risk Prediction ----------------
@app.post("/api/risk")
def get_risk(ctx: RiskContext):
    hour = ctx.time_hour if ctx.time_hour is not None else datetime.utcnow().hour
    zone = nearest_zone(ctx.latitude, ctx.longitude)
    zone_name = zone["name"]
    effective_zone_risk = zone["zone_risk"]

    # Dynamic overrides mirroring the training data's real-world-grounded logic:
    # Hilltop Beacon Ground spikes only during Karthigai Deepam; VOC Nagar spikes
    # only during heavy monsoon rain (per the documented Dec 2024 landslide).
    if zone_name == "Hilltop Beacon Ground (Karthigai Deepam site)" and ctx.is_festival_period:
        effective_zone_risk = "High"
    if zone_name == "VOC Nagar (Hill Base, Landslide-Prone Zone)" and ctx.is_monsoon_heavy_rain:
        effective_zone_risk = "High"

    # zone_name_enc: unmapped locations get the encoder's first known class as a safe fallback
    if zone_name is not None and zone_name in encoders["zone_name"].classes_:
        zone_name_enc = encoders["zone_name"].transform([zone_name])[0]
    else:
        zone_name_enc = 0

    row = {
        "time_hour": hour,
        "is_festival_period": int(ctx.is_festival_period),
        "is_monsoon_heavy_rain": int(ctx.is_monsoon_heavy_rain),
        "previous_incidents": ctx.previous_incidents,
        "crowd_level_enc": encoders["crowd_level"].transform([ctx.crowd_level])[0],
        "weather_risk_enc": encoders["weather_risk"].transform([ctx.weather_risk])[0],
        "tourist_density_enc": encoders["tourist_density"].transform([ctx.tourist_density])[0],
        "zone_risk_enc": encoders["zone_risk"].transform([effective_zone_risk])[0],
        "zone_name_enc": zone_name_enc,
    }
    X = [[row[c] for c in feature_cols]]

    risk_label = clf.predict(X)[0]
    risk_score = float(reg.predict(X)[0])
    risk_score = max(0, min(100, risk_score))

    result = {
        "user_id": ctx.user_id,
        "risk_score": round(risk_score, 1),
        "risk_level": risk_label,
        "zone_name": zone_name,
        "timestamp": datetime.utcnow().isoformat(),
    }
    predictions_db.append(result)
    return result


# ---------------- Safe Route (stub for demo) ----------------
@app.get("/api/safe-route")
def safe_route(lat: float, lng: float, dest_lat: float, dest_lng: float):
    # Demo stub: real version would call a routing API and avoid DEMO_ZONES polygons
    return {
        "route": "safer_route_demo",
        "avoids_zones": [z["name"] for z in DEMO_ZONES],
        "note": "Prototype stub — production version integrates a real routing API and geofenced zone avoidance",
    }


# ---------------- Nearby Help ----------------
from nearby_data import HOSPITALS, POLICE_STATIONS, FIRE_STATIONS
from hotels_places_data import HOTELS, PLACES_TO_VISIT, PUBLIC_TOILETS, TRANSPORT_HUBS, ATMS


def haversine_km(lat1, lng1, lat2, lng2):
    """Real straight-line distance between two coordinates, in kilometers."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest(lat, lng, places):
    best = None
    best_dist = None
    for place in places:
        d = haversine_km(lat, lng, place["latitude"], place["longitude"])
        if best_dist is None or d < best_dist:
            best = place
            best_dist = d
    return {
        "name": best["name"],
        "distance_km": round(best_dist, 2),
        "phone": best.get("phone"),
        "type": best.get("type"),
        "latitude": best["latitude"],
        "longitude": best["longitude"],
    }


def find_nearest_n(lat, lng, places, n=5):
    scored = []
    for place in places:
        d = haversine_km(lat, lng, place["latitude"], place["longitude"])
        scored.append({
            "name": place["name"],
            "distance_km": round(d, 2),
            "phone": place.get("phone"),
            "type": place.get("type"),
            "latitude": place["latitude"],
            "longitude": place["longitude"],
        })
    scored.sort(key=lambda x: x["distance_km"])
    return scored[:n]


@app.get("/api/nearby-help")
def nearby_help(lat: float, lng: float):
    return {
        "nearest_police_station": find_nearest(lat, lng, POLICE_STATIONS),
        "nearest_hospital": find_nearest(lat, lng, HOSPITALS),
        "nearest_fire_station": find_nearest(lat, lng, FIRE_STATIONS),
        "tourist_helpline": "1363",  # national Tourist Helpline (verify current number before deployment)
    }


@app.get("/api/nearby-hospitals")
def nearby_hospitals(lat: float, lng: float, limit: int = 10):
    return {"hospitals": find_nearest_n(lat, lng, HOSPITALS, n=limit)}


@app.get("/api/nearby-police")
def nearby_police(lat: float, lng: float, limit: int = 10):
    return {"police_stations": find_nearest_n(lat, lng, POLICE_STATIONS, n=limit)}


def find_nearest_n_full(lat, lng, places, n=10):
    """Like find_nearest_n, but keeps extra fields (rating, category, description)."""
    scored = []
    for place in places:
        d = haversine_km(lat, lng, place["latitude"], place["longitude"])
        entry = dict(place)
        entry["distance_km"] = round(d, 2)
        scored.append(entry)
    scored.sort(key=lambda x: x["distance_km"])
    return scored[:n]


@app.get("/api/nearby-hotels")
def nearby_hotels(lat: float, lng: float, limit: int = 20):
    return {"hotels": find_nearest_n_full(lat, lng, HOTELS, n=limit)}


@app.get("/api/nearby-places")
def nearby_places(lat: float, lng: float, limit: int = 20):
    return {"places": find_nearest_n_full(lat, lng, PLACES_TO_VISIT, n=limit)}


@app.get("/api/nearby-toilets")
def nearby_toilets(lat: float, lng: float, limit: int = 20):
    return {"toilets": find_nearest_n_full(lat, lng, PUBLIC_TOILETS, n=limit)}


@app.get("/api/nearby-transport")
def nearby_transport(lat: float, lng: float, limit: int = 20):
    return {"transport": find_nearest_n_full(lat, lng, TRANSPORT_HUBS, n=limit)}


@app.get("/api/nearby-atms")
def nearby_atms(lat: float, lng: float, limit: int = 20):
    return {"atms": find_nearest_n_full(lat, lng, ATMS, n=limit)}


# ---------------- SOS / Incidents ----------------
@app.post("/api/sos")
def trigger_sos(req: SOSRequest):
    incident_id = str(uuid.uuid4())[:8]
    incident = {
        "incident_id": incident_id,
        "user_id": req.user_id,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "incident_type": req.incident_type,
        "severity": req.severity,
        "status": "Unassigned",
        "created_at": datetime.utcnow().isoformat(),
        "actions": [],
        "reporter_name": req.name,    # None if the tourist wasn't logged in
        "reporter_phone": req.phone,  # None if the tourist wasn't logged in
        "reporter_photo": req.photo,  # None if no photo was added
        "reporter_blood_group": req.blood_group,  # None if not added
    }
    incidents_db.append(incident)
    return incident


@app.post("/api/incidents/report")
def report_incident(req: SOSRequest):
    return trigger_sos(req)


@app.get("/api/incidents")
def list_incidents():
    return {"incidents": incidents_db}


@app.patch("/api/incidents/{incident_id}")
def update_incident(incident_id: str, update: IncidentUpdate):
    for inc in incidents_db:
        if inc["incident_id"] == incident_id:
            inc["status"] = update.status
            if update.responder or update.action:
                inc["actions"].append({
                    "responder": update.responder,
                    "action": update.action,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            return inc
    raise HTTPException(status_code=404, detail="Incident not found")


@app.get("/")
def root():
    return {"status": "Tourist Safety API running", "docs": "/docs"}
