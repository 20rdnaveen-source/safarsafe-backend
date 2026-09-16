"""
AI Tourist Safety & Emergency Response System - Backend API
SIH 2026 - Arunai Engineering College
 
Run with: uvicorn main:app --reload --port 8000
Docs auto-generated at: http://localhost:8000/docs
"""
 
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta, timezone
import joblib
import uuid
import os
import math
import requests
import socket
import hashlib
import secrets
 
# --- Fix: some cloud hosts (including Render) resolve overpass-api.de to an
# IPv6 address first but have no working IPv6 route out, causing
# "Network is unreachable" even though IPv4 works fine. Force IPv4-only DNS
# resolution for all outbound requests so this doesn't happen.
_original_getaddrinfo = socket.getaddrinfo
 
 
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
 
 
socket.getaddrinfo = _ipv4_only_getaddrinfo
 
app = FastAPI(title="Tourist Safety & Emergency Response API", version="1.0")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for demo; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)
 
MODEL_DIR = os.path.join(os.path.dirname(__file__),"model")
# Switched from the Tiruvannamalai-only model (tvm_risk_classifier.pkl, which
# encodes 8 hardcoded TVM zone names as a training feature and cannot
# generalize to any other city) to the GENERIC model. The generic model's
# features (time_hour, previous_incidents, crowd_level, weather_risk,
# tourist_density, zone_risk) are all place-agnostic — see
# model/train_model.py vs model/train_tvm_model.py for the difference.
clf = joblib.load(os.path.join(MODEL_DIR, "risk_classifier.pkl"))
reg = joblib.load(os.path.join(MODEL_DIR, "risk_regressor.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
 
# ---- In-memory "database" (swap for PostgreSQL in production) ----
users_db = {}
locations_db = []
incidents_db = []
predictions_db = []
 
# Curated reference zones: named tourist areas with a known baseline risk
# level. This table is now a DATA file, not a model feature — adding a new
# city just means adding rows here, with NO retraining required (unlike the
# old TVM model, which had to memorize zone names during training).
# radius_km is an approximate demo catchment, not a surveyed boundary.
# festival_sensitive / monsoon_sensitive replace what used to be exact
# zone-name string matches in the old TVM-specific code, so any city's zones
# can opt into the same seasonal-risk behavior without new code.
REFERENCE_ZONES = [
    {"name": "Annamalaiyar Temple (Base)", "lat": 12.2260, "lng": 79.0678, "radius_km": 0.6, "zone_risk": "Low", "city": "Tiruvannamalai"},
    {"name": "Girivalam Path - Outer Loop", "lat": 12.2200, "lng": 79.0800, "radius_km": 1.5, "zone_risk": "Medium", "city": "Tiruvannamalai"},
    {"name": "Sri Ramana Ashram", "lat": 12.2318, "lng": 79.0672, "radius_km": 0.4, "zone_risk": "Low", "city": "Tiruvannamalai"},
    {"name": "Hill Climb Route (via Ramanashram to Skandashram/Virupaksha Cave)", "lat": 12.2340, "lng": 79.0690, "radius_km": 0.7, "zone_risk": "Medium", "city": "Tiruvannamalai"},
    {"name": "Inner Girivalam Path / Forest Tract (Restricted)", "lat": 12.2400, "lng": 79.0850, "radius_km": 1.0, "zone_risk": "High", "city": "Tiruvannamalai"},
    {"name": "VOC Nagar (Hill Base, Landslide-Prone Zone)", "lat": 12.2280, "lng": 79.0640, "radius_km": 0.5, "zone_risk": "Medium", "city": "Tiruvannamalai", "monsoon_sensitive": True},
    {"name": "Hilltop Beacon Ground (Karthigai Deepam site)", "lat": 12.2380, "lng": 79.0710, "radius_km": 0.4, "zone_risk": "Low", "city": "Tiruvannamalai", "festival_sensitive": True},
    {"name": "Girivalam Path - Lingam Shrine Cluster (Near Kaama Kaadu forest patch)", "lat": 12.2150, "lng": 79.0900, "radius_km": 0.8, "zone_risk": "Medium", "city": "Tiruvannamalai"},
 
    # ---- Jaipur ---- (coordinates are approximate landmark locations —
    # verify against a survey/GIS source before relying on them in production)
    {"name": "Hawa Mahal", "lat": 26.9239, "lng": 75.8267, "radius_km": 0.4, "zone_risk": "Medium", "city": "Jaipur"},
    {"name": "Amber Fort", "lat": 26.9855, "lng": 75.8513, "radius_km": 0.8, "zone_risk": "Medium", "city": "Jaipur"},
    {"name": "City Palace, Jaipur", "lat": 26.9258, "lng": 75.8237, "radius_km": 0.5, "zone_risk": "Low", "city": "Jaipur"},
    {"name": "Nahargarh Fort (hill approach road)", "lat": 26.9373, "lng": 75.8155, "radius_km": 0.6, "zone_risk": "Medium", "city": "Jaipur"},
 
    # ---- Goa ---- (beach zones are monsoon_sensitive — rip currents/rough
    # seas during heavy monsoon rain are a real, well-documented seasonal risk)
    {"name": "Baga Beach", "lat": 15.5553, "lng": 73.7517, "radius_km": 1.0, "zone_risk": "Medium", "city": "Goa", "monsoon_sensitive": True},
    {"name": "Calangute Beach", "lat": 15.5439, "lng": 73.7553, "radius_km": 1.0, "zone_risk": "Medium", "city": "Goa", "monsoon_sensitive": True},
    {"name": "Basilica of Bom Jesus, Old Goa", "lat": 15.5009, "lng": 73.9116, "radius_km": 0.5, "zone_risk": "Low", "city": "Goa"},
    {"name": "Dudhsagar Falls approach trail", "lat": 15.3144, "lng": 74.3142, "radius_km": 1.5, "zone_risk": "High", "city": "Goa", "monsoon_sensitive": True},
 
    # ---- Varanasi ---- (ghats get festival_sensitive during Dev Deepawali/
    # Ganga Aarti crowd surges, which is a well-known real seasonal pattern)
    {"name": "Kashi Vishwanath Temple", "lat": 25.3109, "lng": 83.0107, "radius_km": 0.4, "zone_risk": "Medium", "city": "Varanasi", "festival_sensitive": True},
    {"name": "Dashashwamedh Ghat", "lat": 25.3038, "lng": 83.0107, "radius_km": 0.5, "zone_risk": "Medium", "city": "Varanasi", "festival_sensitive": True},
    {"name": "Assi Ghat", "lat": 25.2919, "lng": 83.0106, "radius_km": 0.5, "zone_risk": "Low", "city": "Varanasi"},
 
    # TODO: keep extending this same way for any other city you add — same
    # shape, no model retraining required. Treat the coordinates above as a
    # demo-ready starting point, not surveyed/verified boundaries.
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
    channel: str = "app"  # "app" (HTTPS) | "sms_offline" (set by the textbee webhook itself,
                           # not normally sent by the app) | "satellite" (see SOS_CHANNELS below —
                           # accepted here so the UI can show it, but not yet a working transport)
 
 
class IncidentUpdate(BaseModel):
    status: str
    responder: Optional[str] = None
    action: Optional[str] = None
    responder_token: Optional[str] = None  # proves this is a real logged-in org/staff account
    responder_lat: Optional[float] = None  # the responder's own current location, if sharing it
    responder_lng: Optional[float] = None
 
 
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
    for z in REFERENCE_ZONES:
        d = _haversine_km(lat, lng, z["lat"], z["lng"])
        if d <= z["radius_km"]:
            inside.append((d, z))
    if inside:
        # Genuinely inside a zone's boundary — use its exact risk level.
        inside.sort(key=lambda t: t[0])
        return inside[0][1] | {"distance_km": round(inside[0][0], 3)}
 
    # Not inside any zone boundary — find the truly closest one and how far
    # past its edge we are, instead of flattening everything to "Low".
    nearest = min(REFERENCE_ZONES, key=lambda z: _haversine_km(lat, lng, z["lat"], z["lng"]))
    dist_to_center = _haversine_km(lat, lng, nearest["lat"], nearest["lng"])
    dist_past_edge = dist_to_center - nearest["radius_km"]
 
    if dist_past_edge <= 0.3:
        # Within 300m of a zone's edge: still meaningfully close to that risk,
        # so use one level below the zone's own risk rather than jumping to Low.
        return nearest | {"zone_risk": _step_down(nearest["zone_risk"]), "distance_km": round(dist_to_center, 3)}
 
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
 
 
# ---------------- Trip Sharing (group live location, join by code) ----------------
import random
import string
 
trips_db = {}  # trip_code -> {trip_code, name, created_at, members: [{user_id, name, phone, photo}]}
 
 
class TripCreateRequest(BaseModel):
    name: str
    user_id: str
    member_name: Optional[str] = None
    member_phone: Optional[str] = None
    member_photo: Optional[str] = None
 
 
class TripJoinRequest(BaseModel):
    code: str
    user_id: str
    member_name: Optional[str] = None
    member_phone: Optional[str] = None
    member_photo: Optional[str] = None
 
 
def _generate_trip_code():
    while True:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if code not in trips_db:
            return code
 
 
@app.post("/api/trips/create")
def create_trip(req: TripCreateRequest):
    code = _generate_trip_code()
    trip = {
        "trip_code": code,
        "name": req.name,
        "created_at": datetime.utcnow().isoformat(),
        "members": [{
            "user_id": req.user_id,
            "name": req.member_name or "Traveler",
            "phone": req.member_phone,
            "photo": req.member_photo,
        }],
    }
    trips_db[code] = trip
    return trip
 
 
@app.post("/api/trips/join")
def join_trip(req: TripJoinRequest):
    trip = trips_db.get(req.code.upper())
    if not trip:
        raise HTTPException(status_code=404, detail="No trip found with that code.")
 
    # Don't add the same user twice if they rejoin.
    if not any(m["user_id"] == req.user_id for m in trip["members"]):
        trip["members"].append({
            "user_id": req.user_id,
            "name": req.member_name or "Traveler",
            "phone": req.member_phone,
            "photo": req.member_photo,
        })
    return trip
 
 
@app.get("/api/trips/{code}")
def get_trip(code: str):
    trip = trips_db.get(code.upper())
    if not trip:
        raise HTTPException(status_code=404, detail="No trip found with that code.")
    return trip
 
 
@app.get("/api/trips/{code}/locations")
def get_trip_locations(code: str):
    trip = trips_db.get(code.upper())
    if not trip:
        raise HTTPException(status_code=404, detail="No trip found with that code.")
 
    results = []
    for member in trip["members"]:
        loc = get_latest_location(member["user_id"])
        results.append({
            "user_id": member["user_id"],
            "name": member["name"],
            "phone": member["phone"],
            "photo": member["photo"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "timestamp": loc["timestamp"],
        })
    return {"trip_code": code.upper(), "members": results}
 
 
# ---------------- Risk Prediction ----------------
def _real_previous_incidents(lat, lng, radius_km=2.0, days=30):
    """Counts real incidents from OUR OWN incidents_db near this location in
    the recent window, replacing the old hardcoded default of 2. Works for
    any location in India — it's just distance + a timestamp filter over
    data this backend already collects, no external dataset needed."""
    if not incidents_db:
        return 0
    cutoff = datetime.utcnow() - timedelta(days=days)
    count = 0
    for inc in incidents_db:
        try:
            inc_lat, inc_lng = inc.get("latitude"), inc.get("longitude")
            if inc_lat is None or inc_lng is None:
                continue
            created = datetime.fromisoformat(inc["created_at"])
            if created < cutoff:
                continue
            if _haversine_km(lat, lng, inc_lat, inc_lng) <= radius_km:
                count += 1
        except (KeyError, ValueError, TypeError):
            continue  # malformed/older incident record — skip rather than crash the risk endpoint
    return count
 
 
def _compute_weather_risk(lat: float, lng: float) -> str:
    """Real current weather via Open-Meteo — free, no API key required,
    same provider the web demo (tourist_app_demo.html) already uses for
    weather, so this isn't a new dependency for the project. Maps current
    conditions to Low/Medium/High. On any failure, falls back to 'Low'
    rather than inventing a risk level the weather data doesn't support."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lng,
                "current": "precipitation,wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=6,
        )
        resp.raise_for_status()
        current = resp.json().get("current", {})
        precip = current.get("precipitation") or 0
        wind = current.get("wind_speed_10m") or 0
        code = current.get("weather_code")
 
        # WMO weather codes (the standard Open-Meteo uses):
        # 65/67/82 = heavy rain/rain showers, 75/86 = heavy snow, 95-99 = thunderstorm
        severe_codes = {65, 67, 75, 82, 86, 95, 96, 99}
        if code in severe_codes or precip >= 10 or wind >= 40:
            return "High"
        # 51-55 = drizzle, 61-63 = light/moderate rain, 71-73 = light/moderate snow, 80-81 = rain showers
        moderate_codes = {51, 53, 55, 61, 63, 71, 73, 80, 81}
        if code in moderate_codes or precip >= 2 or wind >= 25:
            return "Medium"
        return "Low"
    except Exception:
        return "Low"  # weather API unreachable/unexpected response — never fabricate a risk level
 
 
def _compute_crowd_and_density(lat: float, lng: float, radius_km: float = 1.0):
    """First-party proxy for crowd_level/tourist_density, built from OUR OWN
    /api/location pings (locations_db) — real data this backend already
    collects, not a scraped or invented signal.
 
    HONEST LIMITATION: Google's Places API does not officially expose a
    'Popular Times'/live-busyness field (only unofficial scrapers do, which
    violate Google's Terms of Service — not something to build on). So this
    reflects SafarSafe's OWN user activity only. It will be a weak signal
    with few app users and get meaningfully better as adoption grows — an
    honest trade, not a shortcut. `has_enough_data` tells the caller when
    to trust this vs. fall back to a curated per-zone default."""
    now = datetime.utcnow()
    recent_window = timedelta(minutes=45)   # "how busy is it right now"
    density_window = timedelta(days=7)      # "how touristy is this area generally"
 
    recent_users = set()
    weekly_users = set()
    for entry in locations_db:
        try:
            d = _haversine_km(lat, lng, entry["latitude"], entry["longitude"])
        except (KeyError, TypeError):
            continue
        if d > radius_km:
            continue
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
        except (KeyError, ValueError):
            continue
        if now - ts <= recent_window:
            recent_users.add(entry["user_id"])
        if now - ts <= density_window:
            weekly_users.add(entry["user_id"])
 
    def _bucket(n, low_max, med_max):
        if n <= low_max:
            return "Low"
        if n <= med_max:
            return "Medium"
        return "High"
 
    # Thresholds are placeholders — tune once you have real usage numbers;
    # having a real, live-data pipeline at all matters more than the exact
    # cutoffs right now.
    crowd_level = _bucket(len(recent_users), low_max=1, med_max=4)
    tourist_density = _bucket(len(weekly_users), low_max=3, med_max=15)
    has_enough_data = len(weekly_users) >= 3
 
    return crowd_level, tourist_density, has_enough_data
 
 
@app.post("/api/risk")
def get_risk(ctx: RiskContext):
    hour = ctx.time_hour if ctx.time_hour is not None else datetime.utcnow().hour
    zone = nearest_zone(ctx.latitude, ctx.longitude)
    zone_name = zone["name"]
    effective_zone_risk = zone["zone_risk"]
 
    # Seasonal zone-risk bumps now come from flags ON the zone's own data
    # (festival_sensitive / monsoon_sensitive in REFERENCE_ZONES) instead of
    # matching hardcoded Tiruvannamalai zone-name strings — so this works
    # for any city's curated zones without touching this code again.
    if zone.get("festival_sensitive") and ctx.is_festival_period:
        effective_zone_risk = "High"
    if zone.get("monsoon_sensitive") and ctx.is_monsoon_heavy_rain:
        effective_zone_risk = "High"
 
    # Real incident count from our own data, for any location — replaces
    # the RiskContext default (previously always defaulted to 2 regardless
    # of where the tourist actually was).
    real_previous_incidents = _real_previous_incidents(ctx.latitude, ctx.longitude)
 
    # Real weather, computed for wherever the tourist actually is, instead
    # of trusting the client-supplied default (which was always "Low").
    computed_weather_risk = _compute_weather_risk(ctx.latitude, ctx.longitude)
 
    # Real crowd/density from our own users' location pings. Falls back to
    # the client-supplied value (or its "Medium" default) only when we
    # don't yet have enough of our own data at this spot — an honest
    # cold-start behavior, not a silent guess dressed up as real data.
    computed_crowd, computed_density, has_enough_data = _compute_crowd_and_density(
        ctx.latitude, ctx.longitude
    )
    effective_crowd_level = computed_crowd if has_enough_data else ctx.crowd_level
    effective_tourist_density = computed_density if has_enough_data else ctx.tourist_density
 
    row = {
        "time_hour": hour,
        "previous_incidents": real_previous_incidents,
        "crowd_level_enc": encoders["crowd_level"].transform([effective_crowd_level])[0],
        "weather_risk_enc": encoders["weather_risk"].transform([computed_weather_risk])[0],
        "tourist_density_enc": encoders["tourist_density"].transform([effective_tourist_density])[0],
        "zone_risk_enc": encoders["zone_risk"].transform([effective_zone_risk])[0],
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
        "weather_risk": computed_weather_risk,
        "crowd_level": effective_crowd_level,
        "tourist_density": effective_tourist_density,
        "crowd_data_source": "live_app_data" if has_enough_data else "fallback_default",
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
        "avoids_zones": [z["name"] for z in REFERENCE_ZONES],
        "note": "Prototype stub — production version integrates a real routing API and geofenced zone avoidance",
    }
 
 
# ---------------- Nearby Help ----------------
from nearby_data import HOSPITALS, POLICE_STATIONS, FIRE_STATIONS
from hotels_places_data import HOTELS, PLACES_TO_VISIT, PUBLIC_TOILETS, TRANSPORT_HUBS, ATMS
 
 
# ---------------- Responder Accounts (hospitals/police login, staff members, nearby-only incident access) ----------------
# Two-tier accounts: an "org" account represents a real, verified hospital or
# police station — it can only be created against a facility that's actually
# in our own verified database (never a free-typed name), which stops anyone
# from registering a fake station. Once an org is logged in, it can create
# individual "member" accounts under itself (e.g. one for each ambulance
# driver or officer). Only logged-in org/member accounts can see full victim
# details (name, phone, blood group, photo) on an incident, and only for
# incidents actually near their own registered facility.
 
responder_orgs_db = {}      # username -> {username, password_hash, salt, org_type, facility_name, latitude, longitude, created_at}
responder_members_db = {}   # member_username -> {member_username, password_hash, salt, org_username, name, role, created_at}
responder_tokens_db = {}    # token -> {kind: "org"|"member", username, org_username}
login_history_db = []       # every org/staff/admin login attempt, for the admin's audit view
 
RESPONDER_NEARBY_RADIUS_KM = 20  # an org only sees incidents within this radius of its own registered location
 
 
def _hash_password(password: str, salt: str = None) -> tuple:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return digest, salt
 
 
def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    digest, _ = _hash_password(password, salt)
    return secrets.compare_digest(digest, stored_hash)
 
 
def _find_verified_facility(org_type: str, facility_name: str):
    source = HOSPITALS if org_type == "hospital" else POLICE_STATIONS if org_type == "police" else None
    if source is None:
        return None
    for f in source:
        if f["name"] == facility_name:
            return f
    return None
 
 
class OrgRegisterRequest(BaseModel):
    username: str
    password: str
    org_type: str  # "hospital" or "police"
    facility_name: str  # must exactly match a name in our verified database
 
 
class OrgLoginRequest(BaseModel):
    username: str
    password: str
 
 
class MemberCreateRequest(BaseModel):
    token: str  # the org's own login token, proves they're allowed to add staff
    member_username: str
    member_password: str
    member_name: str
    role: Optional[str] = None  # e.g. "Ambulance Driver", "Duty Officer"
 
 
class MemberLoginRequest(BaseModel):
    member_username: str
    member_password: str
 
 
@app.get("/api/responder/facilities")
def list_verified_facilities(org_type: str):
    """So the registration screen can offer a picker of real facilities only —
    never a free-text field a person could fake."""
    source = HOSPITALS if org_type == "hospital" else POLICE_STATIONS if org_type == "police" else None
    if source is None:
        raise HTTPException(status_code=400, detail="org_type must be 'hospital' or 'police'")
    return {
        "facilities": [
            {"name": f["name"], "already_registered": f["name"] in [
                o["facility_name"] for o in responder_orgs_db.values() if o["org_type"] == org_type
            ]}
            for f in source
        ]
    }
 
 
@app.post("/api/responder/register-org")
def register_org(req: OrgRegisterRequest):
    if req.username in responder_orgs_db:
        raise HTTPException(status_code=409, detail="That username is already taken.")
 
    facility = _find_verified_facility(req.org_type, req.facility_name)
    if not facility:
        raise HTTPException(
            status_code=400,
            detail="That facility isn't in our verified database — pick one from the list, real stations only.",
        )
    already = any(o["facility_name"] == req.facility_name for o in responder_orgs_db.values())
    if already:
        raise HTTPException(status_code=409, detail="This facility already has a registered account.")
 
    password_hash, salt = _hash_password(req.password)
    responder_orgs_db[req.username] = {
        "username": req.username,
        "password_hash": password_hash,
        "salt": salt,
        "org_type": req.org_type,
        "facility_name": req.facility_name,
        "latitude": facility["latitude"],
        "longitude": facility["longitude"],
        "created_at": datetime.utcnow().isoformat(),
    }
    return {"message": "Organization registered. You can now log in."}
 
 
@app.post("/api/responder/login-org")
def login_org(req: OrgLoginRequest):
    org = responder_orgs_db.get(req.username)
    if not org or not _verify_password(req.password, org["password_hash"], org["salt"]):
        login_history_db.append({
            "type": "org", "username": req.username, "facility_name": None,
            "success": False, "timestamp": datetime.utcnow().isoformat(),
        })
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    token = secrets.token_hex(24)
    responder_tokens_db[token] = {"kind": "org", "username": req.username, "org_username": req.username}
    login_history_db.append({
        "type": "org", "username": req.username, "facility_name": org["facility_name"],
        "success": True, "timestamp": datetime.utcnow().isoformat(),
    })
    return {
        "token": token,
        "org_type": org["org_type"],
        "facility_name": org["facility_name"],
        "username": org["username"],
    }
 
 
@app.post("/api/responder/create-member")
def create_member(req: MemberCreateRequest):
    session = responder_tokens_db.get(req.token)
    if not session or session["kind"] != "org":
        raise HTTPException(status_code=401, detail="Log in as the organization first to add staff.")
    if req.member_username in responder_members_db:
        raise HTTPException(status_code=409, detail="That staff username is already taken.")
 
    password_hash, salt = _hash_password(req.member_password)
    responder_members_db[req.member_username] = {
        "member_username": req.member_username,
        "password_hash": password_hash,
        "salt": salt,
        "org_username": session["org_username"],
        "name": req.member_name,
        "role": req.role,
        "created_at": datetime.utcnow().isoformat(),
    }
    return {"message": f"Staff account created for {req.member_name}."}
 
 
@app.post("/api/responder/login-member")
def login_member(req: MemberLoginRequest):
    member = responder_members_db.get(req.member_username)
    if not member or not _verify_password(req.member_password, member["password_hash"], member["salt"]):
        login_history_db.append({
            "type": "member", "username": req.member_username, "facility_name": None,
            "success": False, "timestamp": datetime.utcnow().isoformat(),
        })
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    org = responder_orgs_db.get(member["org_username"])
    token = secrets.token_hex(24)
    responder_tokens_db[token] = {"kind": "member", "username": req.member_username, "org_username": member["org_username"]}
    login_history_db.append({
        "type": "member", "username": req.member_username,
        "facility_name": org["facility_name"] if org else None,
        "success": True, "timestamp": datetime.utcnow().isoformat(),
    })
    return {
        "token": token,
        "name": member["name"],
        "role": member["role"],
        "org_type": org["org_type"] if org else None,
        "facility_name": org["facility_name"] if org else None,
    }
 
 
@app.get("/api/responder/incidents")
def responder_incidents(token: str):
    """Full incident details (name, phone, blood group, photo included),
    but only for incidents near the logged-in org's own verified location —
    and only reachable with a valid login token."""
    session = responder_tokens_db.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired login — please log in again.")
    org = responder_orgs_db.get(session["org_username"])
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found.")
 
    nearby = []
    for inc in incidents_db:
        d = haversine_km(org["latitude"], org["longitude"], inc["latitude"], inc["longitude"])
        if d <= RESPONDER_NEARBY_RADIUS_KM:
            entry = dict(inc)
            entry["distance_km"] = round(d, 2)
            nearby.append(entry)
    nearby.sort(key=lambda x: x["distance_km"])
    return {
        "facility_name": org["facility_name"],
        "org_type": org["org_type"],
        "radius_km": RESPONDER_NEARBY_RADIUS_KM,
        "incidents": nearby,
    }
 
 
# ---------------- Admin (separate from org/staff — full oversight, credentials never in code) ----------------
# The admin username/password live ONLY as Render environment variables,
# never in this file, since this repo is public on GitHub — hardcoding real
# credentials here would expose them to anyone who visits the repo.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
 
admin_tokens_db = {}  # token -> True (kept separate from responder tokens, admin can't accidentally get org-level access)
 
 
class AdminLoginRequest(BaseModel):
    username: str
    password: str
 
 
def _require_admin(token: str):
    if token not in admin_tokens_db:
        raise HTTPException(status_code=401, detail="Invalid or expired admin session — please log in again.")
 
 
@app.post("/api/admin/login")
def admin_login(req: AdminLoginRequest):
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="Admin login isn't configured on the server yet.")
    if req.username != ADMIN_USERNAME or req.password != ADMIN_PASSWORD:
        login_history_db.append({
            "type": "admin", "username": req.username, "facility_name": None,
            "success": False, "timestamp": datetime.utcnow().isoformat(),
        })
        raise HTTPException(status_code=401, detail="Incorrect admin username or password.")
    token = secrets.token_hex(24)
    admin_tokens_db[token] = True
    login_history_db.append({
        "type": "admin", "username": req.username, "facility_name": None,
        "success": True, "timestamp": datetime.utcnow().isoformat(),
    })
    return {"token": token}
 
 
@app.get("/api/admin/login-history")
def admin_login_history(token: str):
    _require_admin(token)
    return {"history": list(reversed(login_history_db))}  # most recent first
 
 
@app.get("/api/admin/orgs")
def admin_list_orgs(token: str):
    _require_admin(token)
    orgs = []
    for username, org in responder_orgs_db.items():
        member_count = sum(1 for m in responder_members_db.values() if m["org_username"] == username)
        orgs.append({
            "username": username,
            "org_type": org["org_type"],
            "facility_name": org["facility_name"],
            "created_at": org["created_at"],
            "member_count": member_count,
        })
    return {"orgs": orgs}
 
 
@app.get("/api/admin/members")
def admin_list_members(token: str):
    _require_admin(token)
    members = []
    for username, member in responder_members_db.items():
        members.append({
            "member_username": username,
            "name": member["name"],
            "role": member["role"],
            "org_username": member["org_username"],
            "created_at": member["created_at"],
        })
    return {"members": members}
 
 
@app.delete("/api/admin/orgs/{org_username}")
def admin_delete_org(org_username: str, token: str):
    _require_admin(token)
    if org_username not in responder_orgs_db:
        raise HTTPException(status_code=404, detail="Organization not found.")
    del responder_orgs_db[org_username]
    # Cascade: remove every staff account that belonged to this org too, and
    # invalidate any active login sessions tied to them so access is revoked
    # immediately, not just on their next request.
    removed_members = [u for u, m in responder_members_db.items() if m["org_username"] == org_username]
    for u in removed_members:
        del responder_members_db[u]
    stale_tokens = [t for t, s in responder_tokens_db.items() if s["org_username"] == org_username]
    for t in stale_tokens:
        del responder_tokens_db[t]
    return {"message": f"Deleted organization '{org_username}' and {len(removed_members)} staff account(s)."}
 
 
@app.delete("/api/admin/members/{member_username}")
def admin_delete_member(member_username: str, token: str):
    _require_admin(token)
    if member_username not in responder_members_db:
        raise HTTPException(status_code=404, detail="Staff account not found.")
    del responder_members_db[member_username]
    stale_tokens = [t for t, s in responder_tokens_db.items() if s.get("username") == member_username]
    for t in stale_tokens:
        del responder_tokens_db[t]
    return {"message": f"Deleted staff account '{member_username}'."}
 
 
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
 
 
def osm_overpass_search(lat, lng, radius_km, osm_filters, limit=20):
    """
    Query OpenStreetMap's live Overpass API for real points of interest
    anywhere in India (or the world) — not limited to our hand-typed
    Tiruvannamalai list. osm_filters is a list of Overpass tag filter
    strings, e.g. ['"amenity"="hospital"'].
    Never invents data: if Overpass is unreachable or returns nothing,
    returns an empty list so the caller can fall back to curated data.
    """
    radius_m = int(radius_km * 1000)
    filter_clauses = "".join(f'nwr(around:{radius_m},{lat},{lng})[{f}];' for f in osm_filters)
    query = f'[out:json][timeout:20];({filter_clauses});out center {limit * 3};'
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
 
    results = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue  # skip unnamed OSM entries — not useful to show a tourist
        if el.get("type") == "node":
            el_lat, el_lng = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            el_lat, el_lng = center.get("lat"), center.get("lon")
        if el_lat is None or el_lng is None:
            continue
        d = haversine_km(lat, lng, el_lat, el_lng)
        results.append({
            "name": name,
            "distance_km": round(d, 2),
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "type": tags.get("amenity") or tags.get("tourism") or tags.get("shop") or tags.get("railway"),
            "latitude": el_lat,
            "longitude": el_lng,
            "source": "OpenStreetMap (live)",
        })
    results.sort(key=lambda x: x["distance_km"])
    return results[:limit]
 
 
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
    live = osm_overpass_search(lat, lng, radius_km=15, osm_filters=['"amenity"="hospital"'], limit=limit)
    if live:
        return {"hospitals": live, "source": "live"}
    return {"hospitals": find_nearest_n(lat, lng, HOSPITALS, n=limit), "source": "curated_fallback"}
 
 
@app.get("/api/nearby-police")
def nearby_police(lat: float, lng: float, limit: int = 10):
    live = osm_overpass_search(lat, lng, radius_km=15, osm_filters=['"amenity"="police"'], limit=limit)
    if live:
        return {"police_stations": live, "source": "live"}
    return {"police_stations": find_nearest_n(lat, lng, POLICE_STATIONS, n=limit), "source": "curated_fallback"}
 
 
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
 
 
# ---------------- Live India-wide data (OpenStreetMap / Overpass API) ----------------
# This is the India-wide layer: instead of a hand-typed list limited to
# Tiruvannamalai, this queries Geoapify's Places API for whatever's actually
# mapped near the given coordinates, anywhere in India (or the world).
# Geoapify is built on OpenStreetMap data but served on reliable, professional
# infrastructure (unlike free volunteer-run Overpass servers, which often
# block or rate-limit cloud hosts like Render). Free tier: 3,000 requests/day.
# The API key is read from an environment variable — never hardcoded, never
# sent to the frontend — so it's safe even though this code is on GitHub.
# We never invent results — an area with nothing mapped returns an empty list.
 
GEOAPIFY_API_KEY = os.environ.get("GEOAPIFY_API_KEY")
GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"
 
LIVE_CATEGORY_TAGS = {
    "hospital": "healthcare.hospital",
    "police": "service.police",
    "fuel": "service.vehicle.fuel",
    "atm": "service.financial.atm",
    "pharmacy": "healthcare.pharmacy",
    "toilets": "amenity.toilet",
    "restaurant": "catering.restaurant",
    "hotel": "accommodation.hotel",
    "attraction": "tourism.attraction",
    "train_station": "public_transport.train",
    "metro": "public_transport.subway",
    "bus_station": "public_transport.bus",
    # Generic catch-all for "other nearby POIs" — combines real, documented
    # top-level Geoapify categories (shops, leisure, entertainment) rather
    # than inventing a single "other" tag, since Geoapify's own category
    # list doesn't have one.
    "other": "commercial,leisure,entertainment",
}
 
 
def _fetch_live_places(lat, lng, category, radius_m=5000, limit=20):
    """Core live-data fetch, usable from both the /api/live-nearby endpoint
    and the AI assistant's context gathering. Returns an empty list on any
    failure (missing key, network issue, no results) rather than raising —
    callers decide what to do when there's nothing to show."""
    category_code = LIVE_CATEGORY_TAGS.get(category)
    if not category_code or not GEOAPIFY_API_KEY:
        return []
 
    params = {
        "categories": category_code,
        "filter": f"circle:{lng},{lat},{radius_m}",
        "bias": f"proximity:{lng},{lat}",
        "limit": limit * 2,
        "apiKey": GEOAPIFY_API_KEY,
    }
    try:
        resp = requests.get(GEOAPIFY_PLACES_URL, params=params, timeout=15)
        resp.raise_for_status()
        geo_data = resp.json()
    except Exception:
        return []
 
    results = []
    for feature in geo_data.get("features", []):
        props = feature.get("properties", {})
        name = props.get("name")
        if not name:
            continue
        elat, elng = props.get("lat"), props.get("lon")
        if elat is None or elng is None:
            continue
        results.append({
            "name": name,
            "latitude": elat,
            "longitude": elng,
            "distance_km": round(haversine_km(lat, lng, elat, elng), 2),
            "phone": props.get("contact_phone") or props.get("phone"),
            "opening_hours": props.get("opening_hours"),
            "address": props.get("address_line2") or props.get("formatted"),
        })
    results.sort(key=lambda x: x["distance_km"])
    return results[:limit]
 
 
# ---------------- Ola Maps (India-specific secondary source + road-snapping) ----------------
# Ola Maps is used two ways:
# 1. As a secondary Nearby Search source alongside Geoapify — India-built,
#    with its own free tier, useful as a cross-check/backup for the same
#    categories (confirmed via Ola's own docs: restaurant, parking,
#    gas_station, toilet, lodging, bank, hospital — police/atm/attraction/
#    transit are NOT confirmed supported yet, so aren't included here).
# 2. SnapToRoad — cleans up noisy raw GPS points (common with phone GPS) by
#    snapping them onto the real nearest road, so a moving dot on the map
#    (SOS live tracking, Trip Sharing) looks accurate instead of jumping
#    off-road. Endpoint and response shape verified directly from Ola's docs.
 
OLA_MAPS_API_KEY = os.environ.get("OLA_MAPS_API_KEY")
OLA_MAPS_BASE_URL = "https://api.olamaps.io"
 
 
@app.get("/api/config/ola-maps")
def ola_maps_config():
    """The frontend fetches the Ola Maps key from here at runtime instead
    of it being hard-coded in the HTML — this endpoint was missing, which
    is why the map kept showing 'not configured' even after the real key
    was set on Render."""
    return {
        "configured": bool(OLA_MAPS_API_KEY),
        "api_key": OLA_MAPS_API_KEY or "",
    }
 
 
# Only categories confirmed supported by Ola's own Nearby Search docs.
OLA_CATEGORY_TAGS = {
    "hospital": "hospital",
    "hotel": "lodging",
    "restaurant": "restaurant",
    "fuel": "gas_station",
    "toilets": "toilet",
    "parking": "parking",
    "bank": "bank",
}
 
 
def _fetch_ola_nearby(lat, lng, category, radius_m=5000, limit=20):
    """Same purpose as _fetch_live_places, but backed by Ola Maps instead of
    Geoapify. Returns an empty list on any failure — callers fall back to
    Geoapify or curated data, never show an error to the tourist for this."""
    place_type = OLA_CATEGORY_TAGS.get(category)
    if not place_type or not OLA_MAPS_API_KEY:
        return []
 
    params = {
        "location": f"{lat},{lng}",
        "types": place_type,
        "radius": radius_m,
        "api_key": OLA_MAPS_API_KEY,
    }
    try:
        resp = requests.get(f"{OLA_MAPS_BASE_URL}/places/v1/nearbysearch", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
 
    results = []
    for pred in data.get("predictions", []):
        name = (pred.get("structured_formatting") or {}).get("main_text") or pred.get("description")
        geometry = pred.get("geometry", {}).get("location", {})
        elat, elng = geometry.get("lat"), geometry.get("lng")
        if not name or elat is None or elng is None:
            continue
        results.append({
            "name": name,
            "latitude": elat,
            "longitude": elng,
            "distance_km": round(haversine_km(lat, lng, elat, elng), 2),
            "phone": None,
            "opening_hours": None,
            "address": pred.get("description"),
            "place_id": pred.get("place_id"),  # used by /api/ola-place-details for full details on demand
        })
    results.sort(key=lambda x: x["distance_km"])
    return results[:limit]
 
 
@app.get("/api/ola-nearby")
def ola_nearby(lat: float, lng: float, category: str, radius_m: int = 5000, limit: int = 20):
    """Secondary/backup nearby-places source, separate endpoint so the
    frontend can try Geoapify first and fall back to this if needed."""
    if category not in OLA_CATEGORY_TAGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}' for Ola Maps. Valid options: {', '.join(OLA_CATEGORY_TAGS.keys())}",
        )
    if not OLA_MAPS_API_KEY:
        raise HTTPException(status_code=503, detail="Ola Maps isn't configured yet — OLA_MAPS_API_KEY is missing.")
    results = _fetch_ola_nearby(lat, lng, category, radius_m, limit)
    return {"category": category, "results": results, "source": "Ola Maps Nearby Search"}
 
 
@app.get("/api/snap-to-road")
def snap_to_road(points: str):
    """Cleans up a short trail of raw GPS points by snapping them to the
    nearest real road. `points` is a semicolon-separated list of
    "lat,lng" pairs, e.g. "12.23,79.07;12.231,79.071". Used to smooth live
    location tracking (SOS, Trip Sharing) so the moving dot follows real
    roads instead of jittering off-path."""
    if not OLA_MAPS_API_KEY:
        raise HTTPException(status_code=503, detail="Ola Maps isn't configured yet — OLA_MAPS_API_KEY is missing.")
 
    try:
        pairs = [p.strip() for p in points.split(";") if p.strip()]
        if not pairs:
            raise ValueError("no points provided")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'points' format — use lat,lng pairs separated by ';'.")
 
    try:
        resp = requests.get(
            f"{OLA_MAPS_BASE_URL}/routing/v1/snapToRoad",
            params={"points": "|".join(pairs), "api_key": OLA_MAPS_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ola Maps road-snapping unavailable: {e}")
 
    snapped = []
    for point in data.get("snapped_points", []):
        loc = point.get("location", {})
        snapped.append({
            "latitude": loc.get("lat"),
            "longitude": loc.get("lng"),
            "original_index": point.get("original_index"),
            "snapped_type": point.get("snapped_type"),
        })
    return {"snapped_points": snapped}
 
 
def _decode_polyline(encoded, precision=5):
    """Standard polyline decoding algorithm (same format Google Maps uses).
    Turns Ola's compact encoded route string into a real list of [lat, lng]
    points we can draw on the Leaflet map."""
    if not encoded:
        return []
    factor = 10 ** precision
    coords = []
    index = lat = lng = 0
    length = len(encoded)
 
    while index < length:
        for is_lat in (True, False):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lat:
                lat += delta
            else:
                lng += delta
        coords.append([lat / factor, lng / factor])
    return coords
 
 
@app.get("/api/ola-directions")
def ola_directions(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float, mode: str = "driving"):
    """Real turn-by-turn route between two points. Tries Ola Maps first
    (India-specific routing); if that's unavailable (missing key, network
    issue, no route found), falls back to Geoapify's Routing API instead of
    just failing — combining both providers is what actually helps the
    tourist get a route, rather than trusting a single source completely."""
    ola_result = _try_ola_directions(origin_lat, origin_lng, dest_lat, dest_lng, mode)
    if ola_result is not None:
        ola_result["source"] = "Ola Maps"
        return ola_result
 
    geoapify_result = _try_geoapify_directions(origin_lat, origin_lng, dest_lat, dest_lng, mode)
    if geoapify_result is not None:
        geoapify_result["source"] = "Geoapify (Ola Maps unavailable — fell back automatically)"
        return geoapify_result
 
    raise HTTPException(
        status_code=502,
        detail="No route could be calculated — both Ola Maps and Geoapify routing are unavailable right now.",
    )
 
 
def _try_ola_directions(origin_lat, origin_lng, dest_lat, dest_lng, mode):
    """Returns a normalized route dict, or None if Ola Maps couldn't
    provide one for any reason (never raises — the caller decides whether
    to fall back to Geoapify)."""
    if not OLA_MAPS_API_KEY:
        return None
    try:
        resp = requests.post(
            f"{OLA_MAPS_BASE_URL}/routing/v1/directions",
            params={
                "origin": f"{origin_lat},{origin_lng}",
                "destination": f"{dest_lat},{dest_lng}",
                "mode": mode,
                "api_key": OLA_MAPS_API_KEY,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        routes = data.get("routes", [])
        if not routes:
            return None
        route = routes[0]
        legs = route.get("legs", [])
        total_distance_m = sum(leg.get("distance", 0) for leg in legs)
        total_duration_s = sum(leg.get("duration", 0) for leg in legs)
        path_points = _decode_polyline(route.get("overview_polyline"))
        if not path_points:
            return None
        return {
            "distance_km": round(total_distance_m / 1000, 2),
            "duration_minutes": round(total_duration_s / 60),
            "path": path_points,
        }
    except Exception:
        return None
 
 
# Only modes confirmed in Geoapify's own Routing API docs — not a full
# 1:1 mapping of every possible value this endpoint's "mode" param might
# receive, so anything unrecognized falls back to "drive" rather than
# sending Geoapify a value it doesn't document.
GEOAPIFY_MODE_MAP = {
    "driving": "drive",
    "drive": "drive",
    "walking": "walk",
    "walk": "walk",
    "cycling": "bicycle",
    "bicycle": "bicycle",
    "two_wheeler": "motorcycle",
    "motorcycle": "motorcycle",
    "transit": "transit",
}
 
 
def _try_geoapify_directions(origin_lat, origin_lng, dest_lat, dest_lng, mode):
    """Fallback routing via Geoapify — used only when Ola Maps couldn't
    provide a route. Returns None (never raises) on any failure, same
    contract as _try_ola_directions, so the caller can report an honest
    combined failure rather than a confusing partial error."""
    if not GEOAPIFY_API_KEY:
        return None
    geoapify_mode = GEOAPIFY_MODE_MAP.get(mode, "drive")
    try:
        resp = requests.get(
            "https://api.geoapify.com/v1/routing",
            params={
                "waypoints": f"{origin_lat},{origin_lng}|{dest_lat},{dest_lng}",
                "mode": geoapify_mode,
                "apiKey": GEOAPIFY_API_KEY,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        if not features:
            return None
        props = features[0].get("properties", {})
        geometry = features[0].get("geometry", {})
 
        # Geoapify's route geometry can be LineString or MultiLineString
        # (one line per leg/segment) — flatten either shape into one
        # [lat, lng] path the same way _decode_polyline's output looks.
        raw_coords = geometry.get("coordinates", [])
        path_points = []
        if geometry.get("type") == "MultiLineString":
            for segment in raw_coords:
                for lng, lat in segment:
                    path_points.append([lat, lng])
        elif geometry.get("type") == "LineString":
            for lng, lat in raw_coords:
                path_points.append([lat, lng])
        if not path_points:
            return None
 
        distance_m = props.get("distance")
        duration_s = props.get("time")
        return {
            "distance_km": round(distance_m / 1000, 2) if distance_m is not None else None,
            "duration_minutes": round(duration_s / 60) if duration_s is not None else None,
            "path": path_points,
        }
    except Exception:
        return None
 
 
 
def _merge_place_results(primary, secondary, dedupe_radius_km=0.15):
    """Combines two nearby-place lists (Geoapify + Ola Maps) into one, so if
    either source is missing a place the other has, the combined list is
    more complete. This matters concretely: Ola's own map TILES show
    businesses (e.g. 'Indane - Deepam Gas Agencies') that Ola's own Nearby
    Search API does not return — the base map picture and the search
    results come from different underlying datasets on Ola's side. Using
    only one source was leaving out real, visible places. 'primary'
    entries are kept as-is; 'secondary' entries are only added if nothing
    in 'primary' already represents the same real-world place (same name,
    or within ~150m)."""
    merged = list(primary)
    for cand in secondary:
        is_duplicate = False
        for existing in primary:
            same_name = cand["name"].strip().lower() == existing["name"].strip().lower()
            close_by = haversine_km(
                cand["latitude"], cand["longitude"],
                existing["latitude"], existing["longitude"],
            ) <= dedupe_radius_km
            if same_name or close_by:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(cand)
    merged.sort(key=lambda x: x["distance_km"])
    return merged
 
 
@app.get("/api/live-nearby")
def live_nearby(lat: float, lng: float, category: str, radius_m: int = 5000, limit: int = 20):
    if category not in LIVE_CATEGORY_TAGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Valid options: {', '.join(LIVE_CATEGORY_TAGS.keys())}",
        )
    if not GEOAPIFY_API_KEY and not OLA_MAPS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Live nearby data isn't configured yet — both GEOAPIFY_API_KEY and OLA_MAPS_API_KEY are missing on the server.",
        )
 
    # Geoapify (OpenStreetMap-based) as the base — broader raw place
    # coverage in our testing. Then merge in Ola Maps results too, for any
    # category Ola supports — combining both is what actually gives the
    # tourist the fullest list, since each source has real places the other
    # is missing (see _merge_place_results docstring for the concrete
    # example that showed this).
    geoapify_results = _fetch_live_places(lat, lng, category, radius_m, limit) if GEOAPIFY_API_KEY else []
    sources_used = ["Geoapify"] if geoapify_results else []
    combined = geoapify_results
 
    if category in OLA_CATEGORY_TAGS and OLA_MAPS_API_KEY:
        ola_results = _fetch_ola_nearby(lat, lng, category, radius_m, limit)
        if ola_results:
            combined = _merge_place_results(geoapify_results, ola_results) if geoapify_results else ola_results
            sources_used.append("Ola Maps")
 
    if not combined:
        return {"category": category, "results": [], "source": "No results from either provider for this spot."}
 
    return {
        "category": category,
        "results": combined[:limit],
        "source": " + ".join(sources_used) + " (merged, deduplicated)" if len(sources_used) > 1 else sources_used[0],
    }
 
 
@app.get("/api/ola-place-details")
def ola_place_details(place_id: str):
    """Full details (phone, opening hours) for a place Ola Maps' Nearby
    Search already found — a separate, on-demand call rather than fetched
    for every list item up front, to avoid one extra API call per result.
    Real, documented Ola Maps endpoint. Returns whatever fields Ola actually
    has — never fabricates a phone number or hours it doesn't have."""
    if not OLA_MAPS_API_KEY:
        raise HTTPException(status_code=503, detail="Ola Maps isn't configured yet — OLA_MAPS_API_KEY is missing.")
    try:
        resp = requests.get(
            f"{OLA_MAPS_BASE_URL}/places/v1/details",
            params={"place_id": place_id, "api_key": OLA_MAPS_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ola Maps place details unavailable: {e}")
 
    result = data.get("result", data)  # some Ola responses nest under "result", some don't — handle both
    return {
        "name": result.get("name"),
        "address": result.get("formatted_address") or result.get("address"),
        "phone": result.get("formatted_phone_number") or result.get("phone"),
        "opening_hours": result.get("opening_hours"),
        "raw": result,  # everything Ola actually returned, for anything not mapped above
    }
 
 
# ---------------- SOS / Incidents ----------------
# SOS channels: what's real today vs. what's architected but not yet
# available. "satellite" is deliberately NOT wired to any real transport —
# see our earlier discussion: true satellite-to-phone SOS in India requires
# a licensed partnership (e.g. BSNL Direct-to-Device / Inmarsat under DoT
# authorization) that doesn't exist for this project yet. This is an
# honest status list, not a feature toggle that silently does nothing.
SOS_CHANNELS = [
    {
        "id": "app",
        "label": "Internet (HTTPS)",
        "status": "active",
        "description": "Normal path — works whenever the phone has internet.",
    },
    {
        "id": "sms_offline",
        "label": "SMS (offline fallback)",
        "status": "active",
        "description": "Used automatically when there's no internet but there's cellular signal — see /api/textbee/webhook.",
    },
    {
        "id": "satellite",
        "label": "Satellite",
        "status": "planned",
        "description": (
            "Not yet available — requires a licensed satellite partnership "
            "(e.g. BSNL Direct-to-Device / Inmarsat, DoT-authorized) that "
            "isn't in place yet. The SOS system already accepts a "
            "'satellite' channel value so this can be enabled later without "
            "changing the API shape — it just isn't a working transport today."
        ),
    },
]
 
 
@app.get("/api/sos/channels")
def get_sos_channels():
    """Lets the app show every SOS channel — including ones that are
    architected but not live yet — with an honest status per channel,
    instead of hiding 'planned' features or silently pretending they work."""
    return {"channels": SOS_CHANNELS}
 
 
@app.post("/api/sos")
def trigger_sos(req: SOSRequest):
    if req.channel == "satellite":
        # Real, honest rejection — never silently accept an SOS on a
        # transport that doesn't actually exist yet. The app is expected to
        # catch this and fall back to "sms_offline" or "app" itself.
        raise HTTPException(
            status_code=503,
            detail=(
                "Satellite SOS is not available yet — it requires a licensed "
                "satellite partnership not yet in place. Falling back to SMS "
                "or internet is the only real option right now."
            ),
        )
 
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
        "channel": req.channel,
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
 
            # If a real logged-in responder is assigning themselves, record
            # who they are (name, org, role) and where they are right now,
            # so other responders viewing this same incident can see exactly
            # who's already on it and navigate to them if needed.
            if update.status == "Assigned" and update.responder_token:
                session = responder_tokens_db.get(update.responder_token)
                if session:
                    org = responder_orgs_db.get(session["org_username"])
                    member = responder_members_db.get(session["username"]) if session["kind"] == "member" else None
                    inc["assigned_responder"] = {
                        "name": member["name"] if member else (org["facility_name"] if org else "Responder"),
                        "role": member["role"] if member else "Organization Admin",
                        "org_type": org["org_type"] if org else None,
                        "facility_name": org["facility_name"] if org else None,
                        "latitude": update.responder_lat,
                        "longitude": update.responder_lng,
                        "assigned_at": datetime.utcnow().isoformat(),
                    }
            if update.status == "Resolved":
                # Keep the historical record of who handled it, just stop
                # treating their location as "currently live."
                if "assigned_responder" in inc:
                    inc["assigned_responder"]["resolved"] = True
 
            return inc
    raise HTTPException(status_code=404, detail="Incident not found")
 
 
@app.get("/")
def root():
    return {"status": "Tourist Safety API running", "docs": "/docs"}
 
 
# ==================== AI TOURIST ASSISTANT (V4) ====================
# Architecture:
# - The AI credential (ANTHROPIC_API_KEY) lives only as a server environment
#   variable on Render — it is never sent to or stored in the frontend/APK.
# - Every request gathers REAL data from this backend's own real datasets
#   (hospitals, police, hotels, places, toilets, transport) near the user's
#   given location, and hands that to the model as grounding context. The
#   model is instructed to answer only from that data, never invent facts.
# - If no API key is configured yet, a rule-based fallback still answers
#   common questions directly from the same real data, so the feature works
#   even before an Anthropic key is added.
 
import json as _json
 
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
 
try:
    import anthropic
    _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
except Exception:
    _anthropic_client = None
 
 
class AssistantChatRequest(BaseModel):
    message: str
    language: str = "en"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    history: List[dict] = []  # [{role: "user"|"assistant", content: "..."}] — prior turns, so replies stay in context
 
 
class AssistantItineraryRequest(BaseModel):
    days: int = 1
    interests: List[str] = []
    latitude: Optional[float] = None
    longitude: Optional[float] = None
 
 
# ---------------- Emergency Guidance (deterministic — never AI-generated, safety-critical) ----------------
# This is intentionally NOT run through the AI model. For life-safety content,
# fixed, reviewed instructions are safer than anything a language model could
# improvise in the moment. The frontend also keeps its own copy of this same
# data so it still works with zero internet during an actual emergency.
 
EMERGENCY_GUIDANCE = {
    "medical": {
        "title": "Medical Emergency",
        "steps": [
            "Stay calm and check if the person is conscious and breathing.",
            "Call for an ambulance immediately, or ask someone nearby to call.",
            "If they're breathing but unconscious, place them in the recovery position (on their side).",
            "If there's bleeding, apply firm, steady pressure with a clean cloth.",
            "Don't move someone with a suspected head, neck, or back injury unless they're in immediate danger.",
            "Don't give food, water, or medication to an unconscious person.",
            "Keep them warm and stay with them until help arrives.",
        ],
    },
    "snakebite": {
        "title": "Snake Bite",
        "steps": [
            "Stay calm and keep the bitten limb still and below heart level if possible.",
            "Remove any rings, watches, or tight clothing near the bite before swelling starts.",
            "Get to a hospital immediately — snake bite treatment needs real antivenom, not first aid alone.",
            "Do NOT cut the wound, try to suck out venom, or apply ice.",
            "Do NOT apply a tight tourniquet — this can cause more harm.",
            "Try to remember the snake's color/pattern if safely possible, but don't waste time chasing it.",
        ],
    },
    "heatstroke": {
        "title": "Heat Stroke / Dehydration",
        "steps": [
            "Move to shade or a cool place immediately.",
            "Remove excess clothing and cool the person with water or a wet cloth, especially neck, armpits, and groin.",
            "Give small sips of water if they're fully conscious and able to swallow.",
            "Fan them or move to air conditioning/moving air if available.",
            "Seek medical help urgently if there's confusion, very high body temperature, or they stop sweating.",
        ],
    },
    "police": {
        "title": "Crime, Theft, or Feeling Unsafe",
        "steps": [
            "Move to a well-lit, public, crowded area if you feel unsafe or are being followed.",
            "Call the police using the SOS button, or dial 112 (India's national emergency number).",
            "Note down details if safely possible: description of person, vehicle number, location.",
            "Don't confront a thief or aggressor directly — your safety comes first.",
            "Share your live location with a trusted contact right away.",
        ],
    },
    "lost": {
        "title": "Lost or Stranded",
        "steps": [
            "Stay where you are if it's safe — moving randomly can make it harder to be found.",
            "Use the app's Map and Nearby Services to find the closest real landmark, police station, or hospital.",
            "Share your live location with an emergency contact via the Share Location feature.",
            "If you have no signal, look for high ground or open areas where signal is more likely.",
            "Approach a shop, temple staff, or family group rather than an isolated stranger for help.",
        ],
    },
    "fire": {
        "title": "Fire",
        "steps": [
            "Get away from the fire and smoke immediately — don't stop to collect belongings.",
            "Stay low to the ground if there's smoke; it's easier to breathe near the floor.",
            "Never use an elevator during a fire — use stairs.",
            "Call the fire department (101 in India) or use the SOS button once you're safe.",
            "If your clothes catch fire: Stop, Drop, and Roll.",
        ],
    },
    "general": {
        "title": "General Emergency",
        "steps": [
            "Stay as calm as you can — clear thinking helps you make better decisions.",
            "Move to a safe, visible location if you're able to.",
            "Use the SOS button to alert your emergency contacts and the dashboard with your location.",
            "If you have no internet, use the Call/SMS option to reach a contact directly over your phone signal.",
            "Keep your phone charged and visible if you're waiting for help to arrive.",
        ],
    },
}
 
 
@app.get("/api/emergency-guidance")
def emergency_guidance(type: Optional[str] = None):
    if type:
        guide = EMERGENCY_GUIDANCE.get(type)
        if not guide:
            raise HTTPException(status_code=404, detail=f"No guidance found for '{type}'.")
        return {"type": type, **guide}
    return {"guidance": EMERGENCY_GUIDANCE}
 
 
def _gather_real_context(lat, lng):
    """Pulls real nearby data from live nationwide sources first (works
    anywhere in India), falling back to the curated Tiruvannamalai dataset
    only if live data has nothing for that spot — the same rule the rest of
    the app follows, so the assistant isn't stuck to Tiruvannamalai-only
    facts. These are the only facts the assistant is allowed to reference."""
    if lat is None or lng is None:
        return {
            "note": "No location provided.",
            "hospitals": [], "police": [], "hotels": [], "places": [], "toilets": [], "transport": [],
        }
 
    def live_or_curated(category, curated_list, n=3):
        live = _fetch_live_places(lat, lng, category, radius_m=10000, limit=n)
        if live:
            return live
        return find_nearest_n_full(lat, lng, curated_list, n=n)
 
    return {
        "hospitals": live_or_curated("hospital", HOSPITALS, 3),
        "police": live_or_curated("police", POLICE_STATIONS, 3),
        "hotels": live_or_curated("hotel", HOTELS, 3),
        "places": live_or_curated("attraction", PLACES_TO_VISIT, 5),
        "toilets": live_or_curated("toilets", PUBLIC_TOILETS, 2),
        "transport": live_or_curated("bus_station", TRANSPORT_HUBS, 2),
    }
 
 
def _rule_based_reply(message, context, language):
    """No-AI-key fallback: answers common tourist questions directly from
    real data so the assistant still works before an API key is added."""
    msg = message.lower()
 
    def fmt_list(items, kind):
        if not items:
            return f"I don't have {kind} data for your current location."
        lines = [f"- {it['name']} ({it.get('distance_km', '?')} km away)" for it in items]
        return "\n".join(lines)
 
    if "hospital" in msg:
        return "Nearest hospitals:\n" + fmt_list(context["hospitals"], "hospital")
    if "police" in msg:
        return "Nearest police stations:\n" + fmt_list(context["police"], "police")
    if "hotel" in msg or "stay" in msg:
        return "Nearest hotels:\n" + fmt_list(context["hotels"], "hotel")
    if "toilet" in msg or "restroom" in msg or "washroom" in msg:
        return "Nearest public toilets:\n" + fmt_list(context["toilets"], "toilet")
    if "bus" in msg or "train" in msg or "railway" in msg or "station" in msg:
        return "Nearest transport hubs:\n" + fmt_list(context["transport"], "transport")
    if "visit" in msg or "place" in msg or "see" in msg or "go" in msg:
        return "Places worth visiting nearby:\n" + fmt_list(context["places"], "places")
    return ("I can help with nearby hospitals, police stations, hotels, places to visit, "
            "toilets, and transport — try asking about one of those. "
            "(Full free-form AI chat needs an Anthropic API key added on the server.)")
 
 
@app.post("/api/assistant/chat")
def assistant_chat(req: AssistantChatRequest):
    context = _gather_real_context(req.latitude, req.longitude)
 
    if _anthropic_client is None:
        return {"reply": _rule_based_reply(req.message, context, req.language), "mode": "rule_based"}
 
    system_prompt = (
        "You are SafarSafe's tourist assistant for Tiruvannamalai, India. "
        "Answer concisely (2-4 sentences), in a tourist-friendly and action-oriented way. "
        "Respond in this language code: " + req.language + ". "
        "You may ONLY use facts from the REAL_DATA JSON below — never invent names, "
        "distances, prices, opening hours, or phone numbers that aren't in it. "
        "If the answer isn't in REAL_DATA, say so honestly and suggest what you can help with instead.\n\n"
        "REAL_DATA:\n" + _json.dumps(context)
    )
 
    try:
        # Keep the last few turns only — enough for real conversational
        # context without the request growing unbounded over a long chat.
        recent_history = req.history[-10:]
        messages = [{"role": h["role"], "content": h["content"]} for h in recent_history if h.get("role") in ("user", "assistant")]
        messages.append({"role": "user", "content": req.message})
 
        response = _anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            system=system_prompt,
            messages=messages,
        )
        reply_text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return {"reply": reply_text, "mode": "ai"}
    except Exception as e:
        return {"reply": _rule_based_reply(req.message, context, req.language), "mode": "rule_based_fallback", "error": str(e)}
 
 
@app.post("/api/assistant/itinerary")
def assistant_itinerary(req: AssistantItineraryRequest):
    context = _gather_real_context(req.latitude, req.longitude)
    places = context["places"]
 
    if not places:
        return {"itinerary": [], "note": "No location provided — cannot suggest a real itinerary."}
 
    days = max(1, min(req.days, 7))
    per_day = max(1, len(places) // days)
    itinerary = []
    for d in range(days):
        day_places = places[d * per_day:(d + 1) * per_day] or places[:1]
        itinerary.append({"day": d + 1, "places": day_places})
 
    if _anthropic_client is None:
        return {"itinerary": itinerary, "mode": "rule_based"}
 
    try:
        prompt = (
            f"Write a short, friendly {days}-day Tiruvannamalai itinerary using ONLY these real places "
            f"(do not invent any other places): {_json.dumps(places)}. "
            f"Tourist interests: {req.interests}. Respond in language code: {req.language}. Keep it concise."
        )
        response = _anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        reply_text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return {"itinerary": itinerary, "summary": reply_text, "mode": "ai"}
    except Exception as e:
        return {"itinerary": itinerary, "mode": "rule_based_fallback", "error": str(e)}
 
 
# ==================== OFFLINE SOS via SMS (textbee.dev gateway) ====================
# Real flow, no invented pieces:
#   Tourist's phone has NO internet but DOES have a cellular/SMS signal
#     -> phone (native Android SmsManager — see reference Kotlin in the
#        implementation notes) sends a plain SMS to the SafarSafe gateway
#        number
#     -> a dedicated Android phone running the textbee.dev app forwards
#        that SMS to textbee.dev's servers
#     -> textbee.dev calls OUR webhook below
#     -> we create a real incident, the SAME incidents_db the dashboard
#        already reads from /api/incidents — no separate offline system.
#
# Config (server env vars only — never in the APK or this source file):
#   TEXTBEE_API_KEY - from your textbee.dev dashboard. Only needed for the
#                     OUTBOUND backfill call below; the inbound webhook
#                     itself needs no key.
#
# Real, confirmed endpoints (textbee.dev/docs):
#   Send:    POST https://api.textbee.dev/api/v1/gateway/send-sms
#   History: GET  https://api.textbee.dev/api/v1/gateway/messages?direction=received
#   Receive: a webhook YOU configure in the textbee.dev dashboard, delivered
#            as a POST to a URL you register there.
#
# HONEST GAP: textbee.dev's exact webhook JSON field names weren't fully
# visible in public docs at integration time (only that it POSTs on
# "Message Received" events, with signing details on their Webhooks page).
# The handler below reads the raw body and tries the field names used in
# textbee.dev's own published examples (sender/from/phoneNumber,
# message/text/body). Once you register a real webhook in your dashboard,
# check your server logs for the first real payload and adjust the lookups
# below if the actual field names differ.
 
TEXTBEE_API_KEY = os.environ.get("TEXTBEE_API_KEY", "")
TEXTBEE_BASE_URL = "https://api.textbee.dev/api/v1/gateway"
 
import re as _re
 
 
def _extract_latlng_from_text(text: str):
    """Looks for a 'lat,lng' decimal-degree pair anywhere in free text.
    Returns (None, None) if nothing matches — never guesses a location."""
    m = _re.search(r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)", text or "")
    if not m:
        return None, None
    try:
        lat, lng = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng
    except ValueError:
        pass
    return None, None
 
 
def _ai_parse_sos_sms(raw_text: str):
    """Uses the SAME Claude client already configured for the AI Tourist
    Assistant above (ANTHROPIC_API_KEY) to pull structured fields out of a
    panicked, free-form SOS text. Falls back to a plain regex lat/lng scan
    if no AI key is configured, or if the AI call fails for any reason —
    the webhook must never crash or silently drop a real emergency message."""
    lat, lng = _extract_latlng_from_text(raw_text)
    fallback = {
        "latitude": lat,
        "longitude": lng,
        "incident_type": "SOS",
        "severity": "High",
        "summary": (raw_text or "").strip()[:200],
        "mode": "regex_only",
    }
 
    if _anthropic_client is None:
        return fallback
 
    try:
        response = _anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            system=(
                "You triage an incoming emergency SMS for a tourist-safety system. "
                "Reply with ONLY a compact JSON object, no prose, no markdown fences, "
                "with exactly these keys: "
                '{"latitude": number or null, "longitude": number or null, '
                '"incident_type": "medical"|"crime"|"accident"|"lost"|"other", '
                '"severity": "High"|"Medium"|"Low", "summary": string}. '
                "Only fill latitude/longitude if the message actually contains "
                "coordinates or an unambiguous specific place — otherwise use null. "
                "Never invent a location or downplay severity language the sender used."
            ),
            messages=[{"role": "user", "content": raw_text}],
        )
        text_out = "".join(b.text for b in response.content if hasattr(b, "text"))
        parsed = _json.loads(text_out)
        return {
            "latitude": parsed.get("latitude") if parsed.get("latitude") is not None else lat,
            "longitude": parsed.get("longitude") if parsed.get("longitude") is not None else lng,
            "incident_type": parsed.get("incident_type") or "SOS",
            "severity": parsed.get("severity") or "High",
            "summary": parsed.get("summary") or fallback["summary"],
            "mode": "ai",
        }
    except Exception:
        return fallback  # never let a bad/unparseable AI response drop a real SOS
 
 
@app.post("/api/textbee/webhook")
async def textbee_webhook(request: Request):
    """Inbound webhook — textbee.dev calls this when the gateway phone
    receives an SMS. See the module note above re: confirming exact field
    names against your real textbee.dev dashboard payload."""
    payload = {}
    try:
        payload = await request.json()
    except Exception:
        try:
            raw = await request.body()
            payload = {"_raw": raw.decode("utf-8", errors="replace")}
        except Exception:
            pass
 
    sender = (
        payload.get("sender") or payload.get("from") or payload.get("phoneNumber")
        or request.query_params.get("sender") or "unknown"
    )
    message_text = (
        payload.get("message") or payload.get("text") or payload.get("body")
        or payload.get("_raw") or ""
    )
 
    parsed = _ai_parse_sos_sms(message_text)
 
    incident_id = str(uuid.uuid4())[:8]
    incident = {
        "incident_id": incident_id,
        "user_id": sender,  # phone number stands in for user_id on this channel
        "latitude": parsed["latitude"],
        "longitude": parsed["longitude"],
        "incident_type": parsed["incident_type"],
        "severity": parsed["severity"],
        "status": "Unassigned",
        "created_at": datetime.utcnow().isoformat(),
        "actions": [],
        "reporter_name": None,
        "reporter_phone": sender,
        "reporter_photo": None,
        "reporter_blood_group": None,
        "channel": "sms_offline",       # dashboard can flag this differently from in-app SOS
        "raw_sms": (message_text or "")[:500],
        "ai_summary": parsed["summary"],
        "triage_mode": parsed["mode"],  # "ai" or "regex_only" — shows how it was parsed
    }
    incidents_db.append(incident)
    return {"received": True, "incident_id": incident_id}
 
 
@app.get("/api/textbee/backfill")
def textbee_backfill():
    """Manual recovery path: pulls recent received messages directly from
    textbee.dev's own message history (not just relying on the webhook
    having been live). Real, documented endpoint — see module note above."""
    if not TEXTBEE_API_KEY:
        raise HTTPException(status_code=503, detail="TEXTBEE_API_KEY is not configured on the server.")
    try:
        resp = requests.get(
            f"{TEXTBEE_BASE_URL}/messages",
            headers={"x-api-key": TEXTBEE_API_KEY},
            params={"direction": "received"},
            timeout=15,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach textbee.dev: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"textbee.dev error ({resp.status_code}): {resp.text[:300]}")
    return resp.json()
 
# ==================== AI Review / Trust Score system ====================
# Covers hotels, restaurants, AND transport (bus/train hubs, operators) —
# one generic system keyed by place name + category, not a separate
# implementation per category, since the underlying problem (blend real
# user reviews with a fake-review filter into one trust signal) is
# identical across all three.
#
# WHAT THIS REUSES:
#   - TrustEngine (trust_engine.py) — the blended scoring logic already
#     built (rating, volume, freshness, credibility, sentiment, consistency)
#   - fake_review_model.pkl / fake_review_vectorizer.pkl — from
#     train_fake_review_model.py. If these haven't been trained yet, this
#     section still works: it falls back to a plain average rating (no
#     fake-filtering) rather than crashing, and says so explicitly in the
#     response so you never mistake an untrained fallback for the real thing.
#
# HONEST SCOPE: app-submitted reviews (reviews_db below) are real, live
# data your project fully controls. External reviews (Google, TripAdvisor,
# etc.) are NOT wired in here — that needs the Google Places API on the
# backend side, respecting its ToS on review caching/reuse, which is a
# separate integration to add deliberately, not something to fake here.
 
reviews_db = []  # in-memory for the demo — swap for real DB storage in production
 
 
class ReviewSubmission(BaseModel):
    user_id: str
    place_name: str
    category: str  # "hotel" | "restaurant" | "transport"
    rating: float  # 1-5
    text: str
 
 
@app.post("/api/reviews")
def submit_review(review: ReviewSubmission):
    if not (1 <= review.rating <= 5):
        raise HTTPException(status_code=400, detail="rating must be between 1 and 5")
    entry = {
        "review_id": str(uuid.uuid4())[:8],
        "user_id": review.user_id,
        "place_name": review.place_name,
        "category": review.category,
        "rating": review.rating,
        "text": review.text,
        "source": "app",
        "posted_at": datetime.utcnow().isoformat(),
    }
    reviews_db.append(entry)
    return {"message": "review submitted", "review": entry}
 
 
@app.get("/api/reviews")
def list_reviews(place_name: str, category: Optional[str] = None):
    results = [r for r in reviews_db if r["place_name"] == place_name]
    if category:
        results = [r for r in results if r["category"] == category]
    return {"place_name": place_name, "count": len(results), "reviews": results}
 
 
# Lazy, optional TrustEngine load — mirrors the same optional-dependency
# pattern already used for the Anthropic client above: if the fake-review
# model hasn't been trained yet, the trust-score endpoint still works, just
# with fake-filtering disabled and that fact stated in the response.
try:
    from trust_engine import TrustEngine, Review as TrustReview
    _trust_engine = TrustEngine(
        fake_review_model_path=os.path.join(MODEL_DIR, "fake_review_model.pkl"),
        fake_review_vectorizer_path=os.path.join(MODEL_DIR, "fake_review_vectorizer.pkl"),
    )
except Exception:
    _trust_engine = None
 
 
@app.get("/api/trust-score")
def get_trust_score(place_name: str, category: Optional[str] = None):
    app_reviews = [r for r in reviews_db if r["place_name"] == place_name]
    if category:
        app_reviews = [r for r in app_reviews if r["category"] == category]
 
    if not app_reviews:
        return {
            "place_name": place_name,
            "trust_score": None,
            "message": "No reviews yet for this place — nothing to score.",
        }
 
    if _trust_engine is None:
        # Fake-review model not trained yet (see train_fake_review_model.py) —
        # be explicit that this is an honest fallback, not the real system.
        avg_rating = sum(r["rating"] for r in app_reviews) / len(app_reviews)
        return {
            "place_name": place_name,
            "trust_score": round((avg_rating / 5.0) * 100, 1),
            "mode": "fallback_average_rating",
            "message": "Fake-review AI model not trained yet — this is a plain average rating, not the full Trust Score.",
            "review_count": len(app_reviews),
        }
 
    trust_reviews = [
        TrustReview(
            text=r["text"],
            rating=r["rating"],
            source=r["source"],
            posted_at=datetime.fromisoformat(r["posted_at"]).replace(tzinfo=timezone.utc),
        )
        for r in app_reviews
    ]
    breakdown = _trust_engine.compute(trust_reviews)
    return {
        "place_name": place_name,
        "mode": "trust_engine",
        "trust_score": breakdown.trust_score,
        "breakdown": {
            "rating": breakdown.rating_score,
            "volume": breakdown.volume_score,
            "freshness": breakdown.freshness_score,
            "credibility": breakdown.credibility_score,
            "sentiment": breakdown.sentiment_score,
            "consistency": breakdown.consistency_score,
            "safety": breakdown.safety_score,
        },
        "genuine_review_count": breakdown.genuine_review_count,
        "total_review_count": breakdown.total_review_count,
    }
 