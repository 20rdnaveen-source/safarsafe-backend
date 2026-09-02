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
}
 
 
@app.get("/api/live-nearby")
def live_nearby(lat: float, lng: float, category: str, radius_m: int = 5000, limit: int = 20):
    category_code = LIVE_CATEGORY_TAGS.get(category)
    if not category_code:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Valid options: {', '.join(LIVE_CATEGORY_TAGS.keys())}",
        )
    if not GEOAPIFY_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Live nearby data isn't configured yet — GEOAPIFY_API_KEY is missing on the server.",
        )
 
    params = {
        "categories": category_code,
        "filter": f"circle:{lng},{lat},{radius_m}",
        "bias": f"proximity:{lng},{lat}",
        "limit": limit * 2,  # fetch extra since we drop unnamed entries below
        "apiKey": GEOAPIFY_API_KEY,
    }
 
    try:
        resp = requests.get(GEOAPIFY_PLACES_URL, params=params, timeout=15)
        resp.raise_for_status()
        geo_data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Live map data service unavailable: {e}")
 
    results = []
    for feature in geo_data.get("features", []):
        props = feature.get("properties", {})
        name = props.get("name")
        if not name:
            continue  # skip unnamed/low-quality entries
 
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
    return {
        "category": category,
        "results": results[:limit],
        "source": "Geoapify Places API (OpenStreetMap data, live)",
    }
 
 
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
 
 
class AssistantItineraryRequest(BaseModel):
    days: int = 1
    interests: List[str] = []
    latitude: Optional[float] = None
    longitude: Optional[float] = None
 
 
def _gather_real_context(lat, lng):
    """Pulls real nearby data from this backend's own datasets — the only
    facts the assistant is allowed to reference for this location."""
    if lat is None or lng is None:
        return {
            "note": "No location provided.",
            "hospitals": [], "police": [], "hotels": [], "places": [], "toilets": [], "transport": [],
        }
    return {
        "hospitals": find_nearest_n(lat, lng, HOSPITALS, n=3),
        "police": find_nearest_n(lat, lng, POLICE_STATIONS, n=3),
        "hotels": find_nearest_n_full(lat, lng, HOTELS, n=3),
        "places": find_nearest_n_full(lat, lng, PLACES_TO_VISIT, n=5),
        "toilets": find_nearest_n_full(lat, lng, PUBLIC_TOILETS, n=2),
        "transport": find_nearest_n_full(lat, lng, TRANSPORT_HUBS, n=2),
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
        response = _anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": req.message}],
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
 