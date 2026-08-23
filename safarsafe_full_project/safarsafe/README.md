# SafarSafe — AI Tourist Safety & Emergency Response System
### SIH 2026 — Arunai Engineering College — Internal Hackathon Build

This is a working prototype of the system described in the project plan:
AI risk model, full backend API, and interactive demo screens for the
mobile app and emergency dashboard.

## What's included

```
tourist_safety/
├── data/
│   ├── generate_dataset.py       # generates the synthetic training data
│   └── tourist_risk_dataset.csv  # 4,000-row synthetic dataset (generated)
├── model/
│   ├── train_model.py            # trains the Random Forest model
│   ├── risk_classifier.pkl       # trained classifier (Low/Medium/High)
│   ├── risk_regressor.pkl        # trained regressor (0-100 score)
│   ├── encoders.pkl              # label encoders for categorical features
│   └── feature_cols.pkl
├── backend/
│   └── main.py                   # FastAPI backend — all endpoints from the plan
├── app/
│   ├── tourist_app_demo.html     # interactive tourist app UI (open in any browser)
│   └── dashboard_demo.html       # interactive emergency response dashboard
└── requirements.txt
```

## How to run it

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. (Already done for you, but to regenerate) Build the dataset & train the model**
```bash
python data/generate_dataset.py
python model/train_model.py
```
Current model accuracy: **86.1%** on held-out test data.
Strongest predictors: `time_hour`, `zone_risk`, `previous_incidents`.

**3. Run the backend API**
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Interactive API docs (Swagger) will be at: `http://localhost:8000/docs`
Test any endpoint directly from that page — no separate tool needed.

**4. Open the demo screens**
Just double-click (or drag into a browser):
- `app/tourist_app_demo.html` — the tourist-facing app: try the three
  scenario buttons (Safe / Crowded / High-risk trail at night) to see the
  AI risk score and warning respond instantly, then hit the SOS button.
- `app/dashboard_demo.html` — the emergency response side: click an
  incident to see it on the map, then click "Advance Status" to walk
  through Unassigned → Responder Assigned → Response Initiated → Resolved.

These two HTML files are self-contained (no server needed) — they mirror
the exact logic of the trained model and API, so they're safe to demo
live even without wifi at judging.

## Turning this into a real installable APK

The `app/tourist_app_demo.html` file is your exact reference for screens,
copy, colors, and interaction flow. To get a real `.apk`, your Mobile
teammate should:

1. Install Flutter + Android Studio on their machine
   (`flutter.dev/docs/get-started/install`)
2. Rebuild these same screens as Flutter widgets, using this demo as the
   visual/interaction spec
3. Replace the mock JS logic with real calls to your FastAPI backend
   endpoints (`POST /api/risk`, `POST /api/sos`, etc. — see `backend/main.py`)
4. Use the device's real GPS (`geolocator` package) instead of the demo's
   scenario buttons
5. Build the APK with `flutter build apk`

That build step needs the Android SDK toolchain, which isn't available in
this environment — everything else here (model, API, screens, logic) is
ready to hand off directly.

## Tiruvannamalai-specific version (real zones)

`data/generate_tiruvannamalai_dataset.py` builds a dataset using 8 real
Tiruvannamalai zones (Annamalaiyar Temple, Girivalam Path, Sri Ramana
Ashram, the hill climb route, the forest-restricted inner path, VOC Nagar,
the Karthigai Deepam hilltop site, and the Kaama Kaadu forest-adjacent
shrine cluster) with risk logic grounded in each zone's real, documented
characteristics — including the real Dec 2024 Cyclone Fengal landslide at
VOC Nagar. **Zone names/locations/risk drivers are real; exact incident
counts and scores are synthetic** (no public per-zone incident database
exists for Tiruvannamalai) — say this plainly if judges ask.

`model/train_tvm_model.py` trains on this data (88% accuracy) and
`backend/main.py` is wired to it, using the 8 real zones for its
geofencing lookup. Two seasonal/event flags — `is_festival_period` and
`is_monsoon_heavy_rain` — dynamically push the Karthigai Deepam site and
VOC Nagar into higher risk when triggered.

**Known tuning note:** the model also learns each zone's typical risk
from its identity, so the festival/monsoon overrides currently pull the
score up meaningfully but don't always cross into the "High" bucket. If
you want a starker jump for the demo, either give `zone_risk` a heavier
weight relative to `zone_name` when retraining, or explicitly cap/floor
the score in the API for these two override cases.

## Notes for your team / judges

- Dataset is clearly **synthetic** (see `generate_dataset.py`'s rule-based
  labeling) — say this openly if asked, per the project plan's own
  data-honesty guidance.
- Demo zones (Restricted Forest Trail, Hill View Point, etc.) are
  hardcoded around Tiruvannamalai coordinates for a believable local
  demo — swap in real government-tagged risk zones for production.
- In-memory storage in `backend/main.py` is for demo speed; swap for
  PostgreSQL per the original plan before the national round.
