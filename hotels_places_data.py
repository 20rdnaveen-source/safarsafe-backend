# Real hotels, tourist attractions, public toilets, transport hubs, and ATMs in
# Tiruvannamalai, Tamil Nadu. Coordinates, ratings, and phone numbers sourced from
# public map listings — verify current pricing/availability/hours directly before relying on it.

HOTELS = [
    {"name": "Grand Inn - Hotel Tiruvannamalai", "latitude": 12.2334146, "longitude": 79.0726458, "phone": "+91 94889 00517", "rating": 4.8, "type": "Hotel"},
    {"name": "Ashreya Suites", "latitude": 12.2255953, "longitude": 79.0380603, "phone": "+91 93600 13511", "rating": 4.4, "type": "Hotel"},
    {"name": "Hotel Aalayam Tiruvannamalai (Yatri Nivas / TTDC)", "latitude": 12.2455541, "longitude": 79.0742060, "phone": "+91 73581 00396", "rating": 4.6, "type": "Hotel (Government TTDC)"},
    {"name": "Om Lingeswaran Home Stay", "latitude": 12.2366706, "longitude": 79.0284136, "phone": "+91 96000 60697", "rating": 4.9, "type": "Homestay"},
    {"name": "Aakash Temple View Hotel", "latitude": 12.2333333, "longitude": 79.0679286, "phone": "+91 4175 224 640", "rating": 3.7, "type": "Hotel"},
    {"name": "Srikrishna Paradise", "latitude": 12.2332832, "longitude": 79.0732314, "phone": "+91 96291 50166", "rating": 4.5, "type": "Hotel"},
    {"name": "Hotel Himalayaa, A Bergamont Hotel", "latitude": 12.2525068, "longitude": 79.0695339, "phone": "+91 4175 255 255", "rating": 4.2, "type": "Hotel"},
    {"name": "Hotel Arunachala", "latitude": 12.2316296, "longitude": 79.0701840, "phone": "+91 93452 15204", "rating": 4.2, "type": "Hotel"},
    {"name": "Sawariya Hotels", "latitude": 12.2306531, "longitude": 79.0733324, "phone": "+91 76764 16164", "rating": 4.5, "type": "Hotel"},
    {"name": "Lingaa's Archana Homestay", "latitude": 12.2329734, "longitude": 79.0743847, "phone": "+91 94876 30033", "rating": 4.7, "type": "Homestay"},
    {"name": "Sai Murugan Lodge", "latitude": 12.2231548, "longitude": 79.0749574, "phone": "+91 75400 38540", "rating": 3.8, "type": "Budget Lodge"},
    {"name": "Happy Guest House", "latitude": 12.2324368, "longitude": 79.0747158, "phone": "+91 93617 77376", "rating": 4.8, "type": "Guest House"},
    {"name": "Murugan Guest House", "latitude": 12.2324615, "longitude": 79.0732135, "phone": "+91 97897 07145", "rating": 4.8, "type": "Guest House"},
    {"name": "Sai Guest House", "latitude": 12.2209390, "longitude": 79.0493601, "phone": "+91 73393 77146", "rating": 4.7, "type": "Guest House"},
    {"name": "Aadiyogi Guest House", "latitude": 12.2272839, "longitude": 79.0752538, "phone": "+91 70132 06755", "rating": 4.9, "type": "Guest House"},
    {"name": "Omkara - The Fives Guest House", "latitude": 12.2197958, "longitude": 79.0578423, "phone": "+91 89033 60409", "rating": 4.5, "type": "Guest House"},
    {"name": "RKS Residency", "latitude": 12.2262465, "longitude": 79.0780309, "phone": "+91 4175 299 668", "rating": 4.0, "type": "Hotel"},
    {"name": "Sabarish Residency Budget Lodge", "latitude": 12.2305705, "longitude": 79.0737941, "phone": None, "rating": 3.7, "type": "Budget Lodge"},
    {"name": "Shri Valli Rooms", "latitude": 12.2327101, "longitude": 79.0797740, "phone": "+91 89254 19737", "rating": 4.9, "type": "Guest House"},
    {"name": "SwathiSri Residency", "latitude": 12.2324621, "longitude": 79.0695379, "phone": "+91 94426 14126", "rating": 3.5, "type": "Hotel"},
    {"name": "Shri Bhagavan Lodge Room Stay", "latitude": 12.2495682, "longitude": 79.0712924, "phone": "+91 94438 14150", "rating": 4.4, "type": "Budget Lodge"},
    {"name": "Hotel Temple Stay", "latitude": 12.227535, "longitude": 79.0732478, "phone": None, "rating": 4.3, "type": "Hotel"},
]

PLACES_TO_VISIT = [
    {"name": "Annamalaiyar Temple (Arunachaleswarar Temple)", "latitude": 12.2316461, "longitude": 79.0677399, "phone": "+91 4175 252 438", "rating": 4.7, "category": "Main Temple", "description": "The great Shiva temple at the heart of Tiruvannamalai, one of the Pancha Bhoota Stalams (fire element); open 5:30 AM-12:30 PM and 3:30-9:30 PM."},
    {"name": "Raja Gopuram (East Entrance)", "latitude": 12.2311705, "longitude": 79.0697264, "phone": "+91 4175 252 438", "rating": 4.8, "category": "Temple Landmark", "description": "The towering main eastern entrance tower of Annamalaiyar Temple; also the starting point for free Girivalam buses."},
    {"name": "Sri Ramanasramam", "latitude": 12.2238142, "longitude": 79.0567197, "phone": None, "rating": 4.8, "category": "Ashram / Spiritual", "description": "Ashram of Sri Ramana Maharshi at the base of Arunachala Hill, known for its deep silence and meditation halls."},
    {"name": "Arulmigu Indira Lingam", "latitude": 12.2292841, "longitude": 79.0703001, "phone": "+91 99524 60727", "rating": 4.7, "category": "Temple / Ashta Lingam", "description": "First of the 8 Ashta Lingams on the Girivalam path, dedicated to Lord Indra (east direction)."},
    {"name": "Arulmigu Agni Lingam", "latitude": 12.2241518, "longitude": 79.0591276, "phone": None, "rating": 4.8, "category": "Temple / Ashta Lingam", "description": "Second Ashta Lingam, representing the fire element; the inner sanctum reportedly feels naturally warm."},
    {"name": "Arulmigu Sri Varuna Lingam", "latitude": 12.2493791, "longitude": 79.0347198, "phone": None, "rating": 4.7, "category": "Temple / Ashta Lingam", "description": "West-facing Ashta Lingam shrine on the Girivalam circuit with scenic hill views."},
    {"name": "Arulmigu Niruthi Lingam", "latitude": 12.2345670, "longitude": 79.0326968, "phone": None, "rating": 4.8, "category": "Temple / Ashta Lingam", "description": "South-west Ashta Lingam shrine on the Girivalam path, known for its peaceful energy."},
    {"name": "Arulmigu Shri Pachaiamman Temple", "latitude": 12.2480556, "longitude": 79.0672222, "phone": "+91 95973 47692", "rating": 4.7, "category": "Temple", "description": "Local guardian-goddess temple believed to protect Arunachala Hill, quieter and less commercial than the main temple."},
    {"name": "Arulmigu Shri Srinivasa Perumal Koil (Aadi Thiruvarangam)", "latitude": 12.2312786, "longitude": 79.0696454, "phone": None, "rating": 4.5, "category": "Temple", "description": "One of the 108 Divya Desam Vishnu temples, located along the Girivalam path."},
    {"name": "Idukku Pillaiyar Temple (Moksha Margam)", "latitude": 12.2529328, "longitude": 79.0644696, "phone": None, "rating": 4.7, "category": "Temple", "description": "Known for a narrow rock passage devotees squeeze through, believed to symbolize shedding past karma."},
    {"name": "Virupaksha Cave", "latitude": 12.2334707, "longitude": 79.0620277, "phone": None, "rating": 4.7, "category": "Cave / Meditation Spot", "description": "Sacred cave on Arunachala Hill where Ramana Maharshi meditated for years; reached via a short rocky trek, open 8:30 AM-4:30 PM."},
    {"name": "Skandasramam", "latitude": 12.2345482, "longitude": 79.0606944, "phone": None, "rating": 4.8, "category": "Ashram / Viewpoint", "description": "Hillside ashram above Virupaksha Cave with a panoramic view over Tiruvannamalai town, open 8:30 AM-4:30 PM."},
    {"name": "Skandasramam Trail City View Point", "latitude": 12.2330543, "longitude": 79.0606856, "phone": None, "rating": 4.8, "category": "Viewpoint", "description": "Scenic lookout point along the Skandasramam trail overlooking the temple complex and town."},
    {"name": "Arunagiri Children's Park", "latitude": 12.2238294, "longitude": 79.0529954, "phone": None, "rating": 4.2, "category": "Park / Family", "description": "Eco-friendly children's park opposite the Government Arts College with play areas and greenery; closed Mondays and Thursdays."},
]

PUBLIC_TOILETS = [
    {"name": "Free Public Toilet (Vada Othavadai St)", "latitude": 12.2331175, "longitude": 79.0663702, "phone": None, "notes": "Free, well-reviewed as clean."},
    {"name": "Public Toilet (Kanji Main Road, near Nippon Paint)", "latitude": 12.2514080, "longitude": 79.0679115, "phone": None, "notes": None},
    {"name": "Sri GST Mini Hall Public Bathrooms & Toilet", "latitude": 12.2327279, "longitude": 79.0701796, "phone": None, "notes": "Central, near the temple."},
    {"name": "Public Bath Room & Toilets (Pavazhakundur)", "latitude": 12.2315054, "longitude": 79.0652974, "phone": None, "notes": "Paid; bathing facilities and luggage storage available."},
    {"name": "Public Toilet (Vengikkal)", "latitude": 12.2532322, "longitude": 79.0638990, "phone": None, "notes": "Open 24 hours."},
    {"name": "Public Toilet (Vettavalam Road)", "latitude": 12.2248669, "longitude": 79.0766182, "phone": None, "notes": None},
    {"name": "Toilet (TSR Nagar)", "latitude": 12.2231278, "longitude": 79.0538769, "phone": None, "notes": "Paid, ₹10."},
    {"name": "Pay and Use Toilet", "latitude": 12.2361371, "longitude": 79.0799757, "phone": None, "notes": None},
]

TRANSPORT_HUBS = [
    {"name": "Tiruvannamalai Old Bus Stand", "latitude": 12.2401211, "longitude": 79.0739614, "phone": None, "type": "Bus Stand", "notes": "Serves Vellore, Bangalore, Tirupati, Arni and nearby villages."},
    {"name": "Tiruvannamalai New Bus Stand", "latitude": 12.2367591, "longitude": 79.0793330, "phone": None, "type": "Bus Stand", "notes": "Serves Chennai, Trichy, Vellore, Gingee routes; behind the railway station."},
    {"name": "Tiruvannamalai Railway Station (TNM)", "latitude": 12.2383640, "longitude": 79.0778219, "phone": None, "type": "Railway Station", "notes": "3 platforms; connects toward Villupuram and further south."},
    {"name": "Periyar Bus Stand", "latitude": 12.2300960, "longitude": 79.0705560, "phone": None, "type": "Local Bus Stop", "notes": "Near the temple, local town buses."},
]

ATMS = [
    {"name": "Bank of Baroda ATM (Polur Road)", "latitude": 12.2409951, "longitude": 79.0727164, "phone": None, "bank": "Bank of Baroda"},
    {"name": "HDFC Bank ATM (near Anna Statue)", "latitude": 12.2342538, "longitude": 79.0730644, "phone": None, "bank": "HDFC Bank"},
    {"name": "CSB Bank ATM (Thiruvoodal Street)", "latitude": 12.2295468, "longitude": 79.0671316, "phone": None, "bank": "CSB Bank"},
    {"name": "SBI ATM (Thirumanjana Gopuram St)", "latitude": 12.2247830, "longitude": 79.0660310, "phone": None, "bank": "State Bank of India"},
    {"name": "State Bank of India ATM (Sannathi St)", "latitude": 12.2304054, "longitude": 79.0747161, "phone": None, "bank": "State Bank of India"},
    {"name": "SBI ATM (Rayar Naidu Complex, Polur Rd)", "latitude": 12.2413570, "longitude": 79.0728130, "phone": None, "bank": "State Bank of India"},
]
