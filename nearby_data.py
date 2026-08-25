# Real hospitals, police stations, and fire stations in Tiruvannamalai, Tamil Nadu.
# Coordinates and phone numbers sourced from public map listings — verify before
# any real deployment, as clinics/contact numbers can change.

HOSPITALS = [
    {"name": "Government Medical College and Hospital, Tiruvannamalai", "latitude": 12.2742868, "longitude": 79.0791, "phone": "+91 4175 233 315", "type": "Government Multi-Speciality"},
    {"name": "Tiruvannamalai Old GH (District Health Office)", "latitude": 12.2255618, "longitude": 79.0646590, "phone": None, "type": "Government (Emergency cases only)"},
    {"name": "Suriyan Hospital", "latitude": 12.2305964, "longitude": 79.0807439, "phone": "+91 93427 63427", "type": "Private Emergency & ICU"},
    {"name": "Amudham Hospitals", "latitude": 12.2170665, "longitude": 79.0597419, "phone": "+91 94869 91100", "type": "Private Multi-Speciality"},
    {"name": "Staar Multi Speciality Hospital", "latitude": 12.2361192, "longitude": 79.0732614, "phone": "+91 96778 71782", "type": "Private Multi-Speciality"},
    {"name": "PONNI Hospital, Tiruvannamalai", "latitude": 12.2308009, "longitude": 79.0783453, "phone": "+91 4175 223 222", "type": "Private Multi-Speciality"},
    {"name": "Arunai Medical College and Hospital", "latitude": 12.1900299, "longitude": 79.0817595, "phone": "+91 95855 22226", "type": "Medical College Hospital"},
    {"name": "SWATHI SHREE Hospital", "latitude": 12.2360196, "longitude": 79.0744185, "phone": "+91 86868 63147", "type": "Private General Hospital"},
    {"name": "SS Hospital", "latitude": 12.2366589, "longitude": 79.0748665, "phone": "+91 4175 254 470", "type": "Private Maternity Hospital"},
    {"name": "RMR Hospital", "latitude": 12.226065, "longitude": 79.0257762, "phone": "+91 91768 91769", "type": "Private Multi-Speciality with ICU"},
    {"name": "Dhanabhakiyam Multispeciality Hospital", "latitude": 12.2249644, "longitude": 79.0778047, "phone": "+91 70104 84969", "type": "Private Multi-Speciality"},
    {"name": "KK Hospital", "latitude": 12.2175731, "longitude": 79.0591588, "phone": "+91 63821 91197", "type": "Private Multi-Speciality"},
    {"name": "LITHU Multispeciality Hospital", "latitude": 12.2088648, "longitude": 79.0496285, "phone": None, "type": "Private Multi-Speciality"},
    {"name": "SRM Multicare Hospital (Athiyandal)", "latitude": 12.2276118, "longitude": 79.026939, "phone": "+91 80982 03303", "type": "Private Multi-Speciality"},
    {"name": "SRM Multi Clini Care (Kanji Road)", "latitude": 12.2502712, "longitude": 79.0698079, "phone": "+91 94437 25108", "type": "Private Multi-Speciality"},
    {"name": "KVS Hospital", "latitude": 12.2295476, "longitude": 79.064891, "phone": None, "type": "Private Hospital / Pediatrics"},
    {"name": "Sri Vasavi Hospital", "latitude": 12.2290341, "longitude": 79.0661252, "phone": "+91 94861 28026", "type": "Private Multi-Speciality"},
    {"name": "KVM Hospital", "latitude": 12.2381717, "longitude": 79.0736209, "phone": "+91 99524 84939", "type": "Private General Hospital"},
    {"name": "Vasantha Hospital and Wellness Research Center", "latitude": 12.2353737, "longitude": 79.0711882, "phone": "+91 95664 95040", "type": "Private Maternity Hospital"},
    {"name": "Raj Hospital / Raj ENT Hospital", "latitude": 12.2300631, "longitude": 79.0650511, "phone": "+91 97866 63744", "type": "Private Multi-Speciality"},
    {"name": "Sathya ENT Hospital", "latitude": 12.229721, "longitude": 79.065891, "phone": "+91 4175 225 078", "type": "Specialty (ENT)"},
    {"name": "Nalam Hospital (ENT)", "latitude": 12.2445124, "longitude": 79.0719826, "phone": "+91 93856 86719", "type": "Specialty (ENT)"},
    {"name": "Usharani Diabetic and General Clinic", "latitude": 12.2298381, "longitude": 79.0655196, "phone": "+91 95855 70373", "type": "Specialty Clinic (Diabetic Care)"},
    {"name": "Tamilnadu Neuro & Mind Care (TNM)", "latitude": 12.2299178, "longitude": 79.0658522, "phone": "+91 79043 78416", "type": "Specialty Clinic (Neuro/Psychiatric)"},
    {"name": "Rajam Nursing Home", "latitude": 12.2307969, "longitude": 79.0726088, "phone": "+91 86374 30049", "type": "Nursing Home"},
    {"name": "Balaji Nursing Home", "latitude": 12.2299206, "longitude": 79.0772618, "phone": None, "type": "Nursing Home (Maternity)"},
    {"name": "Theepa Nursing Home", "latitude": 12.234474, "longitude": 79.072733, "phone": "+91 4175 253 652", "type": "Nursing Home"},
    {"name": "Vasantha Nursing Home", "latitude": 12.235313, "longitude": 79.073394, "phone": None, "type": "Nursing Home"},
    {"name": "#25 Nursing Home Care", "latitude": 12.2412242, "longitude": 79.0733384, "phone": "+91 86675 44672", "type": "Home Nursing Care"},
    {"name": "Smaart Health Care", "latitude": 12.2338989, "longitude": 79.0695799, "phone": "+91 74491 44440", "type": "Physiotherapy / Pain Clinic"},
    {"name": "Sri Ramana Maharishi Eye Hospital", "latitude": 12.2283195, "longitude": 79.0640753, "phone": "+91 4175 229 461", "type": "Specialty (Eye)"},
    {"name": "Vivekananda Eye Hospital", "latitude": 12.2253184, "longitude": 79.0810997, "phone": "+91 94425 29783", "type": "Specialty (Eye)"},
    {"name": "Sesha Eye Hospital", "latitude": 12.2365707, "longitude": 79.0729336, "phone": "+91 93456 45258", "type": "Specialty (Eye)"},
    {"name": "Dr Agarwal's Eye Hospital", "latitude": 12.2514746, "longitude": 79.07047, "phone": "+91 95949 02665", "type": "Specialty (Eye)"},
    {"name": "Baba Eye Hospital", "latitude": 12.2250651, "longitude": 79.074478, "phone": None, "type": "Specialty (Eye)"},
    {"name": "Dr Madan Dental", "latitude": 12.2294064, "longitude": 79.0665385, "phone": "+91 86819 56161", "type": "Specialty (Dental)"},
    {"name": "RK Dentistry", "latitude": 12.2511226, "longitude": 79.0702432, "phone": "+91 7397 505 797", "type": "Specialty (Dental)"},
    {"name": "Dwarakamai Siddha Varmam Clinic", "latitude": 12.2249336, "longitude": 79.0611727, "phone": "+91 76958 93994", "type": "Siddha / Alternative Medicine"},
    {"name": "Government Primary Health Centre, Konalur", "latitude": 12.1741958, "longitude": 79.1498196, "phone": None, "type": "Government PHC"},
    {"name": "Government Primary Health Centre, Chinnakolapadi", "latitude": 12.2553626, "longitude": 78.9787824, "phone": None, "type": "Government PHC"},
    {"name": "Government Sub Health Centre, Vengikkal", "latitude": 12.2652707, "longitude": 79.0691088, "phone": None, "type": "Government Sub Centre"},
]

POLICE_STATIONS = [
    {"name": "Tiruvannamalai Town Police Station", "latitude": 12.2302015, "longitude": 79.0703003, "phone": None},
    {"name": "Taluk Police Station, Thiruvannamalai", "latitude": 12.2545868, "longitude": 79.0613874, "phone": "+91 4175 232 274"},
    {"name": "East Police Station", "latitude": 12.2349856, "longitude": 79.0759344, "phone": "+91 4175 222 302"},
    {"name": "West Police Station, Thiruvannamalai", "latitude": 12.2284807, "longitude": 79.041883, "phone": None},
    {"name": "SP Office, Tiruvannamalai (District Police Office)", "latitude": 12.271471, "longitude": 79.0749017, "phone": "+91 4175 233 431"},
    {"name": "Police Station Veraiyur", "latitude": 12.0916738, "longitude": 79.130405, "phone": "+91 4175 245 226"},
    {"name": "Police Station Pachal", "latitude": 12.2699071, "longitude": 78.9417685, "phone": None},
]

FIRE_STATIONS = [
    {"name": "Fire and Rescue Station (Collector Office Campus)", "latitude": 12.2691066, "longitude": 79.0736437, "phone": None},
    {"name": "District Fire Office", "latitude": 12.2688529, "longitude": 79.0738606, "phone": None},
    {"name": "Annamalaiyar Temple Fire and Rescue Station", "latitude": 12.2313622, "longitude": 79.0703751, "phone": None},
]
