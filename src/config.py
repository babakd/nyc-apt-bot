"""Configuration: NYC neighborhoods, amenities, scoring weights, and constants."""

from __future__ import annotations

# --- Bot Constants ---

CONTAINER_IDLE_TIMEOUT = 300
DATA_DIR = "/data"

# --- StreetEasy URLs ---

STREETEASY_BASE = "https://streeteasy.com"
STREETEASY_RENTALS = "https://streeteasy.com/for-rent"

# --- NYC Neighborhoods ---
# Maps display name -> StreetEasy URL slug

NEIGHBORHOODS: dict[str, str] = {
    # Manhattan
    "Upper East Side": "upper-east-side",
    "Upper West Side": "upper-west-side",
    "Midtown": "midtown",
    "Midtown East": "midtown-east",
    "Midtown West": "midtown-west",
    "Chelsea": "chelsea",
    "West Village": "west-village",
    "East Village": "east-village",
    "Greenwich Village": "greenwich-village",
    "SoHo": "soho",
    "Tribeca": "tribeca",
    "Lower East Side": "lower-east-side",
    "Financial District": "financial-district",
    "Hell's Kitchen": "hells-kitchen",
    "Murray Hill": "murray-hill",
    "Gramercy": "gramercy-park",
    "Flatiron": "flatiron",
    "NoHo": "noho",
    "Nolita": "nolita",
    "Harlem": "harlem",
    "East Harlem": "east-harlem",
    "Washington Heights": "washington-heights",
    "Inwood": "inwood",
    "Battery Park City": "battery-park-city",
    "Kips Bay": "kips-bay",
    "Stuyvesant Town": "stuyvesant-town",
    # Brooklyn
    "Williamsburg": "williamsburg",
    "Park Slope": "park-slope",
    "DUMBO": "dumbo",
    "Brooklyn Heights": "brooklyn-heights",
    "Cobble Hill": "cobble-hill",
    "Boerum Hill": "boerum-hill",
    "Fort Greene": "fort-greene",
    "Clinton Hill": "clinton-hill",
    "Bed-Stuy": "bed-stuy",
    "Greenpoint": "greenpoint",
    "Bushwick": "bushwick",
    "Crown Heights": "crown-heights",
    "Prospect Heights": "prospect-heights",
    "Carroll Gardens": "carroll-gardens",
    "Red Hook": "red-hook",
    "Sunset Park": "sunset-park",
    "Bay Ridge": "bay-ridge",
    "Flatbush": "flatbush",
    "Downtown Brooklyn": "downtown-brooklyn",
    "Prospect Lefferts Gardens": "prospect-lefferts-gardens",
    "Windsor Terrace": "windsor-terrace",
    "Gowanus": "gowanus",
    # Queens
    "Astoria": "astoria",
    "Long Island City": "long-island-city",
    "Sunnyside": "sunnyside",
    "Jackson Heights": "jackson-heights",
    "Forest Hills": "forest-hills",
    "Flushing": "flushing",
    "Ridgewood": "ridgewood",
    "Woodside": "woodside",
    # Bronx
    "Riverdale": "riverdale",
    "Mott Haven": "mott-haven",
    # Jersey City (often searched alongside NYC)
    "Jersey City": "jersey-city",
}

# Reverse lookup: slug -> display name
NEIGHBORHOOD_SLUGS: dict[str, str] = {v: k for k, v in NEIGHBORHOODS.items()}

# --- Amenity Maps ---
# Maps user-friendly name -> StreetEasy filter parameter

AMENITIES: dict[str, str] = {
    "Dishwasher": "dishwasher",
    "Laundry in Unit": "washer_dryer",
    "Laundry in Building": "laundry",
    "Doorman": "doorman",
    "Elevator": "elevator",
    "Gym": "gym",
    "Roof Deck": "roof_deck",
    "Outdoor Space": "outdoor_space",
    "Pets Allowed": "pets",
    "Cats Allowed": "cats",
    "Dogs Allowed": "dogs",
    "No Fee": "no_fee",
    "Storage": "storage",
    "Bike Room": "bike_room",
    "Parking": "parking",
    "Pool": "pool",
    "Concierge": "concierge",
    "Balcony": "balcony",
    "Terrace": "terrace",
    "Hardwood Floors": "hardwood_floors",
    "Central AC": "central_ac",
    "Live-in Super": "live_in_super",
    "Pre-war": "prewar",
    "New Development": "new_development",
}

# --- Scoring & Filtering ---

SCORING_MODEL = "claude-opus-4-7"
VISION_MODEL = "claude-haiku-4-5-20251001"
SCORE_FLOOR = 25

# --- Scanner Cache ---

CACHE_MAX_AGE_HOURS = 48
AMENITY_CACHE_MAX_ITEMS = 500
AMENITY_TEXT_MAX_LEN = 360

# --- Listing Filters ---

STALE_LISTING_MONTHS = 3
MAX_PER_BUILDING = 2
MAX_PER_NEIGHBORHOOD = 5
MAX_RESULTS_TO_SEND = 12

# --- Display Limits ---

MAX_PROS_DISPLAYED = 3
MAX_CONS_DISPLAYED = 2
MAX_AMENITIES_DISPLAY = 8

# --- Vision Hero Photos ---

MAX_LISTINGS_PER_BATCH = 12
PHOTO_DOWNLOAD_SEMAPHORE = 20

# --- Apify Actor Polling ---

POLL_INTERVAL_SECS = 10
ABORT_AFTER_SECS_NO_ITEMS = 60
SEARCH_MAX_ITEMS = 120
SEARCH_MAX_WAIT_SECS = 420

# --- Apify Enrichment ---

ENRICHMENT_NO_ITEMS_ABORT_SECS = 120
ENRICHMENT_MAX_WAIT_SECS = 240
ENRICHMENT_RETRY_BATCH_SIZE = 25
ENRICHMENT_TARGET_COVERAGE = 0.95
ENRICHMENT_MAX_URLS = 120

# --- Strict Amenity Verification ---

AMENITY_REQUIRED_COVERAGE = 0.95
AMENITY_ENRICHMENT_MAX_WAIT_SECS = 600
AMENITY_ENRICHMENT_NO_ITEMS_ABORT_SECS = 120

# --- Search Reliability ---

SEARCH_FAILURE_STREAK_FOR_COOLDOWN = 2
SEARCH_FAILURE_COOLDOWN_BASE_SECS = 180
SEARCH_FAILURE_COOLDOWN_MAX_SECS = 1800

# --- Actor Build Health / Pinning ---

APIFY_ACTOR_BUILD_PIN_FILE = "/data/system/apify_actor_build_pin.json"
APIFY_ACTOR_CANARY_FILE = "/data/system/apify_actor_canary_status.json"
APIFY_ACTOR_CANARY_URLS_FILE = "/data/system/apify_actor_canary_urls.json"

# --- Telegraph ---

TELEGRAPH_ACCOUNT_FILE = "/data/system/telegraph_account.json"

# --- Conversation ---

MAX_HISTORY_TURNS = 30

# --- Storage ---

UPDATE_MARKER_RETENTION = 5000
UPDATE_MARKER_PRUNE_EVERY = 100
