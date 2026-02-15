"""Configuration: NYC neighborhoods, amenities, scoring weights, and constants."""

from __future__ import annotations

# --- Bot Constants ---

CONTAINER_IDLE_TIMEOUT = 300
SCAN_CRON_SCHEDULE = "0 8 * * *"
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

