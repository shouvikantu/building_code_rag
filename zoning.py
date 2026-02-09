"""
Portland Zoning and Building Information Query Tool

This script provides functionality to query zoning, building, and taxlot information
for addresses in Portland, Oregon using the PortlandMaps API.

Features:
- Geocode addresses to latitude/longitude
- Query zoning information (base zone, overlays, plan districts)
- Query building information (name, address, type, square footage, etc.)
- Query taxlot/parcel information
- Batch query by ZIP code with optional filters and export to CSV

Usage:
    python zoning.py --zip 97211 --num 10 --base-zone R5 --min-sqft 1000

"""

import requests
from geopy.geocoders import Nominatim
from typing import Dict, Optional, Tuple, List
import csv
import argparse
import time
import threading
from functools import lru_cache

# API URLs
ZONING_QUERY_URL = "https://www.portlandmaps.com/od/rest/services/COP_OpenData_ZoningCode/MapServer/16/query"
BUILDING_QUERY_URL = "https://www.portlandmaps.com/od/rest/services/COP_OpenData_Property/MapServer/184/query"
TAXLOT_QUERY_URLS = [
    "https://www.portlandmaps.com/od/rest/services/COP_OpenData_Property/MapServer/1272/query",
    "https://www.portlandmaps.com/od/rest/services/COP_OpenData_Property/MapServer/47/query",
]

# Request/session config
_session_local = threading.local()
_GEOCODE_MIN_INTERVAL = 1.0
_geocode_lock = threading.Lock()
_last_geocode_time = 0.0


def _get_session() -> requests.Session:
    session = getattr(_session_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=1)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _session_local.session = session
    return session


_geolocator = None


def _get_geolocator() -> Nominatim:
    global _geolocator
    if _geolocator is None:
        _geolocator = Nominatim(user_agent="portland-zoning-lookup")
    return _geolocator


def _normalize_address(address: str) -> str:
    return " ".join(address.strip().upper().split())


def _rate_limit_geocode() -> None:
    global _last_geocode_time
    with _geocode_lock:
        elapsed = time.monotonic() - _last_geocode_time
        if elapsed < _GEOCODE_MIN_INTERVAL:
            time.sleep(_GEOCODE_MIN_INTERVAL - elapsed)
        _last_geocode_time = time.monotonic()


def _round_coords(lat: float, lon: float, precision: int = 6) -> Tuple[float, float]:
    return round(lat, precision), round(lon, precision)


@lru_cache(maxsize=2048)
def _geocode_address_cached(address_norm: str) -> Dict[str, float]:
    _rate_limit_geocode()
    geolocator = _get_geolocator()
    location = geolocator.geocode(address_norm, timeout=10)
    if location is None:
        raise ValueError("Address could not be geocoded")
    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "matched_address": location.address
    }


def geocode_address(address: str) -> Dict[str, float]:
    """
    Geocode an address string to latitude and longitude using Nominatim.

    Args:
        address: The address string to geocode.

    Returns:
        A dictionary with 'latitude', 'longitude', and 'matched_address'.

    Raises:
        ValueError: If the address could not be geocoded.
    """
    address_norm = _normalize_address(address)
    result = _geocode_address_cached(address_norm)
    return dict(result)


@lru_cache(maxsize=4096)
def _query_zoning_cached(lat: float, lon: float) -> Dict:
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "where": "1=1",
        "f": "json"
    }
    resp = _get_session().get(ZONING_QUERY_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    if not features:
        raise ValueError("No zoning data found for this location")
    return features[0]


def query_zoning(lat: float, lon: float) -> Dict:
    """
    Query the PortlandMaps zoning layer for zoning info at the given lat/lon.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.

    Returns:
        The first feature dictionary from the API response.

    Raises:
        ValueError: If no zoning data is found for the location.
        requests.HTTPError: If the API request fails.
    """
    lat, lon = _round_coords(lat, lon)
    return _query_zoning_cached(lat, lon)


def extract_zoning_attrs(feature: Dict) -> Dict:
    """
    Extract relevant zoning attributes from a zoning feature dict.

    Args:
        feature: The feature dictionary from the zoning query.

    Returns:
        A dictionary with base zone, overlays, plan district, source, and raw attributes.
    """
    attrs = feature["attributes"]
    return {
        "base_zone": attrs.get("ZONE"),
        "overlay_zones": attrs.get("OVERLAY"),
        "plan_district": attrs.get("PLAN_DISTRICT"),
        "source": "Portland Maps - Zoning Code",
        "raw_attributes": attrs
    }


def get_zoning_for_address(address: str) -> Dict:
    """
    Geocode an address and query zoning info for that location.

    Args:
        address: The address string.

    Returns:
        A dictionary with input address, matched address, location, and zoning info.
    """
    geo = geocode_address(address)
    feature = query_zoning(geo["latitude"], geo["longitude"])
    zoning = extract_zoning_attrs(feature)
    return {
        "input_address": address,
        "matched_address": geo["matched_address"],
        "location": {"lat": geo["latitude"], "lon": geo["longitude"]},
        "zoning": zoning
    }


def format_zoning_result(result: Dict) -> str:
    """
    Format the zoning result dictionary into a readable string for display.

    Args:
        result: The result dictionary from get_zoning_for_address.

    Returns:
        A formatted string representation of the zoning information.
    """
    zoning = result["zoning"]
    attrs = zoning["raw_attributes"]
    lines = [
        "ADDRESS LOOKUP",
        f"  Input Address   : {result['input_address']}",
        f"  Matched Address : {result['matched_address']}",
        "",
        "LOCATION",
        f"  Latitude  : {result['location']['lat']}",
        f"  Longitude : {result['location']['lon']}",
        "",
        "ZONING SUMMARY",
        f"  Base Zone       : {zoning['base_zone']} ({attrs.get('ZONE_DESC')})",
        f"  Overlay Zone    : {attrs.get('OVRLY')} ({attrs.get('OVRLY_DESC')})",
        f"  Plan District   : {attrs.get('PLDIST')} ({attrs.get('PLDIST_DESC')})",
        f"  Map Label       : {attrs.get('MAPLABEL')}",
        "",
        "COMPREHENSIVE PLAN",
        f"  Designation     : {attrs.get('CMP')} ({attrs.get('CMP_DESC')})",
        "",
        "DATA SOURCE",
        f"  {zoning['source']}",
    ]
    return "\n".join(lines)


@lru_cache(maxsize=4096)
def _query_building_cached(lat: float, lon: float) -> Optional[Dict]:
    """
    Query the PortlandMaps Buildings layer (184) for building info at the given latitude and longitude.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.

    Returns:
        A dictionary of building attributes if found, otherwise None.
    """
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "where": "1=1",
        "f": "json"
    }
    try:
        resp = _get_session().get(BUILDING_QUERY_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        if features:
            return features[0]["attributes"]
    except Exception:
        return None
    return None


def query_building(lat: float, lon: float) -> Optional[Dict]:
    lat, lon = _round_coords(lat, lon)
    return _query_building_cached(lat, lon)


def print_building_info(building_attrs: Optional[Dict]) -> None:
    """
    Print formatted building information from the attributes dictionary.

    Args:
        building_attrs: The building attributes dictionary, or None.
    """
    print("\nBUILDING INFORMATION")
    if building_attrs:
        building_info = {
            "Building Name": building_attrs.get("BLDG_NAME"),
            "Building Address": building_attrs.get("BLDG_ADDR"),
            "Building ID": building_attrs.get("BLDG_ID"),
            "Year Built": building_attrs.get("YEAR_BUILT"),
            "Building Type": building_attrs.get("BLDG_TYPE"),
            "Predominant Use": building_attrs.get("BLDG_USE"),
            "Square Footage": building_attrs.get("BLDG_SQFT"),
            "Number of Stories": building_attrs.get("NUM_STORY"),
            "Residential Units": building_attrs.get("UNITS_RES"),
            "Total Occupancy": building_attrs.get("OCCUP_CAP"),
            "ADA Accessible": building_attrs.get("ADA_ACCESS"),
            "Average Height": building_attrs.get("AVG_HEIGHT"),
            "Maximum Height": building_attrs.get("MAX_HEIGHT"),
            "Minimum Height": building_attrs.get("MIN_HEIGHT"),
            "Roof Elevation": building_attrs.get("ROOF_ELEV"),
            "Structure Type": building_attrs.get("STRUC_TYPE"),
            "Structure Condition": building_attrs.get("STRUC_COND"),
        }
        for key, value in building_info.items():
            print(f"  {key:25}: {value}")
    else:
        print("  (No building info found for this location)")


@lru_cache(maxsize=4096)
def _query_taxlot_cached(lat: float, lon: float) -> Optional[Dict]:
    """
    Query for taxlot/parcel info using available layers.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.

    Returns:
        A dictionary of taxlot attributes if found, otherwise None.
    """
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "where": "1=1",
        "f": "json"
    }
    for url in TAXLOT_QUERY_URLS:
        try:
            resp = _get_session().get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            if features:
                return features[0]["attributes"]
        except Exception:
            continue
    return None


def query_taxlot(lat: float, lon: float) -> Optional[Dict]:
    lat, lon = _round_coords(lat, lon)
    return _query_taxlot_cached(lat, lon)


@lru_cache(maxsize=256)
def _query_taxlots_by_zip_cached(zip_code: str, max_results: int = 10) -> List[Dict]:
    """
    Query taxlots by ZIP code.

    Args:
        zip_code: The ZIP code to query.
        max_results: Maximum number of results to return.

    Returns:
        A list of feature dictionaries.
    """
    params = {
        "where": f"ZIP_CODE = '{zip_code}'",
        "outFields": "*",
        "f": "json",
        "resultRecordCount": max_results
    }
    for url in TAXLOT_QUERY_URLS:
        try:
            resp = _get_session().get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            if features:
                return features[:max_results]
        except Exception:
            continue
    return []


def query_taxlots_by_zip(zip_code: str, max_results: int = 10) -> List[Dict]:
    return list(_query_taxlots_by_zip_cached(zip_code, max_results))


def _coerce_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_lat_lon(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _extract_lat_lon(attrs: Dict) -> Optional[Tuple[float, float]]:
    # Prefer explicit lat/lon fields; fall back to point fields only if values look valid.
    lat_candidates = ("LAT", "LATITUDE")
    lon_candidates = ("LON", "LONGITUDE", "LONG", "LNG")
    lat = next((attrs.get(k) for k in lat_candidates if attrs.get(k) is not None), None)
    lon = next((attrs.get(k) for k in lon_candidates if attrs.get(k) is not None), None)
    if lat is not None and lon is not None:
        lat_f = _coerce_float(lat)
        lon_f = _coerce_float(lon)
        if lat_f is not None and lon_f is not None and _valid_lat_lon(lat_f, lon_f):
            return lat_f, lon_f

    # Some layers expose point fields; only use if they are already lat/lon ranges.
    fallback_pairs = [
        ("POINT_Y", "POINT_X"),
        ("Y", "X"),
        ("Y_COORD", "X_COORD"),
    ]
    for lat_key, lon_key in fallback_pairs:
        if attrs.get(lat_key) is None or attrs.get(lon_key) is None:
            continue
        lat_f = _coerce_float(attrs.get(lat_key))
        lon_f = _coerce_float(attrs.get(lon_key))
        if lat_f is not None and lon_f is not None and _valid_lat_lon(lat_f, lon_f):
            return lat_f, lon_f

    return None


def query_property_by_address(address: str) -> Dict:
    """
    Query zoning, building, and taxlot info for a single address.
    """
    result = get_zoning_for_address(address)
    lat = result["location"]["lat"]
    lon = result["location"]["lon"]
    building_attrs = query_building(lat, lon)
    taxlot_attrs = query_taxlot(lat, lon)
    return {
        "Address": result["matched_address"],
        "Latitude": lat,
        "Longitude": lon,
        "Base Zone": result["zoning"]["raw_attributes"].get("ZONE"),
        "Overlay Zone": result["zoning"]["raw_attributes"].get("OVRLY"),
        "Plan District": result["zoning"]["raw_attributes"].get("PLDIST"),
        "Building Name": building_attrs.get("BLDG_NAME") if building_attrs else None,
        "Building Address": building_attrs.get("BLDG_ADDR") if building_attrs else None,
        "Year Built": building_attrs.get("YEAR_BUILT") if building_attrs else None,
        "Building Type": building_attrs.get("BLDG_TYPE") if building_attrs else None,
        "Square Footage": building_attrs.get("BLDG_SQFT") if building_attrs else None,
        "Number of Stories": building_attrs.get("NUM_STORY") if building_attrs else None,
        "Residential Units": building_attrs.get("UNITS_RES") if building_attrs else None,
        "Taxlot ID": taxlot_attrs.get("TLID") if taxlot_attrs else None,
        "Owner": taxlot_attrs.get("OWNER") if taxlot_attrs else None,
        "Land Use": taxlot_attrs.get("LANDUSE") if taxlot_attrs else None,
    }


def query_properties_by_zip(
    zip_code: str,
    max_results: int = 10,
    base_zones: Optional[List[str]] = None,
    min_sqft: Optional[int] = None,
) -> List[Dict]:
    """
    Query properties by ZIP code and return structured property data.
    """
    taxlot_features = query_taxlots_by_zip(zip_code, max_results)
    if not taxlot_features:
        return []

    properties_data = []
    for feature in taxlot_features:
        attrs = feature.get("attributes", {})
        address = attrs.get("SITE_ADDR") or attrs.get("ADDRESS") or attrs.get("ADDRESS_FULL")
        if not address:
            continue

        try:
            lat_lon = _extract_lat_lon(attrs)
            if lat_lon:
                lat, lon = lat_lon
            else:
                geo = geocode_address(f"{address}, Portland, OR")
                lat = geo["latitude"]
                lon = geo["longitude"]

            zoning_feature = query_zoning(lat, lon)
            zoning = extract_zoning_attrs(zoning_feature)
            building_attrs = query_building(lat, lon)
            taxlot_attrs = attrs

            data = {
                "Address": address,
                "Latitude": lat,
                "Longitude": lon,
                "Base Zone": zoning["base_zone"],
                "Overlay Zone": zoning["raw_attributes"].get("OVRLY"),
                "Plan District": zoning["raw_attributes"].get("PLDIST"),
                "Building Name": building_attrs.get("BLDG_NAME") if building_attrs else None,
                "Building Address": building_attrs.get("BLDG_ADDR") if building_attrs else None,
                "Year Built": building_attrs.get("YEAR_BUILT") if building_attrs else None,
                "Building Type": building_attrs.get("BLDG_TYPE") if building_attrs else None,
                "Square Footage": building_attrs.get("BLDG_SQFT") if building_attrs else None,
                "Number of Stories": building_attrs.get("NUM_STORY") if building_attrs else None,
                "Residential Units": building_attrs.get("UNITS_RES") if building_attrs else None,
                "Taxlot ID": taxlot_attrs.get("TLID") or taxlot_attrs.get("TAXLOT_ID"),
                "Owner": taxlot_attrs.get("OWNER"),
                "Land Use": taxlot_attrs.get("LANDUSE"),
            }
            properties_data.append(data)
        except Exception:
            continue

    if base_zones:
        properties_data = [p for p in properties_data if p.get("Base Zone") in base_zones]
    if min_sqft is not None:
        properties_data = [
            p for p in properties_data
            if p.get("Square Footage")
            and isinstance(p["Square Footage"], (int, float))
            and p["Square Footage"] >= min_sqft
        ]

    return properties_data


def main() -> None:
    """
    Main function to query properties by ZIP code and export to CSV.
    """
    parser = argparse.ArgumentParser(description="Query Portland property information by ZIP code.")
    parser.add_argument("--zip", required=True, help="ZIP code to query")
    parser.add_argument("--num", type=int, default=10, help="Number of properties to query (default: 10)")
    parser.add_argument("--output", default="properties.csv", help="Output CSV file (default: properties.csv)")
    parser.add_argument("--base-zone", action='append', help="Filter by base zone (can be used multiple times, e.g., --base-zone R5 --base-zone CE)")
    parser.add_argument("--min-sqft", type=int, help="Minimum square footage")
    args = parser.parse_args()

    zip_code = args.zip
    num_properties = args.num
    output_file = args.output

    print(f"Querying {num_properties} properties in ZIP {zip_code}...")

    properties_data = query_properties_by_zip(
        zip_code,
        num_properties,
        base_zones=args.base_zone,
        min_sqft=args.min_sqft,
    )

    if not properties_data:
        print("No properties found for this ZIP code (or none match the specified filters).")
        return

    # Write to CSV
    fieldnames = properties_data[0].keys()
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(properties_data)
    print(f"Data exported to {output_file} ({len(properties_data)} properties)")


if __name__ == "__main__":
    main()
