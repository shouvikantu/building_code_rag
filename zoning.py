"""
Portland Zoning and Building Information Query Tool

This script provides functionality to query zoning, building, and taxlot information
for addresses in Portland, Oregon using the PortlandMaps API.

Features:
- Geocode addresses to latitude/longitude
- Query zoning information (base zone, overlays, plan districts)
- Query building information (name, address, type, square footage, etc.)
- Query taxlot/parcel information

Usage:
    python zoning.py

"""

import requests
from geopy.geocoders import Nominatim
from typing import Dict, Optional, Tuple

# API URLs
ZONING_QUERY_URL = "https://www.portlandmaps.com/od/rest/services/COP_OpenData_ZoningCode/MapServer/16/query"
BUILDING_QUERY_URL = "https://www.portlandmaps.com/od/rest/services/COP_OpenData_Property/MapServer/184/query"
TAXLOT_QUERY_URLS = [
    "https://www.portlandmaps.com/od/rest/services/COP_OpenData_Property/MapServer/1272/query",
    "https://www.portlandmaps.com/od/rest/services/COP_OpenData_Property/MapServer/47/query",
]


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
    geolocator = Nominatim(user_agent="portland-zoning-lookup")
    location = geolocator.geocode(address)
    if location is None:
        raise ValueError("Address could not be geocoded")
    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "matched_address": location.address
    }


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
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "where": "1=1",
        "f": "json"
    }
    resp = requests.get(ZONING_QUERY_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    if not features:
        raise ValueError("No zoning data found for this location")
    return features[0]


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


def query_building(lat: float, lon: float) -> Optional[Dict]:
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
        resp = requests.get(BUILDING_QUERY_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        if features:
            return features[0]["attributes"]
    except Exception:
        pass
    return None


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


def query_taxlot(lat: float, lon: float) -> Optional[Dict]:
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
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            if features:
                return features[0]["attributes"]
        except Exception:
            continue
    return None


def main() -> None:
    """
    Main function to demonstrate the tool with an example address.
    """
    # Example usage: get zoning and building info for a Portland address
    address = "935 NE 33RD AVE, Portland, OR"

    try:
        # Get zoning info for the address
        result = get_zoning_for_address(address)
        print(format_zoning_result(result))

        # Extract lat/lon for further queries
        lat = result["location"]["lat"]
        lon = result["location"]["lon"]

        # Query and print building info
        building_attrs = query_building(lat, lon)
        print_building_info(building_attrs)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

