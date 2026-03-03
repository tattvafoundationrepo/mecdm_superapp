"""Map utility functions for building mecdm_viz map blocks from query results."""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ---- Color schemes (matching MVP) ----
BLUE_RAMP = {"minColor": "#dbeafe", "maxColor": "#1e40af"}

# Facility marker colors (matching MVP's FACILITY_LEAFLET)
FACILITY_COLORS = {
    "DH": "#d63e2a",
    "SDH": "#a23336",
    "CHC": "#f69730",
    "PHC": "#3b82f6",
    "SC": "#72b026",
    "DP": "#9b59b6",
}

# AWC marker colors
AWC_COLORS = {
    "Regular": {"color": "#0d9488", "fillColor": "#14b8a6"},
    "Mini": {"color": "#0e7490", "fillColor": "#06b6d4"},
}

# Default join keys per geography level
DEFAULT_JOIN_KEYS = {
    "district": "district_name",
    "block": "block_name",
    "village": "village_code_lgd",
}

# Geometry query templates
GEOMETRY_QUERIES = {
    "district": (
        'SELECT "district_name", "district_code_lgd", '
        'ST_AsGeoJSON("geom") as geojson FROM "districts"'
    ),
    "block": (
        'SELECT "block_name", "block_code_lgd", "district_name", '
        'ST_AsGeoJSON("geom") as geojson FROM "subdistricts"'
    ),
    "village": (
        'SELECT "village_name", "village_code_lgd", "block_name", "district_name", '
        'ST_Y("geom") as lat, ST_X("geom") as lng FROM "villages_point" '
        'WHERE "geom" IS NOT NULL'
    ),
}

FACILITY_QUERY = (
    'SELECT "facility_name", "facility_type", "block_name", "district_name", '
    'ST_Y("geom") as lat, ST_X("geom") as lng '
    'FROM "master_health_facilities" WHERE "geom" IS NOT NULL'
)

AWC_QUERY = (
    'SELECT "anganwadi_centre_name", "anganwadi_centre_type", '
    '"block_name", "district_name", '
    '"latitude" as lat, "longitude" as lng '
    'FROM "anganwadi_centres" '
    'WHERE "latitude" IS NOT NULL AND "longitude" IS NOT NULL'
)


def parse_query_results(raw_results) -> list[dict]:
    """Parse AlloyDB query results into a list of dicts.

    The MCP Toolbox returns results in various formats. This handles:
    - String results (JSON or text table)
    - List of dicts
    - Dict with query_result key
    """
    if isinstance(raw_results, list) and len(raw_results) > 0:
        if isinstance(raw_results[0], dict):
            return raw_results
        return raw_results

    if isinstance(raw_results, dict):
        if "query_result" in raw_results:
            return parse_query_results(raw_results["query_result"])
        return [raw_results]

    if isinstance(raw_results, str):
        # Try JSON parse
        try:
            parsed = json.loads(raw_results)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "query_result" in parsed:
                return parse_query_results(parsed["query_result"])
            return [parsed]
        except json.JSONDecodeError:
            pass

        # Try parsing text table format from MCP Toolbox
        rows = _parse_text_table(raw_results)
        if rows:
            return rows

    logger.warning("Could not parse query results: %s", type(raw_results))
    return []


def _parse_text_table(text: str) -> list[dict]:
    """Attempt to parse a text-formatted table into list of dicts."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return []

    # Try pipe-delimited format
    if "|" in lines[0]:
        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
        rows = []
        for line in lines[1:]:
            if set(line.replace("|", "").strip()) <= {"-", "+", "="}:
                continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) == len(headers):
                row = {}
                for h, c in zip(headers, cells):
                    row[h] = _try_numeric(c)
                rows.append(row)
        return rows

    return []


def _try_numeric(val: str):
    """Try to convert a string to int or float."""
    if val.lower() in ("null", "none", ""):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def compute_color_scale(values: list[float | int]) -> dict:
    """Compute min/max color scale from a list of numeric values."""
    numeric = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not numeric:
        return {"min": 0, "max": 1, **BLUE_RAMP}
    return {
        "min": min(numeric),
        "max": max(numeric),
        **BLUE_RAMP,
    }


def scale_radius(value, min_val, max_val, min_r=3, max_r=12) -> float:
    """Scale a value to a bubble radius between min_r and max_r."""
    if min_val == max_val:
        return (min_r + max_r) / 2
    ratio = max(0, min(1, (value - min_val) / (max_val - min_val)))
    return min_r + ratio * (max_r - min_r)


def build_geojson_features(
    metric_data: list[dict],
    geometry_data: list[dict],
    metric_col: str,
    join_key: str,
) -> list[dict]:
    """Join metric data with geometry data and build GeoJSON features.

    Args:
        metric_data: List of dicts from the metric query (e.g. [{district_name: X, total: 5}])
        geometry_data: List of dicts with geojson column from PostGIS
        metric_col: Name of the metric column to visualize
        join_key: Column name to join on (must exist in both datasets)

    Returns:
        List of GeoJSON Feature dicts
    """
    # Build lookup from geometry data keyed by join_key (case-insensitive for names)
    geo_lookup = {}
    for row in geometry_data:
        key = row.get(join_key)
        if key is not None:
            # Normalize string keys for matching
            normalized = str(key).strip().upper() if isinstance(key, str) else key
            geo_lookup[normalized] = row

    features = []
    for mrow in metric_data:
        key = mrow.get(join_key)
        if key is None:
            continue
        normalized = str(key).strip().upper() if isinstance(key, str) else key
        geo_row = geo_lookup.get(normalized)
        if geo_row is None:
            continue

        geojson_str = geo_row.get("geojson")
        if not geojson_str:
            continue

        try:
            geometry = json.loads(geojson_str) if isinstance(geojson_str, str) else geojson_str
        except json.JSONDecodeError:
            continue

        value = mrow.get(metric_col)
        if value is None:
            continue

        # Build properties: include the metric value and a display name
        name = mrow.get(join_key, "")
        properties = {"name": name, "value": value, metric_col: value}
        # Include any extra columns from metric data
        for k, v in mrow.items():
            if k != join_key and k not in properties:
                properties[k] = v

        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": properties,
        })

    logger.info(
        "Built %d GeoJSON features (from %d metric rows, %d geometry rows)",
        len(features), len(metric_data), len(geometry_data),
    )
    return features


def build_bubble_markers(
    metric_data: list[dict],
    geometry_data: list[dict],
    metric_col: str,
    join_key: str = "village_code_lgd",
) -> list[dict]:
    """Join metric data with village point geometry to build bubble markers.

    Returns:
        List of BubbleMarker dicts: {lat, lng, name, value, district, block}
    """
    # Build geo lookup
    geo_lookup = {}
    for row in geometry_data:
        key = row.get(join_key)
        if key is not None:
            normalized = str(key).strip().upper() if isinstance(key, str) else key
            geo_lookup[normalized] = row

    markers = []
    values = []
    for mrow in metric_data:
        key = mrow.get(join_key)
        if key is None:
            continue
        normalized = str(key).strip().upper() if isinstance(key, str) else key
        geo_row = geo_lookup.get(normalized)
        if geo_row is None:
            continue

        value = mrow.get(metric_col)
        if value is None or not isinstance(value, (int, float)):
            continue

        lat = geo_row.get("lat")
        lng = geo_row.get("lng")
        if lat is None or lng is None:
            continue

        values.append(value)
        markers.append({
            "lat": float(lat),
            "lng": float(lng),
            "name": geo_row.get("village_name", mrow.get("village_name", "")),
            "value": value,
            "district": geo_row.get("district_name", mrow.get("district_name", "")),
            "block": geo_row.get("block_name", mrow.get("block_name", "")),
        })

    logger.info(
        "Built %d bubble markers (from %d metric rows, %d geometry rows)",
        len(markers), len(metric_data), len(geometry_data),
    )
    return markers


def build_facility_overlay(facility_data: list[dict]) -> list[dict]:
    """Build facility overlay markers from query results.

    Returns:
        List of FacilityMarker dicts: {lat, lng, name, type, district, block}
    """
    markers = []
    for row in facility_data:
        lat = row.get("lat")
        lng = row.get("lng")
        if lat is None or lng is None:
            continue
        try:
            lat, lng = float(lat), float(lng)
        except (ValueError, TypeError):
            continue
        if lat == 0 and lng == 0:
            continue

        ftype = str(row.get("facility_type", "")).strip().upper()
        markers.append({
            "lat": lat,
            "lng": lng,
            "name": row.get("facility_name", ""),
            "type": ftype,
            "district": row.get("district_name", ""),
            "block": row.get("block_name", ""),
        })

    logger.info("Built %d facility overlay markers", len(markers))
    return markers


def build_awc_overlay(awc_data: list[dict]) -> list[dict]:
    """Build AWC overlay markers from query results.

    Returns:
        List of AwcMarker dicts: {lat, lng, name, awcType}
    """
    markers = []
    for row in awc_data:
        lat = row.get("lat")
        lng = row.get("lng")
        if lat is None or lng is None:
            continue
        try:
            lat, lng = float(lat), float(lng)
        except (ValueError, TypeError):
            continue
        if lat == 0 and lng == 0:
            continue

        markers.append({
            "lat": lat,
            "lng": lng,
            "name": row.get("anganwadi_centre_name", ""),
            "awcType": row.get("anganwadi_centre_type", "Regular"),
        })

    logger.info("Built %d AWC overlay markers", len(markers))
    return markers


def build_mecdm_viz_block(
    title: str,
    geography_level: str,
    metric_col: str,
    features: list[dict] | None = None,
    bubbles: list[dict] | None = None,
    facility_overlay: list[dict] | None = None,
    awc_overlay: list[dict] | None = None,
    color_scale: dict | None = None,
) -> dict:
    """Assemble a complete mecdm_viz map block."""
    if geography_level == "village":
        map_type = "bubble"
        values = [b["value"] for b in (bubbles or []) if b.get("value") is not None]
    else:
        map_type = "choropleth"
        values = [
            f["properties"].get("value")
            for f in (features or [])
            if f.get("properties", {}).get("value") is not None
        ]

    if color_scale is None:
        color_scale = compute_color_scale(values)

    block: dict = {
        "type": "map",
        "title": title,
        "mapType": map_type,
        "geographyLevel": geography_level,
        "center": [25.5, 91.0],
        "zoom": 8,
        "valueKey": "value",
        "colorScale": color_scale,
        "legend": {
            "title": metric_col.replace("_", " ").title(),
            "min": color_scale["min"],
            "max": color_scale["max"],
            "minColor": color_scale["minColor"],
            "maxColor": color_scale["maxColor"],
        },
    }

    if map_type == "choropleth" and features:
        block["features"] = features
    elif map_type == "bubble" and bubbles:
        block["bubbles"] = bubbles

    if facility_overlay:
        block["facilityOverlay"] = facility_overlay
    if awc_overlay:
        block["awcOverlay"] = awc_overlay

    return block


def format_viz_block_as_markdown(block: dict) -> str:
    """Format a mecdm_viz block as a markdown fenced code block."""
    return f"```mecdm_viz\n{json.dumps(block, default=str)}\n```"


def build_find_nearest_viz_block(
    origin: dict,
    nearest: list[dict],
    title: str = "Nearest Facilities",
) -> dict:
    """Build a mecdm_viz map block for find_nearest results.

    Args:
        origin: Dict with lat, lng, name for the origin village
        nearest: List of dicts with lat, lng, name, type, distance_km
        title: Map title

    Returns:
        mecdm_viz map block dict
    """
    markers = [
        {
            "lat": origin["lat"],
            "lng": origin["lng"],
            "label": f"📍 {origin['name']}",
            "color": "#3b82f6",
        }
    ]
    for item in nearest:
        color = FACILITY_COLORS.get(
            str(item.get("type", "")).upper(), "#6b7280"
        )
        markers.append({
            "lat": item["lat"],
            "lng": item["lng"],
            "label": f"{item.get('name', '')} ({item.get('type', '')})",
            "value": round(item.get("distance_km", 0), 1),
            "color": color,
        })

    distance_lines = []
    for item in nearest:
        distance_lines.append({
            "from": [origin["lat"], origin["lng"]],
            "to": [item["lat"], item["lng"]],
            "distance_km": round(item.get("distance_km", 0), 1),
            "label": f"{item.get('name', '')} - {round(item.get('distance_km', 0), 1)} km",
        })

    return {
        "type": "map",
        "title": title,
        "mapType": "markers",
        "center": [origin["lat"], origin["lng"]],
        "zoom": 10,
        "markers": markers,
        "distanceLines": distance_lines,
    }
