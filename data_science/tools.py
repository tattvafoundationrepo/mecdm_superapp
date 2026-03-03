# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for the ADK Samples Data Science Agent."""

import json
import logging

from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .sub_agents import alloydb_agent, analytics_agent
from .utils.map_utils import (
    FACILITY_QUERY,
    AWC_QUERY,
    GEOMETRY_QUERIES,
    DEFAULT_JOIN_KEYS,
    build_awc_overlay,
    build_bubble_markers,
    build_facility_overlay,
    build_find_nearest_viz_block,
    build_geojson_features,
    build_mecdm_viz_block,
    compute_color_scale,
    format_viz_block_as_markdown,
    parse_query_results,
)

logger = logging.getLogger(__name__)


async def call_alloydb_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call alloydb database (nl2sql) agent."""
    logger.debug("call_alloydb_agent: %s", question)

    agent_tool = AgentTool(agent=alloydb_agent)

    alloydb_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    tool_context.state["alloydb_agent_output"] = alloydb_agent_output
    return alloydb_agent_output


async def call_analytics_agent(
    question: str,
    tool_context: ToolContext,
):
    """
    This tool can generate Python code to process and analyze a dataset.

    Some of the tasks it can do in Python include:
    * Creating graphics for data visualization;
    * Processing or filtering existing datasets;
    * Combining datasets to create a joined dataset for further analysis.

    The Python modules available to it are:
    * io
    * math
    * re
    * matplotlib.pyplot
    * numpy
    * pandas

    The tool DOES NOT have the ability to retrieve additional data from
    a database. Only the data already retrieved will be analyzed.

    Args:
        question (str): Natural language question or analytics request.
        tool_context (ToolContext): The tool context to use for generating the
            SQL query.

    Returns:
        Response from the analytics agent.

    """
    logger.debug("call_analytics_agent: %s", question)

    # if question == "N/A":
    #    return tool_context.state["db_agent_output"]

    if "alloydb_query_result" in tool_context.state:
        alloydb_data = tool_context.state["alloydb_query_result"]

    question_with_data = f"""
  Question to answer: {question}

  Actual data to analyze this question is available in the following data
  tables:

  <ALLOYDB>
  {alloydb_data}
  </ALLOYDB>

  IMPORTANT: Return your computed results as JSON data arrays using print(json.dumps(...)).
  Do NOT generate matplotlib plots or images. The frontend will render charts from your JSON data.
  """

    agent_tool = AgentTool(agent=analytics_agent)

    analytics_agent_output = await agent_tool.run_async(
        args={"request": question_with_data}, tool_context=tool_context
    )
    tool_context.state["analytics_agent_output"] = analytics_agent_output
    return analytics_agent_output


async def get_current_datetime(tool_context: ToolContext) -> str:
    """Returns the current date and time. Use this to grounding relative time queries like 'today' or 'last month'."""
    from datetime import datetime
    return f"The current date and time is: {datetime.now().isoformat()}"


async def get_weather_data(location: str, tool_context: ToolContext) -> str:
    """Fetches the current weather for a given location using the free Open-Meteo API.

    Args:
        location: The name of the city or location (e.g., 'Shillong').

    Returns:
        A string describing the current weather conditions.
    """
    import requests

    # Simple geocoding using Open-Meteo's geocoding API
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
    try:
        geo_response = requests.get(geocode_url)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find coordinates for location: {location}"

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        # Fetch weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code&timezone=auto"
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        precip = current.get("precipitation", "N/A")

        return f"Current weather in {location}: Temperature {temp}°C, Humidity {humidity}%, Precipitation {precip}mm."
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return f"Error fetching weather data: {e}"


async def get_historical_weather_data(location: str, start_date: str, end_date: str, tool_context: ToolContext) -> str:
    """Fetches historical weather data using the Open-Meteo Archive API.

    Args:
        location: The name of the city or location (e.g., 'Shillong').
        start_date: Start date in YYYY-MM-DD format (e.g., '2023-11-01').
        end_date: End date in YYYY-MM-DD format (e.g., '2023-11-30').

    Returns:
        A string summarizing the historical weather data for that period.
    """
    import requests

    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
    try:
        geo_response = requests.get(geocode_url)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find coordinates for location: {location}"

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        # Open-Meteo Archive API URL
        archive_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
        weather_response = requests.get(archive_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        daily = weather_data.get("daily", {})
        if not daily:
            return f"No historical data available for {location} between {start_date} and {end_date}."

        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        precips = daily.get("precipitation_sum", [])

        # Calculate some basic aggregates to summarize
        valid_max = [t for t in max_temps if t is not None]
        valid_min = [t for t in min_temps if t is not None]
        valid_precip = [p for p in precips if p is not None]

        avg_max = round(sum(valid_max) / len(valid_max),
                        2) if valid_max else "N/A"
        avg_min = round(sum(valid_min) / len(valid_min),
                        2) if valid_min else "N/A"
        total_precip = round(sum(valid_precip), 2) if valid_precip else "N/A"

        return f"Historical weather for {location} from {start_date} to {end_date}:\\nAverage Max Temp: {avg_max}°C\\nAverage Min Temp: {avg_min}°C\\nTotal Precipitation: {total_precip}mm."

    except Exception as e:
        logger.error(f"Error fetching historical weather data: {e}")
        return f"Error fetching historical weather data: {e}"


async def export_data_to_csv(data_json: str, filename: str, tool_context: ToolContext) -> str:
    """Exports a JSON string representing a list of records to a CSV file.

    Args:
        data_json: A JSON-formatted string, specifically a list of dictionaries (records).
        filename: The desired name of the output CSV file (e.g., 'report.csv').

    Returns:
        A success message with the file path, or an error message.
    """
    import csv
    import json
    import os

    try:
        data = json.loads(data_json)
        if not data or not isinstance(data, list):
            return "Error: Data must be a JSON array of objects."

        # Write to static folder to make it accessible if needed
        output_dir = "static/exports"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if len(data) > 0:
                keys = data[0].keys()
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(data)

        return f"Successfully exported data to {filepath}"
    except Exception as e:
        logger.error(f"Error exporting data to CSV: {e}")
        return f"Error exporting data: {e}"


async def _run_alloydb_query(question: str, tool_context: ToolContext) -> str:
    """Internal helper to run a query via the AlloyDB sub-agent."""
    agent_tool = AgentTool(agent=alloydb_agent)
    output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    return output


async def generate_map_viz(
    geography_level: str,
    metric_col: str,
    title: str,
    join_key: str = "",
    overlay_facilities: bool = False,
    overlay_awc: bool = False,
    tool_context: ToolContext = None,
) -> str:
    """Generate an interactive map visualization from previously retrieved data.

    Call this AFTER call_alloydb_agent has retrieved metric data. This tool reads the
    query results from state, fetches PostGIS geometry via the AlloyDB agent, joins
    the data, and returns a mecdm_viz map block for the frontend to render.

    Supports district choropleth (12 regions), block choropleth (46 regions), and
    village bubble maps (~2,600 matchable points sized/colored by metric).

    Args:
        geography_level: Geographic level — 'district', 'block', or 'village'.
        metric_col: Column name from the alloydb query result to visualize (must be numeric).
        title: Short descriptive map title.
        join_key: Column to join metric data with geometry. Defaults: district_name,
            block_name, or village_code_lgd based on geography_level.
        overlay_facilities: Set True to overlay health facility locations (DH, CHC, PHC, SC)
            as colored markers with layer toggle controls.
        overlay_awc: Set True to overlay Anganwadi Centre locations as teal circle markers.
        tool_context: ADK tool context (injected automatically).

    Returns:
        A mecdm_viz JSON block string wrapped in markdown fences for the frontend to render.
    """
    logger.info(
        "[generate_map_viz] level=%s metric=%s title=%s overlays=(fac=%s, awc=%s)",
        geography_level, metric_col, title, overlay_facilities, overlay_awc,
    )

    if geography_level not in ("district", "block", "village"):
        return "Error: geography_level must be 'district', 'block', or 'village'."

    # 1. Read metric data from state (set by prior call_alloydb_agent)
    raw_results = tool_context.state.get("alloydb_query_result")
    if not raw_results:
        return (
            "Error: No query results found in state. "
            "Call call_alloydb_agent first to retrieve metric data, "
            "then call generate_map_viz."
        )

    metric_data = parse_query_results(raw_results)
    if not metric_data:
        return "Error: Could not parse metric data from previous query results."

    # Determine join key
    jk = join_key if join_key else DEFAULT_JOIN_KEYS.get(geography_level, "")
    if not jk:
        return f"Error: No join_key specified and no default for geography_level={geography_level}."

    # Validate metric_col exists
    if metric_data and metric_col not in metric_data[0]:
        available = ", ".join(metric_data[0].keys())
        return f"Error: metric_col '{metric_col}' not found in query results. Available columns: {available}"

    # 2. Fetch geometry via AlloyDB sub-agent
    geo_query = GEOMETRY_QUERIES.get(geography_level)
    if not geo_query:
        return f"Error: No geometry query defined for geography_level={geography_level}."

    logger.info("[generate_map_viz] Fetching geometry: %s", geo_query[:100])
    geo_output = await _run_alloydb_query(
        f"Execute this exact SQL query and return all results: {geo_query}",
        tool_context,
    )

    geo_data = parse_query_results(
        tool_context.state.get("alloydb_query_result", geo_output)
    )
    if not geo_data:
        return "Error: Could not fetch geometry data from PostGIS."

    logger.info("[generate_map_viz] Got %d geometry rows", len(geo_data))

    # 3. Build map features
    features = None
    bubbles = None

    if geography_level in ("district", "block"):
        features = build_geojson_features(metric_data, geo_data, metric_col, jk)
        if not features:
            return (
                f"Error: No features could be built. Check that '{jk}' exists in both "
                "metric data and geometry data, and that values match."
            )
    else:  # village
        bubbles = build_bubble_markers(metric_data, geo_data, metric_col, jk)
        if not bubbles:
            return (
                "Error: No village points could be matched. Check that village_code_lgd "
                "values in the metric data match the villages_point table."
            )

    # 4. Optionally fetch facility overlay
    facility_overlay = None
    if overlay_facilities:
        logger.info("[generate_map_viz] Fetching facility overlay")
        await _run_alloydb_query(
            f"Execute this exact SQL query and return all results: {FACILITY_QUERY}",
            tool_context,
        )
        fac_data = parse_query_results(
            tool_context.state.get("alloydb_query_result", "")
        )
        if fac_data:
            facility_overlay = build_facility_overlay(fac_data)

    # 5. Optionally fetch AWC overlay
    awc_overlay = None
    if overlay_awc:
        logger.info("[generate_map_viz] Fetching AWC overlay")
        await _run_alloydb_query(
            f"Execute this exact SQL query and return all results: {AWC_QUERY}",
            tool_context,
        )
        awc_data = parse_query_results(
            tool_context.state.get("alloydb_query_result", "")
        )
        if awc_data:
            awc_overlay = build_awc_overlay(awc_data)

    # 6. Build the mecdm_viz block
    viz_block = build_mecdm_viz_block(
        title=title,
        geography_level=geography_level,
        metric_col=metric_col,
        features=features,
        bubbles=bubbles,
        facility_overlay=facility_overlay,
        awc_overlay=awc_overlay,
    )

    # Build summary
    n_items = len(features) if features else len(bubbles) if bubbles else 0
    summary_parts = [
        f"MAP GENERATED: '{title}'",
        f"{n_items} {geography_level}s with data",
    ]
    if facility_overlay:
        summary_parts.append(f"{len(facility_overlay)} health facilities overlaid")
    if awc_overlay:
        summary_parts.append(f"{len(awc_overlay)} AWCs overlaid")

    summary = " | ".join(summary_parts)
    viz_markdown = format_viz_block_as_markdown(viz_block)

    logger.info("[generate_map_viz] %s", summary)

    return f"{summary}\n\nInclude this visualization block in your response:\n\n{viz_markdown}"


async def find_nearest_facilities(
    from_village: str,
    to_type: str = "ANY_FACILITY",
    count: int = 5,
    from_district: str = "",
    from_block: str = "",
    show_map: bool = True,
    tool_context: ToolContext = None,
) -> str:
    """Find the nearest health facilities or AWCs from a specified village.

    Uses PostGIS spatial distance to find the N closest facilities/AWCs to a village.
    Optionally displays results on an interactive map with distance lines.

    Args:
        from_village: Name of the origin village (fuzzy matched).
        to_type: Target facility type — 'PHC', 'SC', 'CHC', 'DH', 'SDH', 'DP',
            'AWC', 'ANY_FACILITY' (all health facilities), or 'ANY' (facilities + AWCs).
        count: Number of nearest to return (1-20, default 5).
        from_district: Optional district name to disambiguate the village.
        from_block: Optional block name to disambiguate the village.
        show_map: If True, returns a mecdm_viz map block with markers and distance lines.
        tool_context: ADK tool context (injected automatically).

    Returns:
        Ranked list of nearest facilities with distances, plus optional map visualization.
    """
    logger.info(
        "[find_nearest] village=%s type=%s count=%d district=%s block=%s",
        from_village, to_type, count, from_district, from_block,
    )

    count = max(1, min(20, count))

    # 1. Find the origin village
    village_filters = [f"\"village_name\" ILIKE '%{from_village}%'"]
    if from_district:
        village_filters.append(f"\"district_name\" ILIKE '%{from_district}%'")
    if from_block:
        village_filters.append(f"\"block_name\" ILIKE '%{from_block}%'")

    village_where = " AND ".join(village_filters)
    village_query = (
        f'SELECT "village_name", "village_code_lgd", "block_name", "district_name", '
        f'ST_Y("geom") as lat, ST_X("geom") as lng '
        f'FROM "villages_point" WHERE {village_where} AND "geom" IS NOT NULL LIMIT 5'
    )

    await _run_alloydb_query(
        f"Execute this exact SQL query: {village_query}", tool_context,
    )
    village_data = parse_query_results(
        tool_context.state.get("alloydb_query_result", "")
    )

    if not village_data:
        return f"Could not find village matching '{from_village}'. Try a different spelling or add district/block filters."

    origin = village_data[0]
    origin_lat = origin.get("lat")
    origin_lng = origin.get("lng")
    if origin_lat is None or origin_lng is None:
        return f"Village '{from_village}' found but has no coordinates."

    logger.info(
        "[find_nearest] Origin: %s (%s, %s)",
        origin.get("village_name"), origin_lat, origin_lng,
    )

    # 2. Build spatial query for nearest facilities
    to_type_upper = to_type.strip().upper()

    if to_type_upper == "AWC":
        target_query = (
            f'SELECT "anganwadi_centre_name" as facility_name, '
            f'"anganwadi_centre_type" as facility_type, '
            f'"block_name", "district_name", '
            f'"latitude" as lat, "longitude" as lng, '
            f'ST_Distance('
            f'  ST_SetSRID(ST_MakePoint("longitude", "latitude"), 4326)::geography, '
            f"  ST_SetSRID(ST_MakePoint({origin_lng}, {origin_lat}), 4326)::geography"
            f') / 1000.0 as distance_km '
            f'FROM "anganwadi_centres" '
            f'WHERE "latitude" IS NOT NULL AND "longitude" IS NOT NULL '
            f'ORDER BY distance_km LIMIT {count}'
        )
    else:
        type_filter = ""
        if to_type_upper not in ("ANY_FACILITY", "ANY"):
            type_filter = f" AND UPPER(TRIM(\"facility_type\")) = '{to_type_upper}'"

        target_query = (
            f'SELECT "facility_name", "facility_type", '
            f'"block_name", "district_name", '
            f'ST_Y("geom") as lat, ST_X("geom") as lng, '
            f'ST_Distance('
            f'  "geom"::geography, '
            f"  ST_SetSRID(ST_MakePoint({origin_lng}, {origin_lat}), 4326)::geography"
            f') / 1000.0 as distance_km '
            f'FROM "master_health_facilities" '
            f'WHERE "geom" IS NOT NULL{type_filter} '
            f'ORDER BY distance_km LIMIT {count}'
        )

    await _run_alloydb_query(
        f"Execute this exact SQL query: {target_query}", tool_context,
    )
    nearest_data = parse_query_results(
        tool_context.state.get("alloydb_query_result", "")
    )

    if not nearest_data:
        return f"No {to_type} facilities found near '{from_village}'."

    # 3. Build text result table
    result_lines = [
        f"## Nearest {to_type} to {origin.get('village_name', from_village)}",
        f"District: {origin.get('district_name', 'N/A')} | Block: {origin.get('block_name', 'N/A')}",
        "",
        "| Rank | Name | Type | Block | Distance (km) |",
        "|------|------|------|-------|---------------|",
    ]
    for i, item in enumerate(nearest_data, 1):
        dist = round(float(item.get("distance_km", 0)), 1)
        result_lines.append(
            f"| {i} | {item.get('facility_name', 'N/A')} | "
            f"{item.get('facility_type', 'N/A')} | "
            f"{item.get('block_name', 'N/A')} | {dist} |"
        )

    result_text = "\n".join(result_lines)

    # 4. Build map visualization if requested
    if show_map:
        viz_block = build_find_nearest_viz_block(
            origin={
                "lat": float(origin_lat),
                "lng": float(origin_lng),
                "name": origin.get("village_name", from_village),
            },
            nearest=[
                {
                    "lat": float(item.get("lat", 0)),
                    "lng": float(item.get("lng", 0)),
                    "name": item.get("facility_name", ""),
                    "type": str(item.get("facility_type", "")).upper(),
                    "distance_km": float(item.get("distance_km", 0)),
                }
                for item in nearest_data
                if item.get("lat") is not None and item.get("lng") is not None
            ],
            title=f"Nearest {to_type} to {origin.get('village_name', from_village)}",
        )
        viz_markdown = format_viz_block_as_markdown(viz_block)
        result_text += f"\n\nInclude this map in your response:\n\n{viz_markdown}"

    logger.info("[find_nearest] Found %d results", len(nearest_data))
    return result_text


async def search_policy_rag_engine(query: str, tool_context: ToolContext) -> str:
    """Searches the Meghalaya Government Policy Intelligence Engine (Vertex AI RAG API) for policy details.

    Args:
        query: The user's question or search terms regarding policy guidelines.

    Returns:
        The search results or summarized response from the RAG engine.
    """
    import os
    import vertexai
    from vertexai.preview import rag

    project_id = os.getenv("MECDM_POLICY_PROJECT_ID")
    # rag has specific regional availability
    # Based on user's region
    location = os.getenv("MECDM_POLICY_LOCATION", "asia-south1")
    corpus_id = os.getenv("MECDM_POLICY_CORPUS_ID")

    if not all([project_id, corpus_id]):
        return "Error: RAG Engine configuration (MECDM_POLICY_PROJECT_ID, MECDM_POLICY_CORPUS_ID) is missing in the environment."

    try:
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        # Configure the RAG Retrieval
        corpus_name = f"projects/{project_id}/locations/{location}/ragCorpora/{corpus_id}"

        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_name,
                )
            ],
            text=query,
            similarity_top_k=3,  # Optional
            vector_distance_threshold=0.5,  # Optional
        )

        results = []
        if hasattr(response, 'contexts') and hasattr(response.contexts, 'contexts'):
            for context in response.contexts.contexts:
                # context.text contains the snippet
                snippet = context.text if hasattr(
                    context, 'text') else str(context)

                # Try to extract the source filename if available
                source = "Unknown Source"
                if hasattr(context, 'source_uri'):
                    source = context.source_uri
                elif hasattr(context, 'source_display_name'):
                    source = context.source_display_name

                results.append(f"[{source}]\\n{snippet}")

        if not results:
            return "No relevant policy documents found for the query."

        # Combine the top snippets
        combined_results = "\\n\\n---\\n\\n".join(results)
        return f"Search Results from Policy Engine:\\n\\n{combined_results}"

    except Exception as e:
        logger.error(f"Error querying RAG engine: {e}")
        return f"Error querying policy engine: {e}"
