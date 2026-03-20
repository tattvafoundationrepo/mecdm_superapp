"""Tools for the ADK Samples Data Science Agent."""

import json
import logging

from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .sub_agents import alloydb_agent
from .sub_agents.alloydb.tools import get_toolbox_client

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

    from .sub_agents import analytics_agent

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



async def find_nearest_facilities(
    from_village: str,
    to_type: str = "ANY_FACILITY",
    count: int = 5,
    from_district: str = "",
    from_block: str = "",
) -> str:
    """Find the nearest health facilities or AWCs from a specified village.

    Returns a lightweight mecdm_map config block. The frontend handles the spatial
    query via /api/map/nearest and renders the map with markers and distance lines.

    Args:
        from_village: Name of the origin village (fuzzy matched).
        to_type: Target facility type — 'PHC', 'SC', 'CHC', 'DH', 'SDH', 'DP',
            'AWC', 'ANY_FACILITY' (all health facilities), or 'ANY' (facilities + AWCs).
        count: Number of nearest to return (1-20, default 5).
        from_district: Optional district name to disambiguate the village.
        from_block: Optional block name to disambiguate the village.

    Returns:
        A mecdm_map config block for the frontend to execute and render.
    """
    logger.info(
        "[find_nearest] village=%s type=%s count=%d district=%s block=%s",
        from_village, to_type, count, from_district, from_block,
    )

    count = max(1, min(20, count))
    to_type_upper = to_type.strip().upper()

    payload = {
        "nearestQuery": {
            "fromVillage": from_village,
            "toType": to_type_upper,
            "count": count,
        },
        "map": {
            "mapType": "markers",
            "geographyLevel": "village",
            "metricColumn": "distance_km",
            "joinKey": "village_name",
        },
        "title": f"Nearest {to_type_upper} to {from_village}",
    }

    if from_district:
        payload["nearestQuery"]["fromDistrict"] = from_district
    if from_block:
        payload["nearestQuery"]["fromBlock"] = from_block

    map_block = f"```mecdm_map\n{json.dumps(payload)}\n```"

    return f"Include this map in your response:\n\n{map_block}"


STATS_ALLOWLISTED_TABLES = [
    "village_indicators_monthly",
    "mother_journeys",
    "anc_visits",
    "master_districts",
    "master_blocks",
    "master_health_facilities",
    "anganwadi_centres",
    "nfhs_indicators",
]

PREDEFINED_STATS_CATALOG = [
    {"id": "total-registrations", "name": "Total Registrations", "category": "maternal-health", "description": "Total maternal registrations across all districts"},
    {"id": "institutional-delivery-rate", "name": "Institutional Delivery Rate", "category": "maternal-health", "description": "Percentage of deliveries at health institutions"},
    {"id": "health-facilities-count", "name": "Health Facilities", "category": "infrastructure", "description": "Total health facilities across all districts"},
    {"id": "awc-count", "name": "Anganwadi Centres", "category": "infrastructure", "description": "Total Anganwadi centres across all districts"},
    {"id": "district-registrations", "name": "District-wise Registrations", "category": "maternal-health", "description": "Total maternal registrations by district (bar chart)"},
    {"id": "monthly-registrations-trend", "name": "Monthly Registration & Delivery Trends", "category": "maternal-health", "description": "Monthly trends of registrations and deliveries (area chart)"},
    {"id": "facility-type-distribution", "name": "Facility Type Distribution", "category": "infrastructure", "description": "Health facility count by type (pie chart)"},
    {"id": "block-delivery-rate", "name": "Top 20 Blocks: Institutional Deliveries", "category": "maternal-health", "description": "Blocks with highest institutional deliveries (bar chart)"},
    {"id": "maternal-deaths-district", "name": "Maternal Deaths by District", "category": "maternal-health", "description": "Total reported maternal deaths by district (bar chart)"},
    {"id": "monthly-anc-coverage", "name": "Monthly ANC Coverage", "category": "maternal-health", "description": "Monthly ANC visits, IFA recipients, and TT doses (line chart)"},
]

_NUMERIC_TYPES = {"bigint", "integer", "smallint", "numeric", "double precision", "real"}
_META_COLUMNS = {"created_at", "updated_at", "objectid", "geom", "geometry"}
_GEOMETRY_TYPES = {"geometry", "geography", "USER-DEFINED"}


def _infer_column_role(col_name: str, data_type: str) -> str | None:
    """Infer the stats role for a column. Returns None if the column should be skipped."""
    name_lower = col_name.lower()
    type_lower = data_type.lower()

    if name_lower in _META_COLUMNS or type_lower in _GEOMETRY_TYPES:
        return None  # skip
    if name_lower.endswith(("_id", "_code", "_code_lgd")):
        return "identifier"
    if name_lower in ("year_month",) or name_lower.endswith("_date"):
        return "timestamp"
    if type_lower in _NUMERIC_TYPES:
        return "measure"
    return "dimension"


async def get_stats_schema_summary(tool_context: ToolContext) -> str:
    """Returns a summary of queryable tables and their columns for generating StatQuery objects.

    Use this tool when you need to construct a mecdm_stat block. It returns the available
    tables, their columns (with data types and roles like dimension/measure/timestamp),
    and which tables are eligible for the Stats API.

    Returns:
        JSON string with table schemas in StatQuery-compatible format.
    """
    try:
        list_tables_tool = get_toolbox_client().load_tool("list_tables")
        table_names = ",".join(STATS_ALLOWLISTED_TABLES)
        raw_schema = list_tables_tool(schema_names="public", table_names=table_names)

        # Parse the raw schema output into a structured format
        # The list_tables tool returns JSON with table details
        if isinstance(raw_schema, str):
            import json as _json
            try:
                tables_data = _json.loads(raw_schema)
            except _json.JSONDecodeError:
                # Raw text output — return it with instructions
                return f"Stats-eligible tables: {table_names}\n\nRaw schema:\n{raw_schema[:4000]}"
        else:
            tables_data = raw_schema

        # Build condensed schema summary
        summary = {}
        if isinstance(tables_data, list):
            for table_info in tables_data:
                tname = table_info.get("table_name") or table_info.get("name", "")
                if tname not in STATS_ALLOWLISTED_TABLES:
                    continue
                columns = []
                for col in table_info.get("columns", []):
                    col_name = col.get("column_name") or col.get("name", "")
                    col_type = col.get("data_type") or col.get("type", "")
                    role = _infer_column_role(col_name, col_type)
                    if role:
                        columns.append({"name": col_name, "type": col_type, "role": role})
                if columns:
                    summary[tname] = {"columns": columns}
        elif isinstance(tables_data, dict):
            # Handle dict format where keys are table names
            for tname, tinfo in tables_data.items():
                if tname not in STATS_ALLOWLISTED_TABLES:
                    continue
                columns = []
                for col in (tinfo.get("columns", []) if isinstance(tinfo, dict) else []):
                    col_name = col.get("column_name") or col.get("name", "")
                    col_type = col.get("data_type") or col.get("type", "")
                    role = _infer_column_role(col_name, col_type)
                    if role:
                        columns.append({"name": col_name, "type": col_type, "role": role})
                if columns:
                    summary[tname] = {"columns": columns}

        if not summary:
            # Fallback: return table names and raw output
            return f"Stats-eligible tables: {table_names}\n\nSchema data format not recognized. Raw:\n{str(tables_data)[:3000]}"

        return json.dumps(summary, indent=2)

    except Exception as e:
        logger.error("get_stats_schema_summary failed: %s", e)
        return f"Error fetching stats schema: {e}"


async def get_predefined_stats_catalog(tool_context: ToolContext) -> str:
    """Returns the catalog of predefined statistics available in the dashboard.

    Use this when a user asks about available stats, KPIs, or dashboard metrics, or
    when the user's question matches a predefined stat. You can reference these by ID
    in a mecdm_stat block using {"predefined_id": "<id>"} instead of building a new query.

    Returns:
        JSON string listing predefined stat IDs, names, descriptions, and categories.
    """
    return json.dumps(PREDEFINED_STATS_CATALOG, indent=2)


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
