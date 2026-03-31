"""Tools for the ADK Samples Data Science Agent."""

import json
import logging
import os

from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .app_utils.db import execute_readonly_sql
from .app_utils.llm_client import get_llm_client
from .app_utils.stat_query_compiler import compile_stat_query_v2_to_sql
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

    from .sub_agents.analytics.agent import analytics_agent

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


def _get_stats_blocked_tables() -> set[str]:
    """Get blocked tables from dataset config."""
    from .prompts.prompt_builder import load_dataset_config

    return set(load_dataset_config().stats_blocked_tables)


STATS_BLOCKED_TABLES = _get_stats_blocked_tables()

# Cached list of available (non-blocked) tables, populated on first call to MCP
_stats_available_tables: list[str] | None = None


def get_stats_available_tables() -> list[str]:
    """Fetch all tables from MCP Toolbox, exclude blocked ones, cache the result."""
    global _stats_available_tables
    if _stats_available_tables is not None:
        return _stats_available_tables

    try:
        list_tables_tool = get_toolbox_client().load_tool("list_tables")
        raw_schema = list_tables_tool(schema_names="public", table_names="")

        all_tables: list[str] = []
        if isinstance(raw_schema, str):
            import json as _json
            try:
                tables_data = _json.loads(raw_schema)
            except _json.JSONDecodeError:
                tables_data = []
        else:
            tables_data = raw_schema

        if isinstance(tables_data, list):
            for t in tables_data:
                name = t.get("table_name") or t.get("name", "")
                if name:
                    all_tables.append(name)
        elif isinstance(tables_data, dict):
            all_tables = list(tables_data.keys())

        _stats_available_tables = [
            t for t in all_tables if t not in STATS_BLOCKED_TABLES
        ]
        logger.info(
            "Discovered %d stats-available tables (blocked %d)",
            len(_stats_available_tables),
            len(all_tables) - len(_stats_available_tables),
        )
        return _stats_available_tables

    except Exception as e:
        logger.error("Failed to fetch tables from MCP: %s", e)
        return []

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
        available_tables = get_stats_available_tables()
        if not available_tables:
            return "Error: No stats-available tables found. Check MCP Toolbox connection."

        list_tables_tool = get_toolbox_client().load_tool("list_tables")
        table_names_str = ",".join(available_tables)
        raw_schema = list_tables_tool(schema_names="public", table_names=table_names_str)

        # Parse the raw schema output into a structured format
        if isinstance(raw_schema, str):
            import json as _json
            try:
                tables_data = _json.loads(raw_schema)
            except _json.JSONDecodeError:
                return f"Available tables: {table_names_str}\n\nRaw schema:\n{raw_schema[:4000]}"
        else:
            tables_data = raw_schema

        available_set = set(available_tables)

        # Build condensed schema summary
        summary = {}
        if isinstance(tables_data, list):
            for table_info in tables_data:
                tname = table_info.get("table_name") or table_info.get("name", "")
                if tname not in available_set:
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
            for tname, tinfo in tables_data.items():
                if tname not in available_set:
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
            return f"Available tables: {table_names_str}\n\nSchema data format not recognized. Raw:\n{str(tables_data)[:3000]}"

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


# =============================================================================
# StatQuery V2 generation tool
# =============================================================================


_STAT_QUERY_V2_PROMPT = """Generate a StatQuery V2 JSON object for this health data question.

Available tables and their columns (choose the best table for the question):
{SCHEMA}

StatQuery V2 format:
{{
  "version": 2,
  "source": {{ "table": "<table_name>", "joins": [{{ "table": "<t>", "on": {{ "left": "<col>", "right": "<col>" }}, "type": "inner"|"left"|"right", "caseInsensitive": true|false }}] }},
  "dimensions": [{{ "column": "<col>", "alias": "<display_name>", "transform": "date_trunc_month"|"date_trunc_quarter"|"date_trunc_year" }}],
  "measures": [{{ "column": "<col>", "aggregate": "count"|"sum"|"avg"|"min"|"max"|"count_distinct", "alias": "<name>" }}],
  "computedColumns": [{{ "alias": "<name>", "expression": "<safe expression referencing measure aliases>" }}],
  "windows": [{{ "alias": "<name>", "function": "row_number"|"rank"|"dense_rank"|"lag"|"lead"|"sum"|"avg"|"count", "column": "<col>", "partitionBy": ["<col>"], "orderBy": [{{ "column": "<alias>", "direction": "asc"|"desc" }}], "offset": 1 }}],
  "filters": [{{ "column": "<col>", "operator": "eq"|"neq"|"gt"|"gte"|"lt"|"lte"|"in"|"not_in"|"like"|"is_null"|"is_not_null"|"between", "value": <val> }}],
  "having": [{{ "column": "<measure_alias>", "operator": "gt"|"gte"|"lt"|"lte"|"eq"|"neq"|"between", "value": <val> }}],
  "orderBy": [{{ "column": "<alias>", "direction": "asc"|"desc" }}],
  "limit": 1000,
  "timeRange": {{ "column": "<col>", "preset": "last_7d"|"last_30d"|"last_quarter"|"last_year"|"ytd"|"all" }}
}}

Rules:
- version MUST be 2 (integer, not string)
- source.table must be one of the tables shown in the schema above
- dimensions/measures columns must exist in the table schema
- computedColumns expressions can ONLY reference measure/dimension aliases, arithmetic (+,-,*,/), numeric literals, COALESCE, NULLIF, ROUND, CEIL, FLOOR, ABS, GREATEST, LEAST, CAST, and CASE WHEN
- having filters apply to measure aliases (post-aggregation)
- windows: use rank/dense_rank for rankings, lag/lead for period-over-period, sum/avg/count for running aggregates
- For year_month TEXT columns, do NOT use date_trunc transforms — use custom timeRange with "YYYY-MM" strings
- In mother_journeys/anc_visits, ALL columns are TEXT — cast with CAST(col AS numeric) in computedColumns if needed
- District names in mother_journeys/anc_visits are UPPERCASE (e.g., 'EAST KHASI HILLS')
- Use integer code joins (district_code_lgd, block_code_lgd) over name joins
- IMPORTANT: When joining by name across tables with different casing (UPPERCASE vs Title Case), set "caseInsensitive": true on the join. This applies to ANY join between village_indicators_monthly/mother_journeys (UPPERCASE) and nfhs_indicators/master_*/anganwadi_centres (Title Case).
- anganwadi_centres.block_code is NOT the same as block_code_lgd. Use district_code for district-level joins or name joins with caseInsensitive: true.

Common patterns (note: expressions reference measure ALIASES, not raw column names):
- IDR: measures: [{{"column":"institutional_deliveries","aggregate":"sum","alias":"inst_del"}}, {{"column":"total_deliveries","aggregate":"sum","alias":"total_del"}}], computedColumns: [{{"alias":"idr","expression":"inst_del * 100.0 / NULLIF(total_del, 0)"}}]
- MMR: measures: [{{"column":"maternal_deaths","aggregate":"sum","alias":"deaths"}}, {{"column":"total_deliveries","aggregate":"sum","alias":"total_del"}}], computedColumns: [{{"alias":"mmr","expression":"deaths * 100000.0 / NULLIF(total_del, 0)"}}]
- Threshold (having references measure aliases): having: [{{"column":"total_del","operator":"gt","value":100}}]

Question: {QUESTION}

Return ONLY the JSON object. No markdown fencing, no explanation."""


async def generate_stat_query(
    question: str,
    tool_context: ToolContext,
) -> str:
    """Generate a StatQuery V2 JSON for a structured data question AND execute it.

    Use this tool for: district summaries, monthly trends, KPI metrics, rankings,
    rate calculations (IDR/MMR/IMR), facility counts, comparisons with thresholds,
    period-over-period analysis.

    The generated query is structured JSON that the frontend can execute, render
    as an interactive chart, and save to the dashboard. Prefer this over
    call_alloydb_agent for any query that fits the StatQuery V2 format.

    Returns both the validated StatQuery V2 JSON and the actual query results
    so you can provide data-driven insights. Use the STAT_QUERY_JSON in your
    mecdm_stat block (as the "query" field) and the QUERY_RESULTS for your
    textual analysis.

    Args:
        question: The natural language question about health data.

    Returns:
        The validated StatQuery V2 JSON inside <STAT_QUERY_JSON> tags,
        followed by actual query results inside <QUERY_RESULTS> tags.
        On error, returns an error message string.
    """
    try:
        schema_summary = await get_stats_schema_summary(tool_context)

        prompt = _STAT_QUERY_V2_PROMPT.format(
            SCHEMA=schema_summary,
            QUESTION=question,
        )

        response = get_llm_client().models.generate_content(
            model=os.getenv("BASELINE_NL2SQL_MODEL", ""),
            contents=prompt,
            config={"temperature": 0.1},
        )

        raw_text = (response.text or "").strip()
        # Strip markdown fencing if the LLM added it
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0].strip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

        # Parse and validate
        query = json.loads(raw_text)

        # Ensure version is set
        if query.get("version") != 2:
            query["version"] = 2

        # Validate table is not blocked
        table = query.get("source", {}).get("table", "")
        if table in STATS_BLOCKED_TABLES:
            return f"Error: Table '{table}' is blocked from stats queries."
        available = get_stats_available_tables()
        if available and table not in available:
            return f"Error: Table '{table}' not found. Available tables: {available}"

        # Validate computed columns if present
        if query.get("computedColumns"):
            from .app_utils.expression_validator import validate_computed_columns

            allowed_cols: set[str] = set()
            for dim in query.get("dimensions", []):
                allowed_cols.add(dim.get("alias", dim.get("column", "")))
                allowed_cols.add(dim.get("column", ""))
            for m in query.get("measures", []):
                allowed_cols.add(m.get("alias", m.get("column", "")))
                allowed_cols.add(m.get("column", ""))
            # Include window function aliases so computed columns can reference them
            for w in query.get("windows", []):
                if w.get("alias"):
                    allowed_cols.add(w["alias"])

            is_safe, reason = validate_computed_columns(
                query["computedColumns"], allowed_cols
            )
            if not is_safe:
                return f"Error: Unsafe computed column — {reason}"

        tool_context.state["stat_query_v2"] = query
        query_json = json.dumps(query)

        # --- Execute the query to get actual data for agent insights ---
        data_section = ""
        try:
            sql = compile_stat_query_v2_to_sql(query)
            logger.debug("generate_stat_query compiled SQL: %s", sql)

            success, results, error = execute_readonly_sql(sql)

            if not success:
                data_section = f"\n<QUERY_RESULTS>\nExecution error (use the JSON for frontend rendering): {error}\n</QUERY_RESULTS>"
            elif results:
                # Format results as a markdown table for the agent
                if isinstance(results, str):
                    try:
                        rows = json.loads(results)
                    except json.JSONDecodeError:
                        rows = None
                        data_section = f"\n<QUERY_RESULTS>\n{results[:3000]}\n</QUERY_RESULTS>"
                else:
                    rows = results

                if rows and isinstance(rows, list) and len(rows) > 0:
                    # Limit to first 50 rows for context
                    preview_rows = rows[:50]
                    headers = list(preview_rows[0].keys()) if isinstance(preview_rows[0], dict) else []
                    if headers:
                        header_line = "| " + " | ".join(headers) + " |"
                        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
                        row_lines = []
                        for row in preview_rows:
                            vals = []
                            for h in headers:
                                v = row.get(h, "")
                                if v is None:
                                    v = "NULL"
                                elif isinstance(v, float):
                                    v = f"{v:.2f}"
                                else:
                                    v = str(v)
                                vals.append(v)
                            row_lines.append("| " + " | ".join(vals) + " |")
                        table_str = "\n".join([header_line, sep_line] + row_lines)
                        total_note = ""
                        if len(rows) > 50:
                            total_note = f"\n(Showing 50 of {len(rows)} rows)"
                        data_section = f"\n<QUERY_RESULTS>\n{table_str}{total_note}\n</QUERY_RESULTS>"
                    else:
                        data_section = f"\n<QUERY_RESULTS>\n{json.dumps(preview_rows, default=str)[:3000]}\n</QUERY_RESULTS>"
                elif rows is not None and not data_section:
                    data_section = "\n<QUERY_RESULTS>\nQuery returned no rows.\n</QUERY_RESULTS>"
            else:
                data_section = "\n<QUERY_RESULTS>\nQuery returned no rows.\n</QUERY_RESULTS>"

        except Exception as exec_err:
            logger.warning("generate_stat_query execution failed (non-fatal): %s", exec_err)
            data_section = f"\n<QUERY_RESULTS>\nExecution error (use the JSON for frontend rendering): {exec_err}\n</QUERY_RESULTS>"

        return f"<STAT_QUERY_JSON>\n{query_json}\n</STAT_QUERY_JSON>{data_section}"

    except json.JSONDecodeError as e:
        logger.error("generate_stat_query JSON parse error: %s", e)
        return f"Error: Failed to parse generated query as JSON — {e}"
    except Exception as e:
        logger.error("generate_stat_query error: %s", e)
        return f"Error: {e}"

