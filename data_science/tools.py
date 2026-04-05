"""Tools for the ADK Samples Data Science Agent."""

import json
import logging
import os

from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.genai import Client
from google.genai.types import HttpOptions

from .services.file_processor import UPLOAD_BUCKET_NAME, read_extracted_text
from .sub_agents import alloydb_agent
from .sub_agents.alloydb.tools import get_toolbox_client
from .utils.utils import USER_AGENT

logger = logging.getLogger(__name__)


async def call_alloydb_agent(
    question: str,
    tool_context: ToolContext,
):
    """Query AlloyDB using the full NL2SQL pipeline with automatic error correction.

    Use this for complex queries requiring multi-table joins, ambiguous table
    selection, or when execute_sql returned an error. The sub-agent
    selects tables, generates SQL, executes it, and retries on failure.

    Args:
        question: Natural language data question.
    """
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
    """Extract available table names from the schema summary already loaded at startup."""
    global _stats_available_tables
    if _stats_available_tables is not None:
        return _stats_available_tables

    try:
        # Use the schema summary already loaded by get_database_settings() at startup
        # (via list_table_summaries). This avoids a separate MCP call that fails with
        # empty table_names parameter.
        from .sub_agents.alloydb.tools import get_database_settings

        settings = get_database_settings()
        schema_summary_raw = settings.get("schema_summary", "")

        all_tables: list[str] = []
        if isinstance(schema_summary_raw, str) and schema_summary_raw:
            import json as _json
            try:
                tables_data = _json.loads(schema_summary_raw)
            except _json.JSONDecodeError:
                tables_data = []
        elif isinstance(schema_summary_raw, list):
            tables_data = schema_summary_raw
        else:
            tables_data = []

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
        logger.error(
            "Failed to extract stats tables from schema summary: %s", e)
        return []


PREDEFINED_STATS_CATALOG = [
    {"id": "total-registrations", "name": "Total Registrations", "category": "maternal-health",
        "description": "Total maternal registrations across all districts"},
    {"id": "institutional-delivery-rate", "name": "Institutional Delivery Rate",
        "category": "maternal-health", "description": "Percentage of deliveries at health institutions"},
    {"id": "health-facilities-count", "name": "Health Facilities", "category": "infrastructure",
        "description": "Total health facilities across all districts"},
    {"id": "awc-count", "name": "Anganwadi Centres", "category": "infrastructure",
        "description": "Total Anganwadi centres across all districts"},
    {"id": "district-registrations", "name": "District-wise Registrations", "category": "maternal-health",
        "description": "Total maternal registrations by district (bar chart)"},
    {"id": "monthly-registrations-trend", "name": "Monthly Registration & Delivery Trends",
        "category": "maternal-health", "description": "Monthly trends of registrations and deliveries (area chart)"},
    {"id": "facility-type-distribution", "name": "Facility Type Distribution",
        "category": "infrastructure", "description": "Health facility count by type (pie chart)"},
    {"id": "block-delivery-rate", "name": "Top 20 Blocks: Institutional Deliveries",
        "category": "maternal-health", "description": "Blocks with highest institutional deliveries (bar chart)"},
    {"id": "maternal-deaths-district", "name": "Maternal Deaths by District", "category": "maternal-health",
        "description": "Total reported maternal deaths by district (bar chart)"},
    {"id": "monthly-anc-coverage", "name": "Monthly ANC Coverage", "category": "maternal-health",
        "description": "Monthly ANC visits, IFA recipients, and TT doses (line chart)"},
]

_NUMERIC_TYPES = {"bigint", "integer", "smallint",
                  "numeric", "double precision", "real"}
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
        raw_schema = list_tables_tool(
            schema_names="public", table_names=table_names_str)

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
                tname = table_info.get(
                    "table_name") or table_info.get("name", "")
                if tname not in available_set:
                    continue
                columns = []
                for col in table_info.get("columns", []):
                    col_name = col.get("column_name") or col.get("name", "")
                    col_type = col.get("data_type") or col.get("type", "")
                    role = _infer_column_role(col_name, col_type)
                    if role:
                        columns.append(
                            {"name": col_name, "type": col_type, "role": role})
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
                        columns.append(
                            {"name": col_name, "type": col_type, "role": role})
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


def _qi(identifier: str) -> str:
    """Quote a SQL identifier."""
    return f'"{identifier.replace(chr(34), chr(34)+chr(34))}"'


_OPERATOR_SQL = {
    "eq": "=",
    "neq": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "like": "LIKE",
}

_TIME_PRESETS = {
    "last_7d": "7 days",
    "last_30d": "30 days",
    "last_quarter": "3 months",
    "last_year": "1 year",
    "ytd": "ytd",
}


def _compile_stat_query_v2_to_sql(query: dict) -> str:
    """Compile a StatQuery V2 dict into a PostgreSQL string.

    This is a simplified backend compiler for executing stat queries and
    returning preview data. The frontend has a full-featured compiler;
    this handles the common cases so the agent can see actual results.
    """
    table = query["source"]["table"]
    alias = "t0"

    dimensions = query.get("dimensions", [])
    measures = query.get("measures", [])
    computed_columns = query.get("computedColumns", [])
    filters = query.get("filters", [])
    having = query.get("having", [])
    order_by = query.get("orderBy", [])
    limit = min(query.get("limit", 1000), 10_000)
    time_range = query.get("timeRange")

    needs_wrapper = len(computed_columns) > 0

    # --- Inner SELECT ---
    select_parts: list[str] = []
    inner_aliases: list[str] = []

    for dim in dimensions:
        col_ref = f'{_qi(alias)}.{_qi(dim["column"])}'
        transform = dim.get("transform")
        if transform == "date_trunc_month":
            col_ref = f"DATE_TRUNC('month', {col_ref})"
        elif transform == "date_trunc_quarter":
            col_ref = f"DATE_TRUNC('quarter', {col_ref})"
        elif transform == "date_trunc_year":
            col_ref = f"DATE_TRUNC('year', {col_ref})"
        dim_alias = dim.get("alias", dim["column"])
        select_parts.append(f"{col_ref} AS {_qi(dim_alias)}")
        inner_aliases.append(dim_alias)

    # Track alias→aggregate expression for HAVING
    alias_to_agg: dict[str, str] = {}

    for m in measures:
        col_ref = f'{_qi(alias)}.{_qi(m["column"])}'
        agg = m["aggregate"]
        if agg == "count_distinct":
            agg_expr = f"COUNT(DISTINCT {col_ref})"
        else:
            agg_expr = f"{agg.upper()}({col_ref})"
        m_alias = m.get("alias", m["column"])
        select_parts.append(f"{agg_expr} AS {_qi(m_alias)}")
        inner_aliases.append(m_alias)
        alias_to_agg[m_alias] = agg_expr

    # --- FROM ---
    from_clause = f"{_qi(table)} AS {_qi(alias)}"
    joins = query.get("source", {}).get("joins", [])
    for i, join in enumerate(joins):
        j_alias = f"t{i + 1}"
        j_type = join.get("type", "inner").upper()
        left_ref = f'{_qi(alias)}.{_qi(join["on"]["left"])}'
        right_ref = f'{_qi(j_alias)}.{_qi(join["on"]["right"])}'
        if join.get("caseInsensitive"):
            on_clause = f"UPPER({left_ref}) = UPPER({right_ref})"
        else:
            on_clause = f"{left_ref} = {right_ref}"
        from_clause += f" {j_type} JOIN {_qi(join['table'])} AS {_qi(j_alias)} ON {on_clause}"

    # --- WHERE ---
    where_parts: list[str] = []

    if time_range and time_range.get("column"):
        col = _qi(time_range["column"])
        preset = time_range.get("preset", "all")
        custom = time_range.get("custom")
        if preset != "all" and not custom:
            pg_interval = _TIME_PRESETS.get(preset)
            if pg_interval and pg_interval != "ytd":
                where_parts.append(
                    f"{col} >= (CURRENT_DATE - INTERVAL '{pg_interval}')::text"
                )
            elif pg_interval == "ytd":
                where_parts.append(
                    f"{col} >= TO_CHAR(DATE_TRUNC('year', CURRENT_DATE), 'YYYY-MM')"
                )
        elif custom:
            if custom.get("from"):
                where_parts.append(f"{col} >= '{custom['from']}'")
            if custom.get("to"):
                where_parts.append(f"{col} <= '{custom['to']}'")

    for f in filters:
        col = f'{_qi(alias)}.{_qi(f["column"])}'
        op = f.get("operator", "eq")
        val = f.get("value")
        if op in _OPERATOR_SQL:
            if isinstance(val, str):
                where_parts.append(f"{col} {_OPERATOR_SQL[op]} '{val}'")
            else:
                where_parts.append(f"{col} {_OPERATOR_SQL[op]} {val}")
        elif op == "in" and isinstance(val, list):
            vals = ", ".join(
                f"'{v}'" if isinstance(v, str) else str(v) for v in val
            )
            where_parts.append(f"{col} IN ({vals})")
        elif op == "not_in" and isinstance(val, list):
            vals = ", ".join(
                f"'{v}'" if isinstance(v, str) else str(v) for v in val
            )
            where_parts.append(f"{col} NOT IN ({vals})")
        elif op == "is_null":
            where_parts.append(f"{col} IS NULL")
        elif op == "is_not_null":
            where_parts.append(f"{col} IS NOT NULL")
        elif op == "between" and isinstance(val, list) and len(val) == 2:
            lo, hi = val
            lo_str = f"'{lo}'" if isinstance(lo, str) else str(lo)
            hi_str = f"'{hi}'" if isinstance(hi, str) else str(hi)
            where_parts.append(f"{col} BETWEEN {lo_str} AND {hi_str}")

    where_str = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    # --- GROUP BY ---
    group_by_str = ""
    if dimensions:
        group_by_str = f"GROUP BY {', '.join(str(i + 1) for i in range(len(dimensions)))}"

    # --- HAVING ---
    having_parts: list[str] = []
    computed_alias_set = {cc["alias"] for cc in computed_columns}
    for h in having:
        h_col = h["column"]
        h_op = _OPERATOR_SQL.get(h.get("operator", "eq"), "=")
        h_val = h.get("value")
        # HAVING on computed columns must go in outer WHERE
        if h_col in computed_alias_set:
            continue
        agg_expr = alias_to_agg.get(h_col)
        if agg_expr:
            if isinstance(h_val, str):
                having_parts.append(f"{agg_expr} {h_op} '{h_val}'")
            else:
                having_parts.append(f"{agg_expr} {h_op} {h_val}")

    having_str = f"HAVING {' AND '.join(having_parts)}" if having_parts else ""

    if not needs_wrapper:
        # Simple single-layer query
        order_str = ""
        if order_by:
            ob_parts = []
            for o in order_by:
                direction = o.get("direction", "asc").upper()
                ob_parts.append(f"{_qi(o['column'])} {direction}")
            order_str = f"ORDER BY {', '.join(ob_parts)}"

        sql = f"SELECT {', '.join(select_parts)} FROM {from_clause} {where_str} {group_by_str} {having_str} {order_str} LIMIT {limit}"
        return " ".join(sql.split())

    # Two-layer query: inner has dimensions+measures, outer adds computed columns
    inner_sql = f"SELECT {', '.join(select_parts)} FROM {from_clause} {where_str} {group_by_str} {having_str}"

    outer_select = [_qi(a) for a in inner_aliases]
    for cc in computed_columns:
        outer_select.append(f"({cc['expression']}) AS {_qi(cc['alias'])}")

    # Outer HAVING (for computed column filters)
    outer_where_parts: list[str] = []
    for h in having:
        if h["column"] in computed_alias_set:
            h_op = _OPERATOR_SQL.get(h.get("operator", "eq"), "=")
            h_val = h.get("value")
            expr = next(
                (cc["expression"]
                 for cc in computed_columns if cc["alias"] == h["column"]),
                _qi(h["column"]),
            )
            if isinstance(h_val, str):
                outer_where_parts.append(f"({expr}) {h_op} '{h_val}'")
            else:
                outer_where_parts.append(f"({expr}) {h_op} {h_val}")

    outer_where = f"WHERE {' AND '.join(outer_where_parts)}" if outer_where_parts else ""

    order_str = ""
    if order_by:
        ob_parts = []
        for o in order_by:
            direction = o.get("direction", "asc").upper()
            ob_parts.append(f"{_qi(o['column'])} {direction}")
        order_str = f"ORDER BY {', '.join(ob_parts)}"

    sql = f"SELECT {', '.join(outer_select)} FROM ({inner_sql}) AS _inner {outer_where} {order_str} LIMIT {limit}"
    return " ".join(sql.split())


_v2_llm_client = None


def _get_v2_llm_client():
    global _v2_llm_client
    if _v2_llm_client is None:
        vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT", None)
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        http_options = HttpOptions(headers={"user-agent": USER_AGENT})
        _v2_llm_client = Client(
            vertexai=True,
            project=vertex_project,
            location=location,
            http_options=http_options,
        )
    return _v2_llm_client


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
    """Build a StatQuery V2 JSON for frontend chart/table rendering.

    PREREQUISITE: Requires data to already be retrieved and cached in tool_context state via
    execute_sql or call_alloydb_agent. This is a visualization
    formatting tool, not a data retrieval tool.

    Args:
        question: The natural language question about health data.

    Returns:
        The validated StatQuery V2 JSON inside <STAT_QUERY_JSON> tags,
        followed by actual query results inside <QUERY_RESULTS> tags.
        On error, returns an error message string.
    """
    # Runtime guard: reject if called before data retrieval
    if ("alloydb_query_result" not in tool_context.state
            and "alloydb_agent_output" not in tool_context.state):
        return (
            "Error: No data retrieved yet. Call execute_sql or "
            "call_alloydb_agent first to get the data, then call "
            "generate_stat_query to create the visualization."
        )

    try:
        schema_summary = await get_stats_schema_summary(tool_context)

        prompt = _STAT_QUERY_V2_PROMPT.format(
            SCHEMA=schema_summary,
            QUESTION=question,
        )

        response = _get_v2_llm_client().models.generate_content(
            model=os.getenv("BASELINE_NL2SQL_MODEL", ""),
            contents=prompt,
            config={"temperature": 0.1},
        )

        raw_text = (response.text or "").strip()
        # Strip markdown fencing if the LLM added it
        if raw_text.startswith("```"):
            raw_text = raw_text.split(
                "\n", 1)[1] if "\n" in raw_text else raw_text
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0].strip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

        # Parse and validate
        query = json.loads(raw_text)

        # Ensure version is set
        if query.get("version") != 2:
            query["version"] = 2

        # Normalize dimensions: accept plain strings as shorthand
        if "dimensions" in query:
            query["dimensions"] = [
                {"column": d} if isinstance(d, str) else d
                for d in query["dimensions"]
            ]

        # Normalize timeRange: accept start/end as shorthand for custom.from/to
        tr = query.get("timeRange")
        if tr and ("start" in tr or "end" in tr):
            tr["custom"] = {
                "from": tr.pop("start", ""),
                "to": tr.pop("end", ""),
            }

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
            sql = _compile_stat_query_v2_to_sql(query)
            logger.debug("generate_stat_query compiled SQL: %s", sql)

            execute_sql_tool = get_toolbox_client().load_tool("execute_sql")
            results = execute_sql_tool(sql)

            if results:
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
                    headers = list(preview_rows[0].keys()) if isinstance(
                        preview_rows[0], dict) else []
                    if headers:
                        header_line = "| " + " | ".join(headers) + " |"
                        sep_line = "| " + \
                            " | ".join("---" for _ in headers) + " |"
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
                        table_str = "\n".join(
                            [header_line, sep_line] + row_lines)
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
            logger.warning(
                "generate_stat_query execution failed (non-fatal): %s", exec_err)
            data_section = f"\n<QUERY_RESULTS>\nExecution error (use the JSON for frontend rendering): {exec_err}\n</QUERY_RESULTS>"

        return f"<STAT_QUERY_JSON>\n{query_json}\n</STAT_QUERY_JSON>{data_section}"

    except json.JSONDecodeError as e:
        logger.error("generate_stat_query JSON parse error: %s", e)
        return f"Error: Failed to parse generated query as JSON — {e}"
    except Exception as e:
        logger.error("generate_stat_query error: %s", e)
        return f"Error: {e}"


async def read_uploaded_file(
    gcs_uri: str,
    tool_context: ToolContext,
) -> str:
    """Read the content of a previously uploaded file from Google Cloud Storage.

    For documents (DOCX, PPTX, XLSX), returns the extracted text content.
    For images and PDFs, the content is already provided as multimodal input
    in the conversation — this tool confirms that.

    Use this tool when a user references an uploaded Office document and you
    need to read its contents for analysis.

    Args:
        gcs_uri: The GCS URI of the uploaded file (gs://bucket/path/to/file).

    Returns:
        The text content of the file, or a note about multimodal content.
    """
    expected_prefix = f"gs://{UPLOAD_BUCKET_NAME}/"
    if not gcs_uri.startswith(expected_prefix):
        return f"Error: Invalid GCS URI. Must start with {expected_prefix}"

    # Determine file type from extension
    uri_lower = gcs_uri.lower()
    multimodal_exts = (".pdf", ".png", ".jpg", ".jpeg", ".webp")
    if any(uri_lower.endswith(ext) for ext in multimodal_exts):
        return (
            "This file (image or PDF) was already provided as multimodal content "
            "in the user's message. You can analyze it directly from the conversation "
            "context — no need to read it separately."
        )

    try:
        text = read_extracted_text(gcs_uri)
        if len(text) > 50000:
            text = text[:50000] + \
                "\n\n... [Content truncated at 50,000 characters]"
        return text
    except FileNotFoundError as e:
        return f"Error: {e}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("read_uploaded_file error for %s", gcs_uri)
        return f"Error reading file: {e}"


# ============================================================================
# Training Video Recommendation
# ============================================================================

_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "of", "and", "for", "to", "with", "on",
    "at", "by", "from", "is", "what", "does", "about", "how", "why",
    "are", "training", "video", "videos", "show", "me", "available",
    "do", "you", "have",
})

CATEGORY_META = {
    "BREASTFEEDING": {"color": "#e91e63", "icon": "baby"},
    "BREAST_NIPPLE_CONDITIONS": {"color": "#ad1457", "icon": "baby"},
    "NEWBORN_CARE": {"color": "#2196f3", "icon": "heart-pulse"},
    "NUTRITION_MICRONUTRIENTS": {"color": "#ff9800", "icon": "apple"},
    "RECIPES": {"color": "#9c27b0", "icon": "utensils"},
    "COMPLEMENTARY_FEEDING": {"color": "#4caf50", "icon": "salad"},
    "PRE_PREGNANCY_MATERNAL": {"color": "#00bcd4", "icon": "heart"},
    "CHILD_NUTRITION_GROWTH": {"color": "#607d8b", "icon": "ruler"},
    "FOOD_SAFETY": {"color": "#795548", "icon": "hand-sparkles"},
}

_video_library_cache: list[dict] | None = None


def _load_video_library() -> list[dict]:
    """Load and cache the video library from the JSON source file."""
    global _video_library_cache
    if _video_library_cache is not None:
        return _video_library_cache

    video_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "mecdm_dataset", "db_20260226", "output", "video_library.json",
    )
    try:
        with open(video_path, encoding="utf-8") as f:
            _video_library_cache = json.load(f)
        logger.info("Loaded video library: %d videos", len(_video_library_cache))
    except FileNotFoundError:
        logger.warning("Video library not found at %s", video_path)
        _video_library_cache = []
    return _video_library_cache


def _score_video(video: dict, keywords: list[str], category: str | None) -> int:
    """Compute weighted relevance score for a video against search keywords."""
    score = 0
    tags_lower = " ".join(video.get("tags", [])).lower()
    title_lower = (video.get("title") or "").lower()
    indicators_lower = " ".join(video.get("health_indicators", [])).lower()
    desc_lower = (video.get("description") or "").lower()
    cat_lower = (video.get("category") or "").lower()
    summary_lower = (video.get("one_line_summary") or "").lower()

    for kw in keywords:
        if kw in tags_lower:
            score += 4
        if kw in title_lower:
            score += 3
        if kw in indicators_lower:
            score += 2
        if kw in desc_lower:
            score += 2
        if kw in cat_lower:
            score += 1
        if kw in summary_lower:
            score += 1

    if category and cat_lower == category.lower():
        score += 5

    return score


async def recommend_video(
    query: str,
    category: str = None,
    max_results: int = 5,
    tool_context: ToolContext = None,
):
    """Search and recommend health training videos from the CureWell/spoken-tutorial.org
    video library (49 videos) covering breastfeeding, newborn care, nutrition,
    complementary feeding, recipes, and food safety.

    Returns video cards with descriptions, watch links, and language availability.
    Use when users ask for training content, or PROACTIVELY when data shows
    red flags (low breastfeeding rates, high neonatal deaths, anemia, etc.).
    Videos are available in English, Khasi, and Garo (select videos).

    Args:
        query: Search terms describing the topic. Examples:
            'breastfeeding positions', 'iron rich recipes',
            'newborn care', 'complementary feeding 6 months'.
        category: Optional category filter. One of: BREASTFEEDING,
            BREAST_NIPPLE_CONDITIONS, NEWBORN_CARE, NUTRITION_MICRONUTRIENTS,
            RECIPES, COMPLEMENTARY_FEEDING, PRE_PREGNANCY_MATERNAL,
            CHILD_NUTRITION_GROWTH, FOOD_SAFETY.
        max_results: Number of videos to return (default 5, max 10).
    """
    logger.info("recommend_video: query='%s' category=%s", query, category)

    videos = _load_video_library()
    if not videos:
        return "Video library not available."

    # Extract and filter keywords
    keywords = [
        w for w in query.lower().split()
        if w not in _STOP_WORDS and len(w) > 1
    ]
    if not keywords:
        keywords = [query.lower().strip()]

    # Score all videos
    scored = []
    for video in videos:
        score = _score_video(video, keywords, category)
        if score > 0:
            scored.append((score, video))

    scored.sort(key=lambda x: x[0], reverse=True)
    max_n = min(int(max_results), 10)
    top = scored[:max_n]

    if not top:
        return (
            f"No training videos found matching '{query}'. "
            "Try broader terms like 'breastfeeding', 'nutrition', "
            "'newborn care', or 'recipes'."
        )

    # Build text summary for the LLM
    text_parts = [f"TRAINING VIDEOS FOUND ({len(top)} matches):\n"]
    # Build structured cards for frontend
    cards = []

    for score, video in top:
        langs = video.get("languages", {})
        lang_parts = []
        if langs.get("english"):
            lang_parts.append("EN \u2713")
        if langs.get("khasi"):
            lang_parts.append("Khasi \u2713")
        if langs.get("garo"):
            lang_parts.append("Garo \u2713")
        lang_str = " | ".join(lang_parts) if lang_parts else "EN \u2713"

        cat = video.get("category", "")
        meta = CATEGORY_META.get(cat, {"color": "#607d8b", "icon": "book"})

        text_parts.append(
            f"--- {video.get('id', '')} ---\n"
            f"Title: {video.get('title', 'Unknown')}\n"
            f"Category: {cat} | Duration: ~{video.get('duration_min', 0)} min\n"
            f"Languages: {lang_str}\n"
            f"Description: {video.get('description', '')}\n"
            f"URL: {video.get('url', '')}"
        )

        cards.append({
            "id": video.get("id", ""),
            "title": video.get("title", "Unknown"),
            "category": cat,
            "categoryColor": meta["color"],
            "categoryIcon": meta.get("icon"),
            "durationMin": video.get("duration_min", 0),
            "description": video.get("description", ""),
            "url": video.get("url", ""),
            "thumbnailUrl": video.get("thumbnail_url"),
            "languages": langs if isinstance(langs, dict) else {"english": True, "khasi": False, "garo": False},
            "relevanceScore": score,
        })

    text_value = "\n\n".join(text_parts)
    text_value += (
        "\n\n---\n"
        "Present these as training resources for frontline workers (ASHA, ANM, Anganwadi). "
        "Mention language availability where relevant.\n\n"
        "Include the following block VERBATIM in your response to render video cards:\n\n"
    )

    viz_block = json.dumps({
        "type": "video_cards",
        "title": f"Training Videos: {query}",
        "cards": cards,
    }, ensure_ascii=False)

    text_value += f"```mecdm_viz\n{viz_block}\n```"

    return text_value
