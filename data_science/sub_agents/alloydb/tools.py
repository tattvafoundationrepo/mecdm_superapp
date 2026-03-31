"""This file contains the tools used by the AlloyDB agent."""

import logging
import os

from google.adk.tools import ToolContext
from google.genai import Client
from google.genai.types import HttpOptions
from toolbox_core import ToolboxSyncClient, auth_methods
from toolbox_core.protocol import Protocol

from data_science.utils.utils import get_env_var

from ...utils.utils import USER_AGENT

ALLOYDB_TOOLSET = os.getenv("ALLOYDB_TOOLSET", "postgres-database-tools")
# The agent connects to AlloyDB using the MCP Toolbox for Databases
# By default it connects to localhost; change this environment variable
# for a remote deployment.
MCP_TOOLBOX_HOST = os.getenv("MCP_TOOLBOX_HOST", "localhost")
MCP_TOOLBOX_PORT = os.getenv("MCP_TOOLBOX_PORT", "5000")

# MAX_NUM_ROWS = 80

vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT", None)
location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
http_options = HttpOptions(headers={"user-agent": USER_AGENT})
llm_client = Client(
    vertexai=True,
    project=vertex_project,
    location=location,
    http_options=http_options,
)

database_settings = None
toolbox_client = None
toolbox_toolset = None

logger = logging.getLogger(__name__)


def _build_toolbox_url():
    """Build the toolbox URL from environment variables."""
    if MCP_TOOLBOX_HOST in ["localhost", "127.0.0.1"]:
        return f"http://{MCP_TOOLBOX_HOST}:{MCP_TOOLBOX_PORT}"
    toolbox_url = f"https://{MCP_TOOLBOX_HOST}"
    if MCP_TOOLBOX_PORT != "":
        toolbox_url += f":{MCP_TOOLBOX_PORT}"
    return toolbox_url


def get_toolbox_client():
    """Get MCP Toolbox client."""
    global toolbox_client
    if toolbox_client is None:
        toolbox_url = _build_toolbox_url()
        logger.info("Connecting to MCP Toolbox at %s", toolbox_url)

        client_headers = {}
        if MCP_TOOLBOX_HOST not in ["localhost", "127.0.0.1"]:
            auth_token_provider = auth_methods.aget_google_id_token(toolbox_url)
            client_headers["Authorization"] = auth_token_provider

        toolbox_client = ToolboxSyncClient(
            toolbox_url,
            client_headers=client_headers,
            protocol=Protocol.MCP_LATEST,
        )
        toolbox_client.__enter__()

        # Debug: list all available tools on the server
        try:
            all_tools = toolbox_client.load_toolset()
            tool_names = [t.__name__ for t in all_tools]
            logger.info("Toolbox available tools: %s", tool_names)
        except Exception as e:
            logger.warning("Could not list toolbox tools: %s", e)
    return toolbox_client


def get_toolbox_toolset():
    """Get MCP Toolbox toolset."""
    global toolbox_toolset
    if toolbox_toolset is None:
        toolbox_toolset = get_toolbox_client().load_toolset(toolset_name=ALLOYDB_TOOLSET)
        tool_names = [t.__name__ for t in toolbox_toolset]
        logger.info("Loaded toolset '%s' with tools: %s", ALLOYDB_TOOLSET, tool_names)
    return toolbox_toolset


def get_database_settings():
    """Get database settings."""
    global database_settings
    if database_settings is None:
        database_settings = update_database_settings()
    return database_settings


def get_schema_summary():
    get_summary_tool = get_toolbox_client().load_tool("list_table_summaries")
    schema_summary = get_summary_tool(
        schema_names=get_env_var("ALLOYDB_SCHEMA_NAME")
    )
    return schema_summary


def get_table_schema(table_names: str):
    get_schema_tool = get_toolbox_client().load_tool("list_tables")
    schema = get_schema_tool(
        schema_names=get_env_var("ALLOYDB_SCHEMA_NAME"), table_names=table_names
    )
    return schema


def update_database_settings():
    """Update database settings."""
    global database_settings

    schema_summary = get_schema_summary()

    database_settings = {
        "project_id": get_env_var("ALLOYDB_PROJECT_ID"),
        "database": get_env_var("ALLOYDB_DATABASE"),
        "schema_name": get_env_var("ALLOYDB_SCHEMA_NAME"),
        "schema_summary": schema_summary,
    }
    return database_settings


def _fetch_golden_examples() -> str:
    """Fetch golden SQL examples directly from the shared user DB."""
    try:
        from sqlalchemy import create_engine, select

        from data_science.app_utils.models import GoldenSql

        db_url = os.environ.get("DATABASE_URL_USER")
        if not db_url:
            logger.warning("DATABASE_URL_USER not set, skipping golden SQL examples")
            return ""

        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(
                select(GoldenSql).where(GoldenSql.is_active == True).limit(15)  # noqa: E712
            )
            rows = result.all()

        if not rows:
            return ""

        lines = []
        for row in rows:
            lines.append(f"Q: {row.question}")
            lines.append(f"SQL: {row.sql_query}")
            if row.explanation:
                lines.append(f"-- {row.explanation}")
            lines.append("")
        return "\n".join(lines)

    except Exception as e:
        logger.warning("Failed to fetch golden SQL examples: %s", e)
        return ""


def alloydb_nl2sql(
    question: str,
    tool_context: ToolContext,
) -> str:
    """Generates an initial SQL query from a natural language question.

    Args:
        question (str): Natural language question.
        tool_context (ToolContext): The tool context to use for generating the
          SQL query.

    Returns:
        str: An SQL statement to answer this question.
    """

    prompt_template = """
Translate the health data question below into a valid PostgreSQL query using the provided schema.

Rules:
- Reference tables as "table_name" (double-quoted, case-sensitive).
- Use ONLY columns listed in the schema. Associate each column with its correct table.
- Minimize joins. Ensure matching data types on join columns.
- Include all non-aggregated SELECT columns in GROUP BY.
- Apply WHERE/HAVING filters to minimize returned rows.
- In mother_journeys and anc_visits, ALL columns are TEXT type — cast to numeric/date as needed.
- District names in mother_journeys/anc_visits are UPPERCASE (e.g., 'EAST KHASI HILLS').
- Use integer code joins (district_code_lgd, block_code_lgd) instead of name joins when possible.
{EXAMPLES_SECTION}
Schema:
```
{SCHEMA}
```

Question: {QUESTION}

Generate the PostgreSQL query.
   """

    schema_summary = tool_context.state["database_settings"]["alloydb"]["schema_summary"]

    # --- Stage 1: Table Selection ---
    selection_prompt_template = """
Given the following database schema summary (tables and their descriptions):
{SCHEMA_SUMMARY}

Which specific tables are necessary to answer the following question?
Question: {QUESTION}

Return a comma-separated list of EXACT table names ONLY. Do not include any other text, reasoning, or markdown.
If no tables are relevant, return "none".
"""
    selection_prompt = selection_prompt_template.format(
        SCHEMA_SUMMARY=schema_summary,
        QUESTION=question,
    )

    selection_response = llm_client.models.generate_content(
        model=os.getenv("BASELINE_NL2SQL_MODEL", ""),
        contents=selection_prompt,
        config={"temperature": 0.1},
    )

    selected_tables = selection_response.text.strip() if selection_response.text else ""
    logger.debug("Selected tables: %s", selected_tables)

    if not selected_tables or selected_tables.lower() == "none":
        return "SELECT 'No relevant tables found for this question' AS error;"

    # --- Stage 2: Detailed Schema Fetch ---
    schema = get_table_schema(selected_tables)

    # Fetch golden SQL examples for few-shot learning
    golden_examples = _fetch_golden_examples()
    examples_section = ""
    if golden_examples:
        examples_section = f"""
Example Queries (verified correct — follow these patterns):
```
{golden_examples}
```
"""

    prompt = prompt_template.format(
        SCHEMA=schema,
        QUESTION=question,
        EXAMPLES_SECTION=examples_section,
    )

    response = llm_client.models.generate_content(
        model=os.getenv("BASELINE_NL2SQL_MODEL", ""),
        contents=prompt,
        config={"temperature": 0.1},
    )

    sql = response.text or ""
    if sql:
        sql = sql.replace("```sql", "").replace("```", "").strip()

    logger.debug("sql: %s", sql)

    tool_context.state["sql_query"] = sql

    return sql


def run_alloydb_query(
    sql_string: str,
    tool_context: ToolContext,
) -> dict:
    """
    Runs an AlloyDB SQL query.

    This function validates the provided SQL string, then runs it against
    AlloyDB and returns the results.

    It performs the following steps:

    1. **SQL Cleanup:**  Preprocesses the SQL string using a `cleanup_sql`
    function
    2. **DML/DDL Restriction:**  Rejects any SQL queries containing DML or DDL
       statements (e.g., UPDATE, DELETE, INSERT, CREATE, ALTER) to ensure
       read-only operations.
    3. **Syntax and Execution:** Sends the cleaned SQL to BigQuery for
       execution and retrieves the results.

    Args:
        sql_string (str): The SQL query string to validate.
        tool_context (ToolContext): The tool context to use for validation.

    Returns:
        A dict with two keys:
            query_results (list): A list of {key, value} dicts for each element
                in the result set.
            error_message (str): A message indicating the query outcome.
                This includes:
                  - "Valid SQL. Results: ..." if the query is valid and returns
                    data.
                  - "Query executed successfully (no results)." if
                    the query is valid but returns no data.
                  - "Invalid SQL: ..." if the query is invalid, along with the
                    error message from BigQuery.
                  - "Query error: ..." if another error occurs, including any
                  error message.
    """

    def cleanup_sql(sql_string):
        """Processes the SQL string to get a printable, valid SQL string."""

        # 1. Remove backslashes escaping double quotes
        sql_string = sql_string.replace('\\"', '"')

        # 2. Remove newlines
        sql_string = sql_string.replace("\\\n", " ")
        sql_string = sql_string.replace("\n", " ")

        # 3. Replace escaped single quotes
        sql_string = sql_string.replace("\\'", "'")

        # 4. Replace escaped newlines (those not preceded by a backslash)
        # sql_string = sql_string.replace("\\n", "\n")

        # 5. Add limit clause if not present
        # if "limit" not in sql_string.lower():
        #    sql_string = sql_string + " limit " + str(MAX_NUM_ROWS)

        return sql_string

    logger.debug("Executing SQL: %s", sql_string)
    sql_string = cleanup_sql(sql_string)
    logger.debug("Validating SQL (after cleanup): %s", sql_string)

    final_result = {"query_result": "", "error_message": ""}

    # Validate SQL using pglast (PostgreSQL's actual C parser)
    from data_science.app_utils.sql_validator import validate_sql

    is_safe, reason = validate_sql(sql_string)
    if not is_safe:
        final_result["error_message"] = f"Blocked: {reason}"
        return final_result

    try:
        execute_sql_tool = get_toolbox_client().load_tool("execute_sql")
        logger.debug("Sending SQL query: %s", sql_string)
        results = execute_sql_tool(sql_string)
        logger.debug("Received results: %s", results)

        if results:  # Check if query returned data
            final_result["query_result"] = results
            tool_context.state["alloydb_query_result"] = results

        else:
            final_result["error_message"] = (
                "Valid SQL. Query executed successfully (no results)."
            )

    except (
        # Catch generic BQ exceptions  # pylint: disable=broad-exception-caught
        Exception
    ) as e:
        final_result["error_message"] = f"Query error: {e}"

    logger.debug("run_alloydb_query final_result: %s", final_result)

    return final_result
