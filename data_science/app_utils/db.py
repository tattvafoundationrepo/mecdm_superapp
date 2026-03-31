"""Safe read-only SQL execution via MCP Toolbox.

Centralizes the validate-then-execute pipeline that was duplicated in
alloydb/tools.py (run_alloydb_query) and tools.py (generate_stat_query).
"""

import logging

from data_science.sub_agents.alloydb.tools import get_toolbox_client

from .sql_validator import validate_sql

logger = logging.getLogger(__name__)


def execute_readonly_sql(sql: str) -> tuple[bool, list | str | None, str]:
    """Validate and execute a read-only SQL query.

    Uses pglast to ensure only safe SELECT statements reach the database,
    then executes via the MCP Toolbox ``execute_sql`` tool.

    Args:
        sql: The SQL string to validate and execute.

    Returns:
        Tuple of ``(success, results, error_message)``:
        - *success*: ``True`` if the query executed without error.
        - *results*: Query results (list of dicts or raw string), or
          ``None`` on failure / empty result set.
        - *error_message*: Empty string on success, description on failure.
    """
    is_safe, reason = validate_sql(sql)
    if not is_safe:
        return False, None, f"Blocked: {reason}"

    try:
        execute_sql_tool = get_toolbox_client().load_tool("execute_sql")
        results = execute_sql_tool(sql)
        if results:
            return True, results, ""
        return True, None, "Query executed successfully (no results)."
    except Exception as e:
        logger.error("execute_readonly_sql error: %s", e)
        return False, None, f"Query error: {e}"
