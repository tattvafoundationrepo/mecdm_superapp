"""SQL validation using pglast (PostgreSQL's actual C parser).

Ensures only safe SELECT statements reach the database, blocking:
- Non-SELECT statements (INSERT, UPDATE, DELETE, DROP, etc.)
- Dangerous functions (pg_read_file, dblink, lo_export, etc.)
- Multiple statements in a single query
"""

import logging

import pglast
from pglast import ast, visitors

logger = logging.getLogger(__name__)

# Functions that should never appear in AI-generated queries
BLOCKED_FUNCTIONS: set[str] = {
    # File system access
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_ls_logdir",
    "pg_ls_waldir",
    "pg_stat_file",
    # External connections
    "dblink",
    "dblink_exec",
    "dblink_connect",
    # Large object manipulation
    "lo_export",
    "lo_import",
    "lo_create",
    "lo_unlink",
    # Process control
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    # Config manipulation
    "set_config",
    "pg_advisory_lock",
    "pg_advisory_unlock",
    # Notification / listen
    "pg_notify",
    # Copy
    "pg_copy_to",
    "pg_copy_from",
}


class _FunctionChecker(visitors.Visitor):
    """AST visitor that collects all function calls in the query."""

    def __init__(self) -> None:
        self.found_blocked: str | None = None

    def visit_FuncCall(self, _ancestors: visitors.Ancestor, node: ast.FuncCall) -> None:
        if node.funcname:
            # funcname is a tuple of ast.String nodes
            for part in node.funcname:
                if isinstance(part, ast.String) and part.sval.lower() in BLOCKED_FUNCTIONS:
                    self.found_blocked = part.sval.lower()
                    return


def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate SQL using PostgreSQL's actual parser.

    Returns:
        (is_safe, reason) — True if safe to execute, otherwise False with reason.
    """
    # Parse the SQL
    try:
        stmts = pglast.parse_sql(sql)
    except pglast.parser.ParseError as e:
        return False, f"SQL parse error: {e}"

    if not stmts:
        return False, "Empty SQL statement"

    # Only allow a single statement
    if len(stmts) > 1:
        return False, "Multiple statements not allowed"

    stmt = stmts[0].stmt

    # Only allow SELECT statements (no INSERT, UPDATE, DELETE, etc.)
    if not isinstance(stmt, ast.SelectStmt):
        type_name = type(stmt).__name__
        return False, f"Only SELECT statements allowed, got {type_name}"

    # Check for blocked functions in the AST
    checker = _FunctionChecker()
    checker(stmts[0])
    if checker.found_blocked:
        return False, f"Blocked function: {checker.found_blocked}"

    return True, "ok"
