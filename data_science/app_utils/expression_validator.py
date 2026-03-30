"""Expression validator for StatQuery V2 computed columns.

Validates that expressions only contain safe operations:
- Column references (measure/dimension aliases)
- Arithmetic operators: +, -, *, /
- Numeric literals
- String literals (single-quoted)
- Parentheses
- Allowlisted SQL functions
- CASE WHEN ... THEN ... ELSE ... END
- CAST(... AS type)

Blocks: subqueries, semicolons, comments, non-allowlisted functions,
system functions, and any other SQL injection vectors.
"""

import re
from typing import Set

ALLOWED_FUNCTIONS: Set[str] = {
    "COALESCE",
    "NULLIF",
    "ROUND",
    "CEIL",
    "FLOOR",
    "ABS",
    "GREATEST",
    "LEAST",
    "CAST",
}

ALLOWED_CAST_TYPES: Set[str] = {
    "numeric",
    "integer",
    "bigint",
    "real",
    "double precision",
    "text",
    "boolean",
    "date",
}

ALLOWED_KEYWORDS: Set[str] = {
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "AS",
    "AND",
    "OR",
    "NOT",
    "IS",
    "NULL",
    "TRUE",
    "FALSE",
    "IN",
}

# Patterns that indicate SQL injection or dangerous operations
BLOCKED_PATTERNS = [
    re.compile(r";"),                          # multiple statements
    re.compile(r"--"),                         # line comment
    re.compile(r"/\*"),                        # block comment
    re.compile(r"\bSELECT\b", re.I),          # subqueries
    re.compile(r"\bINSERT\b", re.I),
    re.compile(r"\bUPDATE\b", re.I),
    re.compile(r"\bDELETE\b", re.I),
    re.compile(r"\bDROP\b", re.I),
    re.compile(r"\bALTER\b", re.I),
    re.compile(r"\bCREATE\b", re.I),
    re.compile(r"\bEXEC(?:UTE)?\b", re.I),
    re.compile(r"\bUNION\b", re.I),
    re.compile(r"\bINTO\b", re.I),
    re.compile(r"\bFROM\b", re.I),
    re.compile(r"\bWHERE\b", re.I),
    re.compile(r"\bGROUP\b", re.I),
    re.compile(r"\bHAVING\b", re.I),
    re.compile(r"\bORDER\b", re.I),
    re.compile(r"\bLIMIT\b", re.I),
    re.compile(r"\bOFFSET\b", re.I),
    re.compile(r"\bJOIN\b", re.I),
    re.compile(r"\bSET\b", re.I),
    re.compile(r"\bGRANT\b", re.I),
    re.compile(r"\bREVOKE\b", re.I),
    re.compile(r"\bpg_", re.I),                # PostgreSQL system functions
    re.compile(r"\bdblink\b", re.I),
    re.compile(r"\blo_", re.I),                # large object functions
    re.compile(r"\\x[0-9a-f]", re.I),         # hex escapes
    re.compile(r"\bCOPY\b", re.I),
]

_FUNCTION_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CAST_TYPE_RE = re.compile(r"\bCAST\s*\([^)]*\bAS\s+([a-z ]+)\)", re.I)
_STRING_LITERAL_RE = re.compile(r"'[^']*'")
_NUMERIC_LITERAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def validate_expression(expression: str, allowed_columns: Set[str]) -> tuple[bool, str]:
    """Validate a computed column expression for safety.

    Args:
        expression: The SQL expression to validate.
        allowed_columns: Set of valid column names/aliases that can be referenced.

    Returns:
        Tuple of (is_safe, reason). is_safe is True if the expression is safe.
    """
    if not expression or not expression.strip():
        return False, "Expression cannot be empty"

    if len(expression) > 500:
        return False, "Expression too long (max 500 characters)"

    # Check blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(expression):
            return False, f"Expression contains blocked pattern: {pattern.pattern}"

    # Validate function calls
    for match in _FUNCTION_CALL_RE.finditer(expression):
        fn_name = match.group(1).upper()
        if fn_name not in ALLOWED_FUNCTIONS:
            return False, f'Function "{match.group(1)}" is not allowed in expressions'

    # Validate CAST types
    for match in _CAST_TYPE_RE.finditer(expression):
        cast_type = match.group(1).strip().lower()
        if cast_type not in ALLOWED_CAST_TYPES:
            return False, f'CAST type "{cast_type}" is not allowed'

    # Tokenize and validate identifiers
    stripped = expression
    stripped = _STRING_LITERAL_RE.sub("", stripped)      # remove string literals
    stripped = _NUMERIC_LITERAL_RE.sub("", stripped)     # remove numeric literals
    stripped = re.sub(r"[+\-*/(),.:]+", " ", stripped)   # remove operators/punctuation
    stripped = stripped.strip()

    if stripped:
        tokens = stripped.split()
        for token in tokens:
            upper = token.upper()
            if upper in ALLOWED_FUNCTIONS:
                continue
            if upper in ALLOWED_KEYWORDS:
                continue
            if token.lower() in ALLOWED_CAST_TYPES:
                continue
            # Check if it's a valid column reference (with optional table alias prefix)
            col_name = token.split(".")[-1] if "." in token else token
            if col_name not in allowed_columns and token not in allowed_columns:
                return False, (
                    f'Unknown identifier "{token}" in expression. '
                    f"Allowed columns: {', '.join(sorted(allowed_columns))}"
                )

    return True, "OK"


def validate_computed_columns(
    columns: list[dict], allowed_columns: Set[str]
) -> tuple[bool, str]:
    """Validate all computed column expressions.

    Args:
        columns: List of dicts with 'alias' and 'expression' keys.
        allowed_columns: Set of valid column names/aliases.

    Returns:
        Tuple of (is_safe, reason).
    """
    seen_aliases: Set[str] = set()
    for col in columns:
        alias = col.get("alias", "")
        expr = col.get("expression", "")

        if not alias or not alias.strip():
            return False, "Computed column must have an alias"

        if alias in seen_aliases:
            return False, f'Duplicate computed column alias: "{alias}"'
        seen_aliases.add(alias)

        is_safe, reason = validate_expression(expr, allowed_columns)
        if not is_safe:
            return False, f'Computed column "{alias}": {reason}'

    return True, "OK"
