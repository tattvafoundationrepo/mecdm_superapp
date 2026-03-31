"""Compile StatQuery V2 JSON to PostgreSQL SQL.

A simplified backend compiler for executing stat queries and returning
preview data. The frontend has a full-featured compiler in
frontend/src/lib/stats/query-builder.ts; this handles the common cases
so the agent can see actual results before responding.

This module is a pure function (dict in, string out) with no I/O or
framework dependencies.
"""


def quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier."""
    return f'"{identifier.replace(chr(34), chr(34)+chr(34))}"'


OPERATOR_SQL = {
    "eq": "=",
    "neq": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "like": "LIKE",
}

TIME_PRESETS = {
    "last_7d": "7 days",
    "last_30d": "30 days",
    "last_quarter": "3 months",
    "last_year": "1 year",
    "ytd": "ytd",
}


def compile_stat_query_v2_to_sql(query: dict) -> str:
    """Compile a StatQuery V2 dict into a PostgreSQL string.

    Handles: dimensions, measures, computed columns (two-layer query),
    filters, HAVING, JOINs (with caseInsensitive), timeRange presets,
    ORDER BY, and LIMIT.

    Args:
        query: A validated StatQuery V2 dict with at minimum
               ``source.table`` and either dimensions or measures.

    Returns:
        A PostgreSQL SELECT statement string.
    """
    qi = quote_identifier
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
        col_ref = f'{qi(alias)}.{qi(dim["column"])}'
        transform = dim.get("transform")
        if transform == "date_trunc_month":
            col_ref = f"DATE_TRUNC('month', {col_ref})"
        elif transform == "date_trunc_quarter":
            col_ref = f"DATE_TRUNC('quarter', {col_ref})"
        elif transform == "date_trunc_year":
            col_ref = f"DATE_TRUNC('year', {col_ref})"
        dim_alias = dim.get("alias", dim["column"])
        select_parts.append(f"{col_ref} AS {qi(dim_alias)}")
        inner_aliases.append(dim_alias)

    # Track alias→aggregate expression for HAVING
    alias_to_agg: dict[str, str] = {}

    for m in measures:
        col_ref = f'{qi(alias)}.{qi(m["column"])}'
        agg = m["aggregate"]
        if agg == "count_distinct":
            agg_expr = f"COUNT(DISTINCT {col_ref})"
        else:
            agg_expr = f"{agg.upper()}({col_ref})"
        m_alias = m.get("alias", m["column"])
        select_parts.append(f"{agg_expr} AS {qi(m_alias)}")
        inner_aliases.append(m_alias)
        alias_to_agg[m_alias] = agg_expr

    # --- FROM ---
    from_clause = f"{qi(table)} AS {qi(alias)}"
    joins = query.get("source", {}).get("joins", [])
    for i, join in enumerate(joins):
        j_alias = f"t{i + 1}"
        j_type = join.get("type", "inner").upper()
        left_ref = f'{qi(alias)}.{qi(join["on"]["left"])}'
        right_ref = f'{qi(j_alias)}.{qi(join["on"]["right"])}'
        if join.get("caseInsensitive"):
            on_clause = f"UPPER({left_ref}) = UPPER({right_ref})"
        else:
            on_clause = f"{left_ref} = {right_ref}"
        from_clause += f" {j_type} JOIN {qi(join['table'])} AS {qi(j_alias)} ON {on_clause}"

    # --- WHERE ---
    where_parts: list[str] = []

    if time_range and time_range.get("column"):
        col = qi(time_range["column"])
        preset = time_range.get("preset", "all")
        custom = time_range.get("custom")
        if preset != "all" and not custom:
            pg_interval = TIME_PRESETS.get(preset)
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
        col = f'{qi(alias)}.{qi(f["column"])}'
        op = f.get("operator", "eq")
        val = f.get("value")
        if op in OPERATOR_SQL:
            if isinstance(val, str):
                where_parts.append(f"{col} {OPERATOR_SQL[op]} '{val}'")
            else:
                where_parts.append(f"{col} {OPERATOR_SQL[op]} {val}")
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
        h_op = OPERATOR_SQL.get(h.get("operator", "eq"), "=")
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
                ob_parts.append(f"{qi(o['column'])} {direction}")
            order_str = f"ORDER BY {', '.join(ob_parts)}"

        sql = f"SELECT {', '.join(select_parts)} FROM {from_clause} {where_str} {group_by_str} {having_str} {order_str} LIMIT {limit}"
        return " ".join(sql.split())

    # Two-layer query: inner has dimensions+measures, outer adds computed columns
    inner_sql = f"SELECT {', '.join(select_parts)} FROM {from_clause} {where_str} {group_by_str} {having_str}"

    outer_select = [qi(a) for a in inner_aliases]
    for cc in computed_columns:
        outer_select.append(f"({cc['expression']}) AS {qi(cc['alias'])}")

    # Outer HAVING (for computed column filters)
    outer_where_parts: list[str] = []
    for h in having:
        if h["column"] in computed_alias_set:
            h_op = OPERATOR_SQL.get(h.get("operator", "eq"), "=")
            h_val = h.get("value")
            expr = next(
                (cc["expression"] for cc in computed_columns if cc["alias"] == h["column"]),
                qi(h["column"]),
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
            ob_parts.append(f"{qi(o['column'])} {direction}")
        order_str = f"ORDER BY {', '.join(ob_parts)}"

    sql = f"SELECT {', '.join(outer_select)} FROM ({inner_sql}) AS _inner {outer_where} {order_str} LIMIT {limit}"
    return " ".join(sql.split())
