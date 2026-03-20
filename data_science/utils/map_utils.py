"""Map utility functions — retained parsing helpers.

Most map rendering logic has moved to the frontend (see frontend/src/lib/map/).
The backend now emits lightweight mecdm_map config blocks instead of full GeoJSON.
"""

import json
import logging

logger = logging.getLogger(__name__)


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
