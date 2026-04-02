"""
Task-Specific Prompt Blocks for Dynamic Prompting

Each block provides focused guidance that is injected into the system instruction
when the user's message matches the corresponding task type.
"""

TASK_BLOCKS: dict[str, str] = {
    "data_query": """
<TASK_GUIDANCE: DATA_QUERY>
This question requires data retrieval. Follow this optimized path:

1. **Identify the table** from your schema context. For MCH aggregates use `village_indicators_monthly`.
   For individual journeys use `mother_journeys` or `anc_visits` (remember: ALL columns are TEXT — cast with ::NUMERIC or ::DATE).
2. **Direct MCP path (preferred for 1-2 tables):**
   `list_tables(schema_names="public", table_names="<table>")` → inspect columns → `execute_sql(<query>)`
3. **Delegate for complex queries (3+ tables, ambiguous):** `call_alloydb_agent(question)`
4. **Return data first**, then optionally visualize. Never skip showing the actual numbers.

SQL reminders:
- Double-quote table names: `"mother_journeys"`, `"village_indicators_monthly"`
- Use integer code joins (`district_code_lgd`, `block_code_lgd`) over name joins
- District names in mother_journeys/anc_visits are UPPERCASE
- Always include WHERE or LIMIT — never scan full tables
</TASK_GUIDANCE>
""".strip(),

    "analysis": """
<TASK_GUIDANCE: ANALYSIS>
This question requires statistical analysis or trend computation.

Workflow:
1. **Retrieve raw data** first via MCP tools or `call_alloydb_agent`
2. **Delegate computation** to `call_analytics_agent(question)` for:
   - Time-series trends and seasonality
   - Correlations between indicators
   - Growth rates (month-over-month, year-over-year)
   - Rankings with statistical context
   - Outlier detection
3. **Report findings** with:
   - Sample sizes and time periods covered
   - Data quality caveats (missing months, small denominators)
   - Comparison to baselines (state targets, NFHS benchmarks)
4. **Visualize** via `generate_stat_query` if the user benefits from a chart

For trends: always specify the time column (`year_month`) and aggregation level (district/block).
For rates: use NULLIF to avoid division by zero, note when denominators are small (<30).
</TASK_GUIDANCE>
""".strip(),

    "visualization": """
<TASK_GUIDANCE: VISUALIZATION>
This question requests a visual output. Use `generate_stat_query` to build StatQuery V2 JSON.

Chart selection rules:
- **grouped_bar**: Compare ONE metric across MULTIPLE entities over time (DEFAULT for "compare districts by month")
- **stacked_bar**: Show composition/parts-of-whole over time
- **bar**: Single-dimension ranking (no time axis)
- **line**: Single entity trend over time, or ≤3 series
- **pie/donut**: Proportions of a whole (≤7 slices)
- **kpi_card**: Single headline number
- **table**: Detailed multi-column data, exact numbers

Steps:
1. Call `generate_stat_query(question)` — it returns STAT_QUERY_JSON + QUERY_RESULTS
2. Base your textual insights on the actual QUERY_RESULTS, not guesses
3. Embed the returned JSON directly as the "query" field in your `mecdm_stat` block
4. Do NOT rewrite or re-create the query JSON — only add chart, name, description around it
5. For multiple views (table + chart), reuse the SAME query JSON
</TASK_GUIDANCE>
""".strip(),

    "comparison": """
<TASK_GUIDANCE: COMPARISON>
This question compares multiple entities (districts, blocks, indicators).

Patterns:
- **District/block ranking**: Use `bar` chart with `orderBy` desc/asc
- **Entity vs entity over time**: Use `grouped_bar` with `groupBy` = entity dimension
- **Before/after comparison**: Include both time periods in filters, compute difference
- **Top/bottom N**: Use `limit` + `orderBy` in StatQuery V2

Always:
- Include the state average as a reference point when comparing districts
- Highlight entities exceeding red-flag thresholds
- Use consistent ordering (descending by the primary metric unless user specifies otherwise)
- For "which district is best/worst" queries: retrieve data first, then visualize the ranking
</TASK_GUIDANCE>
""".strip(),

    "geographic": """
<TASK_GUIDANCE: GEOGRAPHIC>
This question involves spatial or location-based analysis.

Tools:
- **`find_nearest_facilities`**: For "nearest PHC/AWC to village X". Returns an mecdm_map block — include it verbatim.
  Params: from_village, to_type (PHC/SC/CHC/DH/AWC/ANY_FACILITY/ANY), count, from_district, from_block
- **`mecdm_map` block**: For choropleth/bubble maps showing metric distribution
  - `geographyLevel`: district (12), block (46), village (~2600)
  - `joinKey`: dimension alias for geometry join (usually district_name or block_name)
  - `overlays.facilities`: true to show health facility markers

For map queries:
1. Build the data query (StatQuery V2 format) for the metric
2. Wrap it in an `mecdm_map` block with appropriate mapType and geographyLevel
3. You do NOT need to call call_alloydb_agent first — the frontend executes the query
4. For block-level maps within a district, add `geoFilter` with the district name
</TASK_GUIDANCE>
""".strip(),

    "policy": """
<TASK_GUIDANCE: POLICY>
This question involves policy recommendations or program guidance.

Workflow:
1. **Retrieve supporting data** first — recommendations must be grounded in evidence
2. Call `search_policy_rag_engine(query)` to find relevant policy documents and guidelines
3. **Format recommendations** as:
   **<Priority>**: <Finding with data> → <Action> (<Policy citation>)
   Priorities: Critical, Warning, Moderate. Max 3-5 items, 2 lines each.
4. Compare current metrics against state targets (2025-26):
   - Institutional Delivery Rate: ≥80%
   - 1st Trimester Registration: ≥70%
   - 4+ ANC visits: ≥75%
   - Full immunization: ≥90%

Do NOT call search_policy_rag_engine for simple factual questions. Only invoke it when
the user asks for recommendations, action plans, or when data reveals red-flag thresholds.
</TASK_GUIDANCE>
""".strip(),

    "general": """
<TASK_GUIDANCE: GENERAL>
This is a general/conversational message (greeting, help request, or capabilities inquiry).

- Respond briefly and conversationally
- If asked about capabilities, describe the types of analysis you can perform
- Do NOT invoke data retrieval tools for greetings or meta-questions
- If the user's intent is unclear, ask a clarifying question and suggest example queries
</TASK_GUIDANCE>
""".strip(),
}
