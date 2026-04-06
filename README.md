# MECDM SuperApp Backend

AI-powered health intelligence platform for maternal and child health (MCH) decision-making in Meghalaya, India. Built with **Google ADK**, **Vertex AI Gemini**, and **AlloyDB with PostGIS**.

## What It Does

The backend powers a conversational AI agent that lets users query maternal-child health data using natural language. It translates questions into SQL, runs analytics, generates geographic visualizations, searches policy documents, and returns structured JSON for frontend rendering.

**Three user personas:**
- **Government Officials** -- aggregate analytics, trend analysis, district comparisons
- **Frontline Health Workers (ASHA/ANM)** -- beneficiary tracking, facility-level data
- **Citizens** -- health awareness and service availability

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI + Google ADK (Agent Development Kit) |
| LLM | Vertex AI Gemini 2.5 Pro (root), 2.5 Flash (sub-agents) |
| Primary Database | AlloyDB with PostGIS (spatial queries) |
| Database Access | MCP Toolbox for Databases (`toolbox-core`) |
| User/Session DB | PostgreSQL with SQLAlchemy |
| Policy Search | Vertex AI RAG Engine |
| Code Execution | Vertex AI Code Executor |
| Observability | OpenTelemetry, Google Cloud Logging |
| Deployment | Google Cloud Run, Docker |
| Package Manager | uv |

## Architecture

```
                         User Query (natural language)
                                    |
                              FastAPI Router
                             (chat.py endpoints)
                                    |
                          ADK App / Root Agent
                        (Gemini 2.5 Pro, temp=0.01)
                                    |
                 +------------------+------------------+
                 |                  |                  |
          AlloyDB Agent      Analytics Agent     Direct Tools
          (NL2SQL)           (NL2Py)             (14 tools)
                 |                  |
          MCP Toolbox        Vertex AI Code
          execute_sql        Executor
                 |                  |
            AlloyDB             Python sandbox
         (PostGIS enabled)      (pandas, numpy, scipy)
                                    |
                          Structured JSON Response
                       (mecdm_viz / mecdm_stat blocks)
                                    |
                            Frontend Rendering
```

### Agent Hierarchy

**Root Agent** orchestrates two specialized sub-agents via AgentTools, with multi-persona support (Decision Maker, Frontline Worker, Citizen, Analyst):

- **AlloyDB Agent** -- translates natural language to PostgreSQL, executes queries via MCP Toolbox, returns tabular results
- **Analytics Agent** -- takes query results, generates Python code for statistical analysis, returns structured JSON output

### Root Agent Tools (14)

| Tool | Purpose |
|---|---|
| `call_alloydb_agent` | Route data questions to the NL2SQL sub-agent |
| `call_analytics_agent` | Route data for Python-based statistical analysis |
| `generate_stat_query` | Build StatQuery V2 JSON blocks for interactive frontend charts |
| `find_nearest_facilities` | Spatial distance queries (nearest PHC/SC/CHC/DH/AWC) |
| `search_policy_rag_engine` | Search Meghalaya Government policy documents |
| `recommend_video` | Search and recommend videos from the curated video library |
| `get_stats_schema_summary` | List stats-eligible tables and columns |
| `get_predefined_stats_catalog` | Return 10 predefined KPI templates |
| `read_uploaded_file` | Read and extract text from uploaded files (PDF, DOCX, XLSX, images) |
| `export_data_to_csv` | Export query results to CSV |
| `get_current_datetime` | Current date/time for relative queries |
| `get_weather_data` | Current weather via Open-Meteo API |
| `get_historical_weather_data` | Historical weather trends |
| `google_search` | ADK built-in web search |

## Request Flow

**Data retrieval example** ("Show ANC coverage by district"):

1. User sends natural language query via chat API
2. FastAPI router passes to ADK App / Root Agent
3. Root agent invokes `call_alloydb_agent` with the question
4. AlloyDB agent calls `alloydb_nl2sql` (Gemini 2.5 Flash generates PostgreSQL from schema + golden examples)
5. Agent calls `run_alloydb_query` (executes via MCP Toolbox `execute_sql`)
6. Results stored in `tool_context.state["alloydb_query_result"]`
7. Root agent optionally calls `call_analytics_agent` for further processing
8. Root agent formats response with `mecdm_viz` JSON blocks
9. Response saved to user DB, returned to frontend

**Map generation example** ("Show a map of institutional deliveries by district"):

1. `call_alloydb_agent` retrieves metric data with join key (e.g., `district_code_lgd`)
2. Root agent emits an `mecdm_map` config block specifying geography level, metric column, and join key
3. Frontend fetches PostGIS geometry via API (`ST_AsGeoJSON`)
4. Frontend joins metric data to geometry on LGD codes
5. Optionally includes facility/AWC overlay configuration
6. Frontend renders the interactive choropleth/bubble map

## Visualization Output

The agent returns structured JSON blocks (not images) for the frontend to render:

| Block Type | Description |
|---|---|
| `mecdm_stat` | StatQuery V2 -- structured query builder with source, dimensions, measures, computedColumns, filters, and chart config. Preferred for all aggregations. |
| `mecdm_map` | Map config block -- specifies geography level, metric column, join key, map type (choropleth/bubble), and optional facility/AWC overlays. Frontend fetches geometry and renders. |
| `mecdm_viz:chart` | Pre-computed bar, line, pie, area charts |
| `mecdm_viz:stat_cards` | KPI cards with value, trend, and color |
| `mecdm_viz:table` | Structured data tables |

## Project Structure

```
backend/
├── main.py                              # Cloud Run entry point
├── data_science/
│   ├── agent.py                         # Root agent definition, dataset config loading
│   ├── tools.py                         # 14 root-level tools
│   ├── fast_api_app.py                  # FastAPI app factory (production)
│   │
│   ├── prompts/
│   │   └── prompt_builder.py            # Modular prompt system (Persona, PromptConfig, DatasetConfig)
│   │
│   ├── sub_agents/
│   │   ├── alloydb/
│   │   │   ├── agent.py                 # NL2SQL sub-agent
│   │   │   ├── prompts.py               # NL2SQL instructions
│   │   │   └── tools.py                 # SQL generation + query execution
│   │   └── analytics/
│   │       ├── agent.py                 # Analytics sub-agent (Vertex AI Code Executor)
│   │       └── prompts.py               # NL2Py instructions
│   │
│   ├── routers/
│   │   ├── chat.py                      # Chat session & message endpoints
│   │   ├── feedback.py                  # User feedback endpoints
│   │   ├── upload.py                    # File upload endpoint (GCS)
│   │   └── whatsapp.py                  # WhatsApp webhook (Meta Cloud API)
│   │
│   ├── utils/
│   │   ├── map_utils.py                 # PostGIS geometry joining, GeoJSON building
│   │   └── utils.py                     # General utilities
│   │
│   ├── services/
│   │   ├── file_processor.py            # File upload validation, GCS storage, text extraction
│   │   ├── whatsapp_service.py          # WhatsApp channel: citizen agent, Meta API, sessions
│   │   └── whatsapp_formatter.py        # Agent response → WhatsApp text formatting
│   │
│   └── app_utils/
│       ├── models.py                    # SQLAlchemy models (ChatSession, ChatMessage, Feedback, UserPreferences, GoldenSql)
│       ├── whatsapp_models.py           # Pydantic models for Meta webhook payloads
│       ├── sql_validator.py             # pglast-based SQL injection prevention
│       ├── expression_validator.py      # Expression validation
│       ├── telemetry.py                 # OpenTelemetry + Google Cloud Logging
│       ├── user_db.py                   # Async user DB session factory (asyncpg)
│       └── typing.py                    # Type definitions
│
├── dataset_mecdm.json                   # Dataset config (25 tables, domains, stats rules)
├── cross_dataset_relations.json         # 13 foreign key relationships
├── pyproject.toml                       # Dependencies
├── Makefile                             # Dev, deploy, test commands
├── Dockerfile                           # Container build
├── GEMINI.md                            # Developer guide
├── .env.example                         # Environment template
├── migrate_mecdm.py                     # Data migration script
├── toolbox-alloydb-local.yaml           # MCP Toolbox config (local)
├── toolbox-alloydb-remote.yaml          # MCP Toolbox config (Cloud Run)
├── tests/                               # Unit, integration, load, eval tests
├── eval/                                # Agent evaluation
├── migrations/                          # Database migrations
├── deployment/                          # Deploy automation (deploy.py, terraform/)
├── infra/                               # Infrastructure config
├── notebooks/                           # Jupyter notebooks (ADK testing, evaluation)
└── .github/workflows/                   # CI/CD (pr_checks, staging, prod deploy)
```

## Data Schema

25 tables across 6 categories in AlloyDB:

| Category | Tables |
|---|---|
| Geographic Boundaries (PostGIS) | `states`, `districts`, `subdistricts`, `villages_poly`, `villages_point` |
| Master Reference (LGD codes) | `master_districts`, `master_blocks`, `master_villages`, `master_health_facilities` |
| Health Infrastructure | `master_health_facilities`, `anganwadi_centres` |
| MCH Tracking (core) | `mother_journeys`, `anc_visits`, `village_indicators_monthly` |
| Raw Source Records | `raw_pregnancy_records`, `raw_anc_records`, `raw_child_records` |
| Survey & Reference | `nfhs_indicators`, `research_articles`, `video_library`, `geo_full_mapping` |

**Key schema notes:**
- Census tables use `geometry` column; master tables use `geom`
- Health data uses UPPERCASE names; census uses Title Case
- Prefer integer code joins (`district_code_lgd`, `block_code_lgd`) over name joins
- `mother_journeys` / `anc_visits`: all columns are TEXT -- cast to numeric/date as needed

## WhatsApp Integration (Citizen Channel)

The backend includes a WhatsApp channel that exposes the **Citizen persona** agent via Meta's official WhatsApp Business Cloud API. Citizens can query health facility locations, program eligibility, and general health information directly from WhatsApp -- no app install required.

### Architecture

```
               WhatsApp User
                    |
            Meta Cloud API
                    |
     POST /whatsapp/webhook (FastAPI)
                    |
            +----- 200 OK (immediate) -----+
            |                               |
     BackgroundTask                   (Meta satisfied)
            |
     WhatsAppService (singleton)
            |
    +-------+--------+--------+
    |        |        |        |
  Session  ADK      Format   Meta API
  Lookup   Runner   Response  (send reply)
    |        |
  ADK DB   Citizen Agent
           (Gemini 2.5, Persona=CITIZEN)
```

### How It Works

1. **Meta sends a POST** to `/whatsapp/webhook` with the user's message
2. **Router returns 200 immediately** (Meta requires response within 5 seconds) and queues processing as a `BackgroundTask`
3. **WhatsAppService** looks up or creates an ADK session for the phone number (user ID format: `wa_{phone}`)
4. A **dedicated Citizen-persona agent** (separate from the web agent) processes the message through the same tool pipeline (AlloyDB, policy search, facility finder, etc.)
5. The agent response is **formatted for WhatsApp**: visualization blocks (`mecdm_stat`, `mecdm_viz`, `mecdm_map`) are stripped, markdown is converted to WhatsApp formatting (`*bold*`, `_italic_`), and long messages are split at paragraph boundaries (max 4096 chars per message)
6. The formatted response is **sent back via Meta Cloud API**

### Key Design Decisions

- **Separate agent instance** -- The WhatsApp service creates its own `LlmAgent` with `Persona.CITIZEN` hardcoded, so the web agent's persona configuration is unaffected
- **Session isolation** -- Uses `app_name="whatsapp_citizen"` to namespace WhatsApp sessions separately from web sessions in the same database
- **Per-phone locking** -- An `asyncio.Lock` per phone number prevents race conditions when a user sends multiple messages rapidly
- **No visualization blocks** -- Since WhatsApp can't render charts/maps, the agent's `include_visualization_guide` is set to `False` and any remaining viz blocks are stripped in the formatter
- **Graceful error handling** -- If the agent fails, the user receives a friendly "try again" message rather than silence

### Files

| File | Purpose |
|---|---|
| `data_science/routers/whatsapp.py` | Webhook endpoints (GET for verification, POST for messages) |
| `data_science/services/whatsapp_service.py` | Core service: citizen agent, ADK Runner, session management, Meta API calls |
| `data_science/services/whatsapp_formatter.py` | Response formatting: strip viz blocks, markdown-to-WhatsApp conversion, message splitting |
| `data_science/app_utils/whatsapp_models.py` | Pydantic models for Meta webhook payloads |

### Environment Variables

```bash
# WhatsApp Business API (Meta Cloud API)
WHATSAPP_VERIFY_TOKEN=<random-string-you-choose>       # Webhook verification token
WHATSAPP_ACCESS_TOKEN=<meta-access-token>               # From Meta Developer Console
WHATSAPP_PHONE_NUMBER_ID=<phone-number-id>              # From Meta Business Manager
WHATSAPP_API_VERSION=v21.0                              # Meta Graph API version (optional)
```

### Setup Guide

#### 1. Create a Meta Business App

1. Go to [Meta for Developers](https://developers.facebook.com/) and create a new app (type: **Business**)
2. Add the **WhatsApp** product to your app
3. In the WhatsApp section, you'll find:
   - **Temporary access token** (for testing; generate a permanent one via System User for production)
   - **Phone number ID** (the ID of your WhatsApp Business phone number)
   - **Test phone number** (Meta provides a free test number for development)

#### 2. Configure the Webhook

1. In your Meta App dashboard, go to **WhatsApp > Configuration**
2. Click **Edit** on the Webhook section
3. Enter your webhook URL: `https://<your-domain>/whatsapp/webhook`
   - For local development, use [ngrok](https://ngrok.com/): `ngrok http 8000` and use the generated HTTPS URL
4. Enter your `WHATSAPP_VERIFY_TOKEN` (the same value you set in `.env`)
5. Click **Verify and Save**
6. Subscribe to the **messages** webhook field

#### 3. Add Environment Variables

Add the following to your `.env` file:

```bash
WHATSAPP_VERIFY_TOKEN=my-secret-verify-token
WHATSAPP_ACCESS_TOKEN=EAAxxxxxxxxx...
WHATSAPP_PHONE_NUMBER_ID=123456789012345
```

#### 4. Test the Integration

```bash
# Test webhook verification
curl "http://localhost:8000/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=my-secret-verify-token&hub.challenge=test123"
# Expected: test123

# Start the backend
make local-backend

# Send a test message from WhatsApp to your business number
# Example: "Where is the nearest health facility in Shillong?"
```

#### 5. Production Deployment

For production use with Cloud Run:

1. **Generate a permanent access token**: Create a System User in Meta Business Manager, assign WhatsApp permissions, and generate a token
2. **Set environment variables** in your Cloud Run service (or Secret Manager)
3. **Configure the webhook URL** to point to your Cloud Run service: `https://<cloud-run-url>/whatsapp/webhook`
4. **Request API access**: Submit your app for App Review to get production-level messaging access (beyond the 5 test numbers allowed in development mode)

### Response Formatting

The formatter handles the conversion from rich agent output to WhatsApp-compatible plain text:

| Agent Output | WhatsApp Output |
|---|---|
| `mecdm_stat` / `mecdm_viz` / `mecdm_map` blocks | Replaced with "[View detailed visualization on the MECDM web portal]" |
| `**bold text**` | `*bold text*` (WhatsApp bold) |
| `[link text](url)` | `link text (url)` |
| `### Header` | `*Header*` (bold) |
| `- bullet item` | `bullet item` |
| Messages > 4096 chars | Split at paragraph boundaries into multiple messages |

### Session Continuity

Each WhatsApp phone number gets a persistent ADK session. This means:

- Follow-up questions work naturally ("What about East Khasi Hills?" after asking about district coverage)
- The agent remembers context from previous messages in the conversation
- Sessions are stored in the same PostgreSQL database as web sessions (but namespaced under `app_name="whatsapp_citizen"`)

---

## Design Decisions

- **JSON-first visualization** -- returns structured data, not images. Frontend controls rendering.
- **Privacy by design** -- always aggregates data, never exposes individual identifiers. Read-only database access.
- **SQL injection prevention** -- all generated SQL validated through `pglast` (PostgreSQL C parser) before execution.
- **Spatial-first** -- PostGIS integration for geometry queries, facility proximity, and map generation as first-class features.
- **Few-shot NL2SQL** -- golden SQL examples from a curated database guide the LLM's SQL generation.
- **Multi-persona routing** -- root agent prompt encodes MCH domain knowledge (red flag thresholds, derived metrics) and adapts responses by user role (Decision Maker, Frontline Worker, Citizen, Analyst).
- **Modular prompt system** -- prompts are assembled dynamically from configuration (Persona, DatasetConfig, RelationsConfig) rather than hardcoded strings.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Google Cloud account with Vertex AI enabled
- AlloyDB instance with PostGIS
- MCP Toolbox for Databases (local or Cloud Run)

### Installation

```bash
# Install dependencies
make install

# Configure environment
cp .env.example .env
# Edit .env with your credentials (see below)
```

### Environment Variables

```bash
# Vertex AI
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=your-region

# Dataset config
DATASET_CONFIG_FILE=dataset_mecdm.json

# Models
ROOT_AGENT_MODEL=gemini-2.5-pro
ANALYTICS_AGENT_MODEL=gemini-2.5-pro
BASELINE_NL2SQL_MODEL=gemini-2.5-pro
ALLOYDB_AGENT_MODEL=gemini-2.5-pro

# Agent persona
AGENT_PERSONA=decision_maker  # decision_maker|frontline_worker|citizen|analyst

# AlloyDB / MCP Toolbox
ALLOYDB_TOOLSET=postgres-database-tools
ALLOYDB_SCHEMA_NAME=public
ALLOYDB_DATABASE=your-db-name
ALLOYDB_PROJECT_ID=your-project-id
MCP_TOOLBOX_HOST=localhost
MCP_TOOLBOX_PORT=5000

# User database (auth, chat sessions, feedback)
DATABASE_URL_USER=postgresql://user:pass@host/dbname

# Cross-dataset relations
CROSS_DATASET_RELATIONS_DEFS=./cross_dataset_relations.json

# Privacy & CORS
STRICT_PRIVACY=true
ALLOW_ORIGINS=http://localhost:3000
```

### Running Locally

```bash
# Start the backend (default port 8000)
make local-backend

# Or specify a port
make local-backend PORT=8001

# Launch the ADK playground UI (port 8501)
make playground
```

### Testing

```bash
# Run unit + integration tests
make test

# Run agent evaluation
make eval

# Run all evalsets
make eval-all

# Lint
make lint
```

## Deployment

Deploy to Google Cloud Run:

```bash
# Standard deployment
make deploy

# With Identity-Aware Proxy
make deploy IAP=true
```

Infrastructure setup via Terraform:

```bash
make setup-dev-env
```

## Optimizing and Adjustment Tips

- **Prompt Engineering:** Refine the modular prompt system in `data_science/prompts/prompt_builder.py` and sub-agent prompts (`sub_agents/alloydb/prompts.py`, `sub_agents/analytics/prompts.py`) to improve accuracy. The root agent prompt is built dynamically from Persona, DatasetConfig, and RelationsConfig -- adjust these to match evolving data or requirements.
- **Golden SQL Examples:** Add curated question-SQL pairs to the `GoldenSql` table in the user database. The AlloyDB agent uses these as few-shot examples for NL2SQL generation -- more examples directly improve query accuracy.
- **Model Selection:** Swap models per agent via environment variables (`ROOT_AGENT_MODEL`, `ALLOYDB_AGENT_MODEL`, `ANALYTICS_AGENT_MODEL`). Use a heavier model (e.g., `gemini-2.5-pro`) for the root agent and lighter models (e.g., `gemini-2.5-flash`) for sub-agents to balance quality and latency.
- **Dataset Configuration:** Edit `dataset_mecdm.json` to add/remove tables, update column descriptions, or adjust cross-dataset relationships in `cross_dataset_relations.json`. Clear descriptions on tables and columns significantly boost NL2SQL performance.
- **Adding Tools:** Add new tools to the root agent in `data_science/tools.py` and register them in `data_science/agent.py`. Each tool should return a string result.
- **Adding Sub-Agents:** Create a new directory under `data_science/sub_agents/` with `agent.py`, `prompts.py`, and `tools.py`, then register it as an `AgentTool` in the root agent.
- **Visualization Schemas:** The `mecdm_viz` and `mecdm_stat` JSON schemas are defined in the root agent prompt. Modify them there to add new chart types or data fields.

## Configuration Files

| File | Purpose |
|---|---|
| `dataset_mecdm.json` | Defines all 25 tables, their descriptions, columns, domains, and stats rules |
| `cross_dataset_relations.json` | 13 foreign key relationships between tables (e.g., `mother_journeys` <-> `anc_visits` via `mother_id`) |
| `toolbox-alloydb-local.yaml` | MCP Toolbox config for local AlloyDB connection |
| `toolbox-alloydb-remote.yaml` | MCP Toolbox config for Cloud Run deployment |
