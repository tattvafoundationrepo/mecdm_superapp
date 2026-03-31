# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file initializes a FastAPI application for Data Science agent
using get_fast_api_app() from ADK. Session service URI and a flag
for a web interface configured via environment variables.
It can then be run using Uvicorn, which listens on a port specified by
the PORT environment variable or defaults to 8080.
This approach offers more flexibility, particularly if you want to
embed Data Science agent within a custom FastAPI application.
It is used for Cloud Run deployment with standard gcloud run deploy command.
"""

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.cloud import logging as google_cloud_logging

# Load environment variables from .env file
load_dotenv()

logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get session service URI from environment variables
# Prefer explicit SESSION_SERVICE_URI, fall back to DATABASE_URL_USER
# Note: ADK's DatabaseSessionService uses synchronous create_engine (psycopg2),
# so we pass the postgresql:// URL as-is — do NOT convert to asyncpg.
session_uri = os.getenv("SESSION_SERVICE_URI")
if not session_uri:
    session_uri = os.getenv("DATABASE_URL_USER")

# Get Enable Web interface serving flag from environment variables
# Set web=True if you intend to serve a web interface, False otherwise
web_interface_enabled = os.getenv("SERVE_WEB_INTERFACE", "False").lower() in (
    "true",
    "1",
)

# Prepare arguments for get_fast_api_app
app_args = {"agents_dir": AGENT_DIR, "web": web_interface_enabled}

# Only include session_service_uri if it's provided
if session_uri:
    app_args["session_service_uri"] = session_uri
else:
    logger.log_text(
        "SESSION_SERVICE_URI and DATABASE_URL_USER not provided. "
        "Using in-memory session service instead. "
        "All sessions will be lost when the server restarts.",
        severity="WARNING",
    )

# Memory Bank configuration (Vertex AI Agent Engine)
memory_bank_agent_engine_id = os.getenv("MEMORY_BANK_AGENT_ENGINE_ID")
memory_bank_location = os.getenv("MEMORY_BANK_LOCATION", "asia-south1")
if memory_bank_agent_engine_id:
    app_args["memory_service_uri"] = (
        f"agentengine://projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}"
        f"/locations/{memory_bank_location}"
        f"/reasoningEngines/{memory_bank_agent_engine_id}"
    )

# Create FastAPI app with appropriate arguments
app: FastAPI = get_fast_api_app(**app_args)

app.title = "data_science"
app.description = "Data Science Agent"

# Mount chat and feedback routers
from data_science.routers.chat import router as chat_router
from data_science.routers.feedback import router as feedback_router

app.include_router(chat_router)
app.include_router(feedback_router)

if __name__ == "__main__":
    # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
