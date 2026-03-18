# Copyright 2026 Google LLC
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

import os
from urllib.parse import quote

import google.auth
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.cloud import logging as google_cloud_logging

from data_science.app_utils.telemetry import setup_telemetry
from data_science.routers.chat import router as chat_router
from data_science.routers.feedback import router as feedback_router

setup_telemetry()
_, project_id = google.auth.default()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(
        ",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cloud SQL session configuration
db_user = os.environ.get("DB_USER", "postgres")
db_name = os.environ.get("DB_NAME", "postgres")
db_pass = os.environ.get("DB_PASS")
instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME")

session_service_uri = None

# Prefer DATABASE_URL_USER (direct TCP — works locally and in production)
# Note: ADK's DatabaseSessionService uses synchronous create_engine (psycopg2),
# so we pass the postgresql:// URL as-is — do NOT convert to asyncpg.
db_url_user = os.environ.get("DATABASE_URL_USER")
if db_url_user:
    session_service_uri = db_url_user
elif instance_connection_name and db_pass:
    # Fallback: Cloud SQL Unix socket (only works on Cloud Run with proxy)
    encoded_user = quote(db_user, safe="")
    encoded_pass = quote(db_pass, safe="")
    session_service_uri = (
        f"postgresql://{encoded_user}:{encoded_pass}@"
        f"/{db_name}"
        f"?host=/cloudsql/{instance_connection_name}"
    )

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=False,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=True,
)
app.title = "mecdm-super-agent"
app.description = "API for interacting with the Agent mecdm-super-agent"

# Mount chat and feedback routers
app.include_router(chat_router)
app.include_router(feedback_router)


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
