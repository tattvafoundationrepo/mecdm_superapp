"""Shared Vertex AI GenAI client.

Centralizes the Client creation that was previously duplicated in
alloydb/tools.py (module-level) and tools.py (_get_v2_llm_client).
"""

import os

from google.genai import Client
from google.genai.types import HttpOptions

from data_science.utils.utils import USER_AGENT

_client: Client | None = None


def get_llm_client() -> Client:
    """Return a singleton Vertex AI GenAI client."""
    global _client
    if _client is None:
        _client = Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT", None),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            http_options=HttpOptions(headers={"user-agent": USER_AGENT}),
        )
    return _client
