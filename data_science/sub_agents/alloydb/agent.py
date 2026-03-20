"""Database Agent: get data from database (BigQuery) using NL2SQL."""

import logging
import os

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from . import tools
from .prompts import return_instructions_alloydb

logger = logging.getLogger(__name__)


def setup_before_agent_call(callback_context: CallbackContext) -> None:
    """Setup the agent."""
    logger.debug("setup_before_agent_call")

    if "database_settings" not in callback_context.state:
        callback_context.state["database_settings"] = (
            tools.get_database_settings()
        )


alloydb_agent = LlmAgent(
    model=os.getenv("ALLOYDB_AGENT_MODEL", ""),
    name="alloydb_agent",
    instruction=return_instructions_alloydb(),
    output_key="alloydb_agent_output",
    tools=[
        tools.alloydb_nl2sql,
        tools.run_alloydb_query,
        # tools.get_toolbox_toolset(),
    ],
    # before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
