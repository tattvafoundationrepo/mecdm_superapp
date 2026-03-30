"""
MECDM Insight Agent - Refined Architecture

This agent provides health intelligence for Meghalaya's Early Childhood
Development Mission, supporting decision makers with data-driven insights.

Key Features:
- Modular prompt system loaded from configuration
- Cross-dataset relations awareness
- MCP Toolbox integration for AlloyDB access
- Privacy-first design with aggregation enforcement
- Multi-persona support (Decision Maker, Frontline Worker, Citizen, Analyst)

Environment Variables:
    DATASET_CONFIG_FILE: Path to datasets.json
    CROSS_DATASET_RELATIONS_DEFS: Path to cross_dataset_relations.json (optional)
    ROOT_AGENT_MODEL: Model name (default: gemini-2.5-flash)
    AGENT_PERSONA: Persona type (decision_maker|frontline_worker|citizen|analyst)
"""

import logging
import os
from typing import Optional, Dict, Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.tools import google_search
from google.genai import types

from .prompts.prompt_builder import (
    PromptConfig,
    Persona,
    DatasetConfig,
    RelationsConfig,
    load_dataset_config,
    load_relations_config,
    build_root_instruction,
    build_global_instruction,
)
from .sub_agents.alloydb.tools import (
    get_database_settings as get_alloydb_database_settings,
)
from .tools import (
    call_alloydb_agent,
    call_analytics_agent,
    export_data_to_csv,
    find_nearest_facilities,
    generate_stat_query,
    get_current_datetime,
    get_historical_weather_data,
    get_predefined_stats_catalog,
    get_stats_schema_summary,
    get_weather_data,
    search_policy_rag_engine,
    validate_and_wrap_sql,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_supported_dataset_types = ["alloydb"]


# ============================================================================
# Configuration
# ============================================================================

class AgentConfiguration:
    """Centralized agent configuration manager."""

    def __init__(self):
        self.dataset_config: Optional[DatasetConfig] = None
        self.relations_config: Optional[RelationsConfig] = None
        self.database_settings: Dict[str, Any] = {}
        self.db_schema: str = ""
        self._raw_dataset_config: Dict[str, Any] = {}
        self._initialized = False

    def initialize(self):
        """Load all configurations. Call once at startup."""
        if self._initialized:
            return

        logger.info("Initializing MECDM Agent Configuration...")

        # Load dataset configuration
        dataset_config_file = os.getenv("DATASET_CONFIG_FILE", "")
        if not dataset_config_file:
            logger.fatal("DATASET_CONFIG_FILE env var not set")
            raise ValueError("DATASET_CONFIG_FILE env var not set")

        self.dataset_config = load_dataset_config(dataset_config_file)
        self._raw_dataset_config = self.dataset_config.raw_config
        logger.info("Loaded dataset config: %s", self.dataset_config.name)

        # Validate dataset types
        for dataset in self._raw_dataset_config.get("datasets", []):
            if dataset.get("type") not in _supported_dataset_types:
                logger.fatal("Dataset type '%s' not supported", dataset.get("type"))

        # Load cross-dataset relations
        relations_file = os.getenv("CROSS_DATASET_RELATIONS_DEFS")
        self.relations_config = load_relations_config(relations_file)
        logger.info("Loaded %d relations", len(self.relations_config.relations))

        # Load database settings and schema
        self._init_database_settings()

        self._initialized = True

    def _init_database_settings(self):
        """Initialize database settings for configured datasets."""
        for dataset in self._raw_dataset_config.get("datasets", []):
            if dataset["type"] == "alloydb":
                try:
                    settings = get_alloydb_database_settings()
                    self.database_settings[dataset["type"]] = settings
                    self.db_schema = settings.get("schema", "")
                    logger.info("Loaded database schema from AlloyDB")
                except Exception as e:
                    logger.warning("Could not load database schema: %s", e)

    def get_persona(self) -> Persona:
        """Get persona from environment or default to decision maker."""
        persona_str = os.getenv("AGENT_PERSONA", "decision_maker").lower()
        try:
            return Persona(persona_str)
        except ValueError:
            logger.warning("Invalid persona '%s'; using decision_maker", persona_str)
            return Persona.DECISION_MAKER

    def build_prompt_config(self) -> PromptConfig:
        """Build prompt configuration based on environment."""
        return PromptConfig(
            persona=self.get_persona(),
            include_schema=bool(self.db_schema),
            include_relations=bool(
                self.relations_config and self.relations_config.relations
            ),
            include_domain_knowledge=True,
            include_visualization_guide=True,
            strict_privacy=os.getenv("STRICT_PRIVACY", "true").lower() == "true",
        )


# Global configuration instance
_config = AgentConfiguration()


# ============================================================================
# Callback Handlers
# ============================================================================

def load_database_settings_in_context(callback_context: CallbackContext):
    """Load database settings into the callback context on first use."""
    if "database_settings" not in callback_context.state:
        callback_context.state["database_settings"] = _config.database_settings


# ============================================================================
# Agent Factory
# ============================================================================

def create_root_agent() -> LlmAgent:
    """
    Factory function to create the configured root agent.

    Returns:
        Configured LlmAgent instance
    """
    # Build prompt configuration
    prompt_config = _config.build_prompt_config()

    # Build the instruction prompt
    instruction = build_root_instruction(
        config=prompt_config,
        dataset=_config.dataset_config,
        relations=_config.relations_config,
        db_schema=_config.db_schema,
    )

    logger.info(
        "Built instruction prompt: %d chars, persona=%s",
        len(instruction),
        prompt_config.persona.value,
    )

    # Assemble tools list
    tools = [
        call_analytics_agent,
        find_nearest_facilities,
        generate_stat_query,
        validate_and_wrap_sql,
        get_current_datetime,
        get_weather_data,
        get_historical_weather_data,
        export_data_to_csv,
        search_policy_rag_engine,
        get_stats_schema_summary,
        get_predefined_stats_catalog,
        google_search,
    ]

    # Add dataset-specific tools
    for dataset in _config._raw_dataset_config.get("datasets", []):
        if dataset["type"] == "alloydb":
            tools.append(call_alloydb_agent)

    agent = LlmAgent(
        model=os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash"),
        name="data_science_root_agent",
        instruction=instruction,
        global_instruction=build_global_instruction(),
        sub_agents=[],
        tools=tools,
        before_agent_callback=load_database_settings_in_context,
        generate_content_config=types.GenerateContentConfig(temperature=0.01),
    )

    return agent


# ============================================================================
# Module-Level Initialization
# ============================================================================

_config.initialize()

root_agent = create_root_agent()

app = App(root_agent=root_agent, name="data_science")
