"""WhatsApp channel service for the MECDM Citizen agent.

Creates a dedicated Citizen-persona agent and ADK Runner, receives text
messages from the webhook router, runs them through the agent, formats the
response, and delivers it back via Meta's WhatsApp Cloud API.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import google_search
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

from data_science.prompts.prompt_builder import (
    Persona,
    PromptConfig,
    build_global_instruction,
    build_instruction_provider,
)
from data_science.prompts.intent_classifier import IntentClassifier
from data_science.agent import (
    _config,
    load_database_settings_in_context,
    save_session_to_memory,
)
from data_science.sub_agents.alloydb.tools import get_toolbox_toolset
from data_science.tools import (
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
    read_uploaded_file,
    recommend_video,
    search_policy_rag_engine,
)
from data_science.services.whatsapp_formatter import format_for_whatsapp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v21.0")

_META_API_BASE = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"


# ---------------------------------------------------------------------------
# Citizen Agent Factory
# ---------------------------------------------------------------------------

def _create_citizen_agent() -> LlmAgent:
    """Build a Citizen-persona agent reusing the shared configuration."""
    prompt_config = PromptConfig(
        persona=Persona.CITIZEN,
        include_schema=bool(_config.db_schema),
        include_relations=bool(
            _config.relations_config and _config.relations_config.relations
        ),
        include_domain_knowledge=True,
        include_visualization_guide=False,  # WhatsApp can't render viz
        strict_privacy=True,
    )

    instruction_provider = build_instruction_provider(
        config=prompt_config,
        dataset=_config.dataset_config,
        relations=_config.relations_config,
        db_schema=_config.db_schema,
    )

    tools = [
        PreloadMemoryTool(),
        *get_toolbox_toolset(),
        call_alloydb_agent,
        call_analytics_agent,
        find_nearest_facilities,
        search_policy_rag_engine,
        recommend_video,
        get_current_datetime,
        get_weather_data,
        get_historical_weather_data,
        get_stats_schema_summary,
        get_predefined_stats_catalog,
        google_search,
    ]

    agent = LlmAgent(
        model=os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash"),
        name="whatsapp_citizen_agent",
        instruction=instruction_provider,
        global_instruction=build_global_instruction(),
        sub_agents=[],
        tools=tools,
        before_agent_callback=load_database_settings_in_context,
        after_agent_callback=save_session_to_memory,
        generate_content_config=types.GenerateContentConfig(temperature=0.01),
    )

    logger.info("Created WhatsApp Citizen agent")
    return agent


# ---------------------------------------------------------------------------
# WhatsApp Service (singleton)
# ---------------------------------------------------------------------------

class WhatsAppService:
    """Manages the WhatsApp ↔ Agent lifecycle."""

    _instance: Optional["WhatsAppService"] = None

    def __init__(self) -> None:
        # Dedicated citizen agent (separate from the web root_agent)
        self._agent = _create_citizen_agent()

        # ADK session service (same database, different app_name namespace)
        session_uri = os.getenv("DATABASE_URL_USER")
        if not session_uri:
            raise RuntimeError(
                "DATABASE_URL_USER must be set for WhatsApp session storage"
            )
        self._session_service = DatabaseSessionService(url=session_uri)

        # ADK Runner
        self._runner = Runner(
            agent=self._agent,
            app_name="whatsapp_citizen",
            session_service=self._session_service,
        )

        # Async HTTP client for Meta Cloud API
        self._http = httpx.AsyncClient(
            base_url=_META_API_BASE,
            headers={
                "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

        # Per-phone locks to serialize messages from the same user
        self._locks: dict[str, asyncio.Lock] = {}

        logger.info("WhatsAppService initialized")

    @classmethod
    def get_instance(cls) -> "WhatsAppService":
        """Lazy singleton — created on first access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public entry point (called from background task)
    # ------------------------------------------------------------------

    async def handle_message(
        self, phone: str, text: str, message_id: str
    ) -> None:
        """Process an inbound WhatsApp text message end-to-end."""
        lock = self._locks.setdefault(phone, asyncio.Lock())
        async with lock:
            try:
                # Mark as read immediately
                await self._mark_as_read(message_id)

                # Run agent
                user_id = f"wa_{phone}"
                session = await self._get_or_create_session(user_id)
                agent_response = await self._run_agent(
                    user_id, session.id, text
                )

                # Format and send
                chunks = format_for_whatsapp(agent_response)
                for chunk in chunks:
                    await self._send_text(phone, chunk)

            except Exception:
                logger.exception("Error handling WhatsApp message from %s", phone)
                await self._send_text(
                    phone,
                    "Sorry, I'm having trouble processing your request right "
                    "now. Please try again in a moment.",
                )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_or_create_session(self, user_id: str):
        """Return the existing ADK session for this user, or create one."""
        sessions = await self._session_service.list_sessions(
            app_name="whatsapp_citizen", user_id=user_id
        )
        if sessions and sessions.sessions:
            return sessions.sessions[0]

        return await self._session_service.create_session(
            app_name="whatsapp_citizen", user_id=user_id
        )

    # ------------------------------------------------------------------
    # Agent invocation
    # ------------------------------------------------------------------

    async def _run_agent(
        self, user_id: str, session_id: str, text: str
    ) -> str:
        """Send a user message through the citizen agent and return the
        final text response."""
        content = types.Content(
            role="user", parts=[types.Part(text=text)]
        )

        response_parts: list[str] = []
        async for event in self._runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_parts.append(part.text)

        return "\n".join(response_parts) if response_parts else (
            "I couldn't find an answer for that. Could you rephrase your question?"
        )

    # ------------------------------------------------------------------
    # Meta Cloud API helpers
    # ------------------------------------------------------------------

    async def _send_text(self, to_phone: str, body: str) -> None:
        """Send a text message via Meta WhatsApp Cloud API."""
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_phone,
            "type": "text",
            "text": {"preview_url": False, "body": body},
        }
        try:
            resp = await self._http.post(
                f"/{WHATSAPP_PHONE_NUMBER_ID}/messages", json=payload
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            logger.exception("Failed to send WhatsApp message to %s", to_phone)

    async def _mark_as_read(self, message_id: str) -> None:
        """Mark an inbound message as read so the user sees blue ticks."""
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        try:
            resp = await self._http.post(
                f"/{WHATSAPP_PHONE_NUMBER_ID}/messages", json=payload
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            logger.debug("Could not mark message %s as read", message_id)
