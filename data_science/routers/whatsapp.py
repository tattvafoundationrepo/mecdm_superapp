"""WhatsApp webhook router for Meta Cloud API integration.

Provides two endpoints:
  GET  /whatsapp/webhook  — Verification handshake (Meta calls once during setup)
  POST /whatsapp/webhook  — Inbound messages and status updates
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Query, Request, Response

from data_science.app_utils.whatsapp_models import (
    WhatsAppWebhookPayload,
    extract_messages,
)
from data_science.services.whatsapp_service import (
    WHATSAPP_VERIFY_TOKEN,
    WhatsAppService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])


# ---------------------------------------------------------------------------
# Webhook verification (GET)
# ---------------------------------------------------------------------------

@router.get("/webhook")
async def verify_webhook(
    response: Response,
    hub_mode: str = Query("", alias="hub.mode"),
    hub_verify_token: str = Query("", alias="hub.verify_token"),
    hub_challenge: str = Query("", alias="hub.challenge"),
):
    """Meta sends a GET request to verify the webhook URL.

    We must check the verify token and echo back the challenge value.
    """
    if hub_mode == "subscribe" and hub_verify_token == WHATSAPP_VERIFY_TOKEN:
        logger.info("WhatsApp webhook verified successfully")
        return Response(content=hub_challenge, media_type="text/plain")

    logger.warning("WhatsApp webhook verification failed (token mismatch)")
    response.status_code = 403
    return {"error": "Verification failed"}


# ---------------------------------------------------------------------------
# Inbound webhook (POST)
# ---------------------------------------------------------------------------

@router.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Receive messages / status updates from Meta.

    Returns 200 immediately (Meta requires < 5 s response) and processes
    messages asynchronously via BackgroundTasks.
    """
    body = await request.json()
    payload = WhatsAppWebhookPayload.model_validate(body)

    messages = extract_messages(payload)
    if not messages:
        # Status update or non-text message — acknowledge silently
        return {"status": "ok"}

    service = WhatsAppService.get_instance()

    for msg in messages:
        logger.info(
            "WhatsApp message from %s (%s): %s",
            msg.phone,
            msg.sender_name,
            msg.text[:80],
        )
        background_tasks.add_task(
            service.handle_message, msg.phone, msg.text, msg.message_id
        )

    return {"status": "ok"}
