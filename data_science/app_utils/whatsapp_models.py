"""Pydantic models for Meta WhatsApp Cloud API webhook payloads."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inbound webhook models (Meta → us)
# ---------------------------------------------------------------------------

class WhatsAppProfile(BaseModel):
    name: str = ""


class WhatsAppContact(BaseModel):
    profile: WhatsAppProfile = Field(default_factory=WhatsAppProfile)
    wa_id: str = ""


class WhatsAppTextBody(BaseModel):
    body: str = ""


class WhatsAppInboundMessage(BaseModel):
    """A single message inside the webhook payload."""

    from_: str = Field("", alias="from")
    id: str = ""
    timestamp: str = ""
    type: str = ""
    text: Optional[WhatsAppTextBody] = None

    model_config = {"populate_by_name": True}


class WhatsAppStatus(BaseModel):
    id: str = ""
    status: str = ""  # sent | delivered | read | failed
    timestamp: str = ""
    recipient_id: str = ""


class WhatsAppMetadata(BaseModel):
    display_phone_number: str = ""
    phone_number_id: str = ""


class WhatsAppValue(BaseModel):
    messaging_product: str = ""
    metadata: WhatsAppMetadata = Field(default_factory=WhatsAppMetadata)
    contacts: list[WhatsAppContact] = Field(default_factory=list)
    messages: list[WhatsAppInboundMessage] = Field(default_factory=list)
    statuses: list[WhatsAppStatus] = Field(default_factory=list)


class WhatsAppChange(BaseModel):
    value: WhatsAppValue = Field(default_factory=WhatsAppValue)
    field: str = ""


class WhatsAppEntry(BaseModel):
    id: str = ""
    changes: list[WhatsAppChange] = Field(default_factory=list)


class WhatsAppWebhookPayload(BaseModel):
    """Top-level model for the Meta webhook POST body."""

    object: str = ""
    entry: list[WhatsAppEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Convenience extractor
# ---------------------------------------------------------------------------

class ParsedMessage(BaseModel):
    """Flattened representation of an inbound text message."""

    phone: str
    text: str
    message_id: str
    sender_name: str = ""


def extract_messages(payload: WhatsAppWebhookPayload) -> list[ParsedMessage]:
    """Extract text messages from a webhook payload, ignoring statuses."""
    results: list[ParsedMessage] = []
    for entry in payload.entry:
        for change in entry.changes:
            if change.field != "messages":
                continue
            value = change.value
            contacts_by_id = {c.wa_id: c for c in value.contacts}
            for msg in value.messages:
                if msg.type != "text" or msg.text is None:
                    continue
                contact = contacts_by_id.get(msg.from_)
                results.append(
                    ParsedMessage(
                        phone=msg.from_,
                        text=msg.text.body,
                        message_id=msg.id,
                        sender_name=contact.profile.name if contact else "",
                    )
                )
    return results
