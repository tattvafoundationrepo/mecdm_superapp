"""Chat session and message CRUD endpoints."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, update, delete, asc, desc

from data_science.app_utils.models import ChatMessage, ChatSession
from data_science.app_utils.user_db import get_session_factory

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Request / Response schemas ───────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    id: str | None = None
    user_id: str
    title: str = "New Chat"
    adk_session_id: str | None = None


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    adk_session_id: str | None = None


class MessagePayload(BaseModel):
    id: str
    chat_session_id: str
    role: str
    content: str = ""
    thinking: str | None = None
    tool_calls: dict | list | None = None
    activity_events: dict | list | None = None
    attachments: list | None = None
    sort_order: int


class BatchMessagesRequest(BaseModel):
    messages: list[MessagePayload]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _require_db():
    factory = get_session_factory()
    if factory is None:
        raise HTTPException(
            status_code=503,
            detail="User database not configured (DATABASE_URL_USER missing)",
        )
    return factory


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions(user_id: str = Query(...)):
    factory = _require_db()
    async with factory() as session:
        result = await session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(desc(ChatSession.updated_at))
        )
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "userId": r.user_id,
                "title": r.title,
                "adkSessionId": r.adk_session_id,
                "createdAt": r.created_at.isoformat(),
                "updatedAt": r.updated_at.isoformat(),
            }
            for r in rows
        ]


@router.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest):
    factory = _require_db()
    row = ChatSession(
        id=body.id or str(uuid.uuid4()),
        user_id=body.user_id,
        title=body.title,
        adk_session_id=body.adk_session_id,
    )
    async with factory() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return {
            "id": row.id,
            "userId": row.user_id,
            "title": row.title,
            "adkSessionId": row.adk_session_id,
            "createdAt": row.created_at.isoformat(),
            "updatedAt": row.updated_at.isoformat(),
        }


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    factory = _require_db()
    async with factory() as session:
        row = await session.get(ChatSession, session_id)
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs_result = await session.execute(
            select(ChatMessage)
            .where(ChatMessage.chat_session_id == session_id)
            .order_by(asc(ChatMessage.sort_order))
        )
        messages = [
            {
                "id": m.id,
                "chatSessionId": m.chat_session_id,
                "role": m.role,
                "content": m.content,
                "thinking": m.thinking,
                "toolCalls": m.tool_calls,
                "activityEvents": m.activity_events,
                "attachments": m.attachments,
                "sortOrder": m.sort_order,
                "createdAt": m.created_at.isoformat(),
            }
            for m in msgs_result.scalars().all()
        ]
        return {
            "id": row.id,
            "userId": row.user_id,
            "title": row.title,
            "adkSessionId": row.adk_session_id,
            "createdAt": row.created_at.isoformat(),
            "updatedAt": row.updated_at.isoformat(),
            "messages": messages,
        }


@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, body: UpdateSessionRequest):
    factory = _require_db()
    values: dict = {"updated_at": datetime.now(timezone.utc)}
    if body.title is not None:
        values["title"] = body.title
    if body.adk_session_id is not None:
        values["adk_session_id"] = body.adk_session_id
    async with factory() as session:
        result = await session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(**values)
            .returning(ChatSession)
        )
        row = result.scalar_one_or_none()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        await session.commit()
        return {
            "id": row.id,
            "userId": row.user_id,
            "title": row.title,
            "adkSessionId": row.adk_session_id,
            "createdAt": row.created_at.isoformat(),
            "updatedAt": row.updated_at.isoformat(),
        }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    factory = _require_db()
    async with factory() as session:
        await session.execute(
            delete(ChatSession).where(ChatSession.id == session_id)
        )
        await session.commit()
        return {"ok": True}


@router.post("/messages")
async def batch_upsert_messages(body: BatchMessagesRequest):
    factory = _require_db()
    async with factory() as session:
        for msg in body.messages:
            existing = await session.get(ChatMessage, msg.id)
            if existing:
                existing.content = msg.content
                existing.thinking = msg.thinking
                existing.tool_calls = msg.tool_calls
                existing.activity_events = msg.activity_events
                existing.attachments = msg.attachments
            else:
                session.add(
                    ChatMessage(
                        id=msg.id,
                        chat_session_id=msg.chat_session_id,
                        role=msg.role,
                        content=msg.content,
                        thinking=msg.thinking,
                        tool_calls=msg.tool_calls,
                        activity_events=msg.activity_events,
                        attachments=msg.attachments,
                        sort_order=msg.sort_order,
                    )
                )
        await session.commit()
        return {"ok": True, "count": len(body.messages)}
