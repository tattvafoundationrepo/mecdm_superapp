"""Feedback endpoint — persists to DB and Cloud Logging."""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from data_science.app_utils.models import Feedback
from data_science.app_utils.user_db import get_session_factory

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    score: int = Field(..., ge=1, le=5)
    text: str | None = ""
    user_id: str
    session_id: str | None = None
    message_id: str | None = None


@router.post("/feedback")
async def collect_feedback(body: FeedbackRequest):
    factory = get_session_factory()
    if factory:
        row = Feedback(
            id=str(uuid.uuid4()),
            user_id=body.user_id,
            chat_session_id=body.session_id,
            message_id=body.message_id,
            score=body.score,
            text=body.text,
        )
        async with factory() as session:
            session.add(row)
            await session.commit()

    return {"status": "success"}
