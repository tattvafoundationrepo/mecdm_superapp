"""Upload router for file attachments."""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from data_science.services.file_processor import (
    upload_to_gcs,
    validate_file,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


class UploadResponse(BaseModel):
    file_id: str
    file_name: str
    mime_type: str
    gcs_uri: str
    size_bytes: int
    processing_strategy: str


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    session_id: str = Form(...),
):
    """Upload a file to GCS for agent processing.

    Supports: PDF, DOCX, PPTX, XLSX, PNG, JPG, WEBP.
    Images: max 10 MB. Documents: max 20 MB.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_bytes = await file.read()
    content_type = file.content_type or "application/octet-stream"

    try:
        validate_file(file.filename, content_type, len(file_bytes))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = upload_to_gcs(
            file_bytes=file_bytes,
            filename=file.filename,
            mime_type=content_type,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        logger.exception("File upload failed")
        raise HTTPException(status_code=500, detail="File upload failed")

    return UploadResponse(**result)
