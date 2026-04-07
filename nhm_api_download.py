#!/usr/bin/env python3
"""
NHM Megh formdata API → JSONL downloader

Pulls every record from the NHM Meghalaya formdata API for a given form ID and
writes it to a JSONL file (one JSON document per line). Designed to be the
upstream of `migrate_mecdm.py`'s `nhm_mother` stage.

Pagination is cursor-based on `_id` (the API silently ignores offset/skip/page
/pageSize and caps `limit` at 1000). The default sort is `_id` desc, so we walk
backwards using `_id lt <last_id>`.

The file is APPENDED to. If the destination file already exists, the downloader
resumes from the last `_id` it finds in the file (one tail read), so an
interrupted run can be restarted with no duplicates and no re-fetching.

Usage:
    python nhm_api_download.py                       # full pull, resume if file exists
    python nhm_api_download.py --no-resume           # truncate file, restart from page 1
    python nhm_api_download.py --limit 500           # smaller pages
    python nhm_api_download.py --max-pages 3         # smoke test (3 pages)
    python nhm_api_download.py --out path/to.jsonl   # custom output path
    python nhm_api_download.py --form-id <id>        # different form

Environment variables:
    NHM_API_KEY_ID    (required)
    NHM_API_KEY       (required)

Output (default): backend/mecdm_dataset/nhm_formdata.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nhm_download")

DEFAULT_FORM_ID = "5fa5510a4794b76e71267ebb"
DEFAULT_OUT = Path(__file__).parent / "mecdm_dataset" / "nhm_formdata.jsonl"
ENDPOINT_TEMPLATE = (
    "https://data.nhmmegh.in/api/external/formdata/records/data/{form_id}"
)
MAX_RETRIES = 5
RETRY_BACKOFF = (1, 2, 4, 8, 16)  # seconds


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def _build_body(cursor: Optional[str], limit: int) -> dict:
    """Match-all + optional cursor on `_id` lt <cursor>."""
    if cursor is None:
        return {
            "conditionFilter": {"type": "a", "values": []},
            "limit": limit,
        }
    return {
        "conditionFilter": {
            "type": "a",
            "values": [
                {
                    "type": "c",
                    "value": {
                        "fieldName": "_id",
                        "operator": "lt",
                        "staticValue": cursor,
                        "dataType": "s",
                    },
                }
            ],
        },
        "limit": limit,
    }


def _post_with_retry(url: str, headers: dict, body: dict) -> dict:
    """POST with exponential backoff on network/5xx/429."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 32
            logger.warning(
                "  request failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1, MAX_RETRIES, str(exc)[:160], wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"giving up after {MAX_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _read_last_id(path: Path) -> Optional[str]:
    """
    Read the last `_id` from the existing JSONL file (tail-only, no full read).
    Returns None if the file is empty or unreadable.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    # Tail read: seek backwards in chunks until we find a newline.
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(8192, size)
            f.seek(size - chunk)
            tail = f.read().decode("utf-8", errors="replace")
        last_line = tail.strip().split("\n")[-1]
        if not last_line:
            return None
        rec = json.loads(last_line)
        return rec.get("_id")
    except Exception as exc:
        logger.warning("could not read tail of %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------

def download_all(
    out_path: Path,
    form_id: str,
    limit: int = 1000,
    resume: bool = True,
    max_pages: Optional[int] = None,
) -> int:
    """
    Stream every record into `out_path` as JSONL. Returns total downloaded
    in this run (excluding rows already on disk from a prior run).
    """
    api_key_id = os.environ.get("NHM_API_KEY_ID")
    api_key = os.environ.get("NHM_API_KEY")
    if not api_key_id or not api_key:
        raise RuntimeError(
            "NHM_API_KEY_ID and NHM_API_KEY must be set in env or .env file"
        )

    headers = {
        "X-Api-Key-Id": api_key_id,
        "X-Api-Key": api_key,
        "X-product": "APPVERSE",
        "Content-Type": "application/json",
    }
    url = ENDPOINT_TEMPLATE.format(form_id=form_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cursor: Optional[str] = None
    if resume:
        cursor = _read_last_id(out_path)
        if cursor:
            logger.info("resuming from cursor _id=%s", cursor)
        elif out_path.exists() and out_path.stat().st_size > 0:
            logger.warning(
                "could not parse last _id; will append from page 1 (may dupe)"
            )
    else:
        if out_path.exists():
            logger.info("--no-resume: truncating %s", out_path)
            out_path.unlink()

    mode = "ab"  # binary append → portable line endings, no buffering surprises
    total = 0
    page = 0
    started = time.time()

    with out_path.open(mode) as f:
        while True:
            page += 1
            body = _build_body(cursor, limit)
            resp = _post_with_retry(url, headers, body)
            recs = resp.get("records", [])
            server_count = resp.get("count")

            if not recs:
                logger.info("no more records — done.")
                break

            buf = "\n".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":"))
                for r in recs
            ) + "\n"
            f.write(buf.encode("utf-8"))
            f.flush()

            total += len(recs)
            cursor = recs[-1]["_id"]
            elapsed = time.time() - started
            rate = total / elapsed if elapsed > 0 else 0.0
            logger.info(
                "page %d: +%d rows (total this run=%d, rate=%.0f/s, "
                "filtered_count_remaining=%s, cursor=%s)",
                page, len(recs), total, rate, server_count, cursor,
            )

            if len(recs) < limit:
                logger.info(
                    "page returned fewer than limit (%d<%d) — done.",
                    len(recs), limit,
                )
                break

            if max_pages is not None and page >= max_pages:
                logger.info("hit --max-pages=%d — stopping early.", max_pages)
                break

    logger.info(
        "downloaded %d records in %d page(s) (%.1fs); file=%s (%.1f MB)",
        total, page, time.time() - started, out_path,
        out_path.stat().st_size / 1024 / 1024 if out_path.exists() else 0.0,
    )
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Download all NHM Megh formdata records to JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output JSONL path (default: {DEFAULT_OUT})")
    p.add_argument("--form-id", default=DEFAULT_FORM_ID,
                   help=f"NHM form ID (default: {DEFAULT_FORM_ID})")
    p.add_argument("--limit", type=int, default=1000,
                   help="Records per request (server caps at 1000).")
    p.add_argument("--no-resume", action="store_true",
                   help="Truncate the output file and restart from page 1.")
    p.add_argument("--max-pages", type=int, default=None,
                   help="Stop after N pages (smoke test).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        download_all(
            out_path=args.out,
            form_id=args.form_id,
            limit=args.limit,
            resume=not args.no_resume,
            max_pages=args.max_pages,
        )
    except KeyboardInterrupt:
        logger.warning("interrupted by user — partial file kept for resume.")
        sys.exit(130)
