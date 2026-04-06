"""Format agent responses for WhatsApp delivery.

Handles:
- Stripping visualization blocks (mecdm_stat, mecdm_viz, mecdm_map)
- Converting markdown to WhatsApp-compatible formatting
- Splitting long messages at paragraph / sentence boundaries
"""

from __future__ import annotations

import re

# Max characters per WhatsApp text message
MAX_MESSAGE_LENGTH = 4096

# Regex for fenced code blocks with visualization markers
_VIZ_BLOCK_RE = re.compile(
    r"```(?:mecdm_stat|mecdm_viz|mecdm_map)\b.*?```",
    re.DOTALL,
)

# Fenced code blocks of any language (for general cleanup)
_CODE_BLOCK_RE = re.compile(r"```\w*\n.*?```", re.DOTALL)


def format_for_whatsapp(agent_response: str) -> list[str]:
    """Convert an agent response to a list of WhatsApp-ready messages.

    Returns one or more strings, each ≤ MAX_MESSAGE_LENGTH characters.
    """
    text = _strip_visualization_blocks(agent_response)
    text = _markdown_to_whatsapp(text)
    text = _collapse_blank_lines(text)
    return _split_messages(text.strip())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_visualization_blocks(text: str) -> str:
    """Remove mecdm_stat / mecdm_viz / mecdm_map fenced code blocks."""
    # Replace each viz block with a short placeholder
    text = _VIZ_BLOCK_RE.sub(
        "[View detailed visualization on the MECDM web portal]", text
    )
    # Also strip any remaining generic code blocks (SQL, JSON, etc.)
    # but keep the placeholder if it was already inserted
    return text


def _markdown_to_whatsapp(text: str) -> str:
    """Convert common markdown formatting to WhatsApp equivalents.

    WhatsApp supports: *bold*, _italic_, ~strikethrough~, ```monospace```.
    """
    # Bold: **text** or __text__ → *text*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"__(.+?)__", r"*\1*", text)

    # Italic: *text* (single) or _text_ — WhatsApp uses _text_
    # Be careful not to clobber the bold we just converted.
    # Single-star italic that isn't already bold:
    # (skip — WhatsApp *bold* already looks fine)

    # Links: [text](url) → text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    # Headers: ### Title → *Title*
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)

    # Bullet normalization (- or * at line start → •)
    text = re.sub(r"^[\-\*]\s+", "• ", text, flags=re.MULTILINE)

    # Horizontal rules
    text = re.sub(r"^-{3,}$", "───", text, flags=re.MULTILINE)

    return text


def _collapse_blank_lines(text: str) -> str:
    """Replace 3+ consecutive blank lines with 2."""
    return re.sub(r"\n{3,}", "\n\n", text)


def _split_messages(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split text into chunks that each fit within max_len.

    Tries to split at paragraph boundaries first, then sentence boundaries.
    """
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_len:
            current = candidate
        else:
            # Flush current buffer
            if current:
                chunks.append(current.strip())
            # If the paragraph itself exceeds max_len, split by sentences
            if len(para) > max_len:
                chunks.extend(_split_by_sentences(para, max_len))
                current = ""
            else:
                current = para

    if current:
        chunks.append(current.strip())

    return chunks or [text[:max_len]]


def _split_by_sentences(text: str, max_len: int) -> list[str]:
    """Last-resort split at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # If a single sentence exceeds max_len, hard-split
            if len(sentence) > max_len:
                for i in range(0, len(sentence), max_len):
                    chunks.append(sentence[i : i + max_len])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current.strip())

    return chunks
