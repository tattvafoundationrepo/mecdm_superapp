-- Migration: 001_chat_and_feedback
-- Creates chat_session, chat_message, feedback, user_preferences tables
-- Depends on BetterAuth "user" table already existing in mecdm_superapp_user_db

BEGIN;

-- ── chat_session ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_session (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    title           TEXT NOT NULL DEFAULT 'New Chat',
    adk_session_id  TEXT UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_session_user_id ON chat_session(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_session_updated_at ON chat_session(updated_at DESC);

-- ── chat_message ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_message (
    id               TEXT PRIMARY KEY,
    chat_session_id  TEXT NOT NULL REFERENCES chat_session(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL DEFAULT '',
    thinking         TEXT,
    tool_calls       JSONB,
    activity_events  JSONB,
    sort_order       INTEGER NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_message_session_order ON chat_message(chat_session_id, sort_order);

-- ── feedback ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    chat_session_id TEXT REFERENCES chat_session(id) ON DELETE SET NULL,
    message_id      TEXT,
    score           INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
    text            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_session_id ON feedback(chat_session_id);

-- ── user_preferences ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id         TEXT PRIMARY KEY REFERENCES "user"(id) ON DELETE CASCADE,
    chat_position   TEXT NOT NULL DEFAULT 'dock-right',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMIT;
