from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx
import numpy as np
import psycopg
from psycopg.rows import dict_row

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS spectrum_submissions (
    id              BIGSERIAL PRIMARY KEY,
    color_name      TEXT NOT NULL,
    reflectance     DOUBLE PRECISION[] NOT NULL,
    submitter_email TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_spectrum_submissions_status
    ON spectrum_submissions (status);
CREATE INDEX IF NOT EXISTS idx_spectrum_submissions_created
    ON spectrum_submissions (created_at DESC);
"""


def database_url() -> Optional[str]:
    url = os.environ.get("DATABASE_URL", "").strip()
    return url or None


def connect():
    url = database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(url, row_factory=dict_row, connect_timeout=12)


def ensure_schema() -> None:
    if not database_url():
        return
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_DDL)
        conn.commit()


def fetch_approved_spectra() -> dict[str, np.ndarray]:
    if not database_url():
        return {}
    out: dict[str, np.ndarray] = {}
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (color_name) color_name, reflectance
                FROM spectrum_submissions
                WHERE status = 'approved'
                ORDER BY color_name, id DESC
                """
            )
            rows = cur.fetchall()
    for row in rows:
        name = str(row["color_name"]).strip()
        ref = np.asarray(row["reflectance"], dtype=float).flatten()
        if name:
            out[name] = ref
    return out


def insert_pending(color_name: str, reflectance: np.ndarray, submitter_email: str) -> int:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO spectrum_submissions (color_name, reflectance, submitter_email, status)
                VALUES (%s, %s, %s, 'pending')
                RETURNING id
                """,
                (color_name, reflectance.tolist(), submitter_email.lower().strip()),
            )
            rid = cur.fetchone()["id"]
        conn.commit()
    return int(rid)


def list_pending() -> List[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, color_name, submitter_email, created_at
                FROM spectrum_submissions
                WHERE status = 'pending'
                ORDER BY created_at ASC
                """
            )
            return list(cur.fetchall())


def get_pending_row(row_id: int) -> Optional[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, color_name, reflectance, submitter_email, status, created_at
                FROM spectrum_submissions
                WHERE id = %s
                """,
                (row_id,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def set_status(row_id: int, status: str) -> bool:
    now = datetime.now(timezone.utc)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE spectrum_submissions
                SET status = %s, reviewed_at = %s
                WHERE id = %s AND status = 'pending'
                """,
                (status, now, row_id),
            )
            n = cur.rowcount
        conn.commit()
    return n > 0


def admin_token_ok(provided: str) -> bool:
    expected = os.environ.get("ADMIN_SECRET", "").strip()
    if not expected:
        return False
    return secrets.compare_digest(provided.strip(), expected)


def allowed_submitter_email(email: str) -> bool:
    email = email.strip().lower()
    suffix = os.environ.get("ALLOWED_EMAIL_SUFFIX", "@uwo.ca").strip().lower()
    if not suffix.startswith("@"):
        suffix = "@" + suffix
    return email.endswith(suffix)


def send_new_submission_email(
    *,
    submission_id: int,
    color_name: str,
    submitter_email: str,
) -> None:
    admin_to = os.environ.get("ADMIN_NOTIFY_EMAIL", "").strip()
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    from_addr = os.environ.get("NOTIFY_FROM_EMAIL", "").strip()
    base_url = os.environ.get("PUBLIC_APP_URL", "").rstrip("/")

    if not admin_to:
        return
    subject = f"[SpectOptiBlend] New spectrum submission #{submission_id}: {color_name}"
    lines = [
        f"A student submitted a new spectrum for approval.",
        f"",
        f"ID:            {submission_id}",
        f"Color name:    {color_name}",
        f"Submitter:     {submitter_email}",
        f"",
    ]
    if base_url:
        lines.append(f"Admin UI: {base_url}/admin.html")
    body = "\n".join(lines)

    if api_key and from_addr:
        try:
            r = httpx.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": from_addr,
                    "to": [admin_to],
                    "subject": subject,
                    "text": body,
                },
                timeout=20.0,
            )
            r.raise_for_status()
        except Exception:
            # Do not fail submission if email provider errors
            pass
