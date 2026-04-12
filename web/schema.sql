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
