-- Feedback for each processed job
create table if not exists job_feedback (
    id uuid primary key default gen_random_uuid(),
    job_id text not null,
    reviewer text,                         -- who gave the feedback (e.g. "marko", "kolega")
    preset text,
    density real,
    overall_rating int,                    -- 1-5
    density_feedback text,                 -- too_low | right | too_high
    overall_note text,
    status text not null default 'pending', -- pending | applied | ignored
    dev_note text,                          -- developer's note on what was done
    created_at timestamptz not null default now()
);

-- Per-sound ratings (child rows of job_feedback)
create table if not exists sound_feedback (
    id uuid primary key default gen_random_uuid(),
    feedback_id uuid references job_feedback(id) on delete cascade,
    sound_index int,
    sound_id text,
    sound_name text,
    action_type text,
    timestamp real,
    matched_tier text,
    match_score real,
    rating text,                           -- good | wrong | unnecessary
    created_at timestamptz not null default now()
);

-- Missing-sound markers
create table if not exists missing_sounds (
    id uuid primary key default gen_random_uuid(),
    feedback_id uuid references job_feedback(id) on delete cascade,
    timestamp real,
    note text
);

-- Helpful indexes
create index if not exists idx_sound_feedback_action on sound_feedback(action_type);
create index if not exists idx_sound_feedback_rating on sound_feedback(rating);
create index if not exists idx_job_feedback_status on job_feedback(status);
create index if not exists idx_job_feedback_reviewer on job_feedback(reviewer);
