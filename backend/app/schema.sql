PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  price_cents INTEGER NOT NULL CHECK (price_cents >= 0),
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS captures (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id           INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  role              TEXT NOT NULL CHECK (role IN ('gallery', 'test')),
  session_id        TEXT NOT NULL,
  rel_path          TEXT NOT NULL UNIQUE,
  thumb_rel_path    TEXT NOT NULL,
  mask_rel_path     TEXT NOT NULL,
  sha256            TEXT NOT NULL,
  width             INTEGER NOT NULL,
  height            INTEGER NOT NULL,
  hand_found        INTEGER NOT NULL DEFAULT 0,
  mask_found        INTEGER NOT NULL DEFAULT 0,
  mask_status       TEXT NOT NULL,
  localization_json TEXT NOT NULL,
  created_at        TEXT NOT NULL,
  UNIQUE (item_id, role, sha256)
);
CREATE INDEX IF NOT EXISTS idx_captures_item_role ON captures(item_id, role);

CREATE TABLE IF NOT EXISTS embeddings (
  capture_id         INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
  variant            TEXT NOT NULL,
  model_key          TEXT NOT NULL,
  preprocess_version TEXT NOT NULL,
  dim                INTEGER NOT NULL,
  vector             BLOB NOT NULL,
  created_at         TEXT NOT NULL,
  PRIMARY KEY (capture_id, variant, model_key, preprocess_version)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_lookup
  ON embeddings(variant, model_key, preprocess_version);

CREATE TABLE IF NOT EXISTS eval_runs (
  id             TEXT PRIMARY KEY,
  created_at     TEXT NOT NULL,
  started_at     TEXT,
  finished_at    TEXT,
  status         TEXT NOT NULL CHECK (status IN ('queued','running','completed','failed')),
  request_json   TEXT NOT NULL,
  progress_done  INTEGER NOT NULL DEFAULT 0,
  progress_total INTEGER NOT NULL DEFAULT 0,
  metrics_json   TEXT,
  error          TEXT
);

CREATE TABLE IF NOT EXISTS eval_samples (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id             TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
  capture_id         INTEGER,
  variant            TEXT NOT NULL,
  true_item_id       INTEGER,
  true_item_name     TEXT NOT NULL,
  true_price_cents   INTEGER NOT NULL,
  pred_item_id       INTEGER,
  pred_item_name     TEXT,
  topk_json          TEXT NOT NULL,
  correct_top1       INTEGER NOT NULL DEFAULT 0,
  correct_top5       INTEGER NOT NULL DEFAULT 0,
  sample_status      TEXT NOT NULL,
  error              TEXT,
  timings_json       TEXT NOT NULL,
  rendered_rel_path  TEXT,
  created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_samples_run_variant
  ON eval_samples(run_id, variant, correct_top1);
