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
