"""Refresh the committed catalog from the local SQLite database and media.

Run from the repo root or backend directory:

    cd backend && uv run python ../scripts/export_catalog.py

The app also refreshes an existing ``catalog/`` automatically after item and
gallery-capture changes. This script bootstraps it or repairs a stale snapshot.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.services.catalog import export_catalog, manifest_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--catalog-dir", type=Path)
    args = parser.parse_args()

    overrides = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.catalog_dir:
        overrides["catalog_dir"] = args.catalog_dir
    settings = Settings(**overrides)
    if not settings.db_path.exists():
        raise SystemExit(f"Database not found: {settings.db_path}")

    export_catalog(settings)
    payload = json.loads(manifest_path(settings).read_text())
    print(
        f"Wrote {manifest_path(settings)} "
        f"({len(payload['items'])} items, {len(payload['captures'])} gallery captures)"
    )


if __name__ == "__main__":
    main()
