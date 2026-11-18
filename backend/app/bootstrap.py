"""Must be imported before torch. Sets MPS fallback and cache locations."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AUTO_SCANNER_DATA_DIR", str(PROJECT_ROOT / "data")))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_cache"))
