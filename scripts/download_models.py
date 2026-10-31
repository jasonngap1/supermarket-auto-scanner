import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AUTO_SCANNER_DATA_DIR", str(ROOT / "data")))
os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))
# Xet transfers can stall on some networks; plain HTTP resumes reliably.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
EXPECTED_FACE_SHA256 = (
    "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
)

REPOS = {
    "sam2": "facebook/sam2.1-hiera-tiny",
    "dinov2": "facebook/dinov2-small",
}


def download_repo(repo_id: str) -> dict:
    api = HfApi()
    revision = api.model_info(repo_id).sha
    hub_cache = Path(os.environ["HF_HOME"]) / "hub"
    files = api.list_repo_files(repo_id, revision=revision)
    ignore = ["*.pt", "*.pth", "*.h5", "*.msgpack", "*.onnx"]
    if any(name.endswith(".safetensors") for name in files):
        ignore.append("*.bin")
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(hub_cache),
        ignore_patterns=ignore,
    )
    return {"repo_id": repo_id, "revision": revision, "cache_path": path}


def download_asset(url: str, filename: str, expected_sha256: str) -> dict:
    destination = DATA / "assets" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != expected_sha256:
        destination.unlink(missing_ok=True)
        raise SystemExit(f"Asset checksum mismatch for {filename}: {digest}")
    return {"url": url, "path": str(destination), "sha256": digest}


def download_face() -> dict:
    return download_asset(
        FACE_URL, "blaze_face_short_range.tflite", EXPECTED_FACE_SHA256
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {
        "models": {key: download_repo(repo) for key, repo in REPOS.items()},
        "face": download_face(),
    }
    (DATA / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
