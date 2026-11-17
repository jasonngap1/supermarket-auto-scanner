import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AUTO_SCANNER_DATA_DIR", str(ROOT / "data")))
os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))
# Xet transfers can stall on some networks; plain HTTP resumes reliably.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

REPOS = {
    "dinov2": "facebook/dinov2-small",
}

YOLO_MODEL = "yolo26n-seg.pt"
YOLO_ONNX_MODEL = "yolo26n-seg.onnx"


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


def download_yolo(weights_dir: Path) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required; run: uv sync") from exc

    weights_dir.mkdir(parents=True, exist_ok=True)
    pt_path = weights_dir / YOLO_MODEL
    previous = Path.cwd()
    os.chdir(weights_dir)
    try:
        YOLO(str(pt_path))
        if not pt_path.exists():
            raise SystemExit(f"YOLO download did not produce {pt_path}")

        # Static 640 input keeps ONNX Runtime happy without a dynamic-shape
        # fallback. No simplifier or FP16: both are unreliable on CPU here.
        exported = YOLO(str(pt_path)).export(
            format="onnx",
            imgsz=640,
            dynamic=False,
            simplify=False,
            opset=18,
            quantize=32,
        )
        onnx_path = Path(exported)
    finally:
        os.chdir(previous)

    return {
        "pt": {
            "filename": YOLO_MODEL,
            "path": str(pt_path),
            "sha256": hashlib.sha256(pt_path.read_bytes()).hexdigest(),
        },
        "onnx": {
            "filename": onnx_path.name,
            "path": str(onnx_path),
            "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        },
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {
        "models": {key: download_repo(repo_id) for key, repo_id in REPOS.items()},
        "yolo": download_yolo(DATA / "weights"),
    }
    (DATA / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
