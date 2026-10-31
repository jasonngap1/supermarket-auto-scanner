from dataclasses import dataclass

from fastapi import Request

from app.config import Settings


@dataclass
class AppState:
    settings: Settings
    registry: object | None = None
    pipeline: object | None = None
    captures: object | None = None
    scan: object | None = None


def get_state(request: Request) -> AppState:
    return request.app.state.services


def build_default_state(settings: Settings) -> AppState:
    from app.ml.registry import ModelRegistry
    from app.pipeline.frame_pipeline import FramePipeline
    from app.services.captures import CaptureService
    from app.services.embeddings import EmbeddingService
    from app.services.scan import ScanService

    registry = ModelRegistry(settings)
    pipeline = FramePipeline(registry, settings)
    embeddings = EmbeddingService(settings, registry)
    return AppState(
        settings,
        registry=registry,
        pipeline=pipeline,
        captures=CaptureService(settings, pipeline),
        scan=ScanService(settings, pipeline, embeddings),
    )
