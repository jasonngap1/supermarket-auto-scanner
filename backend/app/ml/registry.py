from threading import RLock
from typing import Callable

from app.config import Settings
from app.errors import ModelUnavailable
from app.ml.devices import resolve_device
from app.pipeline.embedders import DinoV2Embedder
from app.pipeline.face import FaceDetector
from app.pipeline.sam2 import Sam2Segmenter


class ModelRegistry:
    def __init__(
        self,
        settings: Settings,
        *,
        face_factory: Callable = FaceDetector,
        dinov2_factory: Callable = DinoV2Embedder,
        sam2_factory: Callable = Sam2Segmenter,
    ):
        self.settings = settings
        self._gate = RLock()
        self._device = resolve_device(settings.device)
        self._force_cpu: set[str] = set()
        self._factories = {
            "face": face_factory,
            "dinov2": dinov2_factory,
            "sam2": sam2_factory,
        }
        self._face = None
        self._embedder = None
        self._sam2 = None
        self._errors: dict[str, str | None] = {
            "face": None,
            "dinov2": None,
            "sam2": None,
        }

    def gate(self) -> RLock:
        return self._gate

    def _device_for(self, name: str) -> str:
        return "cpu" if name in self._force_cpu else self._device

    def _ensure_face(self) -> None:
        if self._face is not None:
            return
        try:
            self._face = self._factories["face"](asset_path=self.settings.face_asset)
            self._errors["face"] = None
        except Exception as exc:
            self._errors["face"] = str(exc)
            raise ModelUnavailable(str(exc)) from exc

    def _ensure_embedder(self) -> DinoV2Embedder:
        if self._embedder is not None:
            return self._embedder
        try:
            self._embedder = self._factories["dinov2"](
                model_id=self.settings.dinov2_model_id,
                device=self._device_for("dinov2"),
            )
            self._errors["dinov2"] = None
            return self._embedder
        except Exception as exc:
            self._errors["dinov2"] = str(exc)
            raise ModelUnavailable(str(exc)) from exc

    def _ensure_sam2(self) -> Sam2Segmenter:
        if self._sam2 is None:
            try:
                self._sam2 = self._factories["sam2"](
                    model_id=self.settings.sam2_model_id,
                    device=self._device_for("sam2"),
                )
                self._errors["sam2"] = None
            except Exception as exc:
                self._errors["sam2"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc
        return self._sam2

    def warm(self) -> None:
        with self._gate:
            for ensure in (self._ensure_face, self._ensure_embedder):
                try:
                    ensure()
                except ModelUnavailable:
                    pass

    def detect_face(self, image):
        with self._gate:
            self._ensure_face()
            try:
                return self._face.detect(image)
            except Exception as exc:
                self._errors["face"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc

    def embed(self, image):
        with self._gate:
            return self._call_with_fallback(
                "dinov2",
                lambda: self._ensure_embedder().embed(image),
                reload=lambda: setattr(self, "_embedder", None),
            )

    def segment(self, image, prompt):
        with self._gate:
            return self._call_with_fallback(
                "sam2",
                lambda: self._ensure_sam2().segment(image, prompt),
                reload=lambda: setattr(self, "_sam2", None),
            )

    def _call_with_fallback(self, name: str, call, reload):
        try:
            return call()
        except Exception as exc:
            message = str(exc)
            if self._device_for(name) != "mps":
                self._errors[name] = message
                raise ModelUnavailable(message) from exc
            self._errors[name] = message
            self._force_cpu.add(name)
            reload()
            result = call()
            self._errors[name] = f"fell back to CPU: {message}"
            return result

    def status(self) -> dict:
        return {
            "device": {"preference": self.settings.device, "resolved": self._device},
            "models": {
                "face": {
                    "loaded": self._face is not None,
                    "device": "cpu",
                    "error": self._errors["face"],
                },
                "dinov2": {
                    "loaded": self._embedder is not None,
                    "device": self._device_for("dinov2"),
                    "error": self._errors["dinov2"],
                },
                "sam2": {
                    "loaded": self._sam2 is not None,
                    "device": self._device_for("sam2"),
                    "error": self._errors["sam2"],
                },
            },
        }
