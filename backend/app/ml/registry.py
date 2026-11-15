from threading import RLock
from typing import Callable

from app.config import Settings
from app.errors import ModelUnavailable
from app.ml.devices import resolve_device
from app.pipeline.embedders import DinoV2Embedder
from app.pipeline.sam2 import Sam2Segmenter
from app.pipeline.yolo import YoloDetector


class ModelRegistry:
    def __init__(
        self,
        settings: Settings,
        *,
        dinov2_factory: Callable = DinoV2Embedder,
        sam2_factory: Callable = Sam2Segmenter,
        yolo_factory: Callable = YoloDetector,
    ):
        self.settings = settings
        self._gate = RLock()
        self._device = resolve_device(settings.device)
        self._force_cpu: set[str] = set()
        self._factories = {
            "dinov2": dinov2_factory,
            "sam2": sam2_factory,
            "yolo": yolo_factory,
        }
        self._embedder = None
        self._sam2 = None
        self._yolo = None
        self._errors: dict[str, str | None] = {
            "dinov2": None,
            "sam2": None,
            "yolo": None,
        }

    def gate(self) -> RLock:
        return self._gate

    def _device_for(self, name: str) -> str:
        return "cpu" if name in self._force_cpu else self._device

    def _ensure_embedder(self):
        if self._embedder is None:
            try:
                self._embedder = self._factories["dinov2"](
                    model_id=self.settings.dinov2_model_id,
                    device=self._device_for("dinov2"),
                )
                self._errors["dinov2"] = None
            except Exception as exc:
                self._errors["dinov2"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc
        return self._embedder

    def _ensure_sam2(self):
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

    def _ensure_yolo(self):
        if self._yolo is None:
            try:
                self._yolo = self._factories["yolo"](
                    model_id=self.settings.yolo_det_model_id,
                    device=self._device_for("yolo"),
                    conf=self.settings.yolo_conf,
                    iou=self.settings.yolo_iou,
                    imgsz=self.settings.yolo_imgsz,
                    allowed_classes=self.settings.yolo_allowed_classes,
                    weights_dir=self.settings.data_dir / "weights",
                )
                self._errors["yolo"] = None
            except Exception as exc:
                self._errors["yolo"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc
        return self._yolo

    def warm(self) -> None:
        with self._gate:
            for ensure in (self._ensure_embedder, self._ensure_sam2, self._ensure_yolo):
                try:
                    ensure()
                except ModelUnavailable:
                    pass

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

    def detect_objects(self, image):
        with self._gate:
            return self._call_with_fallback(
                "yolo",
                lambda: self._ensure_yolo().predict(image),
                reload=lambda: setattr(self, "_yolo", None),
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
                "dinov2": {
                    "loaded": self._embedder is not None,
                    "device": self._device_for("dinov2"),
                    "error": self._errors["dinov2"],
                    "model_id": self.settings.dinov2_model_id,
                },
                "sam2": {
                    "loaded": self._sam2 is not None,
                    "device": self._device_for("sam2"),
                    "error": self._errors["sam2"],
                    "model_id": self.settings.sam2_model_id,
                },
                "yolo": {
                    "loaded": self._yolo is not None,
                    "device": self._device_for("yolo"),
                    "error": self._errors["yolo"],
                    "model_id": self.settings.yolo_det_model_id,
                },
            },
        }
