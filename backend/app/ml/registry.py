from threading import RLock
from typing import Callable

from app.config import Settings
from app.errors import ModelUnavailable
from app.ml.devices import resolve_device
from app.pipeline.embedders import ClipEmbedder, DinoV2Embedder
from app.pipeline.face import FaceDetector
from app.pipeline.hand import HandDetector
from app.pipeline.sam2 import Sam2Segmenter
from app.pipeline.sam3 import Sam3LiteSegmenter


class ModelRegistry:
    def __init__(
        self,
        settings: Settings,
        *,
        hand_factory: Callable = HandDetector,
        face_factory: Callable = FaceDetector,
        dinov2_factory: Callable = DinoV2Embedder,
        clip_factory: Callable = ClipEmbedder,
        sam2_factory: Callable = Sam2Segmenter,
        sam3_factory: Callable = Sam3LiteSegmenter,
    ):
        self.settings = settings
        self._gate = RLock()
        self._device = resolve_device(settings.device)
        self._force_cpu: set[str] = set()
        self._factories = {
            "hand": hand_factory,
            "face": face_factory,
            "dinov2": dinov2_factory,
            "clip": clip_factory,
            "sam2": sam2_factory,
            "sam3": sam3_factory,
        }
        self._hand = None
        self._face = None
        self._embedders: dict[str, object] = {}
        self._sam2 = None
        self._sam3 = None
        self._errors: dict[str, str | None] = {
            "hand": None,
            "face": None,
            "dinov2": None,
            "clip": None,
            "sam2": None,
            "sam3": None,
        }

    def gate(self) -> RLock:
        return self._gate

    def _device_for(self, name: str) -> str:
        return "cpu" if name in self._force_cpu else self._device

    def _ensure_hand(self) -> None:
        if self._hand is not None:
            return
        try:
            self._hand = self._factories["hand"](asset_path=self.settings.hand_asset)
            self._errors["hand"] = None
        except Exception as exc:
            self._errors["hand"] = str(exc)
            raise ModelUnavailable(str(exc)) from exc

    def _ensure_face(self) -> None:
        if self._face is not None:
            return
        try:
            self._face = self._factories["face"](asset_path=self.settings.face_asset)
            self._errors["face"] = None
        except Exception as exc:
            self._errors["face"] = str(exc)
            raise ModelUnavailable(str(exc)) from exc

    def _ensure_embedder(self, kind: str) -> object:
        if kind in self._embedders:
            return self._embedders[kind]
        if kind == "clip" and not self.settings.enable_clip:
            raise ModelUnavailable("CLIP is disabled (AUTO_SCANNER_ENABLE_CLIP=1 to enable)")
        key = "clip" if kind == "clip" else "dinov2"
        model_id = self.settings.clip_model_id if kind == "clip" else self.settings.dinov2_model_id
        try:
            instance = self._factories[key](model_id=model_id, device=self._device_for(key))
            self._embedders[kind] = instance
            self._errors[key] = None
            return instance
        except Exception as exc:
            self._errors[key] = str(exc)
            raise ModelUnavailable(str(exc)) from exc

    def _ensure_sam2(self):
        if self._sam2 is None:
            try:
                self._sam2 = self._factories["sam2"](
                    model_id=self.settings.sam2_model_id, device=self._device_for("sam2")
                )
                self._errors["sam2"] = None
            except Exception as exc:
                self._errors["sam2"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc
        return self._sam2

    def _ensure_sam3(self):
        if self._sam3 is None:
            try:
                self._sam3 = self._factories["sam3"](
                    model_id=self.settings.sam3_model_id,
                    device=self._device_for("sam3"),
                    min_score=self.settings.sam3_min_score,
                    max_candidates=self.settings.sam3_max_candidates,
                )
                self._errors["sam3"] = None
            except Exception as exc:
                self._errors["sam3"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc
        return self._sam3

    def warm(self) -> None:
        with self._gate:
            for ensure in (
                self._ensure_hand,
                self._ensure_face,
                lambda: self._ensure_embedder("dinov2"),
            ):
                try:
                    ensure()
                except ModelUnavailable:
                    pass

    def detect_hand(self, image):
        with self._gate:
            self._ensure_hand()
            try:
                return self._hand.detect(image)
            except Exception as exc:
                self._errors["hand"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc

    def detect_face(self, image):
        with self._gate:
            self._ensure_face()
            try:
                return self._face.detect(image)
            except Exception as exc:
                self._errors["face"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc

    def embed(self, kind: str, image):
        with self._gate:
            return self._call_with_fallback(
                "clip" if kind == "clip" else "dinov2",
                lambda: self._ensure_embedder(kind).embed(image),
                reload=lambda: self._embedders.pop(kind, None),
            )

    def segment(self, image, prompt):
        with self._gate:
            if self.settings.localizer == "sam3":
                return self._call_with_fallback(
                    "sam3",
                    lambda: self._ensure_sam3().segment(image, prompt),
                    reload=lambda: setattr(self, "_sam3", None),
                )
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
            "localizer": self.settings.localizer,
            "models": {
                "hand": {"loaded": self._hand is not None, "device": "cpu", "error": self._errors["hand"]},
                "face": {"loaded": self._face is not None, "device": "cpu", "error": self._errors["face"]},
                "dinov2": {
                    "loaded": "dinov2" in self._embedders,
                    "device": self._device_for("dinov2"),
                    "error": self._errors["dinov2"],
                },
                "clip": {
                    "loaded": "clip" in self._embedders,
                    "device": self._device_for("clip"),
                    "error": self._errors["clip"],
                },
                "sam2": {
                    "loaded": self._sam2 is not None,
                    "device": self._device_for("sam2"),
                    "error": self._errors["sam2"],
                },
                "sam3": {
                    "loaded": self._sam3 is not None,
                    "device": self._device_for("sam3"),
                    "error": self._errors["sam3"],
                    "model_id": self.settings.sam3_model_id,
                },
            },
        }
