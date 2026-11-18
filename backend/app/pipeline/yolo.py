import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.errors import ModelUnavailable
from app.pipeline.mask_select import select_box_mask, select_detection
from app.pipeline.types import (
    Detection,
    DetectionSelection,
    Localization,
    MaskCandidate,
)


def parse_class_ids(spec: str, names: dict[int, str]) -> list[int] | None:
    if not spec.strip():
        return None
    by_name = {name.lower(): class_id for class_id, name in names.items()}
    ids: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            class_id = int(token)
            if class_id not in names:
                raise ModelUnavailable(f"Unknown YOLO class ID: {token}")
        else:
            class_id = by_name.get(token.lower())
            if class_id is None:
                raise ModelUnavailable(f"Unknown YOLO class name: {token}")
        if class_id not in ids:
            ids.append(class_id)
    return ids or None


def masks_from_result(
    result, count: int, frame_shape: tuple[int, int] | None = None
) -> list[np.ndarray | None]:
    """Convert ultralytics masks to boolean arrays at the frame size.

    Ultralytics returns uint8 masks. Boolean indexing elsewhere expects real
    booleans, so the conversion happens here. Any shape or count mismatch drops
    the masks rather than indexing the wrong detection.
    """
    masks = getattr(result, "masks", None)
    if masks is None:
        return [None] * count

    data = masks.data
    array = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    if array.ndim == 2:
        array = array[None, ...]
    if array.shape[0] != count:
        return [None] * count

    converted: list[np.ndarray | None] = []
    for row in array:
        mask = row.astype(bool)
        if frame_shape is not None and mask.shape != frame_shape:
            resized = Image.fromarray(mask.astype(np.uint8))
            resized = resized.resize((frame_shape[1], frame_shape[0]), Image.NEAREST)
            mask = np.asarray(resized, dtype=bool)
        converted.append(mask)
    return converted


class YoloDetector:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        allowed_classes: str = "",
        weights_dir: Path | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailable("YOLO is unavailable; run: uv sync") from exc

        candidate = Path(model_id)
        if candidate.is_absolute():
            source = str(candidate)
        elif weights_dir is not None and (weights_dir / model_id).exists():
            source = str(weights_dir / model_id)
        else:
            source = model_id

        is_onnx = source.lower().endswith(".onnx")
        if is_onnx:
            try:
                import onnxruntime  # noqa: F401
            except ImportError as exc:
                raise ModelUnavailable(
                    "ONNX Runtime is unavailable; run: uv sync"
                ) from exc

        self.model_id = model_id
        # ONNX Runtime on MPS is not validated here; the detector stays on CPU.
        self.device = "cpu" if is_onnx else device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.allowed_classes = allowed_classes
        self._model = YOLO(source)
        if getattr(self._model, "task", None) != "segment":
            raise ModelUnavailable(
                f"YOLO model {model_id} is not a segmentation model"
            )
        self._class_ids = parse_class_ids(allowed_classes, self._model.names)

    def predict(self, image_rgb: np.ndarray) -> list[Detection]:
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        results = self._model.predict(
            source=bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=self._class_ids,
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []
        names = result.names
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        masks = masks_from_result(result, len(xyxy), image_rgb.shape[:2])
        return [
            Detection(
                index=index,
                class_id=int(class_id),
                class_name=str(names.get(int(class_id), class_id)),
                confidence=float(confidence),
                bbox=tuple(float(value) for value in box[:4]),
                mask=mask,
            )
            for index, (box, confidence, class_id, mask) in enumerate(
                zip(xyxy, confidences, classes, masks)
            )
        ]


def detect_and_select(
    settings, registry, image_rgb: np.ndarray
) -> tuple[list[Detection], DetectionSelection, float]:
    start = time.perf_counter()
    detections = registry.detect_objects(image_rgb)
    elapsed_ms = (time.perf_counter() - start) * 1000

    reject_classes = frozenset(
        token.strip().lower()
        for token in settings.yolo_reject_classes.split(",")
        if token.strip()
    )
    selection = select_detection(
        detections,
        image_rgb.shape[1],
        image_rgb.shape[0],
        min_area_ratio=settings.yolo_min_area_ratio,
        max_area_ratio=settings.yolo_max_area_ratio,
        reject_classes=reject_classes,
        center_weight=settings.yolo_center_weight,
        min_confidence=settings.yolo_conf,
    )
    return detections, selection, elapsed_ms


def localize_with_yolo_seg(settings, registry, image_rgb: np.ndarray) -> Localization:
    """Production chain: one YOLO26n-seg pass supplies the box and the mask."""
    detections, detection_selection, yolo_ms = detect_and_select(
        settings, registry, image_rgb
    )
    timings = {"yolo": yolo_ms}
    selected = detection_selection.selected
    if selected is None:
        return Localization(
            status="no_detection",
            detections=detections,
            detection_selection=detection_selection,
            timings_ms=timings,
        )

    if selected.mask is None:
        return Localization(
            status="mask_rejected",
            bbox=selected.bbox,
            detections=detections,
            detection_selection=detection_selection,
            timings_ms=timings,
        )

    mask_selection = select_box_mask(
        [MaskCandidate(0, selected.mask, 1.0)],
        selected.bbox,
        image_rgb.shape[1],
        image_rgb.shape[0],
        min_area_ratio=settings.yolo_min_area_ratio,
        max_area_ratio=settings.yolo_max_area_ratio,
        min_pred_iou=0.0,
        min_box_iou=settings.mask_min_box_iou,
        min_box_coverage=settings.mask_min_box_coverage,
        max_border_fraction=settings.mask_max_border_fraction,
    )
    selected_mask = mask_selection.selected
    return Localization(
        status="ok" if selected_mask is not None else "mask_rejected",
        mask=selected_mask.mask if selected_mask is not None else None,
        bbox=selected.bbox,
        mask_selection=mask_selection,
        detections=detections,
        detection_selection=detection_selection,
        timings_ms=timings,
    )
