import time
from pathlib import Path

import numpy as np

from app.errors import ModelUnavailable
from app.pipeline.mask_select import select_box_mask, select_detection
from app.pipeline.prompts import build_box_prompt
from app.pipeline.types import Detection, Localization


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

        self.model_id = model_id
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.allowed_classes = allowed_classes
        self._model = YOLO(source)
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
        return [
            Detection(
                index=index,
                class_id=int(class_id),
                class_name=str(names.get(int(class_id), class_id)),
                confidence=float(confidence),
                bbox=tuple(float(value) for value in box[:4]),
            )
            for index, (box, confidence, class_id) in enumerate(
                zip(xyxy, confidences, classes)
            )
        ]


def localize_with_yolo_sam(settings, registry, image_rgb: np.ndarray) -> Localization:
    timings: dict[str, float] = {}

    start = time.perf_counter()
    detections = registry.detect_objects(image_rgb)
    timings["yolo"] = (time.perf_counter() - start) * 1000

    reject_classes = frozenset(
        token.strip().lower()
        for token in settings.yolo_reject_classes.split(",")
        if token.strip()
    )
    detection_selection = select_detection(
        detections,
        image_rgb.shape[1],
        image_rgb.shape[0],
        min_area_ratio=settings.yolo_min_area_ratio,
        max_area_ratio=settings.yolo_max_area_ratio,
        reject_classes=reject_classes,
        center_weight=settings.yolo_center_weight,
        min_confidence=settings.yolo_conf,
    )
    selected = detection_selection.selected
    if selected is None:
        return Localization(
            status="no_detection",
            detections=detections,
            detection_selection=detection_selection,
            timings_ms=timings,
        )

    prompt = build_box_prompt(
        selected.bbox, image_rgb.shape[1], image_rgb.shape[0]
    )
    if prompt.box is None:
        return Localization(
            status="no_detection",
            detections=detections,
            detection_selection=detection_selection,
            timings_ms=timings,
        )

    start = time.perf_counter()
    candidates = registry.segment(image_rgb, prompt)
    timings["sam2"] = (time.perf_counter() - start) * 1000

    mask_selection = select_box_mask(
        candidates,
        prompt.box,
        image_rgb.shape[1],
        image_rgb.shape[0],
        min_area_ratio=settings.yolo_min_area_ratio,
        max_area_ratio=settings.yolo_max_area_ratio,
        min_pred_iou=settings.yolo_sam_min_pred_iou,
        min_box_iou=settings.yolo_sam_min_box_iou,
        min_box_coverage=settings.yolo_sam_min_box_coverage,
        max_border_fraction=settings.yolo_sam_max_border_fraction,
    )
    selected_mask = mask_selection.selected
    return Localization(
        status="ok" if selected_mask is not None else "mask_rejected",
        mask=selected_mask.mask if selected_mask is not None else None,
        bbox=prompt.box,
        prompt=prompt,
        mask_selection=mask_selection,
        detections=detections,
        detection_selection=detection_selection,
        timings_ms=timings,
    )
