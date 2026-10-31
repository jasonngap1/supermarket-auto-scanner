import numpy as np
from PIL import Image

from app.pipeline.types import MaskCandidate, PromptSpec


def prompt_payload(prompt: PromptSpec) -> dict:
    points = [*prompt.positive_points, *prompt.negative_points]
    labels = [1] * len(prompt.positive_points) + [0] * len(prompt.negative_points)
    payload = {
        "input_points": [[[list(map(float, point)) for point in points]]],
        "input_labels": [[labels]],
    }
    if prompt.box is not None:
        payload["input_boxes"] = [[list(map(float, prompt.box))]]
    return payload


class Sam2Segmenter:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.device = device
        self._torch = torch
        self._processor = Sam2Processor.from_pretrained(model_id)
        self._model = Sam2Model.from_pretrained(model_id).to(device).eval()

    def segment(self, image_rgb: np.ndarray, prompt: PromptSpec) -> list[MaskCandidate]:
        payload = prompt_payload(prompt)
        payload.update(images=Image.fromarray(image_rgb), return_tensors="pt")
        inputs = self._processor(**payload).to(self.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs, multimask_output=True)

        # Transformers 4.x expects reshaped_input_sizes; 5.x dropped it and
        # treats the third positional argument as mask_threshold.
        extra = {}
        if inputs.get("reshaped_input_sizes") is not None:
            extra["reshaped_input_sizes"] = inputs["reshaped_input_sizes"]
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"], **extra
        )[0]

        if hasattr(masks, "cpu"):
            mask_array = masks.cpu().numpy()
        else:
            mask_array = np.asarray(masks)
        if mask_array.ndim == 4:
            mask_array = mask_array[0]
        if mask_array.ndim == 2:
            mask_array = mask_array[None, ...]

        # SAM2 predicts IoU as a logit; store calibrated probabilities so the
        # selector's thresholds and ranking use the same scale.
        iou_logits = (
            outputs.iou_scores[0].detach().float().cpu().numpy().reshape(-1)
        )
        iou_array = 1.0 / (1.0 + np.exp(-iou_logits))
        count = min(mask_array.shape[0], iou_array.size)
        return [
            MaskCandidate(index=i, mask=mask_array[i] > 0, pred_iou=float(iou_array[i]))
            for i in range(count)
        ]
