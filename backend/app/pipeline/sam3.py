import numpy as np
from PIL import Image

from app.pipeline.types import MaskCandidate, PromptSpec


class Sam3LiteSegmenter:
    """SAM3-LiteText wrapper for concept + centre-box prompts.

    The model returns instance masks ranked by text-concept similarity. We keep
    only the top candidates and upsample those, rather than every query, to
    keep memory and CPU post-processing bounded.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        *,
        min_score: float = 0.30,
        max_candidates: int = 8,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self._torch = torch
        self._min_score = min_score
        self._max_candidates = max_candidates
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(device).eval()

    def segment(self, image_rgb: np.ndarray, prompt: PromptSpec) -> list[MaskCandidate]:
        height, width = image_rgb.shape[:2]
        payload = {
            "images": Image.fromarray(image_rgb),
            "text": prompt.text or "object",
            "return_tensors": "pt",
        }
        if prompt.box is not None:
            payload["input_boxes"] = [[list(map(float, prompt.box))]]
            payload["input_boxes_labels"] = [[1]]

        inputs = self._processor(**payload).to(self.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        # Move the compact query tensors to CPU before ranking; this avoids
        # advanced indexing on MPS and frees the model outputs promptly.
        pred_masks = outputs.pred_masks[0].detach().float().cpu()
        scores = outputs.pred_logits[0].detach().float().cpu().sigmoid()
        if outputs.presence_logits is not None:
            scores = scores * outputs.presence_logits[0].detach().float().cpu().sigmoid()
        del outputs

        keep = scores >= self._min_score
        order = self._torch.argsort(scores, descending=True)
        order = order[keep[order]][: self._max_candidates]
        if order.numel() == 0:
            return []

        selected = pred_masks[order]
        selected_scores = scores[order]

        # Upsample only the shortlisted instances. CPU keeps the MPS allocator
        # from holding a (candidates x full-resolution) mask buffer.
        upsampled = self._torch.nn.functional.interpolate(
            selected.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        binary = (upsampled.sigmoid() > 0.5).numpy().astype(bool)

        if self.device == "mps" and hasattr(self._torch, "mps"):
            self._torch.mps.empty_cache()

        return [
            MaskCandidate(index=i, mask=binary[i], pred_iou=float(selected_scores[i]))
            for i in range(binary.shape[0])
        ]
