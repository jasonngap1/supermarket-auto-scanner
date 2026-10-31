import numpy as np
from PIL import Image

from app.matching.vector_index import normalize


class DinoV2Embedder:
    model_key = "dinov2-small"
    dim = 384

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(device).eval()

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        inputs = self._processor(images=Image.fromarray(image_rgb), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        vector = outputs.last_hidden_state[:, 0].squeeze(0).float().cpu().numpy()
        return normalize(vector)


class ClipEmbedder:
    model_key = "clip-vit-b32"
    dim = 512

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self._torch = torch
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._model = CLIPModel.from_pretrained(model_id).to(device).eval()

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        inputs = self._processor(images=Image.fromarray(image_rgb), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        # Transformers 5.x returns BaseModelOutputWithPooling; older versions
        # returned the projected embedding tensor directly.
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif hasattr(features, "image_embeds"):
            features = features.image_embeds
        return normalize(features.squeeze(0).float().cpu().numpy())
