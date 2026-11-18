def resolve_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
