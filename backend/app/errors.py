class InvalidImage(ValueError):
    pass


class ImageTooLarge(ValueError):
    pass


class DuplicateCapture(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


class CaptureRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
