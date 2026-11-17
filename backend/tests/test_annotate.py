from io import BytesIO

from PIL import Image

from app.domain import TopItem
from app.pipeline.annotate import annotate_frame
from app.pipeline.frame_pipeline import FramePipeline
from tests.fakes import FakeRegistry, detection, image, make_settings


def test_annotate_returns_decodable_jpeg(tmp_path):
    settings = make_settings(tmp_path)
    pipeline = FramePipeline(
        FakeRegistry(detections=[detection()]),
        settings,
    )
    frame = image()
    output = pipeline.analyze(frame, ["v2"])
    jpeg = annotate_frame(
        frame,
        output.localization,
        TopItem(1, "Milk 1L", 199, 0.91),
        currency_symbol="$",
    )
    decoded = Image.open(BytesIO(jpeg))
    assert decoded.format == "JPEG"
    assert decoded.size == (100, 100)


def test_annotate_accepts_missing_localization():
    jpeg = annotate_frame(image(), None, None, label="true: Milk | pred: none")
    decoded = Image.open(BytesIO(jpeg))
    assert decoded.format == "JPEG"
