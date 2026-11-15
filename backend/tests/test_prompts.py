from app.pipeline.prompts import build_box_prompt, clip_box


def test_clip_box_clamps_to_frame():
    assert clip_box((-10, -5, 120, 80), 100, 60) == (0.0, 0.0, 100.0, 60.0)


def test_build_box_prompt_uses_box_centre():
    prompt = build_box_prompt((20, 30, 60, 90), 100, 100)
    assert prompt.box == (20.0, 30.0, 60.0, 90.0)
    assert prompt.positive_points == [(40.0, 60.0)]


def test_build_box_prompt_rejects_empty_box():
    prompt = build_box_prompt((50, 50, 50, 50), 100, 100)
    assert prompt.box is None
    assert prompt.positive_points == []
