from app.pipeline.prompts import build_center_prompt, center_roi_box


def test_center_roi_is_centred_and_clipped():
    box = center_roi_box(200, 100, width_ratio=0.5, height_ratio=0.6)
    assert box == (50.0, 20.0, 150.0, 80.0)
    clamped = center_roi_box(200, 100, width_ratio=2.0, height_ratio=2.0)
    assert clamped == (0.0, 0.0, 199.0, 99.0)


def test_center_prompt_uses_frame_centre_point_and_box():
    prompt = build_center_prompt(200, 100, roi_width=0.5, roi_height=0.6)
    assert prompt.mode == "sam2_center"
    assert prompt.positive_points == [(100.0, 50.0)]
    assert len(prompt.negative_points) == 12
    assert all(
        not (p[0] >= 50 and p[0] <= 150 and p[1] >= 20 and p[1] <= 80)
        for p in prompt.negative_points
    )
    assert prompt.box == (50.0, 20.0, 150.0, 80.0)


def test_center_prompt_adds_visible_face_negatives():
    prompt = build_center_prompt(
        200, 100, roi_width=0.5, roi_height=0.6, face_bbox=(160, 5, 195, 40)
    )
    assert len(prompt.negative_points) == 17
    covering = build_center_prompt(
        200, 100, roi_width=0.5, roi_height=0.6, face_bbox=(80, 30, 120, 70)
    )
    assert len(covering.negative_points) == 12
