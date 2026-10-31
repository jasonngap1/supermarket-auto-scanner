from app.pipeline.prompts import (
    build_center_prompt,
    build_prompt,
    build_sam3_prompt,
    center_roi_box,
    hand_crop_box,
)


def landmarks(thumb=(40, 40), index=(60, 40), middle=(65, 50)):
    points = [(50.0, 90.0)] * 21
    points[0] = (50.0, 90.0)
    points[4] = thumb
    points[5] = (45.0, 70.0)
    points[8] = index
    points[9] = (55.0, 70.0)
    points[12] = middle
    points[13] = (60.0, 72.0)
    points[17] = (65.0, 75.0)
    return points


def test_open_pinch_uses_pinch_center_and_excludes_fingertips_from_negatives():
    prompt = build_prompt(landmarks(), 200, 200)
    assert prompt.positive_points == [(50.0, 40.0)]
    assert (40.0, 40.0) not in prompt.negative_points
    assert (60.0, 40.0) not in prompt.negative_points
    assert (50.0, 90.0) in prompt.negative_points
    assert prompt.box is not None
    x1, y1, x2, y2 = prompt.box
    assert x1 <= 50 <= x2 and y1 <= 40 <= y2


def test_closed_pinch_uses_grasp_center():
    prompt = build_prompt(landmarks(thumb=(50, 40), index=(52, 40)), 200, 200)
    assert prompt.positive_points == [(54.25, 42.5)]


def test_box_mode_none_and_clipping():
    prompt = build_prompt(landmarks(thumb=(2, 2), index=(4, 2)), 100, 100, mode="none")
    assert prompt.box is None
    expanded = build_prompt(landmarks(thumb=(2, 2), index=(4, 2)), 100, 100)
    assert expanded.box[0] >= 0 and expanded.box[1] >= 0
    crop = hand_crop_box(landmarks(thumb=(2, 2), index=(4, 2)), 100, 100)
    assert crop[0] >= 0 and crop[1] >= 0


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


def test_sam3_prompt_carries_text_and_optional_box():
    prompt = build_sam3_prompt(200, 100, "product")
    assert prompt.mode == "sam3_center"
    assert prompt.text == "product"
    assert prompt.positive_points == []
    assert prompt.box == (40.0, 15.0, 160.0, 85.0)
    without_box = build_sam3_prompt(200, 100, "product", include_box=False)
    assert without_box.box is None
    assert without_box.text == "product"
