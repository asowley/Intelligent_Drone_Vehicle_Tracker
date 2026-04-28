from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from training.scripts.prepare_visdrone import (
    VISDRONE_CLASS_MAP,
    YOLO_CLASS_IDS,
    build_split,
    convert_bbox_to_yolo,
    format_summary,
    parse_annotation_line,
    prepare_visdrone_dataset,
    process_annotation_file,
    summarize_dataset,
    validate_split,
)


def create_split_dirs(root: Path, split_name: str) -> Path:
    split_root = root / split_name
    (split_root / "images").mkdir(parents=True)
    (split_root / "annotations").mkdir(parents=True)
    return split_root


def create_image(path: Path, size: tuple[int, int] = (100, 50)) -> None:
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def test_build_split_uses_expected_layout(tmp_path: Path) -> None:
    split = build_split(tmp_path, "VisDrone2019-DET-train")

    assert split.name == "VisDrone2019-DET-train"
    assert split.source_dir == tmp_path / "VisDrone2019-DET-train"
    assert split.images_dir == tmp_path / "VisDrone2019-DET-train" / "images"
    assert split.annotations_dir == tmp_path / "VisDrone2019-DET-train" / "annotations"


def test_validate_split_raises_when_required_paths_are_missing(tmp_path: Path) -> None:
    split = build_split(tmp_path, "VisDrone2019-DET-train")

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_split(split)

    assert "Missing expected VisDrone paths" in str(exc_info.value)


def test_parse_annotation_line_reads_all_fields() -> None:
    annotation = parse_annotation_line("708,471,74,33,1,4,0,1")

    assert annotation.bbox_left == 708
    assert annotation.bbox_top == 471
    assert annotation.bbox_width == 74
    assert annotation.bbox_height == 33
    assert annotation.score == 1
    assert annotation.object_category == 4
    assert annotation.truncation == 0
    assert annotation.occlusion == 1


def test_parse_annotation_line_accepts_trailing_comma() -> None:
    annotation = parse_annotation_line("440,541,271,152,1,6,0,0,")

    assert annotation.bbox_left == 440
    assert annotation.bbox_top == 541
    assert annotation.bbox_width == 271
    assert annotation.bbox_height == 152
    assert annotation.object_category == 6


def test_convert_bbox_to_yolo_normalizes_correctly() -> None:
    annotation = parse_annotation_line("10,5,20,10,1,4,0,0")

    x_center, y_center, width, height = convert_bbox_to_yolo(
        annotation,
        image_width=100,
        image_height=50,
    )

    assert x_center == pytest.approx(0.2)
    assert y_center == pytest.approx(0.2)
    assert width == pytest.approx(0.2)
    assert height == pytest.approx(0.2)


def test_process_annotation_file_filters_non_vehicle_classes(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    annotation_path = tmp_path / "sample.txt"
    create_image(image_path, size=(100, 50))
    annotation_path.write_text(
        "\n".join(
            [
                "10,5,20,10,1,4,0,0",
                "1,1,10,10,1,1,0,0",
            ]
        )
    )

    labels = process_annotation_file(annotation_path, image_path)

    assert labels == ["0 0.200000 0.200000 0.200000 0.200000"]


def test_summarize_dataset_returns_expected_vehicle_classes(tmp_path: Path) -> None:
    create_split_dirs(tmp_path, "VisDrone2019-DET-train")
    create_split_dirs(tmp_path, "VisDrone2019-DET-val")

    summary = summarize_dataset(tmp_path)

    assert len(summary.splits) == 2
    assert summary.class_map == VISDRONE_CLASS_MAP
    assert summary.class_map[4] == "car"
    assert summary.class_map[5] == "van"
    assert summary.class_map[6] == "truck"
    assert summary.class_map[9] == "bus"


def test_format_summary_contains_expected_sections(tmp_path: Path) -> None:
    create_split_dirs(tmp_path, "VisDrone2019-DET-train")
    create_split_dirs(tmp_path, "VisDrone2019-DET-val")

    summary = summarize_dataset(tmp_path)
    output = format_summary(summary)

    assert "VisDrone raw dataset structure looks correct." in output
    assert "Detected splits:" in output
    assert "Kept vehicle classes:" in output
    assert "Next implementation step:" in output


def test_prepare_visdrone_dataset_creates_processed_yolo_outputs(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    train_split = create_split_dirs(raw_root, "VisDrone2019-DET-train")
    val_split = create_split_dirs(raw_root, "VisDrone2019-DET-val")

    create_image(train_split / "images" / "frame1.jpg", size=(100, 50))
    (train_split / "annotations" / "frame1.txt").write_text(
        "\n".join(
            [
                "10,5,20,10,1,4,0,0",
                "40,10,10,20,1,9,0,0",
                "5,5,10,10,1,1,0,0",
            ]
        )
    )

    create_image(val_split / "images" / "frame2.jpg", size=(200, 100))
    (val_split / "annotations" / "frame2.txt").write_text("20,10,40,20,1,6,0,0")

    results = prepare_visdrone_dataset(raw_root, processed_root)

    assert results == {"train": 1, "val": 1}
    assert (processed_root / "images" / "train" / "frame1.jpg").exists()
    assert (processed_root / "images" / "val" / "frame2.jpg").exists()
    assert (processed_root / "labels" / "train" / "frame1.txt").exists()
    assert (processed_root / "labels" / "val" / "frame2.txt").exists()

    train_labels = (processed_root / "labels" / "train" / "frame1.txt").read_text().splitlines()
    val_labels = (processed_root / "labels" / "val" / "frame2.txt").read_text().splitlines()

    assert train_labels == [
        f"{YOLO_CLASS_IDS[4]} 0.200000 0.200000 0.200000 0.200000",
        f"{YOLO_CLASS_IDS[9]} 0.450000 0.400000 0.100000 0.400000",
    ]
    assert val_labels == [f"{YOLO_CLASS_IDS[6]} 0.200000 0.200000 0.200000 0.200000"]


def test_prepare_visdrone_dataset_skips_files_without_target_vehicles(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    train_split = create_split_dirs(raw_root, "VisDrone2019-DET-train")
    create_split_dirs(raw_root, "VisDrone2019-DET-val")

    create_image(train_split / "images" / "frame1.jpg", size=(100, 50))
    (train_split / "annotations" / "frame1.txt").write_text("5,5,10,10,1,1,0,0")

    results = prepare_visdrone_dataset(raw_root, processed_root)

    assert results["train"] == 0
    assert not (processed_root / "labels" / "train" / "frame1.txt").exists()
