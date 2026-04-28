from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Iterable

from PIL import Image


VISDRONE_CLASS_MAP = {
    4: "car",
    5: "van",
    6: "truck",
    9: "bus",
}

YOLO_CLASS_IDS = {
    4: 0,
    5: 1,
    6: 2,
    9: 3,
}


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    source_dir: Path
    images_dir: Path
    annotations_dir: Path


@dataclass(frozen=True)
class DatasetSummary:
    splits: tuple[DatasetSplit, ...]
    class_map: dict[int, str]


@dataclass(frozen=True)
class VisDroneAnnotation:
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    score: int
    object_category: int
    truncation: int
    occlusion: int


def build_split(root: Path, split_name: str) -> DatasetSplit:
    source_dir = root / split_name
    return DatasetSplit(
        name=split_name,
        source_dir=source_dir,
        images_dir=source_dir / "images",
        annotations_dir=source_dir / "annotations",
    )


def validate_split(split: DatasetSplit) -> None:
    missing = [
        path
        for path in (split.source_dir, split.images_dir, split.annotations_dir)
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing expected VisDrone paths for {split.name}: {missing_text}"
        )


def validate_splits(splits: Iterable[DatasetSplit]) -> None:
    for split in splits:
        validate_split(split)


def get_default_splits(visdrone_root: Path) -> tuple[DatasetSplit, DatasetSplit]:
    train_split = build_split(visdrone_root, "VisDrone2019-DET-train")
    val_split = build_split(visdrone_root, "VisDrone2019-DET-val")
    return train_split, val_split


def summarize_dataset(visdrone_root: Path) -> DatasetSummary:
    splits = get_default_splits(visdrone_root)
    validate_splits(splits)
    return DatasetSummary(splits=splits, class_map=VISDRONE_CLASS_MAP)


def parse_annotation_line(line: str) -> VisDroneAnnotation:
    parts = [part.strip() for part in line.split(",")]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    if len(parts) != 8:
        raise ValueError(f"Expected 8 comma-separated fields, got {len(parts)}: {line!r}")

    return VisDroneAnnotation(
        bbox_left=float(parts[0]),
        bbox_top=float(parts[1]),
        bbox_width=float(parts[2]),
        bbox_height=float(parts[3]),
        score=int(parts[4]),
        object_category=int(parts[5]),
        truncation=int(parts[6]),
        occlusion=int(parts[7]),
    )


def is_target_vehicle(annotation: VisDroneAnnotation) -> bool:
    return annotation.object_category in YOLO_CLASS_IDS


def convert_bbox_to_yolo(
    annotation: VisDroneAnnotation,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x_center = (annotation.bbox_left + annotation.bbox_width / 2.0) / image_width
    y_center = (annotation.bbox_top + annotation.bbox_height / 2.0) / image_height
    width = annotation.bbox_width / image_width
    height = annotation.bbox_height / image_height
    return x_center, y_center, width, height


def format_yolo_label(class_id: int, bbox: tuple[float, float, float, float]) -> str:
    x_center, y_center, width, height = bbox
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def get_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def ensure_processed_dirs(processed_root: Path) -> None:
    for relative_dir in (
        Path("images/train"),
        Path("images/val"),
        Path("labels/train"),
        Path("labels/val"),
    ):
        (processed_root / relative_dir).mkdir(parents=True, exist_ok=True)


def process_annotation_file(annotation_path: Path, image_path: Path) -> list[str]:
    image_width, image_height = get_image_size(image_path)
    yolo_labels: list[str] = []

    for raw_line in annotation_path.read_text().splitlines():
        if not raw_line.strip():
            continue

        annotation = parse_annotation_line(raw_line)
        if not is_target_vehicle(annotation):
            continue

        class_id = YOLO_CLASS_IDS[annotation.object_category]
        bbox = convert_bbox_to_yolo(annotation, image_width=image_width, image_height=image_height)
        yolo_labels.append(format_yolo_label(class_id, bbox))

    return yolo_labels


def process_split(split: DatasetSplit, processed_root: Path, output_split: str) -> int:
    images_out_dir = processed_root / "images" / output_split
    labels_out_dir = processed_root / "labels" / output_split
    processed_count = 0

    for annotation_path in sorted(split.annotations_dir.glob("*.txt")):
        image_path = split.images_dir / f"{annotation_path.stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for annotation file: {annotation_path}")

        yolo_labels = process_annotation_file(annotation_path, image_path)
        if not yolo_labels:
            continue

        copy2(image_path, images_out_dir / image_path.name)
        label_path = labels_out_dir / f"{annotation_path.stem}.txt"
        label_path.write_text("\n".join(yolo_labels) + "\n")
        processed_count += 1

    return processed_count


def prepare_visdrone_dataset(
    visdrone_root: Path,
    processed_root: Path,
) -> dict[str, int]:
    splits = get_default_splits(visdrone_root)
    validate_splits(splits)
    ensure_processed_dirs(processed_root)

    results = {
        "train": process_split(splits[0], processed_root, "train"),
        "val": process_split(splits[1], processed_root, "val"),
    }
    return results


def format_summary(summary: DatasetSummary) -> str:
    lines = [
        "VisDrone raw dataset structure looks correct.",
        "Detected splits:",
    ]
    for split in summary.splits:
        lines.append(f"- {split.name}: {split.source_dir}")
    lines.append("Kept vehicle classes:")
    for original_id, class_name in summary.class_map.items():
        lines.append(f"- VisDrone class {original_id}: {class_name}")
    lines.extend(
        [
            "",
            "Next implementation step:",
            "- parse annotation txt files",
            "- filter to target vehicle classes",
            "- convert boxes to YOLO format",
            "- write labels and image manifests to data/processed/visdrone/",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    visdrone_root = repo_root / "data" / "raw" / "visdrone" / "extracted"
    processed_root = repo_root / "data" / "processed" / "visdrone"

    summary = summarize_dataset(visdrone_root)
    print(format_summary(summary))
    results = prepare_visdrone_dataset(visdrone_root, processed_root)
    print("")
    print("Processed output written to:")
    print(f"- {processed_root}")
    print("Files created:")
    print(f"- train samples: {results['train']}")
    print(f"- val samples:   {results['val']}")


if __name__ == "__main__":
    main()
