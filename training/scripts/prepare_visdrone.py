from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VISDRONE_CLASS_MAP = {
    4: "car",
    5: "van",
    6: "truck",
    9: "bus",
}


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    source_dir: Path
    images_dir: Path
    annotations_dir: Path


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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    visdrone_root = repo_root / "data" / "raw" / "visdrone" / "extracted"

    train_split = build_split(visdrone_root, "VisDrone2019-DET-train")
    val_split = build_split(visdrone_root, "VisDrone2019-DET-val")

    for split in (train_split, val_split):
        validate_split(split)

    print("VisDrone raw dataset structure looks correct.")
    print("Detected splits:")
    print(f"- train: {train_split.source_dir}")
    print(f"- val:   {val_split.source_dir}")
    print("Kept vehicle classes:")
    for original_id, class_name in VISDRONE_CLASS_MAP.items():
        print(f"- VisDrone class {original_id}: {class_name}")
    print("")
    print("Next implementation step:")
    print("- parse annotation txt files")
    print("- filter to target vehicle classes")
    print("- convert boxes to YOLO format")
    print("- write labels and image manifests to data/processed/visdrone/")


if __name__ == "__main__":
    main()
