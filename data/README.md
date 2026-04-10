# Data Layout

This project keeps raw downloads, extracted source datasets, and processed training outputs separate.

## Folder Structure

```text
data/
  raw/
    visdrone/
      downloads/
      extracted/
        VisDrone2019-DET-train/
          images/
          annotations/
        VisDrone2019-DET-val/
          images/
          annotations/
    uavdt/
      downloads/
      extracted/
  processed/
    visdrone/
      images/
        train/
        val/
      labels/
        train/
        val/
```

## VisDrone Download

Download the official archives from:

- `https://aiskyeye.com/submit-2023/object-detection-2/`
- `https://drive.google.com/file/d/1iNht3v0D2r-O4xjnmEGVZk6fB8tdUj-v/view?usp=sharing`
- `https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing`

Save the zip files in:

```text
data/raw/visdrone/downloads/
```

Then extract them so that these two directories exist:

```text
data/raw/visdrone/extracted/VisDrone2019-DET-train/
data/raw/visdrone/extracted/VisDrone2019-DET-val/
```

Each extracted split should contain:

- `images/`
- `annotations/`

## Class Scope

For v1, the prep pipeline will keep only these VisDrone classes:

- `car`
- `van`
- `truck`
- `bus`

The prep script will remap those classes into a compact YOLO class index for training.

## Important Notes

- Do not commit raw or processed datasets to git.
- Keep your own sample drone videos outside version control as well.
- VisDrone is intended for academic, non-commercial use, so treat trained weights as portfolio/research assets unless the training data strategy changes later.
