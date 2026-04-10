# Intelligent Drone Vehicle Tracker

This project aims to let a user tag a vehicle in drone footage, then track and highlight that vehicle throughout the video, including reacquiring it after temporary tracking loss.

## Current Status

This repository is in the setup phase. The first milestone is preparing the training data pipeline for aerial vehicle detection using VisDrone.

## Planned Stack

- Detector: lightweight YOLO fine-tuned for aerial vehicle classes
- Tracking: BoT-SORT plus ReID-assisted target reacquisition
- Backend: FastAPI
- Frontend: React + Vite
- Runtime: Dockerized services for API, worker, and frontend

## Dataset Setup

We are starting with the official VisDrone object detection dataset.

- Official page: `https://aiskyeye.com/submit-2023/object-detection-2/`
- Train zip: `https://drive.google.com/file/d/1iNht3v0D2r-O4xjnmEGVZk6fB8tdUj-v/view?usp=sharing`
- Val zip: `https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing`

Download the archives manually and place them here:

```text
data/raw/visdrone/downloads/
```

After downloading, extract the archives into:

```text
data/raw/visdrone/extracted/
```

The exact expected folder layout is documented in `data/README.md`.

## Repository Layout

```text
data/
  raw/
    visdrone/
      downloads/
      extracted/
    uavdt/
      downloads/
      extracted/
  processed/
training/
  scripts/
```

## First Prep Script

The first data-prep scaffold lives at:

```text
training/scripts/prepare_visdrone.py
```

Its role is to:

- validate the expected raw dataset layout
- locate train and validation images and annotations
- filter to vehicle classes
- convert VisDrone annotations into YOLO-format labels
- write processed outputs into `data/processed/`

## License Note

VisDrone is published for academic, non-commercial use. That makes it suitable for learning and portfolio work, but not a safe default for commercial model release. Keep that note visible in future documentation and demos.
