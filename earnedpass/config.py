"""Paths and the constants fixed in the implementation plan (Agent A tab)."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "EARNEDPASS_DATA",
        REPO.parent / "Multimodal Dataset for Cinnamon Powder Adulteratio" / "Dataset",
    )
)
ARTIFACTS = Path(os.environ.get("EARNEDPASS_ARTIFACTS", REPO / "artifacts"))
TWINS_CSV = REPO / "data" / "image_twins.csv"
THUMBS = ARTIFACTS / "thumbs"

FOLDERS = {"P": "Peanut", "W": "Walnut", "D": "Wood_Dust"}
FOLDER_LABEL = {"P": "Peanut shell", "W": "Walnut shell", "D": "Wood dust"}
LEVELS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24]
LOCKED_LEVELS = [3, 10, 18]  # untouched until the one-time Sunday scoring
BANDS = ["zero", "low", "mid", "high"]  # 0 | 1-4 | 6-12 | 15-24 (% w/w)
BAND_RANGE = {"zero": "0%", "low": "1–4%", "mid": "6–12%", "high": "15–24%"}
BAND_FLOOR = {"zero": 0, "low": 1, "mid": 6, "high": 15}

SENSOR_SETS = {
    "colour": ("colour",),
    "ftir": ("ftir",),
    "image": ("image",),
    "colour+ftir": ("colour", "ftir"),
    "colour+image": ("colour", "image"),
    "ftir+image": ("ftir", "image"),
    "all": ("colour", "ftir", "image"),
}
SCREEN_SET = "colour+image"  # what the inspector has at receiving
LAB_SET = "ftir"  # what "Send to lab" reveals
TASKS = ["detect", "band", "level", "type"]
RUNGS = ["1_sample_id", "2_twin_aware", "3_leave_cell_out", "4_leave_folder_out"]
RUNG_LABEL = {
    "1_sample_id": "1 · sample-ID split",
    "2_twin_aware": "2 · twin-aware",
    "3_leave_cell_out": "3 · leave-one-cell-out",
    "4_leave_folder_out": "4 · leave-one-folder-out",
}

# Defaults the QC manager can change in the app. Illustrative, not regulatory.
ALPHA = 0.10
MUST_CATCH = 6
PASS_BAR = 0.90
SEED = 0


def band_of(level: int) -> str:
    if level == 0:
        return "zero"
    if level <= 4:
        return "low"
    if level <= 12:
        return "mid"
    return "high"
