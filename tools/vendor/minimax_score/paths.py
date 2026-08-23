"""Paths for the Hugging Face Space package (self-contained)."""

from __future__ import annotations

from pathlib import Path

# Space root = parent of the minimax package
ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"

SYSTEM_BASE_FILE = PROMPTS / "system_base.txt"
SYSTEM_REF_FILE = PROMPTS / "system_ref.txt"
SYSTEM_BASE_TASK_FILES = {
    "T2VA": PROMPTS / "system_base_t2va.txt",
    "I2VA": PROMPTS / "system_base_i2va.txt",
    "FL2VA": PROMPTS / "system_base_fl2va.txt",
    "L2VA": PROMPTS / "system_base_l2va.txt",
}

# Default Hub ids for this Space (2.6B champion)
DEFAULT_MODEL = "geocine/minimax-video-prompt-enhancer-2.6b"
DEFAULT_GGUF_REPO = "geocine/minimax-video-prompt-enhancer-2.6b-gguf"
DEFAULT_GGUF_FILE = "minimax-video-prompt-enhancer-2.6b-Q4_K_M.gguf"
BASE_MODEL = "LiquidAI/LFM2.5-2.6B"
CHAMPION_DIR = DEFAULT_MODEL
CHAMPION_2P6B_DIR = DEFAULT_MODEL
GGUF_DIR = ROOT / "models"
CHAMPION_GGUF = GGUF_DIR / DEFAULT_GGUF_FILE
