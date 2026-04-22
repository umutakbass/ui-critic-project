"""RICO dataset yukleyici."""

import json
from pathlib import Path
from typing import Dict, Optional

from PIL import Image


class RicoLoader:
    """RICO dataset yukleyici."""

    def __init__(self, rico_dir):
        self.rico_dir = Path(rico_dir)
        self.combined_dir = self.rico_dir / "combined"

    def load_image(self, rico_id):
        img_path = self.combined_dir / f"{rico_id}.jpg"
        if not img_path.exists():
            return None
        return Image.open(img_path).convert("RGB")

    def load_hierarchy(self, rico_id):
        json_path = self.combined_dir / f"{rico_id}.json"
        if not json_path.exists():
            return None
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def image_exists(self, rico_id):
        return (self.combined_dir / f"{rico_id}.jpg").exists()

    def hierarchy_exists(self, rico_id):
        return (self.combined_dir / f"{rico_id}.json").exists()
