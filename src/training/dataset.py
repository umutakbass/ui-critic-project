"""PyTorch Dataset — eğitim kayıtlarını adapter formatına dönüştürür."""

import json
from pathlib import Path
from typing import Dict

from PIL import Image
from torch.utils.data import Dataset


class UICriticDataset(Dataset):
    """Eğitim verisi için PyTorch Dataset.

    Her __getitem__ çağrısında:
      1. Görsel okunur ve gerekirse küçültülür.
      2. Görev tipine göre doğru instruction + target oluşturulur.
      3. Adapter'ın format_prompt() ile model girdisi hazırlanır.
    """

    def __init__(
        self,
        records_path: str,
        task: str,
        adapter,
        max_image_size: int = 1024,
    ):
        with open(records_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.task = task
        self.adapter = adapter
        self.max_image_size = max_image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        from .prompts import (
            MODEL1_USER_INSTRUCTION,
            MODEL2_USER_INSTRUCTION,
            MODEL3_USER_INSTRUCTION_TEMPLATE,
        )

        rec = self.records[idx]

        img = Image.open(rec["image_path"]).convert("RGB")
        if max(img.size) > self.max_image_size:
            img.thumbnail((self.max_image_size, self.max_image_size))

        if self.task == "model1":
            instruction = MODEL1_USER_INSTRUCTION
            target = json.dumps(
                {
                    "critiques": rec["critiques"],
                    "overall_feedback": rec.get("overall_feedback", ""),
                },
                ensure_ascii=False,
            )
        elif self.task == "model2":
            instruction = MODEL2_USER_INSTRUCTION
            target = json.dumps(rec["hierarchy"], ensure_ascii=False)
        elif self.task == "model3":
            instruction = MODEL3_USER_INSTRUCTION_TEMPLATE.format(
                hierarchy_json=json.dumps(rec["predicted_hierarchy"], ensure_ascii=False)
            )
            target = json.dumps(
                {
                    "critiques": rec["critiques"],
                    "overall_feedback": rec.get("overall_feedback", ""),
                },
                ensure_ascii=False,
            )
        else:
            raise ValueError(f"Bilinmeyen task: {self.task}")

        inputs = self.adapter.format_prompt(instruction, img)
        return {"inputs": inputs, "target": target, "rico_id": rec["rico_id"]}
