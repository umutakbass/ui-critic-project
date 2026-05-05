"""PyTorch Dataset — eğitim kayıtlarını VLM eğitim formatına dönüştürür."""

import json
from typing import Dict, Tuple

from PIL import Image
from torch.utils.data import Dataset


class UICriticDataset(Dataset):
    """Her __getitem__ çağrısında:
    1. Görsel okunur ve küçültülür.
    2. Tam konuşma (kullanıcı + asistan) oluşturulur.
    3. Processor ile tokenize edilir ve labels eklenir.
    """

    def __init__(self, records_path: str, task: str, adapter, max_image_size: int = 1024):
        with open(records_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.task = task
        self.adapter = adapter
        self.max_image_size = max_image_size

    def __len__(self) -> int:
        return len(self.records)

    def _get_instruction_target(self, rec: Dict) -> Tuple[str, str]:
        from .prompts import (
            MODEL1_USER_INSTRUCTION,
            MODEL2_USER_INSTRUCTION,
            MODEL3_USER_INSTRUCTION_TEMPLATE,
        )
        if self.task == "model1":
            instruction = MODEL1_USER_INSTRUCTION
            target = json.dumps(
                {"critiques": rec["critiques"], "overall_feedback": rec.get("overall_feedback", "")},
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
                {"critiques": rec["critiques"], "overall_feedback": rec.get("overall_feedback", "")},
                ensure_ascii=False,
            )
        else:
            raise ValueError(f"Bilinmeyen task: {self.task}")
        return instruction, target

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]

        img = Image.open(rec["image_path"]).convert("RGB")
        if max(img.size) > self.max_image_size:
            img.thumbnail((self.max_image_size, self.max_image_size))

        instruction, target = self._get_instruction_target(rec)

        # Adapter'a özgü eğitim formatına dönüştür (labels masking dahil)
        inputs = self.adapter.prepare_training_inputs(instruction, target, img, max_length=2048)

        # Batch boyutunu kaldır (her örnek tek)
        result = {k: v.squeeze(0) for k, v in inputs.items()}
        # Adapter zaten labels döndürmüyorsa input_ids'den kopyala
        if "labels" not in result:
            result["labels"] = result["input_ids"].clone()
        return result
