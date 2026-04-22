"""Model 3 preprocessing: UICrit + Model 2 tahminleri → eğitim kayıtları."""

import json
from pathlib import Path
from typing import Dict, List

from ..uicrit_loader import UICritLoader
from .model1_prep import build_model1_records


def build_model3_records(
    uicrit: UICritLoader,
    rico_ids: List[int],
    rico_image_dir: str,
    predicted_hierarchies_path: str,
) -> List[Dict]:
    """Model 3 eğitim verisi: UI + Model 2'nin tahmin ettiği VH + kritikler.

    Args:
        uicrit: Yüklenmiş UICritLoader.
        rico_ids: Kullanılacak rico_id listesi.
        rico_image_dir: Görsel dosyalarının bulunduğu klasör.
        predicted_hierarchies_path: Model 2 tarafından üretilmiş VH JSON'u.
            Format: {"<rico_id>": <hierarchy_dict>, ...}

    Returns:
        Model 3 eğitim kayıtlarının listesi.
    """
    with open(predicted_hierarchies_path, "r", encoding="utf-8") as f:
        pred_vhs: Dict[str, Dict] = json.load(f)

    base_records = build_model1_records(uicrit, rico_ids, rico_image_dir)

    model3_records = []
    for rec in base_records:
        rid_str = str(rec["rico_id"])
        if rid_str not in pred_vhs:
            continue
        rec = rec.copy()
        rec["predicted_hierarchy"] = pred_vhs[rid_str]
        model3_records.append(rec)

    return model3_records
