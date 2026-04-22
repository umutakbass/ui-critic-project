"""Model 1 preprocessing: UICrit → eğitim kayıtları."""

import json
from pathlib import Path
from typing import Dict, List

from ..uicrit_loader import UICritLoader


def build_model1_records(
    uicrit: UICritLoader,
    rico_ids: List[int],
    rico_image_dir: str,
) -> List[Dict]:
    """UICrit kayıtlarını Model 1 eğitim formatına dönüştür.

    Her çıktı kaydı bir rico_id'ye karşılık gelir; o UI için tüm
    annotator'ların yorumları tek bir critiques listesinde birleştirilir.
    """
    df = uicrit.load()
    records = []

    for rid in rico_ids:
        group = df[df["rico_id"] == rid]
        if group.empty:
            continue

        all_critiques = []
        for _, row in group.iterrows():
            comments = row["comments"]      # List[Dict] — uicrit_loader parse etti
            sources = row["comments_source"]  # List[str]

            for comment_dict, src in zip(comments, sources):
                all_critiques.append({
                    "comment": comment_dict.get("comment", ""),
                    "bounding_box": comment_dict.get("bounding_box"),
                    "source": src,
                })

        records.append({
            "rico_id": int(rid),
            "image_path": str(Path(rico_image_dir) / f"{rid}.jpg"),
            "task": str(group.iloc[0].get("task", "") or ""),
            "critiques": all_critiques,
            "aesthetics_rating": float(group["aesthetics_rating"].mean()),
            "learnability": float(group["learnability"].mean()),
        })

    return records


def save_records(records: List[Dict], output_path: str) -> None:
    """Kayıtları JSON dosyasına yaz."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
