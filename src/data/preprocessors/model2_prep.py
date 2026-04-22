"""Model 2 preprocessing: RICO view hierarchy → sadeleştirilmiş eğitim kayıtları."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..rico_loader import RicoLoader


def simplify_hierarchy(node: Dict, max_depth: int = 10, current_depth: int = 0) -> Optional[Dict]:
    """VH ağacını modelin işleyebileceği sade formata indirge.

    Görünmeyen elementleri atlar, yalnızca type/bounds/text/clickable tutar.
    max_depth aşılırsa None döner (sonsuz özyinelemeyi önler).
    """
    if current_depth >= max_depth or not node:
        return None

    if not node.get("visible-to-user", True):
        return None

    simplified: Dict = {
        "type": node.get("class", "Unknown").split(".")[-1],
        "bounds": node.get("bounds", [0, 0, 0, 0]),
    }
    if node.get("text"):
        simplified["text"] = node["text"]
    if node.get("clickable"):
        simplified["clickable"] = True

    children = []
    for child in node.get("children", []) or []:
        child_simplified = simplify_hierarchy(child, max_depth, current_depth + 1)
        if child_simplified is not None:
            children.append(child_simplified)

    simplified["children"] = children
    return simplified


def build_model2_records(
    rico: RicoLoader,
    rico_ids: List[int],
    rico_image_dir: str,
    max_depth: int = 10,
) -> List[Dict]:
    """RICO ID listesinden Model 2 eğitim kayıtları oluştur.

    Her kayıt: image_path + sadeleştirilmiş view hierarchy.
    """
    records = []
    for rid in rico_ids:
        if not rico.image_exists(rid):
            continue

        raw_vh = rico.load_hierarchy(rid)
        if raw_vh is None:
            continue

        root = raw_vh.get("activity", {}).get("root")
        if root is None:
            continue

        simplified = simplify_hierarchy(root, max_depth)
        if simplified is None:
            continue

        records.append({
            "rico_id": int(rid),
            "image_path": str(Path(rico_image_dir) / f"{rid}.jpg"),
            "hierarchy": simplified,
        })

    return records
