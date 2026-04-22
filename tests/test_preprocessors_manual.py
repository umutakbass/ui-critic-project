"""Preprocessing modülleri canlı testi."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.uicrit_loader import UICritLoader
from data.rico_loader import RicoLoader
from data.alignment import UICritRicoAligner
from data.splitter import split_ids
from data.preprocessors.model1_prep import build_model1_records, save_records
from data.preprocessors.model2_prep import build_model2_records, simplify_hierarchy
from data.preprocessors.model3_prep import build_model3_records

UICRIT_CSV = Path(__file__).parent.parent / "data" / "uicrit" / "uicrit_public.csv"
RICO_DIR = Path(__file__).parent.parent / "data" / "archive" / "unique_uis"
COMBINED_DIR = RICO_DIR / "combined"

uicrit = UICritLoader(str(UICRIT_CSV))
rico = RicoLoader(str(RICO_DIR))
aligner = UICritRicoAligner(uicrit, rico)

all_ids = aligner.get_all_aligned_ids()
train_ids, val_ids, test_ids = split_ids(all_ids, 0.7, 0.15, 0.15, seed=42)
sample_ids = all_ids[:10]  # ilk 10 ID ile hızlı test

print("=== Model 1 Preprocessing ===")
m1_records = build_model1_records(uicrit, sample_ids, str(COMBINED_DIR))
print(f"  Kayıt sayısı: {len(m1_records)} (beklenen: {len(sample_ids)})")
assert len(m1_records) == len(sample_ids)
rec = m1_records[0]
print(f"  İlk kayıt rico_id: {rec['rico_id']}")
print(f"  image_path var mı: {Path(rec['image_path']).exists()}")
print(f"  critiques sayısı: {len(rec['critiques'])}")
print(f"  İlk critique: {rec['critiques'][0]}")
assert len(rec["critiques"]) > 0
assert "comment" in rec["critiques"][0]
assert "bounding_box" in rec["critiques"][0]

print()
print("=== save_records testi ===")
with tempfile.TemporaryDirectory() as tmpdir:
    out_path = Path(tmpdir) / "model1_test.json"
    save_records(m1_records, str(out_path))
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert len(loaded) == len(m1_records)
    print(f"  JSON dosyası yazıldı ve okundu: {len(loaded)} kayıt")

print()
print("=== simplify_hierarchy testi ===")
raw_vh = rico.load_hierarchy(sample_ids[0])
root = raw_vh["activity"]["root"]
simplified = simplify_hierarchy(root, max_depth=10)
assert simplified is not None
assert "type" in simplified
assert "bounds" in simplified
assert "children" in simplified
print(f"  Kök tip: {simplified['type']}")
print(f"  Kök bounds: {simplified['bounds']}")
print(f"  Doğrudan çocuk sayısı: {len(simplified['children'])}")

print()
print("=== Model 2 Preprocessing ===")
m2_records = build_model2_records(rico, sample_ids, str(COMBINED_DIR))
print(f"  Kayıt sayısı: {len(m2_records)}")
assert len(m2_records) > 0
rec2 = m2_records[0]
assert "hierarchy" in rec2
assert rec2["hierarchy"] is not None
print(f"  İlk kayıt hierarchy type: {rec2['hierarchy']['type']}")

print()
print("=== Model 3 Preprocessing ===")
# Sahte predicted_hierarchies dosyası oluştur
fake_preds = {str(rid): m2_records[i]["hierarchy"]
              for i, rid in enumerate(sample_ids[:len(m2_records)])}
with tempfile.TemporaryDirectory() as tmpdir:
    pred_path = Path(tmpdir) / "pred_vh.json"
    pred_path.write_text(json.dumps(fake_preds))
    m3_records = build_model3_records(uicrit, sample_ids, str(COMBINED_DIR), str(pred_path))
    print(f"  Kayıt sayısı: {len(m3_records)}")
    assert len(m3_records) > 0
    assert "predicted_hierarchy" in m3_records[0]
    assert "critiques" in m3_records[0]
    print(f"  İlk kayıt predicted_hierarchy type: {m3_records[0]['predicted_hierarchy']['type']}")

print()
print("Tüm preprocessing testleri geçti!")
