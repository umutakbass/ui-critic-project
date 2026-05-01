"""
Processed JSON'lardaki 1000 kaydın görsellerini ve JSON'ları zip'ler.
Kullanım: python scripts/create_mini_dataset.py
Çıktı: data/mini_dataset.zip
"""

import json
import shutil
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_ZIP = DATA_DIR / "mini_dataset.zip"

splits = ["model1_train.json", "model1_val.json", "model1_test.json"]

image_paths = []
for split in splits:
    path = PROCESSED_DIR / split
    if not path.exists():
        print(f"Bulunamadı: {path}")
        continue
    records = json.loads(path.read_text())
    for rec in records:
        image_paths.append(rec["image_path"])
    print(f"{split}: {len(records)} kayıt")

image_paths = list(set(image_paths))
print(f"\nToplam benzersiz görsel: {len(image_paths)}")

print("ZIP oluşturuluyor...")
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    # Processed JSON'lar
    for split in splits:
        p = PROCESSED_DIR / split
        if p.exists():
            zf.write(p, f"processed/{split}")
            print(f"  + processed/{split}")

    # Görseller
    for i, img_path in enumerate(image_paths):
        p = Path(img_path)
        if p.exists():
            zf.write(p, f"images/{p.name}")
        if (i + 1) % 100 == 0:
            print(f"  Görseller: {i+1}/{len(image_paths)}")

print(f"\nTamamlandı: {OUTPUT_ZIP}")
print(f"Boyut: {OUTPUT_ZIP.stat().st_size / 1024 / 1024:.1f} MB")
