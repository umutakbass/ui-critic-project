"""RICO loader canlı test — rico_id=15 ile."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.rico_loader import RicoLoader

RICO_DIR = Path(__file__).parent.parent / "data" / "archive" / "unique_uis"

loader = RicoLoader(RICO_DIR)

print(f"RICO dir: {loader.rico_dir}")
print(f"Combined dir: {loader.combined_dir}")
print(f"Combined exists: {loader.combined_dir.exists()}")
print()

# rico_id=15 ile test (UICrit'teki ilk ID)
test_id = 15
print(f"=== rico_id={test_id} testi ===")
print(f"image_exists({test_id}): {loader.image_exists(test_id)}")
print(f"hierarchy_exists({test_id}): {loader.hierarchy_exists(test_id)}")

img = loader.load_image(test_id)
if img:
    print(f"Gorsel yuklendi: mode={img.mode}, size={img.size}")
else:
    print("HATA: Gorsel yuklenemedi!")

vh = loader.load_hierarchy(test_id)
if vh:
    keys = list(vh.keys()) if isinstance(vh, dict) else type(vh)
    print(f"View hierarchy yuklendi: ust seviye anahtarlar={keys}")
else:
    print("HATA: View hierarchy yuklenemedi!")

# Birkac ek ID ile saglama
print()
print("=== Ek ID kontrolleri ===")
for rid in [0, 1, 100, 1000, 99999]:
    img_ok = loader.image_exists(rid)
    vh_ok = loader.hierarchy_exists(rid)
    print(f"  rico_id={rid}: image={img_ok}, hierarchy={vh_ok}")
