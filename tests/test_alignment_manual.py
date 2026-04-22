"""Alignment modülü canlı testi."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.uicrit_loader import UICritLoader
from data.rico_loader import RicoLoader
from data.alignment import UICritRicoAligner

UICRIT_CSV = Path(__file__).parent.parent / "data" / "uicrit" / "uicrit_public.csv"
RICO_DIR = Path(__file__).parent.parent / "data" / "archive" / "unique_uis"

uicrit = UICritLoader(str(UICRIT_CSV))
rico = RicoLoader(str(RICO_DIR))
aligner = UICritRicoAligner(uicrit, rico)

print("=== Coverage Raporu ===")
report = aligner.coverage_report()
for k, v in report.items():
    print(f"  {k}: {v}")

print()
print("=== rico_id=15 aligned record testi ===")
rec = aligner.get_aligned_record(15)
if rec:
    print(f"  rico_id: {rec['rico_id']}")
    print(f"  image: {rec['image'].size if rec['image'] else None}")
    print(f"  hierarchy keys: {list(rec['hierarchy'].keys()) if rec['hierarchy'] else None}")
    print(f"  critiques_records count: {len(rec['critiques_records'])}")
    first = rec["critiques_records"][0]
    print(f"  ilk kayıt comments sayısı: {len(first['comments'])}")
    print(f"  ilk yorum: {first['comments'][0]}")
else:
    print("  HATA: None döndü!")

print()
print("=== Tüm aligned ID'ler ===")
aligned = aligner.get_all_aligned_ids()
print(f"  Toplam: {len(aligned)}")
print(f"  İlk 5: {aligned[:5]}")
