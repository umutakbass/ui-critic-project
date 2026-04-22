"""Splitter modülü testi."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.splitter import split_ids

print("=== split_ids testi ===")

ids = list(range(1000))
train, val, test = split_ids(ids, 0.7, 0.15, 0.15, seed=42)

print(f"  train: {len(train)} (beklenen 700)")
print(f"  val:   {len(val)} (beklenen 150)")
print(f"  test:  {len(test)} (beklenen 150)")

assert len(train) == 700, f"train boyutu hatalı: {len(train)}"
assert len(val) == 150, f"val boyutu hatalı: {len(val)}"
assert len(test) == 150, f"test boyutu hatalı: {len(test)}"
assert set(train).isdisjoint(set(val)), "train-val çakışıyor!"
assert set(train).isdisjoint(set(test)), "train-test çakışıyor!"
assert set(val).isdisjoint(set(test)), "val-test çakışıyor!"

# Tekrarlanabilirlik kontrolü
train2, val2, test2 = split_ids(ids, 0.7, 0.15, 0.15, seed=42)
assert train == train2, "Aynı seed farklı sonuç verdi!"

print("  Tüm testler geçti!")

# Küçük liste kenarlık durumu
small_train, small_val, small_test = split_ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"  10 eleman: train={len(small_train)}, val={len(small_val)}, test={len(small_test)}")
