"""Model registry ve adapter syntax testi (model yüklenmez)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.registry import MODEL_REGISTRY, get_model_config, create_adapter
from models.base_adapter import BaseVLMAdapter

print("=== Registry içerik testi ===")
for name in ["qwen2.5-vl-7b", "gemma-4-4b", "llava-1.6-7b"]:
    assert name in MODEL_REGISTRY, f"{name} registry'de yok!"
    print(f"  {name}: {MODEL_REGISTRY[name]['hf_id']}")

print()
print("=== get_model_config testi ===")
cfg = get_model_config("qwen2.5-vl-7b")
assert cfg["hf_id"] == "Qwen/Qwen2.5-VL-7B-Instruct"
assert cfg["family"] == "qwen"
print(f"  qwen2.5-vl-7b config: {cfg}")

print()
print("=== Bilinmeyen model ValueError testi ===")
try:
    get_model_config("bilinmeyen-model")
    print("  HATA: ValueError fırlatılmadı!")
except ValueError as e:
    print(f"  Beklenen hata alındı: {e}")

print()
print("=== Adapter sınıf import testi ===")
from models.adapters.qwen_adapter import QwenVLAdapter
from models.adapters.gemma_adapter import Gemma4Adapter
from models.adapters.llava_adapter import LLaVAAdapter

for cls in [QwenVLAdapter, Gemma4Adapter, LLaVAAdapter]:
    assert issubclass(cls, BaseVLMAdapter), f"{cls.__name__} BaseVLMAdapter'dan türemiyor!"
    print(f"  {cls.__name__}: BaseVLMAdapter alt sınıfı ✓")

print()
print("=== create_adapter örnek oluşturma ===")
adapter = create_adapter("qwen2.5-vl-7b")
print(f"  adapter: {type(adapter).__name__}")
print(f"  hf_id: {adapter.hf_id}")
assert adapter.model is None, "Modeli henüz yüklememeliydi"
lora_modules = adapter.get_lora_target_modules()
print(f"  lora_modules: {lora_modules}")

print()
print("Tüm registry testleri geçti!")
