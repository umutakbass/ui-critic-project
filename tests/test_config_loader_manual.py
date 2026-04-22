"""Config loader ve schema testi — tüm 9 YAML dosyasını doğrula."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.config_loader import load_config, apply_overrides
from models.registry import MODEL_REGISTRY

VALID_MODELS = set(MODEL_REGISTRY.keys())
CONFIGS_DIR = Path(__file__).parent.parent / "configs"

print("=== Tüm config dosyaları yükleme testi ===")
yaml_files = sorted(CONFIGS_DIR.glob("model*.yaml"))
assert len(yaml_files) == 9, f"9 config bekleniyor, {len(yaml_files)} bulundu"

for yaml_path in yaml_files:
    cfg = load_config(str(yaml_path))
    assert cfg.model.name in VALID_MODELS, f"{yaml_path.name}: bilinmeyen model {cfg.model.name}"
    assert cfg.experiment.task in ("model1", "model2", "model3")
    assert cfg.lora.r > 0
    assert cfg.training.num_epochs > 0
    print(f"  {yaml_path.name}: task={cfg.experiment.task}, model={cfg.model.name} ✓")

print()
print("=== apply_overrides testi ===")
cfg = load_config(str(CONFIGS_DIR / "model1_qwen.yaml"))
assert cfg.model.name == "qwen2.5-vl-7b"
assert cfg.training.num_epochs == 3

cfg2 = apply_overrides(cfg, ["model.name=gemma-4-4b", "training.num_epochs=5"])
assert cfg2.model.name == "gemma-4-4b", f"Beklenen gemma-4-4b, alınan {cfg2.model.name}"
assert cfg2.training.num_epochs == 5, f"Beklenen 5, alınan {cfg2.training.num_epochs}"
assert cfg.model.name == "qwen2.5-vl-7b"  # orijinal değişmemeli
print(f"  Override sonrası model: {cfg2.model.name}")
print(f"  Override sonrası num_epochs: {cfg2.training.num_epochs}")

print()
print("=== Config şema doğrulama testi ===")
from pydantic import ValidationError
try:
    from training.config_schema import FullConfig
    # Geçersiz task değeri
    FullConfig(**{
        "experiment": {"name": "test", "seed": 42, "task": "model99"},
        "model": {"name": "qwen2.5-vl-7b"},
        "lora": {"r": 16, "alpha": 32},
        "data": {"train_path": "x", "val_path": "x", "test_path": "x", "image_dir": "x"},
        "training": {"num_epochs": 3, "batch_size": 2, "gradient_accumulation_steps": 8,
                     "learning_rate": 2e-4, "warmup_steps": 100, "weight_decay": 0.01,
                     "save_steps": 200, "eval_steps": 200, "logging_steps": 20, "max_grad_norm": 1.0},
        "output": {"dir": "x", "logging_dir": "x"},
    })
    print("  HATA: ValidationError fırlatılmadı!")
except ValidationError as e:
    print(f"  Beklenen ValidationError alındı (geçersiz task) ✓")

print()
print("Tüm config testleri geçti!")
