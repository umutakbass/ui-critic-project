"""Eğitim CLI giriş noktası.

Kullanım:
    python scripts/train.py --config configs/model1_qwen.yaml
    python scripts/train.py --config configs/model1_qwen.yaml --override model.name=gemma-4-4b
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config_loader import load_config, apply_overrides
from src.training.trainer import train


# Drive file ID'leri — dosya yoksa otomatik indir
DRIVE_FILES = {
    "data/processed/model2_train_10k.json": "1h4lc_IPzojvQfe-e6H5bOThCc30Ceqly",
    "data/processed/model2_val_10k.json": "1i7VmS8Ksglx67vVx4_DLZj4-6XYoOigo",
    "data/processed/model2_test_10k.json": "12mI2wfNgvLum0m77xsVcc59ZFenmODWR",
}


def ensure_data_files(paths: list[str]) -> None:
    """Listedeki yollar yoksa Drive'dan gdown ile indir."""
    missing = [p for p in paths if not Path(p).exists()]
    if not missing:
        return
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown yükleniyor...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)

    for p in missing:
        if p not in DRIVE_FILES:
            raise FileNotFoundError(f"{p} yok ve DRIVE_FILES'ta tanımlı değil")
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        print(f"İndiriliyor: {p}")
        subprocess.run(["gdown", DRIVE_FILES[p], "-O", p], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM eğitimini başlat")
    parser.add_argument("--config", required=True, help="YAML config dosyası yolu")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="'key.subkey=value' formatında config override'ları",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    print(f"Deney: {config.experiment.name}")
    print(f"Model: {config.model.name}  |  Task: {config.experiment.task}")

    # Data dosyaları yoksa Drive'dan indir
    ensure_data_files([config.data.train_path, config.data.val_path, config.data.test_path])

    train(config)


if __name__ == "__main__":
    main()
