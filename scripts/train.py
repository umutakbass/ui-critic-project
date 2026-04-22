"""Eğitim CLI giriş noktası.

Kullanım:
    python scripts/train.py --config configs/model1_qwen.yaml
    python scripts/train.py --config configs/model1_qwen.yaml --override model.name=gemma-4-4b
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config_loader import load_config, apply_overrides
from src.training.trainer import train


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
    train(config)


if __name__ == "__main__":
    main()
