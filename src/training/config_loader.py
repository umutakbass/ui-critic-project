"""YAML config dosyasını yükleyen ve override uygulayan modül."""

from typing import List

import yaml

from .config_schema import FullConfig


def load_config(path: str) -> FullConfig:
    """YAML dosyasını oku ve FullConfig nesnesi döndür."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return FullConfig(**raw)


def apply_overrides(config: FullConfig, overrides: List[str]) -> FullConfig:
    """'key.subkey=value' formatındaki CLI override'larını uygula.

    Örnek:
        apply_overrides(config, ["model.name=gemma-4-4b", "training.num_epochs=5"])
    """
    config_dict = config.model_dump()

    for ovr in overrides:
        key, value = ovr.split("=", 1)
        keys = key.split(".")
        d = config_dict
        for k in keys[:-1]:
            d = d[k]

        # Basit tip çıkarımı
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif "." in value:
            try:
                value = float(value)
            except ValueError:
                pass
        else:
            try:
                value = int(value)
            except ValueError:
                pass

        d[keys[-1]] = value

    return FullConfig(**config_dict)
