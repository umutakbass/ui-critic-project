"""Model registry: desteklenen VLM'lerin merkezi kaydı."""

from typing import Dict

from .base_adapter import BaseVLMAdapter

MODEL_REGISTRY: Dict[str, Dict] = {
    "qwen2.5-vl-3b": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "adapter_class": "QwenVLAdapter",
        "family": "qwen",
    },
    "qwen2.5-vl-7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "adapter_class": "QwenVLAdapter",
        "family": "qwen",
    },
    "gemma-4-e4b": {
        "hf_id": "google/gemma-4-E4B-it",
        "adapter_class": "Gemma4Adapter",
        "family": "gemma",
    },
    "gemma-4-31b": {
        "hf_id": "google/gemma-4-31b-it",
        "adapter_class": "Gemma4Adapter",
        "family": "gemma",
    },
    "gemma-4-4b": {
        "hf_id": "google/gemma-4-4b-it",
        "adapter_class": "Gemma4Adapter",
        "family": "gemma",
    },
    "gemma-4-12b": {
        "hf_id": "google/gemma-4-12b-it",
        "adapter_class": "Gemma4Adapter",
        "family": "gemma",
    },
    "llava-1.6-7b": {
        "hf_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "adapter_class": "LLaVAAdapter",
        "family": "llava",
    },
    "llava-1.6-13b": {
        "hf_id": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "adapter_class": "LLaVAAdapter",
        "family": "llava",
    },
}


def get_model_config(name: str) -> Dict:
    """Registry'den model konfigürasyonunu döndür."""
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' bilinmiyor. Desteklenen: {available}")
    return MODEL_REGISTRY[name]


def create_adapter(name: str) -> BaseVLMAdapter:
    """Model adına göre doğru adapter örneğini oluştur ve döndür."""
    config = get_model_config(name)
    from .adapters import qwen_adapter, gemma_adapter, llava_adapter

    adapter_classes = {
        "QwenVLAdapter": qwen_adapter.QwenVLAdapter,
        "Gemma4Adapter": gemma_adapter.Gemma4Adapter,
        "LLaVAAdapter": llava_adapter.LLaVAAdapter,
    }
    adapter_cls = adapter_classes[config["adapter_class"]]
    return adapter_cls(config)
