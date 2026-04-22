"""Tüm VLM adapter'larının uyması gereken soyut temel sınıf."""

from abc import ABC, abstractmethod
from typing import Dict, List

from PIL import Image


class BaseVLMAdapter(ABC):
    """VLM adapter arayüzü.

    Config'teki model.name değeri değiştiğinde bu arayüzü uygulayan
    adapter otomatik seçilir; eğitim ve inference kodu hiç değişmez.
    """

    def __init__(self, model_config: Dict):
        self.config = model_config
        self.hf_id: str = model_config["hf_id"]
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self, load_in_4bit: bool = True, torch_dtype: str = "bfloat16") -> None:
        """Modeli ve processor'ı HuggingFace'ten yükle."""

    @abstractmethod
    def format_prompt(self, instruction: str, image: Image.Image, **kwargs) -> Dict:
        """Model'e özgü giriş formatına dönüştür (eğitim + inference için ortak)."""

    @abstractmethod
    def get_lora_target_modules(self) -> List[str]:
        """LoRA'nın uygulanacağı modül isimlerini döndür."""

    @abstractmethod
    def generate(self, inputs: Dict, max_new_tokens: int = 512) -> str:
        """Verilen girdilerden metin üret (inference)."""
