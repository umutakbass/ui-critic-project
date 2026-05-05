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

    def prepare_training_inputs(self, instruction: str, target: str, image: Image.Image, max_length: int = 2048) -> Dict:
        """Eğitim için tam konuşma (user + assistant) girdisini hazırla.
        Sadece asistan cevabı tokenları loss'a katılır (label masking).
        Alt sınıflar model'e özgü format için bu metodu override edebilir.
        """
        import torch

        # Tam konuşma
        messages_full = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
            {"role": "assistant", "content": target},
        ]
        text_full = self.processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        # Sadece user bölümü (asistan başlangıcını bulmak için)
        messages_user = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
        ]
        text_user = self.processor.apply_chat_template(messages_user, tokenize=False, add_generation_prompt=True)
        inputs_user = self.processor(
            text=[text_user],
            images=[image],
            return_tensors="pt",
        )
        user_len = inputs_user["input_ids"].shape[1]

        # Label masking
        labels = inputs["input_ids"].clone()
        labels[0, :user_len] = -100
        inputs["labels"] = labels

        return inputs
