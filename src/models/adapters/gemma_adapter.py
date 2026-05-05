"""Gemma 4 adapter."""

from typing import Dict, List

from PIL import Image

from ..base_adapter import BaseVLMAdapter


class Gemma4Adapter(BaseVLMAdapter):
    """Gemma-4-E4B/31B-it için adapter."""

    def load_model(self, load_in_4bit: bool = True, torch_dtype: str = "bfloat16") -> None:
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        import torch

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[torch_dtype]

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map="cuda",
        )
        self.processor = AutoProcessor.from_pretrained(self.hf_id)

    def format_prompt(self, instruction: str, image: Image.Image, **kwargs) -> Dict:
        """Inference için prompt formatla."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs

    def prepare_training_inputs(self, instruction: str, target: str, image: Image.Image, max_length: int = 2048) -> Dict:
        """Eğitim için tam konuşma girdisini hazırla.
        Sadece asistan cevabı tokenları loss'a katılır (label masking).
        Gemma 4 chat template: <start_of_turn>model\n ile başlayan asistan bölümünü bulur.
        """
        import torch

        messages_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages_full,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            processor_kwargs={"truncation": True, "max_length": max_length},
        )

        import torch

        full_ids = inputs["input_ids"][0]  # [seq_len]

        # Asistan cevabını ayrıca tokenize et (kaç text token tuttuğunu öğren)
        target_ids = self.processor.tokenizer(
            target, add_special_tokens=False
        )["input_ids"]
        target_len = len(target_ids)

        # Önce her şeyi maskele
        labels = torch.full_like(full_ids, -100)

        if "mm_token_type_ids" in inputs:
            # mm_token_type_ids: 0 = metin tokeni, diğer = görsel tokeni
            mm_type = inputs["mm_token_type_ids"][0]  # [seq_len]
            text_positions = (mm_type == 0).nonzero(as_tuple=True)[0]

            # Text tokenların sadece son target_len tanesini unmask et
            # (asistan cevabı her zaman sondaki text tokenlar)
            if len(text_positions) >= target_len:
                assistant_positions = text_positions[-target_len:]
                labels[assistant_positions] = full_ids[assistant_positions]
        else:
            # Fallback: sondan target_len+2 token unmask et
            full_len = len(full_ids)
            mask_until = max(0, full_len - target_len - 2)
            labels[mask_until:] = full_ids[mask_until:]

        inputs["labels"] = labels.unsqueeze(0)  # [1, seq_len]
        return inputs

    def get_lora_target_modules(self) -> str:
        # Gemma 4'ün vision encoder'ı Gemma4ClippableLinear kullanıyor (PEFT desteklemiyor).
        # Regex ile sadece language_model katmanlarını hedef al.
        return r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

    def generate(self, inputs: Dict, max_new_tokens: int = 512) -> str:
        input_len = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
