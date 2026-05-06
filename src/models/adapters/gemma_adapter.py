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
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=dtype,
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
        """Qwen gibi diğer modeller bu metodu kullanır.
        Gemma 4 için get_collate_fn() kullanılır, bu metod çağrılmaz.
        """
        raise NotImplementedError(
            "Gemma 4 için prepare_training_inputs kullanılmaz. "
            "Trainer get_collate_fn() ile çalışır."
        )

    def get_collate_fn(self, max_length: int = 2048):
        """Resmi Google Gemma 4 fine-tuning yaklaşımı.
        Kaynaklar: https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
        """
        processor = self.processor

        def collate_fn(examples):
            texts = []
            images = []

            for ex in examples:
                msgs = ex["messages"]

                # Görseli messages içinden çıkar
                img = None
                for msg in msgs:
                    for content in msg.get("content", []):
                        if isinstance(content, dict) and content.get("type") == "image":
                            img = content["image"]
                            break

                images.append([img] if img is not None else [])

                # Chat template ile metin oluştur (tokenize=False)
                text = processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)

            # Processor ile tokenize et
            batch = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # Label masking — resmi Google yaklaşımı
            labels = batch["input_ids"].clone()
            tokenizer = processor.tokenizer

            # Pad tokenini maskele
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100

            # Görsel özel tokenlarını maskele (boi, image_token, eoi)
            for attr in ("boi_token_id", "image_token_id", "eoi_token_id"):
                token_id = getattr(tokenizer, attr, None)
                if token_id is not None:
                    labels[labels == token_id] = -100

            # mm_token_type_ids ile tüm görsel tokenları maskele (ek güvence)
            if "mm_token_type_ids" in batch:
                labels[batch["mm_token_type_ids"] == 1] = -100

            batch["labels"] = labels
            return batch

        return collate_fn

    def get_lora_target_modules(self) -> str:
        # Gemma 4'ün vision encoder'ı Gemma4ClippableLinear kullanıyor (PEFT desteklemiyor).
        # Regex ile sadece language_model katmanlarını hedef al.
        return r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

    def generate(self, inputs: Dict, max_new_tokens: int = 512) -> str:
        input_len = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
