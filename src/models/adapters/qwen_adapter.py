"""Qwen2.5-VL adapter."""

from typing import Dict, List

from PIL import Image

from ..base_adapter import BaseVLMAdapter


class QwenVLAdapter(BaseVLMAdapter):
    """Qwen2.5-VL-3B/7B-Instruct için adapter."""

    def load_model(self, load_in_4bit: bool = True, torch_dtype: str = "bfloat16") -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
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

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.hf_id,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.hf_id)

    def format_prompt(self, instruction: str, image: Image.Image, **kwargs) -> Dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        return inputs

    def get_lora_target_modules(self) -> List[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def generate(self, inputs: Dict, max_new_tokens: int = 512) -> str:
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
