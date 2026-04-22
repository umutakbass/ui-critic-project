"""LLaVA-1.6 adapter."""

from typing import Dict, List

from PIL import Image

from ..base_adapter import BaseVLMAdapter


class LLaVAAdapter(BaseVLMAdapter):
    """LLaVA-1.6-Mistral-7B / Vicuna-13B için adapter."""

    def load_model(self, load_in_4bit: bool = True, torch_dtype: str = "bfloat16") -> None:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
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

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.hf_id,
            torch_dtype=dtype,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.processor = LlavaNextProcessor.from_pretrained(self.hf_id)

    def format_prompt(self, instruction: str, image: Image.Image, **kwargs) -> Dict:
        # LLaVA-1.6 Mistral chat format
        prompt = f"[INST] <image>\n{instruction} [/INST]"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return inputs

    def get_lora_target_modules(self) -> List[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def generate(self, inputs: Dict, max_new_tokens: int = 512) -> str:
        input_len = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, input_len:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
