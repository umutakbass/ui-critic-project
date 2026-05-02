"""
Tek bir görsel için kritik üretir.
Kullanım: python scripts/inference.py --config configs/model1_qwen.yaml --image path/to/image.jpg
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True, help="Görsel dosyası yolu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    from src.training.config_loader import load_config
    config = load_config(args.config)

    checkpoint = args.checkpoint or str(__import__("pathlib").Path(config.output.dir) / "final")
    print(f"Model yükleniyor: {checkpoint}\n")

    from src.models.registry import create_adapter
    adapter = create_adapter(config.model.name)
    adapter.load_model(
        load_in_4bit=config.model.load_in_4bit,
        torch_dtype=config.model.torch_dtype,
    )

    from peft import PeftModel
    adapter.model = PeftModel.from_pretrained(adapter.model, checkpoint)
    adapter.model.eval()

    from PIL import Image
    import torch

    img = Image.open(args.image).convert("RGB")
    if max(img.size) > config.data.max_image_size:
        img.thumbnail((config.data.max_image_size, config.data.max_image_size))

    from src.training.prompts import MODEL1_USER_INSTRUCTION
    inputs = adapter.format_prompt(MODEL1_USER_INSTRUCTION, img)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    print("Kritik üretiliyor...\n")
    with torch.no_grad():
        output = adapter.generate(inputs, max_new_tokens=args.max_new_tokens)

    print("=== MODEL ÇIKTISI ===")
    print(output)

    # JSON parse dene
    try:
        parsed = json.loads(output)
        print("\n=== PARSE EDİLMİŞ KRİTİK ===")
        if "critiques" in parsed:
            for i, c in enumerate(parsed["critiques"], 1):
                print(f"\n{i}. {c}")
        if "overall_feedback" in parsed:
            print(f"\nGenel: {parsed['overall_feedback']}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
