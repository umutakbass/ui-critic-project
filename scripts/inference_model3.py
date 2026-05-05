"""
Model3 inference: görsel → (model2 ile) hierarchy → (model3 ile) critique
Kullanım:
  python scripts/inference_model3.py \
    --image path/to/image.jpg \
    --config2 configs/model2_qwen.yaml \
    --config3 configs/model3_qwen.yaml
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Görsel dosyası yolu")
    parser.add_argument("--config2", required=True, help="Model2 config yolu")
    parser.add_argument("--config3", required=True, help="Model3 config yolu")
    parser.add_argument("--checkpoint2", default=None, help="Model2 checkpoint (varsayılan: config'deki final)")
    parser.add_argument("--checkpoint3", default=None, help="Model3 checkpoint (varsayılan: config'deki final)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def load_adapter(config, checkpoint):
    from src.models.registry import create_adapter
    from peft import PeftModel

    adapter = create_adapter(config.model.name)
    adapter.load_model(
        load_in_4bit=config.model.load_in_4bit,
        torch_dtype=config.model.torch_dtype,
    )
    adapter.model = PeftModel.from_pretrained(adapter.model, checkpoint)
    adapter.model.eval()
    return adapter


def main():
    args = parse_args()

    from src.training.config_loader import load_config
    from pathlib import Path
    from PIL import Image
    import torch

    # Görseli yükle
    img = Image.open(args.image).convert("RGB")

    # ── ADIM 1: Model2 → hierarchy ──────────────────────────────────────
    print("=== ADIM 1: Model2 ile hierarchy üretiliyor ===")
    config2 = load_config(args.config2)
    ckpt2 = args.checkpoint2 or str(Path(config2.output.dir) / "final")
    print(f"Model2 yükleniyor: {ckpt2}")
    adapter2 = load_adapter(config2, ckpt2)

    img2 = img.copy()
    if max(img2.size) > config2.data.max_image_size:
        img2.thumbnail((config2.data.max_image_size, config2.data.max_image_size))

    from src.training.prompts import MODEL2_USER_INSTRUCTION
    inputs2 = adapter2.format_prompt(MODEL2_USER_INSTRUCTION, img2)
    inputs2 = {k: v.to("cuda") for k, v in inputs2.items()}

    with torch.no_grad():
        hierarchy_raw = adapter2.generate(inputs2, max_new_tokens=args.max_new_tokens)

    try:
        hierarchy = json.loads(hierarchy_raw)
        print("Hierarchy başarıyla parse edildi.")
    except Exception:
        hierarchy = {"raw_output": hierarchy_raw}
        print("Hierarchy JSON parse edilemedi, ham çıktı kullanılıyor.")

    print("\n--- Üretilen Hierarchy ---")
    print(json.dumps(hierarchy, ensure_ascii=False, indent=2)[:500], "...\n")

    # Model2'yi bellekten boşalt
    del adapter2
    torch.cuda.empty_cache()

    # ── ADIM 2: Model3 → critique ────────────────────────────────────────
    print("=== ADIM 2: Model3 ile critique üretiliyor ===")
    config3 = load_config(args.config3)
    ckpt3 = args.checkpoint3 or str(Path(config3.output.dir) / "final")
    print(f"Model3 yükleniyor: {ckpt3}")
    adapter3 = load_adapter(config3, ckpt3)

    img3 = img.copy()
    if max(img3.size) > config3.data.max_image_size:
        img3.thumbnail((config3.data.max_image_size, config3.data.max_image_size))

    from src.training.prompts import MODEL3_USER_INSTRUCTION_TEMPLATE
    instruction3 = MODEL3_USER_INSTRUCTION_TEMPLATE.format(
        hierarchy_json=json.dumps(hierarchy, ensure_ascii=False)
    )
    inputs3 = adapter3.format_prompt(instruction3, img3)
    inputs3 = {k: v.to("cuda") for k, v in inputs3.items()}

    with torch.no_grad():
        critique_raw = adapter3.generate(inputs3, max_new_tokens=args.max_new_tokens)

    print("\n=== MODEL3 ÇIKTISI ===")
    print(critique_raw)

    try:
        parsed = json.loads(critique_raw)
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
