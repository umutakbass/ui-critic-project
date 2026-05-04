"""
Model2 ile UICrit görsellerinin view hierarchy tahminlerini üretir.
Çıktı: data/processed/pred_vh.json  →  {"<rico_id>": <hierarchy_dict>, ...}

Kullanım: python scripts/generate_predicted_vh.py --config configs/model2_qwen.yaml
"""

import argparse
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="data/processed/pred_vh.json")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--uicrit_csv", default="data/uicrit/uicrit_public.csv")
    parser.add_argument("--rico_dir", default="data/archive/unique_uis")
    return parser.parse_args()


def main():
    args = parse_args()

    from src.training.config_loader import load_config
    config = load_config(args.config)

    checkpoint = args.checkpoint or str(Path(config.output.dir) / "final")
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

    from src.data.uicrit_loader import UICritLoader
    from src.data.rico_loader import RicoLoader
    from src.data.alignment import UICritRicoAligner
    from src.training.prompts import MODEL2_USER_INSTRUCTION
    from PIL import Image
    import torch

    uicrit = UICritLoader(args.uicrit_csv)
    rico = RicoLoader(args.rico_dir)
    aligner = UICritRicoAligner(uicrit, rico)
    all_ids = aligner.get_all_aligned_ids()

    rico_image_dir = str(Path(args.rico_dir) / "combined")
    print(f"Toplam {len(all_ids)} görsel için hierarchy üretiliyor...\n")

    pred_vhs = {}
    failed = 0

    for rico_id in tqdm(all_ids):
        img_path = Path(rico_image_dir) / f"{rico_id}.jpg"
        if not img_path.exists():
            failed += 1
            continue

        img = Image.open(img_path).convert("RGB")
        if max(img.size) > config.data.max_image_size:
            img.thumbnail((config.data.max_image_size, config.data.max_image_size))

        inputs = adapter.format_prompt(MODEL2_USER_INSTRUCTION, img)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output = adapter.generate(inputs, max_new_tokens=args.max_new_tokens)

        try:
            hierarchy = json.loads(output)
        except Exception:
            hierarchy = {"raw_output": output}

        pred_vhs[str(rico_id)] = hierarchy

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pred_vhs, f, ensure_ascii=False)

    print(f"\nTamamlandı: {len(pred_vhs)} tahmin kaydedildi → {output_path}")
    if failed:
        print(f"Atlanılan: {failed} görsel bulunamadı")


if __name__ == "__main__":
    main()
