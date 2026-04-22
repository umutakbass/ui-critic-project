"""Veri hazırlama script'i: ham veriyi eğitim formatına dönüştürür.

Kullanım:
    python scripts/prepare_data.py --task model1
    python scripts/prepare_data.py --task model2
    python scripts/prepare_data.py --task model3 --predicted_vh_path data/processed/pred_vh.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.uicrit_loader import UICritLoader
from src.data.rico_loader import RicoLoader
from src.data.alignment import UICritRicoAligner
from src.data.splitter import split_ids
from src.data.preprocessors.model1_prep import build_model1_records, save_records
from src.data.preprocessors.model2_prep import build_model2_records
from src.data.preprocessors.model3_prep import build_model3_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Eğitim verisi hazırla")
    parser.add_argument("--task", required=True, choices=["model1", "model2", "model3"])
    parser.add_argument("--uicrit_csv", default="data/uicrit/uicrit_public.csv")
    parser.add_argument(
        "--rico_dir",
        default="data/archive/unique_uis",
        help="RicoLoader'ın beklediği üst klasör (combined/ burada olmalı)",
    )
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--predicted_vh_path", default=None, help="Model 3 için gerekli")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rico_image_dir = str(Path(args.rico_dir) / "combined")

    uicrit = UICritLoader(args.uicrit_csv)
    rico = RicoLoader(args.rico_dir)

    if args.task == "model1":
        aligner = UICritRicoAligner(uicrit, rico)
        all_ids = aligner.get_all_aligned_ids()
        print(f"Eşleşen ID sayısı: {len(all_ids)}")
        train, val, test = split_ids(all_ids, 0.7, 0.15, 0.15, seed=args.seed)
        for split_name, ids in [("train", train), ("val", val), ("test", test)]:
            records = build_model1_records(uicrit, ids, rico_image_dir)
            out = f"{args.output_dir}/model1_{split_name}.json"
            save_records(records, out)
            print(f"model1_{split_name}: {len(records)} kayıt → {out}")

    elif args.task == "model2":
        all_jpg = list(Path(args.rico_dir, "combined").glob("*.jpg"))
        all_rico_ids = [int(p.stem) for p in all_jpg]
        print(f"RICO toplam görsel: {len(all_rico_ids)}")
        train, val, test = split_ids(all_rico_ids, 0.7, 0.15, 0.15, seed=args.seed)
        for split_name, ids in [("train", train), ("val", val), ("test", test)]:
            records = build_model2_records(rico, ids, rico_image_dir)
            out = f"{args.output_dir}/model2_{split_name}.json"
            save_records(records, out)
            print(f"model2_{split_name}: {len(records)} kayıt → {out}")

    elif args.task == "model3":
        if not args.predicted_vh_path:
            parser.error("Model 3 için --predicted_vh_path gerekli")
        aligner = UICritRicoAligner(uicrit, rico)
        all_ids = aligner.get_all_aligned_ids()
        train, val, test = split_ids(all_ids, 0.7, 0.15, 0.15, seed=args.seed)
        for split_name, ids in [("train", train), ("val", val), ("test", test)]:
            records = build_model3_records(uicrit, ids, rico_image_dir, args.predicted_vh_path)
            out = f"{args.output_dir}/model3_{split_name}.json"
            save_records(records, out)
            print(f"model3_{split_name}: {len(records)} kayıt → {out}")


if __name__ == "__main__":
    main()
