"""
Model değerlendirme scripti — test seti üzerinde BLEU, ROUGE, BERTScore hesaplar.
Kullanım: python scripts/evaluate_model.py --config configs/model1_qwen.yaml
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config YAML dosyası")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint yolu (varsayılan: config'deki output dir/final)")
    parser.add_argument("--max_samples", type=int, default=50, help="Test edilecek örnek sayısı")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def load_config(config_path):
    from src.training.config_loader import load_config
    return load_config(config_path)


def compute_metrics(predictions, references):
    results = {}

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge1, rouge2, rougeL = [], [], []
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1.append(scores["rouge1"].fmeasure)
            rouge2.append(scores["rouge2"].fmeasure)
            rougeL.append(scores["rougeL"].fmeasure)
        results["rouge1"] = sum(rouge1) / len(rouge1)
        results["rouge2"] = sum(rouge2) / len(rouge2)
        results["rougeL"] = sum(rougeL) / len(rougeL)
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")
        print(f"ROUGE-L: {results['rougeL']:.4f}")
    except ImportError:
        print("rouge_score yüklü değil: pip install rouge-score")

    # BLEU
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        import nltk
        nltk.download("punkt", quiet=True)
        refs = [[ref.split()] for ref in references]
        hyps = [pred.split() for pred in predictions]
        smoothie = SmoothingFunction().method4
        results["bleu"] = corpus_bleu(refs, hyps, smoothing_function=smoothie)
        print(f"BLEU: {results['bleu']:.4f}")
    except ImportError:
        print("nltk yüklü değil: pip install nltk")

    # JSON parse rate
    valid_json = 0
    for pred in predictions:
        try:
            json.loads(pred)
            valid_json += 1
        except Exception:
            pass
    results["json_parse_rate"] = valid_json / len(predictions)
    print(f"JSON Parse Rate: {results['json_parse_rate']:.2%} ({valid_json}/{len(predictions)})")

    return results


def main():
    args = parse_args()
    config = load_config(args.config)

    checkpoint = args.checkpoint or str(Path(config.output.dir) / "final")
    print(f"Checkpoint: {checkpoint}")

    # Model yükle
    from src.models.registry import create_adapter
    adapter = create_adapter(config.model.name)
    adapter.load_model(
        load_in_4bit=config.model.load_in_4bit,
        torch_dtype=config.model.torch_dtype,
    )

    # LoRA ağırlıklarını yükle
    from peft import PeftModel
    adapter.model = PeftModel.from_pretrained(adapter.model, checkpoint)
    adapter.model.eval()

    # Test verisi
    from src.training.dataset import UICriticDataset
    from src.training.prompts import MODEL1_USER_INSTRUCTION, MODEL2_USER_INSTRUCTION

    with open(config.data.test_path) as f:
        records = json.load(f)

    records = records[:args.max_samples]
    print(f"\n{len(records)} örnek test ediliyor...\n")

    predictions = []
    references = []

    instruction_map = {
        "model1": MODEL1_USER_INSTRUCTION,
        "model2": MODEL2_USER_INSTRUCTION,
    }
    instruction = instruction_map[config.experiment.task]

    from PIL import Image
    for i, rec in enumerate(tqdm(records)):
        img = Image.open(rec["image_path"]).convert("RGB")
        if max(img.size) > config.data.max_image_size:
            img.thumbnail((config.data.max_image_size, config.data.max_image_size))

        inputs = adapter.format_prompt(instruction, img)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            pred = adapter.generate(inputs, max_new_tokens=args.max_new_tokens)

        if config.experiment.task == "model1":
            ref = json.dumps(
                {"critiques": rec["critiques"], "overall_feedback": rec.get("overall_feedback", "")},
                ensure_ascii=False,
            )
        else:
            ref = json.dumps(rec.get("hierarchy", {}), ensure_ascii=False)

        predictions.append(pred)
        references.append(ref)

    print("\n=== SONUÇLAR ===")
    compute_metrics(predictions, references)

    # Örnek tahminleri kaydet
    output_path = Path(config.output.dir) / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"prediction": p, "reference": r} for p, r in zip(predictions[:5], references[:5])],
            f, ensure_ascii=False, indent=2
        )
    print(f"\nÖrnek tahminler: {output_path}")


if __name__ == "__main__":
    main()
