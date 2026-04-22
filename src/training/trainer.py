"""Ana eğitim fonksiyonu — config'ten modeli alır, LoRA uygular, eğitir."""

from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

from .config_schema import FullConfig
from .dataset import UICriticDataset
from ..models.registry import create_adapter


class VLMDataCollator:
    """Değişken uzunluktaki VLM dizilerini padding ile toplu hale getirir."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        if not batch:
            return {}

        max_len = max(item["input_ids"].shape[0] for item in batch)
        result = {}

        for key in batch[0].keys():
            tensors = [item[key] for item in batch]

            if key == "input_ids":
                padded = [
                    torch.cat([t, t.new_full((max_len - t.shape[0],), self.pad_token_id)])
                    for t in tensors
                ]
                result[key] = torch.stack(padded)
            elif key == "attention_mask":
                padded = [
                    torch.cat([t, t.new_zeros(max_len - t.shape[0])])
                    for t in tensors
                ]
                result[key] = torch.stack(padded)
            elif key == "labels":
                padded = [
                    torch.cat([t, t.new_full((max_len - t.shape[0],), -100)])
                    for t in tensors
                ]
                result[key] = torch.stack(padded)
            else:
                try:
                    result[key] = torch.stack(tensors)
                except Exception:
                    result[key] = tensors

        return result


def train(config: FullConfig) -> None:
    """Verilen config ile tam eğitim döngüsünü çalıştır."""

    # 1. Adapter ve model yükle
    adapter = create_adapter(config.model.name)
    adapter.load_model(
        load_in_4bit=config.model.load_in_4bit,
        torch_dtype=config.model.torch_dtype,
    )

    # 2. LoRA hazırla
    target_modules = (
        adapter.get_lora_target_modules()
        if config.lora.target_modules == "auto"
        else config.lora.target_modules.split(",")
    )

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        target_modules=target_modules,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(adapter.model, lora_config)
    model.print_trainable_parameters()
    adapter.model = model

    # 3. Dataset'leri hazırla
    train_ds = UICriticDataset(
        config.data.train_path,
        config.experiment.task,
        adapter,
        config.data.max_image_size,
    )
    val_ds = UICriticDataset(
        config.data.val_path,
        config.experiment.task,
        adapter,
        config.data.max_image_size,
    )

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir=config.output.dir,
        logging_dir=config.output.logging_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        max_grad_norm=config.training.max_grad_norm,
        max_steps=config.training.max_steps,
        save_total_limit=config.output.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=(config.model.torch_dtype == "bfloat16"),
        fp16=(config.model.torch_dtype == "float16"),
        report_to=["tensorboard"],
        seed=config.experiment.seed,
        remove_unused_columns=False,
    )

    # 5. Collator ve Trainer
    pad_token_id = (
        adapter.processor.tokenizer.pad_token_id
        or adapter.processor.tokenizer.eos_token_id
    )
    collator = VLMDataCollator(pad_token_id=pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    # 6. Eğit ve kaydet
    trainer.train()

    final_path = Path(config.output.dir) / "final"
    trainer.save_model(str(final_path))
    print(f"Model kaydedildi: {final_path}")
