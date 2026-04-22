"""Pydantic tabanlı eğitim konfigürasyon şeması."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 42
    task: Literal["model1", "model2", "model3"]


class ModelConfig(BaseModel):
    name: str  # MODEL_REGISTRY anahtarı
    load_in_4bit: bool = True
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"


class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "auto"  # "auto" → adapter.get_lora_target_modules()


class DataConfig(BaseModel):
    train_path: str
    val_path: str
    test_path: str
    image_dir: str
    max_image_size: int = 1024
    output_format: Literal["hybrid_json", "plain_text", "structured_json"] = "hybrid_json"


class TrainingConfig(BaseModel):
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 20
    max_grad_norm: float = 1.0
    use_unsloth: bool = True


class OutputConfig(BaseModel):
    dir: str
    logging_dir: str
    save_total_limit: int = 2


class FullConfig(BaseModel):
    experiment: ExperimentConfig
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    training: TrainingConfig
    output: OutputConfig
