# UI Design Critique Project

Mobil UI ekran görüntülerinden otomatik tasarım kritiği (design critique) üretmek için 3 farklı model eğiten karşılaştırmalı çalışma. Proje mimarisi **model-agnostik** tasarlanmıştır: config dosyasındaki tek bir satırı değiştirerek Qwen2.5-VL, Gemma 4 veya LLaVA arasında geçiş yapabilirsiniz.

---

## İçindekiler

1. [Projeye Genel Bakış](#1-projeye-genel-bakış)
2. [Araştırma Sorusu](#2-araştırma-sorusu)
3. [3 Model Mimarisi](#3-3-model-mimarisi)
4. [Projenin Teknik Yapısı](#4-projenin-teknik-yapısı)
5. [Dizin Yapısı](#5-dizin-yapısı)
6. [Cloud GPU Kurulumu](#6-cloud-gpu-kurulumu)
7. [Ortam Kurulumu](#7-ortam-kurulumu)
8. [Datasetlerin Hazırlanması](#8-datasetlerin-hazırlanması)
9. [Config Sistemi: Model Nasıl Değiştirilir?](#9-config-sistemi-model-nasıl-değiştirilir)
10. [Desteklenen Modeller](#10-desteklenen-modeller)
11. [Model 1 Eğitimi: UI → Kritik](#11-model-1-eğitimi-ui--kritik)
12. [Model 2 Eğitimi: UI → View Hierarchy](#12-model-2-eğitimi-ui--view-hierarchy)
13. [Model 3 Eğitimi: UI + Hiyerarşi → Kritik](#13-model-3-eğitimi-ui--hiyerarşi--kritik)
14. [Değerlendirme (Evaluation)](#14-değerlendirme-evaluation)
15. [Inference (Eğitilmiş Modeli Kullanma)](#15-inference-eğitilmiş-modeli-kullanma)
16. [Ollama'ya Aktarma (İsteğe Bağlı)](#16-ollamaya-aktarma-isteğe-bağlı)
17. [Sık Karşılaşılan Hatalar](#17-sık-karşılaşılan-hatalar)
18. [Proje Adım Adım Takibi](#18-proje-adım-adım-takibi)
19. [Kaynakça](#19-kaynakça)

---

## 1. Projeye Genel Bakış

### 1.1. Amaç

Mobil uygulama UI'larının ekran görüntülerinden doğal dilde **tasarım kritiği** üreten yapay zekâ modelleri eğitmek. Kritik = "bu UI'da neler iyi, neler kötü, nasıl iyileştirilmeli" şeklindeki profesyonel tasarımcı yorumları.

### 1.2. Kullanılacak Datasetler

| Dataset | Boyut | Ne İçerir? | Ne İçin Kullanılacak? |
|---|---|---|---|
| **UICrit** | 983 UI, 11.344 kritik | UI görseli + tasarımcı kritikleri + bounding box'lar | Model 1 ve Model 3 eğitimi |
| **RICO** | 66.000+ UI | UI görseli + view hierarchy (element ağacı) | Model 2 eğitimi |
| **CLAY** (opsiyonel) | 59.555 UI | RICO'nun temizlenmiş hiyerarşileri | Model 2 eğitim kalitesi için |

UICrit'teki UI'lar zaten RICO'dan geldiği için `rico_id` alanı ile her iki dataset eşleştirilebilir.

### 1.3. Hedef Çıktı

Her model için hem **JSON formatında yapılandırılmış** hem de **doğal dilde okunabilir** karma çıktı:

```json
{
  "critiques": [
    {
      "comment": "Ana aksiyon butonu ekranın alt kenarında ve parmakla ulaşılması zor",
      "bounding_box": [120, 1800, 200, 80],
      "severity": "medium",
      "category": "accessibility"
    }
  ],
  "overall_feedback": "Bu ekranda erişilebilirlik iyileştirmeleri gerekli. Özellikle buton boyutları Material Design rehberindeki minimum 48dp değerinin altında..."
}
```

---

## 2. Araştırma Sorusu

> **"Tahmin edilmiş view hierarchy (yapısal bilgi) modele verildiğinde, UI tasarım kritiği kalitesi ne kadar artar?"**

Bu soruya Model 1 ile Model 3 arasındaki performans farkı cevap verecek.

**Alt sorular:**
- Model 2'nin hiyerarşi çıkarım hatası, Model 3 çıktı kalitesini ne kadar bozuyor?
- Hangi base model (Qwen, Gemma, LLaVA) bu görevde en iyi?
- Görsel grounding (kritiğin doğru bölgeye işaret etmesi) hiyerarşi ile iyileşiyor mu?

---

## 3. 3 Model Mimarisi

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                   │
│   MODEL 1: Baseline                                               │
│   ┌─────────────┐                                                 │
│   │  UI Image   │ ────────────────▶ [VLM fine-tune] ──▶ Kritik   │
│   └─────────────┘                                                 │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   MODEL 2: View Hierarchy Extraction                              │
│   ┌─────────────┐                                                 │
│   │  UI Image   │ ────────────────▶ [VLM fine-tune] ──▶ VH JSON  │
│   └─────────────┘                                                 │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   MODEL 3: Hierarchy-Aware Critic                                 │
│   ┌─────────────┐                                                 │
│   │  UI Image   │ ──┐                                             │
│   └─────────────┘   │                                             │
│                     ├──▶ [VLM fine-tune] ──▶ Kritik              │
│   ┌─────────────┐   │                                             │
│   │ Predicted   │ ──┘                                             │
│   │ VH (M2'den) │                                                 │
│   └─────────────┘                                                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

Her üç model de **aynı 3 base VLM** (Qwen2.5-VL, Gemma 4, LLaVA-1.6) ile ayrı ayrı eğitilebilir. Yani toplamda **9 eğitim deneyi** yapılabilir (3 model × 3 VLM). Config sistemi bunu kolayca yönetir.

---

## 4. Projenin Teknik Yapısı

### 4.1. Kullanılan Framework'ler

| Framework | Rolü |
|---|---|
| **Hugging Face Transformers + PEFT** | Ana motor: model yükleme, LoRA fine-tuning, inference |
| **Unsloth** | Hızlandırıcı: fine-tuning'i 2-5x hızlandırır, VRAM'i %50 düşürür |
| **Ollama** | Günlük inference için kolay arayüz (eğitim sonrası) |
| **PyTorch** | Alt katman (HF'nin altında) |

### 4.2. Adapter Pattern — Model Değiştirilebilirliğin Sırrı

Her VLM'in kendine has prompt formatı var:

- **Qwen2.5-VL** ister: `<|im_start|>user\n<image>\nPrompt<|im_end|>`
- **Gemma 4** ister: `<start_of_turn>user\n<image>Prompt<end_of_turn>`
- **LLaVA-1.6** ister: `USER: <image>\nPrompt ASSISTANT:`

Kodda her model için ayrı bir **adapter dosyası** var (`src/models/adapters/`). Config'te `model.name: "qwen2.5-vl-7b"` yazdığında, `registry.py` otomatik olarak Qwen adapter'ını yükler. Sen kodun geri kalanını değiştirmezsin.

### 4.3. Çıktı Formatı: Hibrit JSON

Hem yapılandırılmış (programla işlenebilir) hem de doğal dil (insan okuyabilir) alanlar var. UICrit'in orijinal formatıyla da uyumlu (bounding box + doğal dil yorum).

---

## 5. Dizin Yapısı

```
ui-critic-project/
│
├── README.md                          # Bu dosya
├── requirements.txt                   # Python paketleri
├── .gitignore                         # Git dışında kalacaklar
│
├── configs/                           # Tüm deney konfigürasyonları
│   ├── model1_qwen.yaml
│   ├── model1_gemma.yaml
│   ├── model1_llava.yaml
│   ├── model2_qwen.yaml
│   ├── model2_gemma.yaml
│   ├── model2_llava.yaml
│   ├── model3_qwen.yaml
│   ├── model3_gemma.yaml
│   ├── model3_llava.yaml
│   └── eval.yaml
│
├── data/                              # Datasetler (gitignore'da)
│   ├── rico/
│   │   ├── combined/                  # UI .jpg + .json VH çiftleri
│   │   └── semantic_annotations/
│   ├── uicrit/
│   │   └── uicrit_public.csv
│   ├── clay/                          # Opsiyonel, önerilen
│   │   └── clay_labels.csv
│   └── processed/                     # Kodun üreteceği temiz veri
│       ├── model1_train.json
│       ├── model1_val.json
│       ├── model1_test.json
│       ├── model2_train.json
│       ├── model2_val.json
│       ├── model2_test.json
│       ├── model3_train.json
│       ├── model3_val.json
│       └── model3_test.json
│
├── src/                               # Ana Python kodu
│   ├── __init__.py
│   │
│   ├── models/                        # Model yükleme katmanı
│   │   ├── __init__.py
│   │   ├── registry.py                # "model.name" → doğru adapter
│   │   ├── base_adapter.py            # Ortak arayüz (abstract class)
│   │   └── adapters/
│   │       ├── __init__.py
│   │       ├── qwen_adapter.py        # Qwen2.5-VL
│   │       ├── gemma_adapter.py       # Gemma 4
│   │       └── llava_adapter.py       # LLaVA-1.6
│   │
│   ├── data/                          # Veri yükleyiciler
│   │   ├── __init__.py
│   │   ├── rico_loader.py
│   │   ├── uicrit_loader.py
│   │   ├── alignment.py               # UICrit ↔ RICO eşleştirme (rico_id)
│   │   └── preprocessors/
│   │       ├── model1_prep.py
│   │       ├── model2_prep.py
│   │       └── model3_prep.py
│   │
│   ├── training/                      # Eğitim pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── lora_config.py
│   │   └── collator.py                # Batch oluşturma
│   │
│   ├── inference/                     # Inference pipeline
│   │   ├── __init__.py
│   │   └── inferencer.py
│   │
│   └── evaluation/                    # Değerlendirme
│       ├── __init__.py
│       ├── metrics.py                 # BLEU, ROUGE, BERTScore, IoU
│       ├── evaluator.py
│       └── judge.py                   # Şimdilik boş (LLM-as-a-judge için)
│
├── scripts/                           # Komut satırı script'leri
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train.py                       # Ana eğitim komutu
│   ├── infer.py
│   ├── evaluate.py
│   └── run_model2_on_uicrit.py        # Model 3 için Model 2 çıktılarını üretir
│
├── outputs/                           # Tüm çıktılar (gitignore'da)
│   ├── checkpoints/                   # Eğitilmiş model ağırlıkları
│   │   ├── model1_qwen_v1/
│   │   ├── model1_gemma_v1/
│   │   ├── model2_qwen_v1/
│   │   ├── model3_qwen_v1/
│   │   └── ...
│   ├── logs/                          # Eğitim logları
│   └── predictions/                   # Inference çıktıları
│
└── notebooks/                         # Keşif amaçlı (opsiyonel)
    └── data_exploration.ipynb
```

---

## 6. Cloud GPU Kurulumu

### 6.1. Sağlayıcı Seçimi

| Sağlayıcı | Avantaj | Dezavantaj |
|---|---|---|
| **RunPod** | En kolay arayüz, hazır şablonlar | Biraz pahalı |
| **Vast.ai** | En ucuz | Güvenilirlik değişken |
| **Lambda Labs** | Profesyonel, stabil | Talep yüksek, boş bulması zor |

### 6.2. Önerilen Donanım

| Bileşen | Minimum | Önerilen |
|---|---|---|
| GPU | RTX 4090 (24 GB) | **RTX 5090 (32 GB)** veya A100 (40 GB) |
| RAM | 16 GB | **32 GB+** |
| Disk | 100 GB | **200 GB+** (RICO ~6 GB, modeller 15-50 GB) |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Template | PyTorch 2.x + CUDA 12.x | PyTorch 2.5 + CUDA 12.4 |

### 6.3. İlk Bağlantı

Sunucu kiralandıktan sonra SSH bilgileri verilir. Örnek:

```bash
ssh root@<sunucu-ip> -p <port> -i <ssh-key.pem>
```

### 6.4. Sistem Güncellemesi

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git wget curl unzip build-essential tmux htop

# GPU'yu kontrol et
nvidia-smi
# Çıktıda GPU modelin, VRAM miktarın ve CUDA sürümün görünmeli
```

---

## 7. Ortam Kurulumu

### 7.1. Miniconda Kurulumu

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda init bash
# Terminali kapat ve yeniden aç
```

### 7.2. Proje Klonlama / Oluşturma

```bash
cd ~
mkdir ui-critic-project
cd ui-critic-project
git init
```

### 7.3. Python Ortamı

```bash
conda create -n uicritic python=3.11 -y
conda activate uicritic
```

Bundan sonra her terminal açışında `conda activate uicritic` çalıştırmayı unutma.

### 7.4. requirements.txt Dosyası

Proje kökünde `requirements.txt` dosyası oluştur:

```txt
# ===== PyTorch (CUDA 12.4) =====
torch==2.5.1
torchvision==0.20.1

# ===== HuggingFace Ekosistemi =====
transformers==4.46.0
accelerate==1.0.0
peft==0.13.0
bitsandbytes==0.44.0
datasets==3.0.0

# ===== Hızlandırma =====
unsloth==2025.1.0

# ===== Quantization =====
auto-gptq==0.7.0
optimum==1.23.0

# ===== Görüntü İşleme =====
pillow==11.0.0
opencv-python-headless==4.10.0.84

# ===== Değerlendirme Metrikleri =====
evaluate==0.4.3
sacrebleu==2.4.0
rouge-score==0.1.2
bert-score==0.3.13
nltk==3.9.1

# ===== Yardımcı =====
pyyaml==6.0.2
pandas==2.2.3
numpy==2.1.0
tqdm==4.66.5
wandb==0.18.0                # Eğitim izleme (opsiyonel)
jsonschema==4.23.0

# ===== HuggingFace Hub =====
huggingface-hub==0.25.0
```

### 7.5. Kurulum

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Kurulum 5-10 dakika sürebilir. Hata alırsan section 17'deki Troubleshooting bölümüne bak.

### 7.6. Ollama (Eğitim Sonrası İçin)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

### 7.7. HuggingFace Girişi

Gemma 4 gibi bazı modeller lisans gerektirir:

```bash
huggingface-cli login
# https://huggingface.co/settings/tokens adresinden token oluştur, yapıştır
```

Gemma 4 için ek olarak: https://huggingface.co/google/gemma-4-4b-it adresine git, "Accept license" butonuna tıkla.

---

## 8. Datasetlerin Hazırlanması

### 8.1. UICrit

```bash
cd data
git clone https://github.com/google-research-datasets/uicrit.git
# data/uicrit/uicrit_public.csv hazır
```

### 8.2. RICO

RICO ~6 GB. Kaggle'dan indirmek en kolayı:

```bash
# Kaggle CLI kur
pip install kaggle

# API anahtarını ayarla:
# https://www.kaggle.com/settings → "Create New API Token" → kaggle.json indir
mkdir -p ~/.kaggle
# kaggle.json dosyasını ~/.kaggle/'a kopyala (scp, SFTP vs.)
chmod 600 ~/.kaggle/kaggle.json

# RICO indir
cd data
kaggle datasets download -d onurgunes1993/rico-dataset
unzip rico-dataset.zip -d rico/
```

İndirme sonrası yapı:
```
data/rico/
├── combined/             # .jpg (görsel) + .json (VH) çiftleri
├── semantic_annotations/
└── ui_details.csv
```

### 8.3. CLAY (Opsiyonel ama Önerilen)

RICO'nun ham hiyerarşilerinde %37.4 geçersiz element var. Model 2'yi temiz veriyle eğitmek için:

```bash
cd data
git clone https://github.com/google-research-datasets/clay.git
# data/clay/clay_labels.csv
```

### 8.4. Veri Ön-İşleme

Tüm datasetler indirildikten sonra proje formatına dönüştürülür:

```bash
# Proje kökünden
python scripts/prepare_data.py --task model1
python scripts/prepare_data.py --task model2

# Model 3 için Model 2 önce eğitilmeli, sonraya kalır
```

Bu komutlar `data/processed/` klasörünü doldurur.

---

## 9. Config Sistemi: Model Nasıl Değiştirilir?

### 9.1. Temel Prensip

**Kodda hiçbir şey değişmez.** Farklı model denemek için farklı bir config dosyası kullanırsın ya da mevcut config'in içindeki `model.name` satırını değiştirirsin.

### 9.2. Config Örneği (`configs/model1_qwen.yaml`)

```yaml
# ============================================================
# MODEL 1 - UI → Kritik (Qwen2.5-VL)
# ============================================================

experiment:
  name: "model1_qwen_v1"             # Çıktı klasörü adı
  seed: 42
  task: "model1"                      # model1 | model2 | model3

# ------------------------------------------------------------
# 🔑 MODEL SEÇİMİ — BURAYI DEĞİŞTİR, GERİSİ OTOMATİK ÇALIŞIR
# ------------------------------------------------------------
model:
  name: "qwen2.5-vl-7b"               # Tanımlı modeller için section 10'a bak
  load_in_4bit: true                  # QLoRA: 4-bit quantization
  torch_dtype: "bfloat16"

# ------------------------------------------------------------
# LoRA AYARLARI
# ------------------------------------------------------------
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: "auto"              # Adapter otomatik seçer

# ------------------------------------------------------------
# VERİ
# ------------------------------------------------------------
data:
  train_path: "data/processed/model1_train.json"
  val_path: "data/processed/model1_val.json"
  test_path: "data/processed/model1_test.json"
  image_dir: "data/rico/combined/"
  max_image_size: 1024
  output_format: "hybrid_json"        # hybrid_json | plain_text | structured_json

# ------------------------------------------------------------
# EĞİTİM
# ------------------------------------------------------------
training:
  num_epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 8      # Etkili batch = 2 × 8 = 16
  learning_rate: 2e-4
  warmup_steps: 100
  weight_decay: 0.01
  save_strategy: "steps"
  save_steps: 200
  eval_steps: 200
  logging_steps: 20
  max_grad_norm: 1.0
  use_unsloth: true                   # Hız için

# ------------------------------------------------------------
# ÇIKTI
# ------------------------------------------------------------
output:
  dir: "outputs/checkpoints/model1_qwen_v1"
  logging_dir: "outputs/logs/model1_qwen_v1"
  save_total_limit: 2                 # Son 2 checkpoint tut, eskileri sil
```

### 9.3. Başka Modele Geçiş — 3 Yol

**Yol A: Yeni config dosyası (önerilen)**

```bash
cp configs/model1_qwen.yaml configs/model1_gemma.yaml
```

`configs/model1_gemma.yaml` içinde sadece 2 satır değiştir:
```yaml
experiment:
  name: "model1_gemma_v1"            # ← değişti
model:
  name: "gemma-4-4b"                 # ← değişti
# Geri kalan aynı
```

Çalıştır:
```bash
python scripts/train.py --config configs/model1_gemma.yaml
```

**Yol B: Mevcut config'i düzenle**

`configs/model1_qwen.yaml` içinde:
```yaml
model:
  name: "llava-1.6-7b"               # ← sadece bunu değiştir
```

**Yol C: Komut satırından override**

```bash
python scripts/train.py \
  --config configs/model1_qwen.yaml \
  --override model.name=gemma-4-4b
```

---

## 10. Desteklenen Modeller

`src/models/registry.py` içinde tanımlı. Config'te `model.name` için kullanabileceğin değerler:

| Config Değeri | Parametre | HF Model ID | Min VRAM (QLoRA) |
|---|---|---|---|
| `qwen2.5-vl-3b` | 3B | `Qwen/Qwen2.5-VL-3B-Instruct` | ~8 GB |
| `qwen2.5-vl-7b` | 7B | `Qwen/Qwen2.5-VL-7B-Instruct` | ~16 GB |
| `gemma-4-4b` | 4B | `google/gemma-4-4b-it` | ~10 GB |
| `gemma-4-12b` | 12B | `google/gemma-4-12b-it` | ~24 GB |
| `llava-1.6-7b` | 7B | `llava-hf/llava-v1.6-mistral-7b-hf` | ~16 GB |
| `llava-1.6-13b` | 13B | `llava-hf/llava-v1.6-vicuna-13b-hf` | ~28 GB |

### Yeni Model Eklemek

Diyelim ki Qwen3-VL çıktı ve denemek istiyorsun:

1. `src/models/adapters/qwen_adapter.py` dosyasını kopyala → `qwen3_adapter.py`
2. Prompt formatını Qwen3'ün dokümantasyonuna göre düzenle
3. `src/models/registry.py` içine ekle:
   ```python
   "qwen3-vl-7b": {
       "hf_id": "Qwen/Qwen3-VL-7B-Instruct",
       "adapter": "Qwen3Adapter"
   }
   ```
4. Artık `model.name: "qwen3-vl-7b"` yazabilirsin.

---

## 11. Model 1 Eğitimi: UI → Kritik

### 11.1. Görev

**Girdi:** UI ekran görüntüsü  
**Çıktı:** JSON formatında kritik listesi + doğal dil özet

### 11.2. Eğitim Verisi Formatı (`data/processed/model1_train.json`)

```json
[
  {
    "rico_id": 12345,
    "image_path": "data/rico/combined/12345.jpg",
    "task": "E-commerce checkout",
    "critiques": [
      {
        "comment": "Ana aksiyon butonu ekranın altında, parmakla ulaşmak zor",
        "bounding_box": [120, 1800, 200, 80],
        "severity": "medium",
        "category": "accessibility"
      },
      {
        "comment": "Başlık ve alt metin arasında kontrast yetersiz",
        "bounding_box": [50, 100, 600, 60],
        "severity": "high",
        "category": "readability"
      }
    ],
    "aesthetics_rating": 6,
    "learnability": 3,
    "overall_feedback": "Bu ekranda erişilebilirlik iyileştirmeleri gerekli..."
  }
]
```

### 11.3. Eğitim Komutu

```bash
# Qwen2.5-VL ile
python scripts/train.py --config configs/model1_qwen.yaml

# Gemma 4 ile
python scripts/train.py --config configs/model1_gemma.yaml

# LLaVA-1.6 ile
python scripts/train.py --config configs/model1_llava.yaml
```

### 11.4. Beklenen Süre

RTX 5090 (32 GB) üzerinde, 983 UI ile 3 epoch:

| Model | Süre |
|---|---|
| Qwen2.5-VL-3B | ~1.5 saat |
| Qwen2.5-VL-7B | ~3-4 saat |
| Gemma 4-4B | ~2 saat |
| Gemma 4-12B | ~5-6 saat |
| LLaVA-1.6-7B | ~3-4 saat |

### 11.5. Eğitim Sırasında İzleme

Ayrı bir terminalde:
```bash
watch -n 2 nvidia-smi          # GPU kullanımı (%80+ olmalı)
tail -f outputs/logs/model1_qwen_v1/trainer.log
```

`wandb` aktifse: https://wandb.ai adresinden canlı grafikler.

---

## 12. Model 2 Eğitimi: UI → View Hierarchy

### 12.1. Görev

**Girdi:** UI ekran görüntüsü  
**Çıktı:** View hierarchy (JSON tree)

### 12.2. Eğitim Verisi Formatı (`data/processed/model2_train.json`)

```json
[
  {
    "rico_id": 54321,
    "image_path": "data/rico/combined/54321.jpg",
    "hierarchy": {
      "type": "FrameLayout",
      "bounds": [0, 0, 1080, 1920],
      "children": [
        {
          "type": "TextView",
          "bounds": [50, 100, 1030, 180],
          "text": "Hoş Geldiniz",
          "children": []
        },
        {
          "type": "Button",
          "bounds": [400, 1700, 680, 1800],
          "text": "Devam Et",
          "clickable": true,
          "children": []
        }
      ]
    }
  }
]
```

### 12.3. Config Özel Ayarları

Model 2 config'inde (`configs/model2_qwen.yaml`) ek parametreler:

```yaml
data:
  hierarchy_simplification: "clay"    # none | clay | minimal
  max_depth: 10                        # Hiyerarşi derinlik sınırı
  include_invisible: false             # Görünmeyen elementleri atla
```

- `none`: RICO'nun ham (gürültülü) VH'si
- `clay`: CLAY ile temizlenmiş (**önerilen**)
- `minimal`: Sadece görünür leaf elementler

### 12.4. Eğitim Komutu

```bash
python scripts/train.py --config configs/model2_qwen.yaml
python scripts/train.py --config configs/model2_gemma.yaml
python scripts/train.py --config configs/model2_llava.yaml
```

### 12.5. Model 2 Değerlendirmesi

Model 2'nin kalitesini ölçmek için özel metrikler:

- **Element Detection AP @ IoU>0.5**: Doğru elementleri doğru yerde mi tespit ediyor?
- **Tree Edit Distance**: Çıktı ağacı gerçek ağaca ne kadar benziyor?
- **Type Accuracy**: Element tiplerini doğru sınıflandırabiliyor mu? (Button, TextView vs.)

```bash
python scripts/evaluate.py --config configs/model2_qwen.yaml
```

### 12.6. En İyi Model 2'yi Seçme

3 farklı VLM ile eğitip test sonuçlarına göre en iyisini seç. Bu, Model 3 eğitiminde kullanılacak hiyerarşileri üretecek.

---

## 13. Model 3 Eğitimi: UI + Hiyerarşi → Kritik

### 13.1. Görev

**Girdi:** UI ekran görüntüsü + Model 2'nin ürettiği hiyerarşi  
**Çıktı:** Tasarım kritiği (Model 1 ile aynı format)

### 13.2. Önkoşullar (SIRAYLA!)

```bash
# 1. Önce Model 2'yi eğit
python scripts/train.py --config configs/model2_qwen.yaml

# 2. En iyi Model 2'yi UICrit'in 983 UI'ına uygula
python scripts/run_model2_on_uicrit.py \
    --config configs/model2_qwen.yaml \
    --checkpoint outputs/checkpoints/model2_qwen_v1/best \
    --output data/processed/uicrit_predicted_hierarchies.json

# 3. Model 3 eğitim verisini oluştur (UICrit kritikleri + predicted VH)
python scripts/prepare_data.py --task model3

# 4. Şimdi Model 3'ü eğit
python scripts/train.py --config configs/model3_qwen.yaml
```

### 13.3. Eğitim Verisi Formatı (`data/processed/model3_train.json`)

```json
[
  {
    "rico_id": 12345,
    "image_path": "data/rico/combined/12345.jpg",
    "predicted_hierarchy": {
      "type": "FrameLayout",
      "children": [
        {"type": "Button", "bounds": [120, 1800, 320, 1880], "text": "Satın Al"}
      ]
    },
    "critiques": [
      {
        "comment": "Ana aksiyon butonu ekranın altında, parmakla ulaşmak zor",
        "bounding_box": [120, 1800, 200, 80],
        "severity": "medium"
      }
    ]
  }
]
```

### 13.4. Prompt Yapısı

Model 3 prompt'u hem görseli hem hiyerarşiyi içerir:

```
<image>

Bu UI ekranının yapısal hiyerarşisi:
{predicted_hierarchy JSON}

Yukarıdaki bilgiyi kullanarak bu UI için profesyonel tasarım kritiği üret.
Çıktıyı aşağıdaki JSON formatında ver: ...
```

### 13.5. Önerilen Ablation Study

Tezinin değerini artırmak için 4 konfigürasyonlu deney öneririm:

| Config | Girdi | Neyi Ölçer? | Komut |
|---|---|---|---|
| **A** | Sadece UI görseli | Model 1 baseline | `train.py --config configs/model1_qwen.yaml` |
| **B** | UI + **oracle** VH (RICO ground truth) | Hiyerarşinin teorik üst sınırı | `train.py --config configs/model3_qwen_oracle.yaml` |
| **C** | UI + **predicted** VH (Model 2 çıktısı) | Model 3, pratik versiyon | `train.py --config configs/model3_qwen.yaml` |
| **D** | Sadece predicted VH (görsel yok) | Hiyerarşinin tek başına katkısı | `train.py --config configs/model3_qwen_noimg.yaml` |

**Sonuçların yorumu:**
- B - C farkı → Model 2 hatalarının Model 3'e etkisi
- A - C farkı → Hiyerarşinin pratik faydası (ASIL ARAŞTIRMA SORUSU)
- A - B farkı → Hiyerarşinin teorik tavanı
- D düşük çıkarsa → Görsel bilgi kritik, hiyerarşi tek başına yetmiyor

---

## 14. Değerlendirme (Evaluation)

### 14.1. Kullanılan Metrikler

| Metrik | Neyi Ölçer? | Nerede Kullanılır? |
|---|---|---|
| **BLEU-1,2,3,4** | N-gram örtüşmesi (kelime benzerliği) | Model 1, 3 |
| **ROUGE-1, ROUGE-2, ROUGE-L** | Recall-tabanlı metin benzerliği | Model 1, 3 |
| **METEOR** | Sinonim-farkında benzerlik | Model 1, 3 |
| **BERTScore** | Semantik benzerlik (mBERT ile) | Model 1, 3 |
| **IoU (Intersection over Union)** | Bounding box doğruluğu | Model 1, 3 |
| **AP @ IoU>0.5** | Ortalama lokalizasyon doğruluğu | Model 2 |
| **Tree Edit Distance** | Ağaç yapısı benzerliği | Model 2 |
| **Type Accuracy** | Element tipi doğruluğu | Model 2 |

### 14.2. Tek Model Değerlendirmesi

```bash
python scripts/evaluate.py --config configs/model1_qwen.yaml
```

Çıktı örneği:
```
Model: model1_qwen_v1
-----------------------------------
BLEU-1:      0.412
BLEU-4:      0.187
ROUGE-L:     0.356
BERTScore:   0.724
Mean IoU:    0.623
```

### 14.3. Karşılaştırma Tablosu

```bash
python scripts/evaluate.py \
    --configs configs/model1_qwen.yaml \
              configs/model1_gemma.yaml \
              configs/model1_llava.yaml \
              configs/model3_qwen.yaml \
              configs/model3_gemma.yaml \
    --output outputs/comparison_table.csv
```

### 14.4. LLM-as-a-Judge (Şimdilik Devre Dışı)

Gelecekte eklemek için `src/evaluation/judge.py` yer tutucu olarak duruyor. İhtiyaç olduğunda Claude veya GPT-4 ile kritikleri puanlatan kod eklenecek.

---

## 15. Inference (Eğitilmiş Modeli Kullanma)

### 15.1. Tek UI için Kritik Üretme

```bash
python scripts/infer.py \
    --config configs/model1_qwen.yaml \
    --checkpoint outputs/checkpoints/model1_qwen_v1/best \
    --image test_ui.png \
    --output prediction.json
```

Çıktı (`prediction.json`):
```json
{
  "image": "test_ui.png",
  "critiques": [
    {
      "comment": "Giriş alanları için yeterli dikey boşluk bırakılmamış",
      "bounding_box": [50, 400, 980, 300],
      "severity": "medium",
      "category": "visual_hierarchy"
    }
  ],
  "overall_feedback": "Bu form ekranında elementler arasında nefes alanı oluşturmak gerekli..."
}
```

### 15.2. Toplu Inference

```bash
python scripts/infer.py \
    --config configs/model1_qwen.yaml \
    --checkpoint outputs/checkpoints/model1_qwen_v1/best \
    --image_dir my_uis/ \
    --output_dir predictions/
```

---

## 16. Ollama'ya Aktarma (İsteğe Bağlı)

Eğitim bitince modeli günlük kullanıma almak istersen.

### 16.1. GGUF Dönüşümü

```bash
# llama.cpp klonla
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# Eğitilmiş modeli GGUF'a çevir
python convert_hf_to_gguf.py \
    ~/ui-critic-project/outputs/checkpoints/model1_qwen_v1/best \
    --outfile ~/ui-critic-project/outputs/model1_qwen.gguf \
    --outtype q4_k_m                   # 4-bit quantization, hızlı + küçük
```

### 16.2. Modelfile Oluştur

Proje kökünde `Modelfile`:

```
FROM ./outputs/model1_qwen.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """
Sen bir UI/UX tasarım uzmanısın. Verilen UI ekran görüntüsünü analiz edip
JSON formatında yapılandırılmış tasarım kritiği üretirsin.
"""
```

### 16.3. Ollama'ya Kur

```bash
ollama create ui-critic-qwen -f Modelfile
ollama run ui-critic-qwen
```

### 16.4. ⚠️ Önemli Uyarı

Vision model GGUF dönüşümü Qwen2.5-VL ve Gemma 4 için **hâlâ deneysel**. Ollama'nın multimodal desteği genişliyor ama stabil değil. Bu adım projenin ana akışını etkilemez — tez bittikten sonra yapılabilir.

---

## 17. Sık Karşılaşılan Hatalar

### 17.1. CUDA Out of Memory (OOM)

**Belirti:** `RuntimeError: CUDA out of memory`

**Çözüm sırası:**

```yaml
# config.yaml - Sırayla dene:

# 1. Batch size düşür
training:
  batch_size: 1                        # 2 → 1
  gradient_accumulation_steps: 16      # 8 → 16 (etkili batch aynı kalır)

# 2. 4-bit quantization açık mı?
model:
  load_in_4bit: true

# 3. Görsel boyutunu düşür
data:
  max_image_size: 512                  # 1024 → 512

# 4. Daha küçük model
model:
  name: "qwen2.5-vl-3b"                # 7B yerine 3B
  # veya
  name: "gemma-4-4b"                   # 12B yerine 4B
```

### 17.2. "Model not found on the Hub"

**Sebep:** HuggingFace giriş yapılmamış veya model lisansı kabul edilmemiş.

**Çözüm:**
```bash
huggingface-cli login
# Token'ı yapıştır
```

Gemma 4 için ayrıca: https://huggingface.co/google/gemma-4-4b-it adresinde "Accept license".

### 17.3. "bitsandbytes not compiled with CUDA"

```bash
pip uninstall bitsandbytes -y
pip install bitsandbytes --upgrade --force-reinstall
```

Hala olmazsa:
```bash
pip install bitsandbytes==0.44.0 --no-cache-dir
```

### 17.4. UICrit CSV Parse Hatası

**Sebep:** `comments` sütunu string olarak okuyor, JSON çevirme gerekiyor.

**Çözüm:** `src/data/uicrit_loader.py` içinde:
```python
import json
import pandas as pd

df = pd.read_csv("data/uicrit/uicrit_public.csv")
df['comments'] = df['comments'].apply(json.loads)
df['comments_source'] = df['comments_source'].apply(json.loads)
```

### 17.5. Eğitim Çok Yavaş

**Belirti:** `nvidia-smi` GPU kullanımını %10-20 gösteriyor.

**Çözüm:**
- Config'te `training.use_unsloth: true` olduğunu kontrol et
- Data loading bottleneck'i olabilir: `num_workers: 4` ekle
- CPU'da çalışıyor olabilirsin — `torch.cuda.is_available()` kontrol et:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  # True dönmeli
  ```

### 17.6. RICO JSON'u Bozuk

**Sebep:** RICO'nun bazı view hierarchy dosyaları bozuk.

**Çözüm:** `src/data/rico_loader.py` içinde:
```python
import json

def load_hierarchy(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None  # Bu UI'ı atla
```

### 17.7. Checkpoint Diski Doldurdu

**Sebep:** Her eval step'te checkpoint kaydediliyor, disk doluyor.

**Çözüm:**
```yaml
output:
  save_total_limit: 2                  # Son 2 checkpoint kalsın
```

Disk temizlemek için:
```bash
du -sh outputs/checkpoints/*
# Gereksizleri sil
rm -rf outputs/checkpoints/model1_qwen_v0_eski/
```

### 17.8. Unsloth "Model Desteklenmiyor" Hatası

**Sebep:** Unsloth bazı modelleri henüz desteklemiyor.

**Çözüm:** Config'te Unsloth'u devre dışı bırak, HF'ye geri dön:
```yaml
training:
  use_unsloth: false
```

Eğitim yavaşlar ama çalışır.

### 17.9. HuggingFace Download Çok Yavaş

**Sebep:** Bazı bölgelerde HF CDN yavaş.

**Çözüm:** `hf_transfer` kullan:
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 18. Proje Adım Adım Takibi

### Aşama 0: Altyapı
- [ ] Cloud GPU sunucu kiralandı
- [ ] SSH erişimi sağlandı
- [ ] `nvidia-smi` çalışıyor
- [ ] Conda environment (`uicritic`) hazır
- [ ] `requirements.txt` kuruldu (hata yok)
- [ ] HuggingFace login yapıldı
- [ ] Gemma 4 lisansı kabul edildi (gerekiyorsa)
- [ ] Antigravity IDE sunucuya bağlandı

### Aşama 1: Veri
- [ ] RICO indirildi
- [ ] UICrit indirildi
- [ ] CLAY indirildi (opsiyonel)
- [ ] `scripts/prepare_data.py --task model1` çalıştı
- [ ] `scripts/prepare_data.py --task model2` çalıştı
- [ ] Train/val/test split'leri kontrol edildi

### Aşama 2: Model 1 Eğitimleri
- [ ] Qwen2.5-VL ile Model 1 eğitildi
- [ ] Gemma 4 ile Model 1 eğitildi
- [ ] LLaVA-1.6 ile Model 1 eğitildi
- [ ] Üç model test setinde değerlendirildi
- [ ] Karşılaştırma tablosu oluşturuldu
- [ ] En iyi Model 1 belirlendi

### Aşama 3: Model 2 Eğitimleri
- [ ] Qwen2.5-VL ile Model 2 eğitildi
- [ ] Gemma 4 ile Model 2 eğitildi
- [ ] LLaVA-1.6 ile Model 2 eğitildi
- [ ] Üç model değerlendirildi
- [ ] En iyi Model 2 belirlendi
- [ ] Model 2'nin UICrit'teki 983 UI için çıktıları üretildi

### Aşama 4: Model 3 Eğitimleri
- [ ] Model 3 veri hazırlama (`prepare_data.py --task model3`)
- [ ] Qwen2.5-VL ile Model 3 eğitildi
- [ ] Gemma 4 ile Model 3 eğitildi
- [ ] LLaVA-1.6 ile Model 3 eğitildi
- [ ] Üç model değerlendirildi
- [ ] Model 1 vs Model 3 karşılaştırıldı

### Aşama 5: Ablation Study (Opsiyonel)
- [ ] Config A (sadece UI) — Model 1 zaten yapıldı
- [ ] Config B (UI + oracle VH) eğitildi
- [ ] Config C (UI + predicted VH) — Model 3 zaten yapıldı
- [ ] Config D (sadece VH) eğitildi
- [ ] 4 config karşılaştırma tablosu

### Aşama 6: Rapor/Tez Yazımı
- [ ] Tüm sonuçlar tabloya döküldü
- [ ] Literatürle karşılaştırma yapıldı
- [ ] Discussion bölümü yazıldı
- [ ] Conclusion yazıldı

### Aşama 7 (Opsiyonel): Deploy
- [ ] En iyi model GGUF'a çevrildi
- [ ] Ollama Modelfile hazırlandı
- [ ] Lokal inference test edildi
- [ ] LLM-as-a-judge eklendi (opsiyonel)

---

## 19. Kaynakça

### Temel Makaleler

**UICrit:**
- Duan et al. (2024). "UICrit: Enhancing Automated Design Evaluation with a UI Critique Dataset." UIST '24.
- arXiv: https://arxiv.org/abs/2407.08850
- GitHub: https://github.com/google-research-datasets/uicrit
- DOI: 10.1145/3654777.3676381

**Iterative Visual Prompting (UICrit takibi):**
- Duan et al. (2024). "Visual Prompting with Iterative Refinement for Design Critique Generation."
- arXiv: https://arxiv.org/abs/2412.16829

**Screen Parsing (Model 2 için temel referans):**
- Wu et al. (2021). "Screen Parsing: Towards Reverse Engineering of UI Models from Screenshots." UIST '21.
- arXiv: https://arxiv.org/abs/2109.08763

**CLAY:**
- Li et al. (2022). "Learning to Denoise Raw Mobile UI Layouts for Improving Datasets at Scale." CHI '22.
- arXiv: https://arxiv.org/abs/2201.04100

**Screen2Words (multimodal UI temsili):**
- Wang et al. (2021). "Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning." UIST '21.
- arXiv: https://arxiv.org/abs/2108.03353

**RICO:**
- Deka et al. (2017). "Rico: A Mobile App Dataset for Building Data-Driven Design Applications." UIST '17.

### Model Kaynakları

- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- Gemma 4: https://ai.google.dev/gemma
- LLaVA-1.6: https://llava-vl.github.io/blog/2024-01-30-llava-next/

### Framework Kaynakları

- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PEFT (LoRA): https://huggingface.co/docs/peft
- Unsloth: https://docs.unsloth.ai
- Ollama: https://ollama.com/docs
- llama.cpp: https://github.com/ggerganov/llama.cpp

---

**Son güncelleme:** Nisan 2026  
**Proje durumu:** Geliştirme aşamasında  
**Lisans:** (Daha sonra eklenecek — MIT önerilir)
