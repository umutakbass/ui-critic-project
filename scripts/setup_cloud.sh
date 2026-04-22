#!/bin/bash
# RunPod cloud kurulum script'i
# Kullanım: bash scripts/setup_cloud.sh <HF_TOKEN>

set -e

HF_TOKEN=$1

if [ -z "$HF_TOKEN" ]; then
    echo "Kullanım: bash scripts/setup_cloud.sh <HF_TOKEN>"
    exit 1
fi

echo "=== 1/6 Gereksinimler kuruluyor ==="
pip install -q gdown huggingface-hub
pip install -r requirements-cloud.txt || pip install -r requirements-cloud.txt --ignore-requires-python
pip install unsloth 2>/dev/null && echo "   unsloth kuruldu (opsiyonel)" || echo "   unsloth atlandı (opsiyonel)"

echo "=== 2/6 HuggingFace login ==="
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo "=== 3/6 UICrit indiriliyor ==="
mkdir -p data/uicrit
gdown 1FcewVy9PJ5RiqC6L1FjICdY4VJoMvuPf -O data/uicrit/uicrit.zip
unzip -q data/uicrit/uicrit.zip -d data/uicrit/
rm data/uicrit/uicrit.zip

# Zip iç içe klasör oluşturduysa düzelt
if [ ! -f "data/uicrit/uicrit_public.csv" ]; then
    FOUND=$(find data/uicrit -name "uicrit_public.csv" | head -1)
    if [ -n "$FOUND" ]; then
        mv "$FOUND" data/uicrit/uicrit_public.csv
        find data/uicrit -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
    else
        echo "HATA: uicrit_public.csv bulunamadı!" && exit 1
    fi
fi
echo "   UICrit hazır: $(wc -l < data/uicrit/uicrit_public.csv) satır"

echo "=== 4/6 RICO (archive) indiriliyor (~25 GB) ==="
gdown 1cEAXOnr6efRkzAUaFbZIm372j9R9dFEq -O data/archive.zip
echo "   ZIP açılıyor..."
unzip -q data/archive.zip -d data/
rm data/archive.zip

# Zip yapısını otomatik tespit et ve doğru konuma taşı
if [ ! -d "data/archive/unique_uis/combined" ]; then
    COMBINED=$(find data -type d -name "combined" | head -1)
    if [ -n "$COMBINED" ]; then
        PARENT=$(dirname "$COMBINED")
        GRANDPARENT=$(dirname "$PARENT")
        if [ "$GRANDPARENT" != "data/archive" ]; then
            mkdir -p data/archive
            mv "$GRANDPARENT"/* data/archive/ 2>/dev/null || mv "$PARENT" data/archive/
        fi
    else
        echo "HATA: combined/ klasörü bulunamadı!" && exit 1
    fi
fi
echo "   RICO hazır: $(ls data/archive/unique_uis/combined | wc -l) dosya"

echo "=== 5/6 Kurulum doğrulanıyor ==="
python scripts/verify_setup.py

echo "=== 6/6 Eğitim verisi hazırlanıyor ==="
python scripts/prepare_data.py --task model1
python scripts/prepare_data.py --task model2

echo ""
echo "Kurulum tamamlandı! Eğitimi başlatmak için:"
echo "  tmux new -s training"
echo "  python scripts/train.py --config configs/model1_qwen.yaml"
