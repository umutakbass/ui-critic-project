#!/bin/bash
# RunPod cloud kurulum script'i
# Kullanım: bash scripts/setup_cloud.sh <HF_TOKEN>
#
# Örnek:
#   bash scripts/setup_cloud.sh hf_xxx

set -e

HF_TOKEN=$1

if [ -z "$HF_TOKEN" ]; then
    echo "Kullanım: bash scripts/setup_cloud.sh <HF_TOKEN>"
    exit 1
fi

echo "=== 1/5 Gereksinimler kuruluyor ==="
pip install -q gdown
pip install -r requirements-cloud.txt

echo "=== 2/5 HuggingFace login ==="
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo "=== 3/5 UICrit indiriliyor (Drive) ==="
mkdir -p data/uicrit
gdown 1FcewVy9PJ5RiqC6L1FjICdY4VJoMvuPf -O data/uicrit/uicrit.zip
unzip -q data/uicrit/uicrit.zip -d data/uicrit/
rm data/uicrit/uicrit.zip
echo "   UICrit hazır: $(ls data/uicrit/)"

echo "=== 4/5 RICO (archive) indiriliyor (~25 GB, uzun sürer) ==="
mkdir -p data
gdown 1cEAXOnr6efRkzAUaFbZIm372j9R9dFEq -O data/archive.zip
echo "   ZIP açılıyor..."
unzip -q data/archive.zip -d data/
rm data/archive.zip
echo "   RICO hazır: $(ls data/archive/unique_uis/combined | wc -l) dosya"

echo "=== 5/5 Eğitim verisi hazırlanıyor ==="
python scripts/prepare_data.py --task model1
python scripts/prepare_data.py --task model2

echo ""
echo "Kurulum tamamlandı! Eğitimi başlatmak için:"
echo "  tmux new -s training"
echo "  python scripts/train.py --config configs/model1_qwen.yaml"
