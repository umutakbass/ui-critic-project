#!/bin/bash
# RunPod cloud kurulum script'i
# Kullanım: bash scripts/setup_cloud.sh <KAGGLE_USERNAME> <KAGGLE_KEY> <HF_TOKEN>
#
# Örnek:
#   bash scripts/setup_cloud.sh umut abc123 hf_xxx

set -e  # herhangi bir hata olursa dur

KAGGLE_USERNAME=$1
KAGGLE_KEY=$2
HF_TOKEN=$3

if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ] || [ -z "$HF_TOKEN" ]; then
    echo "Kullanım: bash scripts/setup_cloud.sh <KAGGLE_USERNAME> <KAGGLE_KEY> <HF_TOKEN>"
    exit 1
fi

echo "=== 1/6 Kaggle kimlik bilgileri ayarlanıyor ==="
mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "=== 2/6 HuggingFace login ==="
pip install -q huggingface-hub
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

echo "=== 3/6 Gereksinimler kuruluyor ==="
pip install -r requirements-cloud.txt

echo "=== 4/6 UICrit CSV indiriliyor ==="
mkdir -p data/uicrit
wget -q -O data/uicrit/uicrit_public.csv \
    https://raw.githubusercontent.com/google-research-datasets/uicrit/main/uicrit_public.csv
echo "   uicrit_public.csv indirildi: $(wc -l < data/uicrit/uicrit_public.csv) satır"

echo "=== 5/6 RICO dataset indiriliyor (~6 GB, birkaç dakika sürer) ==="
mkdir -p data/archive
cd data/archive
kaggle datasets download -d onurgunes1993/rico-dataset
echo "   ZIP açılıyor..."
unzip -q rico-dataset.zip
rm rico-dataset.zip
cd ../..
echo "   RICO hazır: $(ls data/archive/unique_uis/combined | wc -l) dosya"

echo "=== 6/6 Eğitim verisi hazırlanıyor ==="
python scripts/prepare_data.py --task model1
python scripts/prepare_data.py --task model2

echo ""
echo "Kurulum tamamlandı!"
echo "Eğitimi başlatmak için:"
echo "  tmux new -s training"
echo "  python scripts/train.py --config configs/model1_qwen.yaml"
