#!/bin/bash
# Mini dataset ile hızlı pipeline testi
# Kullanım: bash scripts/setup_mini.sh <HF_TOKEN>

set -e

HF_TOKEN=$1

if [ -z "$HF_TOKEN" ]; then
    echo "Kullanım: bash scripts/setup_mini.sh <HF_TOKEN>"
    exit 1
fi

echo "=== 1/5 Gereksinimler kuruluyor ==="
pip install -q gdown huggingface-hub
pip install -r requirements-cloud.txt || pip install -r requirements-cloud.txt --ignore-requires-python
pip install unsloth 2>/dev/null && echo "   unsloth kuruldu (opsiyonel)" || echo "   unsloth atlandı (opsiyonel)"

echo "=== 2/5 HuggingFace login ==="
if command -v hf >/dev/null 2>&1; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

echo "=== 3/5 UICrit indiriliyor ==="
mkdir -p data/uicrit
gdown 1FcewVy9PJ5RiqC6L1FjICdY4VJoMvuPf -O data/uicrit/uicrit.zip
unzip -q data/uicrit/uicrit.zip -d data/uicrit/
rm data/uicrit/uicrit.zip

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

echo "=== 4/5 Mini dataset indiriliyor (~79 MB) ==="
gdown 1LN_K_xczON-vRRMeYpTL9J_mWn_SNGri -O data/mini_dataset.zip
echo "   ZIP açılıyor..."
mkdir -p data/archive/unique_uis/combined
mkdir -p data/processed

# Görselleri doğru konuma çıkar
unzip -q data/mini_dataset.zip -d data/mini_tmp/
mv data/mini_tmp/images/* data/archive/unique_uis/combined/ 2>/dev/null || true
mv data/mini_tmp/processed/* data/processed/ 2>/dev/null || true
rm -rf data/mini_tmp data/mini_dataset.zip

echo "   Mini dataset hazır: $(ls data/archive/unique_uis/combined | wc -l) görsel"
echo "   Processed: $(ls data/processed | wc -l) dosya"

echo "=== 5/5 Kurulum doğrulanıyor ==="
python scripts/verify_setup.py

echo ""
echo "Mini kurulum tamamlandı! Test eğitimini başlatmak için:"
echo "  python scripts/train.py --config configs/model1_qwen.yaml"
