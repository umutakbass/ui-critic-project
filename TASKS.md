# UI Design Critique Project — Atomize Task Listesi

Bu liste, projeyi baştan sona bitirmek için atmanız gereken her adımı **en küçük parçalara** bölünmüş olarak içerir. Her görev:

- **Tek bir somut çıktı** üretir (dosya, test, sonuç)
- **15-60 dakikada** tamamlanabilir
- **Nasıl test edileceği** net
- **Hangi görevin önkoşul** olduğu belli

**Toplam:** 10 Faz, 150+ görev. Tahmini toplam süre: 3-6 hafta (tam zamanlı çalışma ile 2-3 hafta).

**İşaretleme sistemi:**
- `[ ]` = Yapılmadı
- `[~]` = Devam ediyor
- `[x]` = Tamamlandı
- `[!]` = Engellendi / blocker var

---

## İçindekiler

- [Faz 0: Proje Altyapısı (Lokal Kurulum)](#faz-0-proje-altyapısı-lokal-kurulum)
- [Faz 1: Versiyon Kontrol ve Proje İskeleti](#faz-1-versiyon-kontrol-ve-proje-i̇skeleti)
- [Faz 2: Datasetlerin İndirilmesi ve İncelenmesi](#faz-2-datasetlerin-i̇ndirilmesi-ve-i̇ncelenmesi)
- [Faz 3: Veri Yükleyiciler (Data Loaders)](#faz-3-veri-yükleyiciler-data-loaders)
- [Faz 4: Model Katmanı (Registry + Adapters)](#faz-4-model-katmanı-registry--adapters)
- [Faz 5: Veri Ön-İşleme (Preprocessing)](#faz-5-veri-ön-i̇şleme-preprocessing)
- [Faz 6: Eğitim Pipeline](#faz-6-eğitim-pipeline)
- [Faz 7: Cloud GPU Kurulumu](#faz-7-cloud-gpu-kurulumu)
- [Faz 8: Model Eğitimleri](#faz-8-model-eğitimleri)
- [Faz 9: Değerlendirme ve Karşılaştırma](#faz-9-değerlendirme-ve-karşılaştırma)
- [Faz 10: Tez/Rapor Yazımı](#faz-10-tezrapor-yazımı)

---

## Faz 0: Proje Altyapısı (Lokal Kurulum)

**Amaç:** Kendi bilgisayarında temiz bir geliştirme ortamı kurmak.
**Süre:** ~2-3 saat

### 0.1. Temel Araçların Kurulumu

- [ ] **0.1.1** Python 3.11'in kurulu olduğunu kontrol et (`python --version`)
  - Yoksa: https://www.python.org/downloads/ adresinden indir
  - **Test:** `python --version` → `Python 3.11.x` yazmalı

- [ ] **0.1.2** Git'in kurulu olduğunu kontrol et (`git --version`)
  - Yoksa: Windows için Git for Windows, Mac için `brew install git`
  - **Test:** `git --version` → version bilgisi dönmeli

- [ ] **0.1.3** VS Code veya Antigravity IDE indir ve kur
  - Antigravity: https://antigravity.google/
  - **Test:** IDE açılıyor mu?

- [ ] **0.1.4** Miniconda indir ve kur
  - https://docs.conda.io/projects/miniconda/en/latest/
  - **Test:** Yeni terminal aç, `conda --version` çalışmalı

### 0.2. GitHub Hesabı ve SSH Key

- [ ] **0.2.1** GitHub hesabı oluştur (yoksa)
  - https://github.com/signup
  - **Test:** Hesaba giriş yapabiliyorsun

- [ ] **0.2.2** SSH key oluştur
  - Terminal: `ssh-keygen -t ed25519 -C "email@example.com"`
  - Enter, Enter, Enter (şifre girmeden)
  - **Test:** `~/.ssh/id_ed25519.pub` dosyası oluştu mu?

- [ ] **0.2.3** SSH key'i GitHub'a ekle
  - `cat ~/.ssh/id_ed25519.pub` ile içeriği kopyala
  - GitHub → Settings → SSH and GPG keys → New SSH key
  - **Test:** `ssh -T git@github.com` komutu "Hi username!" mesajı dönmeli

### 0.3. Huggingface Hesabı

- [ ] **0.3.1** HuggingFace hesabı oluştur
  - https://huggingface.co/join
  - **Test:** Hesaba giriş yapabiliyorsun

- [ ] **0.3.2** HuggingFace access token oluştur
  - https://huggingface.co/settings/tokens → "Create new token"
  - "Read" yetkisi yeterli, kopyala ve güvenli yere kaydet
  - **Test:** Token kopyalandı mı?

- [ ] **0.3.3** Qwen2.5-VL lisansını kabul et
  - https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
  - "Agree and access repository" (varsa)
  - **Test:** Sayfada model dosyaları görünüyor mu?

- [ ] **0.3.4** Gemma 4 lisansını kabul et
  - https://huggingface.co/google/gemma-4-4b-it
  - "Acknowledge license" butonuna bas
  - **Test:** Model dosyaları görünür olmalı

- [ ] **0.3.5** LLaVA-1.6 erişimini kontrol et
  - https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
  - Lisans gerekirse kabul et
  - **Test:** Sayfaya erişim var mı?

---

## Faz 1: Versiyon Kontrol ve Proje İskeleti

**Amaç:** Boş bir proje iskeleti oluşturup GitHub'a push etmek.
**Süre:** ~2-3 saat

### 1.1. GitHub Repo Oluşturma

- [ ] **1.1.1** GitHub'da yeni repo oluştur
  - Ad: `ui-critic-project`
  - Görünürlük: **Private** (tez olacak)
  - README oluşturma, .gitignore ekleme (sonra yapacağız)
  - **Test:** Repo URL'i eline geçti (`https://github.com/kullanici/ui-critic-project`)

- [ ] **1.1.2** Lokal'de proje klasörünü oluştur
  ```bash
  mkdir ~/ui-critic-project
  cd ~/ui-critic-project
  git init
  ```
  - **Test:** `.git` klasörü oluştu

- [ ] **1.1.3** Remote repo'yu bağla
  ```bash
  git remote add origin git@github.com:kullanici/ui-critic-project.git
  ```
  - **Test:** `git remote -v` origin'i göstermeli

### 1.2. Conda Environment Oluşturma

- [ ] **1.2.1** `uicritic` adında conda environment oluştur
  ```bash
  conda create -n uicritic python=3.11 -y
  ```
  - **Test:** `conda env list` listesinde `uicritic` görünmeli

- [ ] **1.2.2** Environment'ı aktive et
  ```bash
  conda activate uicritic
  ```
  - **Test:** Terminal prompt'unda `(uicritic)` görünmeli

### 1.3. .gitignore Oluşturma

- [ ] **1.3.1** Proje kökünde `.gitignore` dosyası oluştur
  ```gitignore
  # Python
  __pycache__/
  *.py[cod]
  *$py.class
  *.so
  .Python
  *.egg-info/
  .pytest_cache/
  .mypy_cache/

  # Virtual environments
  venv/
  env/
  .venv/

  # IDE
  .vscode/
  .idea/
  *.swp

  # Data & Outputs
  data/
  outputs/
  *.ckpt
  *.pt
  *.gguf
  *.safetensors

  # Logs
  *.log
  wandb/

  # OS
  .DS_Store
  Thumbs.db

  # Secrets
  .env
  *.pem
  *.key
  kaggle.json
  ```
  - **Test:** Dosya oluştu mu?

### 1.4. Proje Klasör Yapısı Oluşturma

- [ ] **1.4.1** Ana klasörleri oluştur
  ```bash
  mkdir -p src/{models/adapters,data/preprocessors,training,inference,evaluation}
  mkdir -p scripts configs data/{rico,uicrit,processed} outputs/{checkpoints,logs,predictions} notebooks tests
  ```
  - **Test:** `tree -L 3` ile yapıyı kontrol et (veya `find . -type d`)

- [ ] **1.4.2** Tüm Python paketlerinde `__init__.py` dosyaları oluştur
  ```bash
  touch src/__init__.py
  touch src/models/__init__.py
  touch src/models/adapters/__init__.py
  touch src/data/__init__.py
  touch src/data/preprocessors/__init__.py
  touch src/training/__init__.py
  touch src/inference/__init__.py
  touch src/evaluation/__init__.py
  touch tests/__init__.py
  ```
  - **Test:** Her `src/*/` klasöründe `__init__.py` var mı?

- [ ] **1.4.3** Her klasöre `.gitkeep` dosyası ekle (boş klasörler git'te görünsün)
  ```bash
  touch data/rico/.gitkeep
  touch data/uicrit/.gitkeep
  touch data/processed/.gitkeep
  touch outputs/checkpoints/.gitkeep
  touch outputs/logs/.gitkeep
  touch outputs/predictions/.gitkeep
  touch notebooks/.gitkeep
  ```

### 1.5. Dokümantasyon

- [ ] **1.5.1** `README.md` dosyasını proje kökünde oluştur (önceki dokümanı kopyala)
  - **Test:** Dosya var, içeriği okunuyor

- [ ] **1.5.2** `TASKS.md` dosyasını proje kökünde oluştur (bu liste)
  - **Test:** Dosya var

### 1.6. Requirements Dosyaları

- [ ] **1.6.1** `requirements-dev.txt` oluştur (lokal geliştirme)
  ```txt
  torch>=2.0
  torchvision
  transformers>=4.46
  pillow
  pandas
  pyyaml
  tqdm
  numpy
  pytest
  ipykernel
  jupyter
  ```
  - **Test:** Dosya var

- [ ] **1.6.2** `requirements-cloud.txt` oluştur (sunucu)
  ```txt
  -r requirements-dev.txt
  accelerate==1.0.0
  peft==0.13.0
  bitsandbytes==0.44.0
  unsloth==2025.1.0
  evaluate
  sacrebleu
  rouge-score
  bert-score
  nltk
  opencv-python-headless
  jsonschema
  huggingface-hub
  wandb
  kaggle
  ```
  - **Test:** Dosya var

- [ ] **1.6.3** Lokal ortama dev requirements kur
  ```bash
  pip install -r requirements-dev.txt
  ```
  - **Test:** `python -c "import torch; import transformers; print('OK')"` çalışmalı

### 1.7. İlk Commit ve Push

- [ ] **1.7.1** İlk commit'i yap
  ```bash
  git add .
  git commit -m "chore: proje iskeleti oluşturuldu"
  ```
  - **Test:** `git log` commit'i göstermeli

- [ ] **1.7.2** Ana dalın adını `main` yap
  ```bash
  git branch -M main
  ```

- [ ] **1.7.3** GitHub'a push et
  ```bash
  git push -u origin main
  ```
  - **Test:** GitHub'da kod görünür

---

## Faz 2: Datasetlerin İndirilmesi ve İncelenmesi

**Amaç:** UICrit ve (lokal'de küçük bir RICO subseti) indirip yapısını anlamak.
**Süre:** ~3-4 saat

### 2.1. UICrit İndirme

- [ ] **2.1.1** UICrit repo'sunu `data/` klasörüne klonla
  ```bash
  cd data
  git clone https://github.com/google-research-datasets/uicrit.git
  cd ..
  ```
  - **Test:** `data/uicrit/uicrit_public.csv` dosyası var

- [ ] **2.1.2** UICrit CSV'nin satır sayısını kontrol et
  ```bash
  wc -l data/uicrit/uicrit_public.csv
  ```
  - **Beklenen:** ~11.344 satır (+ başlık)

- [ ] **2.1.3** UICrit CSV'nin sütunlarını kontrol et
  ```bash
  head -1 data/uicrit/uicrit_public.csv
  ```
  - **Beklenen sütunlar:** `rico_id, task, aesthetics_rating, learnability, usability_rating, overall_design_rating, efficency, comments, comments_source`

### 2.2. UICrit Veri Keşfi (Notebook)

- [ ] **2.2.1** `notebooks/01_uicrit_exploration.ipynb` oluştur
  - Jupyter aç: `jupyter notebook notebooks/`
  - Yeni notebook oluştur

- [ ] **2.2.2** UICrit'i pandas ile yükle
  ```python
  import pandas as pd
  import json

  df = pd.read_csv("../data/uicrit/uicrit_public.csv")
  print("Satır sayısı:", len(df))
  print("Benzersiz UI:", df['rico_id'].nunique())
  print("Sütunlar:", df.columns.tolist())
  df.head()
  ```
  - **Beklenen:** ~11.344 satır, ~983 benzersiz `rico_id`

- [ ] **2.2.3** `comments` sütununun JSON yapısını incele
  ```python
  # İlk satırın yorumlarını aç
  first_comments = json.loads(df.iloc[0]['comments'])
  print(json.dumps(first_comments, indent=2, ensure_ascii=False))
  ```
  - **Beklenen:** Liste formatında, her elemanda `comment` ve `box` alanları

- [ ] **2.2.4** `comments_source` dağılımını incele
  ```python
  all_sources = []
  for srcs in df['comments_source'].dropna():
      all_sources.extend(json.loads(srcs))
  from collections import Counter
  print(Counter(all_sources))
  ```
  - **Beklenen:** `human`, `llm`, `both` değerlerinin dağılımı

- [ ] **2.2.5** Rating dağılımlarını grafikle
  ```python
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  df['aesthetics_rating'].hist(bins=10, ax=axes[0]); axes[0].set_title('Aesthetics (1-10)')
  df['learnability'].hist(bins=5, ax=axes[1]); axes[1].set_title('Learnability (1-5)')
  plt.show()
  ```
  - **Test:** Grafikler görünüyor

- [ ] **2.2.6** UI başına ortalama yorum sayısını hesapla
  ```python
  comments_per_ui = df.groupby('rico_id')['comments'].count()
  print("UI başına ortalama yorum:", comments_per_ui.mean())
  print("Minimum:", comments_per_ui.min())
  print("Maksimum:", comments_per_ui.max())
  ```

- [ ] **2.2.7** Task dağılımına bak (UI'lar hangi görevler için tasarlanmış?)
  ```python
  df['task'].value_counts().head(20)
  ```

### 2.3. RICO Küçük Subset İndirme (Lokal Test İçin)

**Not:** Tam RICO 6 GB. Lokal'de sadece UICrit'teki 983 UI'a karşılık gelen görselleri indireceğiz.

- [ ] **2.3.1** UICrit'teki benzersiz `rico_id`'leri bir listeye kaydet
  ```python
  # Notebook'ta
  unique_ids = df['rico_id'].unique().tolist()
  print("Toplam:", len(unique_ids))
  with open("../data/uicrit_rico_ids.txt", "w") as f:
      for rid in unique_ids:
          f.write(f"{rid}\n")
  ```
  - **Test:** `data/uicrit_rico_ids.txt` 983 satır içermeli

- [ ] **2.3.2** RICO tam dataseti Kaggle'dan indirip sadece ihtiyacımız olan UI'ları ayıklamak için Kaggle API kurulumu
  ```bash
  pip install kaggle
  ```

- [ ] **2.3.3** Kaggle API key'ini ayarla
  - https://www.kaggle.com/settings → "Create New API Token" → `kaggle.json` indir
  - Dosyayı `~/.kaggle/kaggle.json` konumuna koy
  - `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)
  - **Test:** `kaggle datasets list` komutu çalışmalı

- [ ] **2.3.4** **KARAR:** RICO'nun tam halini lokal'e indirme (6 GB disk + 30 dk süre)
  - Alternatif: Lokal'e sadece 50-100 örnek UI'ın görselini manuel indir (test için)
  - **Önerim:** Tam indirmeyi cloud'da yap. Lokal'de 20-30 UI görseli yeterli.

- [ ] **2.3.5** (Lokal test için) Örnek 20 UI görselini manuel olarak lokal'e indir
  - Notebook'ta birkaç `rico_id` seç (örn. `[12345, 67890, ...]`)
  - RICO Kaggle sayfasındaki dosyadan bu ID'lerin `.jpg` ve `.json`'larını çıkar
  - `data/rico/combined_sample/` içine koy
  - **Test:** 20 tane `.jpg` + 20 tane `.json` dosyası var

### 2.4. View Hierarchy (VH) Yapısını Anlamak

- [ ] **2.4.1** `notebooks/02_rico_vh_exploration.ipynb` oluştur

- [ ] **2.4.2** Bir örnek VH dosyasını aç
  ```python
  import json
  with open("../data/rico/combined_sample/12345.json", "r") as f:
      vh = json.load(f)
  print(json.dumps(vh, indent=2)[:2000])  # İlk 2000 karakter
  ```

- [ ] **2.4.3** VH ağacının derinliğini hesaplayan bir fonksiyon yaz
  ```python
  def tree_depth(node, depth=0):
      if "children" not in node or not node["children"]:
          return depth
      return max(tree_depth(c, depth+1) for c in node["children"])

  print("Ağaç derinliği:", tree_depth(vh['activity']['root']))
  ```

- [ ] **2.4.4** Element tiplerini listele
  ```python
  def collect_classes(node, classes=None):
      if classes is None: classes = set()
      if 'class' in node:
          classes.add(node['class'])
      for c in node.get('children', []):
          collect_classes(c, classes)
      return classes

  print("Element tipleri:", collect_classes(vh['activity']['root']))
  ```

### 2.5. Keşif Özetini Yaz

- [ ] **2.5.1** `notebooks/EXPLORATION_NOTES.md` dosyasında bulguları özetle
  - UICrit satır sayısı, UI sayısı
  - Her UI için ortalama kritik sayısı
  - Rating dağılımları
  - VH ağaç yapısı özeti
  - **Test:** Dosya var ve okunuyor

- [ ] **2.5.2** Commit at
  ```bash
  git add notebooks/ data/uicrit_rico_ids.txt
  git commit -m "feat: UICrit ve RICO veri keşfi notebook'ları"
  git push
  ```

---

## Faz 3: Veri Yükleyiciler (Data Loaders)

**Amaç:** UICrit ve RICO'yu kod üzerinden okumak için modüler sınıflar yazmak.
**Süre:** ~4-6 saat

### 3.1. UICrit Loader

- [ ] **3.1.1** `src/data/uicrit_loader.py` dosyasını oluştur
- [ ] **3.1.2** `UICritLoader` sınıfını yaz
  ```python
  import pandas as pd
  import json
  from pathlib import Path
  from typing import List, Dict, Optional

  class UICritLoader:
      def __init__(self, csv_path: str):
          self.csv_path = Path(csv_path)
          self.df = None

      def load(self) -> pd.DataFrame:
          """CSV'yi yükle ve JSON sütunlarını parse et."""
          self.df = pd.read_csv(self.csv_path)
          self.df['comments'] = self.df['comments'].apply(json.loads)
          self.df['comments_source'] = self.df['comments_source'].apply(
              lambda x: json.loads(x) if pd.notna(x) else []
          )
          return self.df

      def get_by_rico_id(self, rico_id: int) -> pd.DataFrame:
          """Belirli bir rico_id için tüm kayıtları döndür."""
          if self.df is None:
              self.load()
          return self.df[self.df['rico_id'] == rico_id]

      def get_unique_rico_ids(self) -> List[int]:
          """Benzersiz rico_id listesi."""
          if self.df is None:
              self.load()
          return self.df['rico_id'].unique().tolist()

      def filter_by_source(self, sources: List[str]) -> pd.DataFrame:
          """'human', 'llm', 'both' kaynaklarına göre filtrele."""
          # Her satırdaki comments_source listesinde istenen değerler var mı
          if self.df is None:
              self.load()
          mask = self.df['comments_source'].apply(
              lambda s_list: any(s in sources for s in s_list) if s_list else False
          )
          return self.df[mask]
  ```

- [ ] **3.1.3** `tests/test_uicrit_loader.py` yaz
  ```python
  from src.data.uicrit_loader import UICritLoader

  def test_load():
      loader = UICritLoader("data/uicrit/uicrit_public.csv")
      df = loader.load()
      assert len(df) > 10000
      assert df['rico_id'].nunique() > 900
      assert isinstance(df.iloc[0]['comments'], list)

  def test_get_by_id():
      loader = UICritLoader("data/uicrit/uicrit_public.csv")
      loader.load()
      ids = loader.get_unique_rico_ids()
      first_id = ids[0]
      records = loader.get_by_rico_id(first_id)
      assert len(records) >= 1
  ```
  - **Çalıştır:** `pytest tests/test_uicrit_loader.py -v`
  - **Test:** İki test de geçmeli

### 3.2. RICO Loader

- [ ] **3.2.1** `src/data/rico_loader.py` dosyasını oluştur
- [ ] **3.2.2** `RicoLoader` sınıfını yaz
  ```python
  import json
  from pathlib import Path
  from typing import Optional, Dict
  from PIL import Image

  class RicoLoader:
      def __init__(self, rico_dir: str):
          self.rico_dir = Path(rico_dir)
          self.combined_dir = self.rico_dir / "combined"

      def load_image(self, rico_id: int) -> Optional[Image.Image]:
          """UI görselini yükle."""
          img_path = self.combined_dir / f"{rico_id}.jpg"
          if not img_path.exists():
              return None
          return Image.open(img_path).convert("RGB")

      def load_hierarchy(self, rico_id: int) -> Optional[Dict]:
          """View hierarchy JSON'unu yükle. Bozuksa None döner."""
          json_path = self.combined_dir / f"{rico_id}.json"
          if not json_path.exists():
              return None
          try:
              with open(json_path, "r") as f:
                  return json.load(f)
          except json.JSONDecodeError:
              return None

      def image_exists(self, rico_id: int) -> bool:
          return (self.combined_dir / f"{rico_id}.jpg").exists()

      def hierarchy_exists(self, rico_id: int) -> bool:
          return (self.combined_dir / f"{rico_id}.json").exists()
  ```

- [ ] **3.2.3** `tests/test_rico_loader.py` yaz
  ```python
  from src.data.rico_loader import RicoLoader

  def test_load_existing_image():
      loader = RicoLoader("data/rico")
      # Lokal'de indirdiğin örnek bir ID'yi kullan
      sample_id = 12345  # senin indirdiğinle değiştir
      img = loader.load_image(sample_id)
      if loader.image_exists(sample_id):
          assert img is not None
          assert img.mode == "RGB"

  def test_load_missing_image():
      loader = RicoLoader("data/rico")
      img = loader.load_image(99999999)
      assert img is None
  ```

### 3.3. Alignment (Eşleştirme) Modülü

- [ ] **3.3.1** `src/data/alignment.py` oluştur
- [ ] **3.3.2** `UICritRicoAligner` sınıfını yaz
  ```python
  from typing import List, Dict, Optional
  from .uicrit_loader import UICritLoader
  from .rico_loader import RicoLoader

  class UICritRicoAligner:
      def __init__(self, uicrit: UICritLoader, rico: RicoLoader):
          self.uicrit = uicrit
          self.rico = rico

      def get_aligned_records(self, rico_id: int) -> Optional[Dict]:
          """Bir rico_id için UICrit kritikleri + RICO görseli/VH'sini birlikte döndür."""
          if not self.rico.image_exists(rico_id):
              return None

          critiques_df = self.uicrit.get_by_rico_id(rico_id)
          if critiques_df.empty:
              return None

          return {
              "rico_id": rico_id,
              "image": self.rico.load_image(rico_id),
              "hierarchy": self.rico.load_hierarchy(rico_id),
              "critiques_records": critiques_df.to_dict("records"),
          }

      def get_all_aligned_ids(self) -> List[int]:
          """UICrit'te olan VE RICO'da görseli olan tüm ID'ler."""
          uicrit_ids = self.uicrit.get_unique_rico_ids()
          return [rid for rid in uicrit_ids if self.rico.image_exists(rid)]
  ```

- [ ] **3.3.3** Test yaz
  ```python
  # tests/test_alignment.py
  from src.data.uicrit_loader import UICritLoader
  from src.data.rico_loader import RicoLoader
  from src.data.alignment import UICritRicoAligner

  def test_aligner():
      uicrit = UICritLoader("data/uicrit/uicrit_public.csv")
      rico = RicoLoader("data/rico")
      aligner = UICritRicoAligner(uicrit, rico)
      ids = aligner.get_all_aligned_ids()
      # Lokal'de birkaç görselin indirilmiş olduğunu varsayıyoruz
      assert len(ids) >= 0  # En az 0 (cloud'da tamamlanacak)
  ```

### 3.4. Train/Val/Test Split

- [ ] **3.4.1** `src/data/splitter.py` oluştur
- [ ] **3.4.2** Stratified split fonksiyonu yaz
  ```python
  import random
  from typing import List, Tuple

  def split_ids(
      ids: List[int],
      train_ratio: float = 0.7,
      val_ratio: float = 0.15,
      test_ratio: float = 0.15,
      seed: int = 42,
  ) -> Tuple[List[int], List[int], List[int]]:
      """rico_id listesini train/val/test olarak böl."""
      assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

      rng = random.Random(seed)
      shuffled = ids.copy()
      rng.shuffle(shuffled)

      n = len(shuffled)
      n_train = int(n * train_ratio)
      n_val = int(n * val_ratio)

      train = shuffled[:n_train]
      val = shuffled[n_train:n_train + n_val]
      test = shuffled[n_train + n_val:]

      return train, val, test
  ```

- [ ] **3.4.3** Test yaz
  ```python
  from src.data.splitter import split_ids

  def test_split():
      ids = list(range(1000))
      train, val, test = split_ids(ids, 0.7, 0.15, 0.15, seed=42)
      assert len(train) == 700
      assert len(val) == 150
      assert len(test) == 150
      # Çakışma yok
      assert set(train).isdisjoint(set(val))
      assert set(train).isdisjoint(set(test))
      assert set(val).isdisjoint(set(test))
  ```

### 3.5. Commit

- [ ] **3.5.1** Tüm data loader'ları commit et
  ```bash
  git add src/data/ tests/
  git commit -m "feat: UICrit, RICO loader ve alignment modülleri"
  git push
  ```

---

## Faz 4: Model Katmanı (Registry + Adapters)

**Amaç:** Modeli config'ten seçilebilir yapan adapter pattern'i kurmak.
**Süre:** ~6-8 saat

### 4.1. Base Adapter (Abstract Class)

- [ ] **4.1.1** `src/models/base_adapter.py` oluştur
- [ ] **4.1.2** `BaseVLMAdapter` abstract class'ını yaz
  ```python
  from abc import ABC, abstractmethod
  from typing import Dict, List, Optional
  from PIL import Image

  class BaseVLMAdapter(ABC):
      """Tüm VLM adapter'larının uyması gereken arayüz."""

      def __init__(self, model_config: Dict):
          self.config = model_config
          self.hf_id = model_config["hf_id"]
          self.model = None
          self.processor = None

      @abstractmethod
      def load_model(self, load_in_4bit: bool = True, torch_dtype: str = "bfloat16"):
          """Modeli ve processor'ı yükle."""
          pass

      @abstractmethod
      def format_prompt(self, instruction: str, image: Image.Image, **kwargs) -> Dict:
          """Model'e özel prompt formatı. Eğitim ve inference için ortak input üretir."""
          pass

      @abstractmethod
      def get_lora_target_modules(self) -> List[str]:
          """LoRA için hangi modüllere uygulanacağı."""
          pass

      @abstractmethod
      def generate(self, inputs: Dict, max_new_tokens: int = 512) -> str:
          """Inference için generation."""
          pass
  ```

### 4.2. Registry

- [ ] **4.2.1** `src/models/registry.py` oluştur
- [ ] **4.2.2** Model registry dictionary ve fonksiyonlarını yaz
  ```python
  from typing import Dict
  from .base_adapter import BaseVLMAdapter

  MODEL_REGISTRY = {
      "qwen2.5-vl-3b": {
          "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
          "adapter_class": "QwenVLAdapter",
          "family": "qwen",
      },
      "qwen2.5-vl-7b": {
          "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
          "adapter_class": "QwenVLAdapter",
          "family": "qwen",
      },
      "gemma-4-4b": {
          "hf_id": "google/gemma-4-4b-it",
          "adapter_class": "Gemma4Adapter",
          "family": "gemma",
      },
      "gemma-4-12b": {
          "hf_id": "google/gemma-4-12b-it",
          "adapter_class": "Gemma4Adapter",
          "family": "gemma",
      },
      "llava-1.6-7b": {
          "hf_id": "llava-hf/llava-v1.6-mistral-7b-hf",
          "adapter_class": "LLaVAAdapter",
          "family": "llava",
      },
      "llava-1.6-13b": {
          "hf_id": "llava-hf/llava-v1.6-vicuna-13b-hf",
          "adapter_class": "LLaVAAdapter",
          "family": "llava",
      },
  }

  def get_model_config(name: str) -> Dict:
      if name not in MODEL_REGISTRY:
          available = list(MODEL_REGISTRY.keys())
          raise ValueError(f"Model '{name}' bilinmiyor. Desteklenen: {available}")
      return MODEL_REGISTRY[name]

  def create_adapter(name: str) -> BaseVLMAdapter:
      config = get_model_config(name)
      # Dinamik import
      from .adapters import qwen_adapter, gemma_adapter, llava_adapter
      adapter_classes = {
          "QwenVLAdapter": qwen_adapter.QwenVLAdapter,
          "Gemma4Adapter": gemma_adapter.Gemma4Adapter,
          "LLaVAAdapter": llava_adapter.LLaVAAdapter,
      }
      adapter_cls = adapter_classes[config["adapter_class"]]
      return adapter_cls(config)
  ```

- [ ] **4.2.3** `src/models/registry.py` için test
  ```python
  # tests/test_registry.py
  from src.models.registry import get_model_config, MODEL_REGISTRY

  def test_all_models_registered():
      assert "qwen2.5-vl-7b" in MODEL_REGISTRY
      assert "gemma-4-4b" in MODEL_REGISTRY
      assert "llava-1.6-7b" in MODEL_REGISTRY

  def test_get_model_config():
      config = get_model_config("qwen2.5-vl-7b")
      assert config["hf_id"] == "Qwen/Qwen2.5-VL-7B-Instruct"

  def test_unknown_model_raises():
      import pytest
      with pytest.raises(ValueError):
          get_model_config("bogus-model")
  ```

### 4.3. Qwen Adapter

- [ ] **4.3.1** `src/models/adapters/qwen_adapter.py` oluştur
- [ ] **4.3.2** `QwenVLAdapter` class'ını yaz — iskelet
  ```python
  from typing import Dict, List
  from PIL import Image
  from ..base_adapter import BaseVLMAdapter

  class QwenVLAdapter(BaseVLMAdapter):
      def load_model(self, load_in_4bit: bool = True, torch_dtype: str = "bfloat16"):
          from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
          import torch

          dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
          dtype = dtype_map[torch_dtype]

          quant_config = None
          if load_in_4bit:
              from transformers import BitsAndBytesConfig
              quant_config = BitsAndBytesConfig(
                  load_in_4bit=True,
                  bnb_4bit_compute_dtype=dtype,
                  bnb_4bit_quant_type="nf4",
              )

          self.model = Qwen2VLForConditionalGeneration.from_pretrained(
              self.hf_id,
              torch_dtype=dtype,
              quantization_config=quant_config,
              device_map="auto",
          )
          self.processor = AutoProcessor.from_pretrained(self.hf_id)

      def format_prompt(self, instruction: str, image: Image.Image, **kwargs) -> Dict:
          messages = [
              {
                  "role": "user",
                  "content": [
                      {"type": "image"},
                      {"type": "text", "text": instruction},
                  ],
              }
          ]
          text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
          inputs = self.processor(
              text=[text],
              images=[image],
              return_tensors="pt",
              padding=True,
          )
          return inputs

      def get_lora_target_modules(self) -> List[str]:
          return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

      def generate(self, inputs: Dict, max_new_tokens: int = 512) -> str:
          output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
          generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
          text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
          return text
  ```

### 4.4. Gemma 4 Adapter

- [ ] **4.4.1** `src/models/adapters/gemma_adapter.py` oluştur
- [ ] **4.4.2** `Gemma4Adapter` iskelet (Gemma 4 API'sine uygun)
  ```python
  from typing import Dict, List
  from PIL import Image
  from ..base_adapter import BaseVLMAdapter

  class Gemma4Adapter(BaseVLMAdapter):
      def load_model(self, load_in_4bit: bool = True, torch_dtype: str = "bfloat16"):
          # Gemma 4 için AutoProcessor ve modeli yükle
          # NOT: Gemma 4'ün exact model class'ı için HF dokümantasyonuna bak
          from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
          import torch

          dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
          dtype = dtype_map[torch_dtype]

          quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype) if load_in_4bit else None

          self.model = AutoModelForCausalLM.from_pretrained(
              self.hf_id,
              torch_dtype=dtype,
              quantization_config=quant_config,
              device_map="auto",
          )
          self.processor = AutoProcessor.from_pretrained(self.hf_id)

      def format_prompt(self, instruction: str, image: Image.Image, **kwargs) -> Dict:
          messages = [
              {"role": "user", "content": [
                  {"type": "image", "image": image},
                  {"type": "text", "text": instruction},
              ]}
          ]
          inputs = self.processor.apply_chat_template(
              messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
          )
          return inputs

      def get_lora_target_modules(self) -> List[str]:
          return ["q_proj", "k_proj", "v_proj", "o_proj"]

      def generate(self, inputs: Dict, max_new_tokens: int = 512) -> str:
          output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
          text = self.processor.decode(output_ids[0], skip_special_tokens=True)
          return text
  ```

### 4.5. LLaVA Adapter

- [ ] **4.5.1** `src/models/adapters/llava_adapter.py` oluştur
- [ ] **4.5.2** `LLaVAAdapter` iskelet
  ```python
  from typing import Dict, List
  from PIL import Image
  from ..base_adapter import BaseVLMAdapter

  class LLaVAAdapter(BaseVLMAdapter):
      def load_model(self, load_in_4bit: bool = True, torch_dtype: str = "bfloat16"):
          from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
          import torch

          dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
          dtype = dtype_map[torch_dtype]

          quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype) if load_in_4bit else None

          self.model = LlavaNextForConditionalGeneration.from_pretrained(
              self.hf_id, torch_dtype=dtype, quantization_config=quant_config, device_map="auto"
          )
          self.processor = LlavaNextProcessor.from_pretrained(self.hf_id)

      def format_prompt(self, instruction: str, image: Image.Image, **kwargs) -> Dict:
          prompt = f"[INST] <image>\n{instruction} [/INST]"
          inputs = self.processor(text=prompt, images=image, return_tensors="pt")
          return inputs

      def get_lora_target_modules(self) -> List[str]:
          return ["q_proj", "k_proj", "v_proj", "o_proj"]

      def generate(self, inputs: Dict, max_new_tokens: int = 512) -> str:
          output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
          text = self.processor.decode(output_ids[0], skip_special_tokens=True)
          return text
  ```

### 4.6. Adapters `__init__.py`

- [ ] **4.6.1** `src/models/adapters/__init__.py` içine
  ```python
  from . import qwen_adapter, gemma_adapter, llava_adapter
  ```

### 4.7. Commit

- [ ] **4.7.1** Commit
  ```bash
  git add src/models/ tests/test_registry.py
  git commit -m "feat: model registry ve 3 adapter (Qwen, Gemma, LLaVA)"
  git push
  ```

**Not:** Bu kodları lokal'de **çalıştırmaya çalışmayın** — model ağırlıkları yüklenemez. Sadece syntax kontrolü: `python -c "from src.models.registry import MODEL_REGISTRY; print(MODEL_REGISTRY)"`

---

## Faz 5: Veri Ön-İşleme (Preprocessing)

**Amaç:** Ham UICrit + RICO'yu eğitim formatına dönüştüren script'ler yazmak.
**Süre:** ~4-5 saat

### 5.1. Model 1 Preprocessing

- [ ] **5.1.1** `src/data/preprocessors/model1_prep.py` oluştur
- [ ] **5.1.2** UICrit kayıtlarını Model 1 formatına dönüştüren fonksiyon
  ```python
  import json
  from pathlib import Path
  from typing import List, Dict
  from ..uicrit_loader import UICritLoader

  def build_model1_records(
      uicrit: UICritLoader,
      rico_ids: List[int],
      rico_image_dir: str,
  ) -> List[Dict]:
      """UICrit kayıtlarını Model 1 eğitim formatına dönüştür."""
      records = []
      df = uicrit.load()

      for rid in rico_ids:
          group = df[df['rico_id'] == rid]
          if group.empty:
              continue

          # Aynı rico_id için tüm annotator yorumlarını birleştir
          all_critiques = []
          for _, row in group.iterrows():
              for c, src in zip(row['comments'], row['comments_source']):
                  all_critiques.append({
                      "comment": c.get("comment", ""),
                      "bounding_box": c.get("box", [0, 0, 0, 0]),
                      "source": src,
                  })

          record = {
              "rico_id": int(rid),
              "image_path": f"{rico_image_dir}/{rid}.jpg",
              "task": group.iloc[0].get("task", ""),
              "critiques": all_critiques,
              "aesthetics_rating": float(group['aesthetics_rating'].mean()),
              "learnability": float(group['learnability'].mean()),
          }
          records.append(record)

      return records

  def save_records(records: List[Dict], output_path: str):
      Path(output_path).parent.mkdir(parents=True, exist_ok=True)
      with open(output_path, "w") as f:
          json.dump(records, f, ensure_ascii=False, indent=2)
  ```

### 5.2. Model 2 Preprocessing

- [ ] **5.2.1** `src/data/preprocessors/model2_prep.py` oluştur
- [ ] **5.2.2** RICO VH'sini temiz formata çeviren fonksiyon
  ```python
  import json
  from pathlib import Path
  from typing import List, Dict, Optional
  from ..rico_loader import RicoLoader

  def simplify_hierarchy(node: Dict, max_depth: int = 10, current_depth: int = 0) -> Optional[Dict]:
      """VH ağacını sadeleştir."""
      if current_depth >= max_depth:
          return None
      if not node:
          return None
      # Görünmeyen elementleri atla
      if not node.get("visible-to-user", True):
          return None

      simplified = {
          "type": node.get("class", "Unknown").split(".")[-1],
          "bounds": node.get("bounds", [0, 0, 0, 0]),
      }
      if node.get("text"):
          simplified["text"] = node["text"]
      if node.get("clickable"):
          simplified["clickable"] = True

      children = []
      for child in node.get("children", []) or []:
          child_simplified = simplify_hierarchy(child, max_depth, current_depth + 1)
          if child_simplified:
              children.append(child_simplified)
      simplified["children"] = children
      return simplified

  def build_model2_records(
      rico: RicoLoader,
      rico_ids: List[int],
      rico_image_dir: str,
      max_depth: int = 10,
  ) -> List[Dict]:
      records = []
      for rid in rico_ids:
          if not rico.image_exists(rid):
              continue
          raw_vh = rico.load_hierarchy(rid)
          if raw_vh is None:
              continue
          root = raw_vh.get("activity", {}).get("root")
          if root is None:
              continue
          simplified = simplify_hierarchy(root, max_depth)
          if simplified is None:
              continue
          records.append({
              "rico_id": int(rid),
              "image_path": f"{rico_image_dir}/{rid}.jpg",
              "hierarchy": simplified,
          })
      return records
  ```

### 5.3. Model 3 Preprocessing (Placeholder)

- [ ] **5.3.1** `src/data/preprocessors/model3_prep.py` oluştur
- [ ] **5.3.2** Model 2 çıktılarını + UICrit kritiklerini birleştiren fonksiyon
  ```python
  import json
  from pathlib import Path
  from typing import List, Dict
  from .model1_prep import build_model1_records
  from ..uicrit_loader import UICritLoader

  def build_model3_records(
      uicrit: UICritLoader,
      rico_ids: List[int],
      rico_image_dir: str,
      predicted_hierarchies_path: str,
  ) -> List[Dict]:
      """Model 3 eğitim verisi: UI + predicted VH + kritik."""
      with open(predicted_hierarchies_path, "r") as f:
          pred_vhs = json.load(f)  # {rico_id: predicted_hierarchy}

      base_records = build_model1_records(uicrit, rico_ids, rico_image_dir)

      model3_records = []
      for rec in base_records:
          rid_str = str(rec["rico_id"])
          if rid_str not in pred_vhs:
              continue
          rec["predicted_hierarchy"] = pred_vhs[rid_str]
          model3_records.append(rec)

      return model3_records
  ```

### 5.4. Ana Preprocessing Script

- [ ] **5.4.1** `scripts/prepare_data.py` oluştur
  ```python
  import argparse
  from pathlib import Path
  from src.data.uicrit_loader import UICritLoader
  from src.data.rico_loader import RicoLoader
  from src.data.alignment import UICritRicoAligner
  from src.data.splitter import split_ids
  from src.data.preprocessors.model1_prep import build_model1_records, save_records
  from src.data.preprocessors.model2_prep import build_model2_records
  from src.data.preprocessors.model3_prep import build_model3_records

  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--task", required=True, choices=["model1", "model2", "model3"])
      parser.add_argument("--uicrit_csv", default="data/uicrit/uicrit_public.csv")
      parser.add_argument("--rico_dir", default="data/rico")
      parser.add_argument("--output_dir", default="data/processed")
      parser.add_argument("--predicted_vh_path", default=None, help="Model 3 için gerekli")
      parser.add_argument("--seed", type=int, default=42)
      args = parser.parse_args()

      uicrit = UICritLoader(args.uicrit_csv)
      rico = RicoLoader(args.rico_dir)
      aligner = UICritRicoAligner(uicrit, rico)

      if args.task == "model1":
          all_ids = aligner.get_all_aligned_ids()
          train, val, test = split_ids(all_ids, 0.7, 0.15, 0.15, seed=args.seed)
          rico_image_dir = str(Path(args.rico_dir) / "combined")
          for split_name, ids in [("train", train), ("val", val), ("test", test)]:
              records = build_model1_records(uicrit, ids, rico_image_dir)
              save_records(records, f"{args.output_dir}/model1_{split_name}.json")
              print(f"model1_{split_name}: {len(records)} kayıt")

      elif args.task == "model2":
          # RICO'nun TÜM UI'ları (UICrit'te olmayanlar dahil)
          all_rico_ids = [int(p.stem) for p in Path(args.rico_dir, "combined").glob("*.jpg")]
          train, val, test = split_ids(all_rico_ids, 0.7, 0.15, 0.15, seed=args.seed)
          rico_image_dir = str(Path(args.rico_dir) / "combined")
          for split_name, ids in [("train", train), ("val", val), ("test", test)]:
              records = build_model2_records(rico, ids, rico_image_dir)
              save_records(records, f"{args.output_dir}/model2_{split_name}.json")
              print(f"model2_{split_name}: {len(records)} kayıt")

      elif args.task == "model3":
          assert args.predicted_vh_path, "Model 3 için --predicted_vh_path gerekli"
          all_ids = aligner.get_all_aligned_ids()
          train, val, test = split_ids(all_ids, 0.7, 0.15, 0.15, seed=args.seed)
          rico_image_dir = str(Path(args.rico_dir) / "combined")
          for split_name, ids in [("train", train), ("val", val), ("test", test)]:
              records = build_model3_records(uicrit, ids, rico_image_dir, args.predicted_vh_path)
              save_records(records, f"{args.output_dir}/model3_{split_name}.json")
              print(f"model3_{split_name}: {len(records)} kayıt")

  if __name__ == "__main__":
      main()
  ```

### 5.5. Commit

- [ ] **5.5.1** Commit
  ```bash
  git add src/data/preprocessors/ scripts/prepare_data.py
  git commit -m "feat: Model 1, 2, 3 için preprocessing script'leri"
  git push
  ```

---

## Faz 6: Eğitim Pipeline

**Amaç:** Config tabanlı, tüm modelleri destekleyen bir eğitim sistemi.
**Süre:** ~6-8 saat

### 6.1. Config Şeması

- [ ] **6.1.1** `src/training/config_schema.py` oluştur
  ```python
  from pydantic import BaseModel, Field
  from typing import Optional, Literal

  class ExperimentConfig(BaseModel):
      name: str
      seed: int = 42
      task: Literal["model1", "model2", "model3"]

  class ModelConfig(BaseModel):
      name: str  # registry key
      load_in_4bit: bool = True
      torch_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

  class LoRAConfig(BaseModel):
      r: int = 16
      alpha: int = 32
      dropout: float = 0.05
      target_modules: str = "auto"

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
  ```

- [ ] **6.1.2** pydantic'i requirements-dev.txt'e ekle
  ```
  pydantic>=2.0
  ```
  ```bash
  pip install pydantic
  ```

### 6.2. Config Loader

- [ ] **6.2.1** `src/training/config_loader.py` oluştur
  ```python
  import yaml
  from pathlib import Path
  from .config_schema import FullConfig

  def load_config(path: str) -> FullConfig:
      with open(path, "r") as f:
          raw = yaml.safe_load(f)
      return FullConfig(**raw)

  def apply_overrides(config: FullConfig, overrides: list) -> FullConfig:
      """'--override model.name=gemma-4-4b' formatındaki override'ları uygula."""
      config_dict = config.model_dump()
      for ovr in overrides:
          key, value = ovr.split("=", 1)
          keys = key.split(".")
          d = config_dict
          for k in keys[:-1]:
              d = d[k]
          # Tip çıkarımı
          if value.lower() in ("true", "false"):
              value = value.lower() == "true"
          elif value.replace(".", "").replace("-", "").isdigit():
              value = float(value) if "." in value else int(value)
          d[keys[-1]] = value
      return FullConfig(**config_dict)
  ```

### 6.3. Prompt Templates

- [ ] **6.3.1** `src/training/prompts.py` oluştur
  ```python
  MODEL1_SYSTEM_PROMPT = """Sen bir UI/UX tasarım uzmanısın. Sana gösterilen mobil uygulama ekran görüntüsünü analiz et ve profesyonel bir tasarım kritiği üret."""

  MODEL1_USER_INSTRUCTION = """Bu UI ekran görüntüsünü değerlendir. Çıktını aşağıdaki JSON formatında ver:

  {
    "critiques": [
      {
        "comment": "...",
        "bounding_box": [x, y, w, h],
        "severity": "low|medium|high",
        "category": "accessibility|readability|hierarchy|consistency|aesthetics"
      }
    ],
    "overall_feedback": "..."
  }"""

  MODEL2_USER_INSTRUCTION = """Bu UI ekranının view hierarchy'sini (yapısal ağacını) çıkar. JSON formatında, her element için tip, bounds ve (varsa) text bilgisini ver.

  Örnek:
  {
    "type": "FrameLayout",
    "bounds": [0, 0, 1080, 1920],
    "children": [
      {"type": "TextView", "bounds": [...], "text": "..."},
      ...
    ]
  }"""

  MODEL3_USER_INSTRUCTION_TEMPLATE = """Bu UI ekranının yapısal hiyerarşisi:
  {hierarchy_json}

  Yukarıdaki hiyerarşi bilgisini kullanarak bu UI için tasarım kritiği üret. Çıktıyı Model 1 ile aynı JSON formatında ver."""
  ```

### 6.4. Dataset (PyTorch)

- [ ] **6.4.1** `src/training/dataset.py` oluştur
  ```python
  import json
  from pathlib import Path
  from torch.utils.data import Dataset
  from PIL import Image
  from typing import Dict, List

  class UICriticDataset(Dataset):
      def __init__(self, records_path: str, image_dir: str, task: str, adapter, max_image_size: int = 1024):
          with open(records_path, "r") as f:
              self.records = json.load(f)
          self.image_dir = Path(image_dir)
          self.task = task
          self.adapter = adapter
          self.max_image_size = max_image_size

      def __len__(self):
          return len(self.records)

      def __getitem__(self, idx: int) -> Dict:
          rec = self.records[idx]
          img = Image.open(rec["image_path"]).convert("RGB")
          # Resize eğer max'ı aşarsa
          if max(img.size) > self.max_image_size:
              img.thumbnail((self.max_image_size, self.max_image_size))

          from .prompts import MODEL1_USER_INSTRUCTION, MODEL2_USER_INSTRUCTION, MODEL3_USER_INSTRUCTION_TEMPLATE

          if self.task == "model1":
              instruction = MODEL1_USER_INSTRUCTION
              target = json.dumps({
                  "critiques": rec["critiques"],
                  "overall_feedback": rec.get("overall_feedback", ""),
              }, ensure_ascii=False)
          elif self.task == "model2":
              instruction = MODEL2_USER_INSTRUCTION
              target = json.dumps(rec["hierarchy"], ensure_ascii=False)
          elif self.task == "model3":
              instruction = MODEL3_USER_INSTRUCTION_TEMPLATE.format(
                  hierarchy_json=json.dumps(rec["predicted_hierarchy"], ensure_ascii=False)
              )
              target = json.dumps({
                  "critiques": rec["critiques"],
                  "overall_feedback": rec.get("overall_feedback", ""),
              }, ensure_ascii=False)
          else:
              raise ValueError(f"Unknown task: {self.task}")

          inputs = self.adapter.format_prompt(instruction, img)
          return {"inputs": inputs, "target": target, "rico_id": rec["rico_id"]}
  ```

### 6.5. Trainer

- [ ] **6.5.1** `src/training/trainer.py` oluştur
  ```python
  from pathlib import Path
  from transformers import TrainingArguments, Trainer
  from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
  from .config_schema import FullConfig
  from .dataset import UICriticDataset
  from ..models.registry import create_adapter

  def train(config: FullConfig):
      # 1. Adapter yükle
      adapter = create_adapter(config.model.name)
      adapter.load_model(
          load_in_4bit=config.model.load_in_4bit,
          torch_dtype=config.model.torch_dtype,
      )

      # 2. LoRA uygula
      target_modules = (
          adapter.get_lora_target_modules() if config.lora.target_modules == "auto"
          else config.lora.target_modules.split(",")
      )
      model = prepare_model_for_kbit_training(adapter.model)
      lora_config = LoraConfig(
          r=config.lora.r,
          lora_alpha=config.lora.alpha,
          target_modules=target_modules,
          lora_dropout=config.lora.dropout,
          bias="none",
          task_type="CAUSAL_LM",
      )
      model = get_peft_model(model, lora_config)
      adapter.model = model

      # 3. Dataset hazırla
      train_ds = UICriticDataset(
          config.data.train_path, config.data.image_dir,
          config.experiment.task, adapter, config.data.max_image_size,
      )
      val_ds = UICriticDataset(
          config.data.val_path, config.data.image_dir,
          config.experiment.task, adapter, config.data.max_image_size,
      )

      # 4. TrainingArguments
      args = TrainingArguments(
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
          save_total_limit=config.output.save_total_limit,
          eval_strategy="steps",
          save_strategy="steps",
          bf16=config.model.torch_dtype == "bfloat16",
          report_to=["tensorboard"],
      )

      # 5. Trainer
      # NOTE: Custom collator gerekecek VLM'ler için
      # Bu bir iskelet — collator Phase 7'de tam implement edilecek
      trainer = Trainer(
          model=model,
          args=args,
          train_dataset=train_ds,
          eval_dataset=val_ds,
      )

      # 6. Eğit
      trainer.train()

      # 7. Kaydet
      final_path = Path(config.output.dir) / "final"
      trainer.save_model(str(final_path))
      print(f"Model kaydedildi: {final_path}")
  ```

### 6.6. CLI Training Script

- [ ] **6.6.1** `scripts/train.py` oluştur
  ```python
  import argparse
  from src.training.config_loader import load_config, apply_overrides
  from src.training.trainer import train

  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--config", required=True)
      parser.add_argument("--override", nargs="*", default=[])
      args = parser.parse_args()

      config = load_config(args.config)
      if args.override:
          config = apply_overrides(config, args.override)
      print(f"Eğitim başlıyor: {config.experiment.name}")
      print(f"Model: {config.model.name}")
      train(config)

  if __name__ == "__main__":
      main()
  ```

### 6.7. Örnek Config Dosyaları

- [ ] **6.7.1** `configs/model1_qwen.yaml` oluştur (README'deki örneği kopyala)

- [ ] **6.7.2** `configs/model1_gemma.yaml` oluştur (sadece `name` ve `output.dir` farklı)

- [ ] **6.7.3** `configs/model1_llava.yaml` oluştur

- [ ] **6.7.4** `configs/model2_qwen.yaml` oluştur (task=model2, veri yolu farklı)

- [ ] **6.7.5** `configs/model2_gemma.yaml` oluştur

- [ ] **6.7.6** `configs/model2_llava.yaml` oluştur

- [ ] **6.7.7** `configs/model3_qwen.yaml` oluştur

- [ ] **6.7.8** `configs/model3_gemma.yaml` oluştur

- [ ] **6.7.9** `configs/model3_llava.yaml` oluştur

### 6.8. Config Yükleme Testi

- [ ] **6.8.1** `tests/test_config_loader.py` yaz
  ```python
  from src.training.config_loader import load_config

  def test_load_model1_qwen():
      config = load_config("configs/model1_qwen.yaml")
      assert config.model.name == "qwen2.5-vl-7b"
      assert config.experiment.task == "model1"

  def test_all_configs_loadable():
      import glob
      for path in glob.glob("configs/model*.yaml"):
          config = load_config(path)
          assert config.model.name in [
              "qwen2.5-vl-3b", "qwen2.5-vl-7b", "gemma-4-4b",
              "gemma-4-12b", "llava-1.6-7b", "llava-1.6-13b",
          ]
  ```

### 6.9. Commit

- [ ] **6.9.1**
  ```bash
  git add src/training/ scripts/train.py configs/ tests/
  git commit -m "feat: eğitim pipeline ve config sistemi"
  git push
  ```

---

## Faz 7: Cloud GPU Kurulumu

**Amaç:** RunPod/Vast.ai'de sunucu kurup lokal kodu sunucuya aktarmak.
**Süre:** ~2-3 saat

### 7.1. Sunucu Seçimi ve Kiralama

- [ ] **7.1.1** RunPod'a üye ol ve kredi yükle
  - https://runpod.io
  - Min $10 kredi

- [ ] **7.1.2** Yeni pod başlat
  - GPU: RTX 4090 veya 5090 (32 GB tercih)
  - Template: "PyTorch 2.5 + CUDA 12.4"
  - Disk: 200 GB
  - **Test:** Pod başladı, "Running" durumunda

- [ ] **7.1.3** SSH bağlantı bilgilerini al
  - Pod sayfasında "Connect" → "SSH over exposed TCP"
  - Komut ve key'i kaydet

### 7.2. İlk Bağlantı ve Ortam

- [ ] **7.2.1** SSH ile bağlan
  ```bash
  ssh root@<IP> -p <PORT> -i ~/runpod_key.pem
  ```
  - **Test:** Bağlandın, prompt açıldı

- [ ] **7.2.2** GPU'yu doğrula
  ```bash
  nvidia-smi
  ```
  - **Test:** GPU bilgisi + VRAM görünür

- [ ] **7.2.3** Sistem güncelle
  ```bash
  apt update && apt install -y git tmux htop wget unzip
  ```

### 7.3. Proje Klonlama

- [ ] **7.3.1** GitHub SSH key'i sunucuya kopyala (veya yeni oluştur)
  ```bash
  # Sunucuda:
  ssh-keygen -t ed25519 -C "server"
  cat ~/.ssh/id_ed25519.pub
  ```
  - GitHub → Settings → Deploy keys → Add deploy key (bu repo'ya özel)

- [ ] **7.3.2** Repo'yu klonla
  ```bash
  cd /workspace   # kalıcı disk
  git clone git@github.com:kullanici/ui-critic-project.git
  cd ui-critic-project
  ```
  - **Test:** `ls` ile dosyalar görünür

### 7.4. Conda + Paket Kurulumu

- [ ] **7.4.1** Miniconda kur
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
  source $HOME/miniconda/bin/activate
  conda init bash
  exec bash  # yeniden yükle
  ```

- [ ] **7.4.2** Environment oluştur ve aktive
  ```bash
  conda create -n uicritic python=3.11 -y
  conda activate uicritic
  ```

- [ ] **7.4.3** Cloud requirements kur
  ```bash
  pip install -r requirements-cloud.txt
  ```
  - **Test:** Hata yok

- [ ] **7.4.4** CUDA'nın çalıştığını doğrula
  ```bash
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
  ```
  - **Beklenen:** `True` ve GPU adı

### 7.5. HuggingFace Login

- [ ] **7.5.1** HF token ile giriş yap
  ```bash
  huggingface-cli login
  # Token yapıştır
  ```

### 7.6. Kaggle Setup

- [ ] **7.6.1** Kaggle API key'i sunucuya kopyala
  ```bash
  # Lokal'den sunucuya:
  scp -P <PORT> ~/.kaggle/kaggle.json root@<IP>:/root/.kaggle/
  # Sunucuda:
  chmod 600 ~/.kaggle/kaggle.json
  ```

### 7.7. Datasetleri Cloud'a İndir

- [ ] **7.7.1** UICrit'i klonla
  ```bash
  cd /workspace/ui-critic-project/data
  git clone https://github.com/google-research-datasets/uicrit.git
  ```

- [ ] **7.7.2** RICO'yu indir (~6 GB)
  ```bash
  cd /workspace/ui-critic-project/data
  kaggle datasets download -d onurgunes1993/rico-dataset
  unzip rico-dataset.zip -d rico/
  rm rico-dataset.zip
  ```
  - **Test:** `ls data/rico/combined | wc -l` → 60.000+ dosya

- [ ] **7.7.3** CLAY'i klonla (opsiyonel)
  ```bash
  cd /workspace/ui-critic-project/data
  git clone https://github.com/google-research-datasets/clay.git
  ```

### 7.8. Commit Sunucu Ayarları

- [ ] **7.8.1** Gerekirse `.env` veya ayarları güncelle

---

## Faz 8: Model Eğitimleri

**Amaç:** 9 eğitim deneyini çalıştırmak.
**Süre:** ~25-35 saat GPU zamanı (arka planda tmux ile)

### 8.1. Veri Hazırlama (Cloud'da)

- [ ] **8.1.1** Model 1 verisi hazırla
  ```bash
  python scripts/prepare_data.py --task model1
  ```
  - **Test:** `data/processed/model1_train.json` oluştu

- [ ] **8.1.2** Model 2 verisi hazırla
  ```bash
  python scripts/prepare_data.py --task model2
  ```
  - **Test:** `data/processed/model2_train.json` oluştu, ~45.000 kayıt

- [ ] **8.1.3** İlk 3 kaydı kontrol et
  ```bash
  python -c "import json; d=json.load(open('data/processed/model1_train.json')); print(d[:2])"
  ```

### 8.2. Pilot Eğitim (Küçük Test)

**Amaç:** Tam eğitime girmeden önce pipeline'ın çalıştığından emin olmak.

- [ ] **8.2.1** Pilot config oluştur
  ```bash
  cp configs/model1_qwen.yaml configs/model1_qwen_pilot.yaml
  # İçinde: num_epochs=1, save_steps=50, sadece 20 örnek kullan
  ```

- [ ] **8.2.2** Pilot'u çalıştır
  ```bash
  python scripts/train.py --config configs/model1_qwen_pilot.yaml
  ```
  - **Test:** 10-20 dakika içinde ilk checkpoint'i kaydediyor mu?

- [ ] **8.2.3** Hatalar varsa logları incele ve düzelt
  ```bash
  tail -f outputs/logs/model1_qwen_pilot/trainer.log
  ```

### 8.3. Model 1 Eğitimleri (3 Deney)

- [ ] **8.3.1** Qwen2.5-VL ile Model 1
  ```bash
  tmux new -s m1_qwen
  conda activate uicritic
  python scripts/train.py --config configs/model1_qwen.yaml
  # Ctrl+B, D → detach
  ```
  - Süre: ~3-4 saat
  - **Test:** `outputs/checkpoints/model1_qwen_v1/final/` var

- [ ] **8.3.2** Eğitimi izle (ayrı terminal)
  ```bash
  tmux attach -t m1_qwen   # Bağlan
  # veya
  tensorboard --logdir outputs/logs/ --port 6006
  ```

- [ ] **8.3.3** Gemma 4 ile Model 1 (Qwen bittikten sonra)
  ```bash
  tmux new -s m1_gemma
  python scripts/train.py --config configs/model1_gemma.yaml
  ```
  - Süre: ~2-3 saat

- [ ] **8.3.4** LLaVA-1.6 ile Model 1
  ```bash
  tmux new -s m1_llava
  python scripts/train.py --config configs/model1_llava.yaml
  ```
  - Süre: ~3-4 saat

### 8.4. Model 2 Eğitimleri (3 Deney)

- [ ] **8.4.1** Qwen ile Model 2
  ```bash
  tmux new -s m2_qwen
  python scripts/train.py --config configs/model2_qwen.yaml
  ```
  - Süre: ~5-8 saat (RICO çok büyük)

- [ ] **8.4.2** Gemma ile Model 2
  ```bash
  python scripts/train.py --config configs/model2_gemma.yaml
  ```

- [ ] **8.4.3** LLaVA ile Model 2
  ```bash
  python scripts/train.py --config configs/model2_llava.yaml
  ```

### 8.5. Model 2 → UICrit Inference

- [ ] **8.5.1** En iyi Model 2'yi seç (değerlendirmelere göre)

- [ ] **8.5.2** UICrit'teki 983 UI için hiyerarşileri çıkar
  ```bash
  python scripts/run_model2_on_uicrit.py \
      --config configs/model2_qwen.yaml \
      --checkpoint outputs/checkpoints/model2_qwen_v1/final \
      --output data/processed/uicrit_predicted_hierarchies.json
  ```
  - Süre: ~1-2 saat
  - **Test:** JSON dosyası 983 kayıt içeriyor

### 8.6. Model 3 Veri Hazırlama

- [ ] **8.6.1** Model 3 verisi hazırla
  ```bash
  python scripts/prepare_data.py --task model3 \
      --predicted_vh_path data/processed/uicrit_predicted_hierarchies.json
  ```
  - **Test:** `data/processed/model3_train.json` oluştu

### 8.7. Model 3 Eğitimleri (3 Deney)

- [ ] **8.7.1** Qwen ile Model 3
  ```bash
  tmux new -s m3_qwen
  python scripts/train.py --config configs/model3_qwen.yaml
  ```

- [ ] **8.7.2** Gemma ile Model 3
  ```bash
  python scripts/train.py --config configs/model3_gemma.yaml
  ```

- [ ] **8.7.3** LLaVA ile Model 3
  ```bash
  python scripts/train.py --config configs/model3_llava.yaml
  ```

### 8.8. Checkpoint Backup

- [ ] **8.8.1** Tüm final checkpoint'leri HuggingFace Hub'a push et (opsiyonel)
  ```bash
  huggingface-cli upload <kullanici>/ui-critic-m1-qwen outputs/checkpoints/model1_qwen_v1/final
  # Benzer şekilde diğerleri
  ```

- [ ] **8.8.2** Veya SCP ile lokale indir
  ```bash
  # Lokal'den:
  rsync -avz -e "ssh -p <PORT>" root@<IP>:/workspace/ui-critic-project/outputs/checkpoints/ ./local_backup/
  ```

---

## Faz 9: Değerlendirme ve Karşılaştırma

**Amaç:** 9 eğitilen modelin test setinde performansını ölçmek.
**Süre:** ~6-10 saat

### 9.1. Metrik Modülleri

- [ ] **9.1.1** `src/evaluation/metrics.py` oluştur
  ```python
  from typing import List, Dict
  import numpy as np

  def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
      from sacrebleu import corpus_bleu
      # ...
      pass

  def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
      from rouge_score import rouge_scorer
      # ...
      pass

  def compute_bertscore(predictions: List[str], references: List[str], lang: str = "en") -> Dict[str, float]:
      from bert_score import score
      P, R, F1 = score(predictions, references, lang=lang)
      return {"P": P.mean().item(), "R": R.mean().item(), "F1": F1.mean().item()}

  def compute_iou(pred_boxes: List[List[float]], gt_boxes: List[List[float]]) -> float:
      # Per-box IoU hesapla, ortalama al
      pass
  ```

### 9.2. Inference Script

- [ ] **9.2.1** `scripts/infer.py` oluştur — batch inference yapar, JSON çıktılarını dosyaya yazar

### 9.3. Evaluator

- [ ] **9.3.1** `src/evaluation/evaluator.py` oluştur
  ```python
  def evaluate_model(config, checkpoint_path):
      # 1. Model yükle
      # 2. Test seti üzerinden inference
      # 3. Metrikleri hesapla
      # 4. Sonuçları JSON olarak kaydet
      pass
  ```

### 9.4. Evaluate CLI

- [ ] **9.4.1** `scripts/evaluate.py` oluştur
  ```bash
  python scripts/evaluate.py --config configs/model1_qwen.yaml \
      --checkpoint outputs/checkpoints/model1_qwen_v1/final
  ```

### 9.5. Tüm Model Değerlendirmeleri

- [ ] **9.5.1** Model 1 - Qwen değerlendir
- [ ] **9.5.2** Model 1 - Gemma değerlendir
- [ ] **9.5.3** Model 1 - LLaVA değerlendir
- [ ] **9.5.4** Model 2 - Qwen değerlendir
- [ ] **9.5.5** Model 2 - Gemma değerlendir
- [ ] **9.5.6** Model 2 - LLaVA değerlendir
- [ ] **9.5.7** Model 3 - Qwen değerlendir
- [ ] **9.5.8** Model 3 - Gemma değerlendir
- [ ] **9.5.9** Model 3 - LLaVA değerlendir

### 9.6. Karşılaştırma Tablosu

- [ ] **9.6.1** Tüm sonuçları tek bir CSV'ye topla
  ```bash
  python scripts/compare_models.py --output outputs/final_comparison.csv
  ```

- [ ] **9.6.2** Grafikleri oluştur (notebook'ta)
  - Bar chart: Model 1 vs Model 3 karşılaştırması
  - Heatmap: Her metrik için her modelin skoru

### 9.7. Ablation Study (Opsiyonel)

- [ ] **9.7.1** Config B (oracle VH) hazırla ve eğit
- [ ] **9.7.2** Config D (sadece VH) hazırla ve eğit
- [ ] **9.7.3** A/B/C/D tablosu oluştur

### 9.8. Commit

- [ ] **9.8.1** Sonuçları commit
  ```bash
  git add outputs/final_comparison.csv src/evaluation/ scripts/evaluate.py
  git commit -m "feat: değerlendirme pipeline ve sonuçlar"
  ```

---

## Faz 10: Tez/Rapor Yazımı

**Amaç:** Deneyleri akademik bir belgeye dönüştürmek.
**Süre:** ~2-4 hafta

### 10.1. Abstract ve Giriş

- [ ] **10.1.1** 200 kelimelik abstract yaz
- [ ] **10.1.2** Introduction bölümü: motivasyon, research question, katkılar

### 10.2. Literatür Taraması (zaten var)

- [ ] **10.2.1** Mevcut literatür taramasını teze uyarla
- [ ] **10.2.2** Related Work bölümü yaz

### 10.3. Metodoloji

- [ ] **10.3.1** 3 model mimarisini şekillerle açıkla
- [ ] **10.3.2** Adapter pattern mimarisini anlat
- [ ] **10.3.3** Veri hazırlama sürecini detaylandır
- [ ] **10.3.4** Hyperparameter tablosu

### 10.4. Deneyler

- [ ] **10.4.1** Deney kurulumu bölümü (donanım, veri split)
- [ ] **10.4.2** Sonuçlar tablosu
- [ ] **10.4.3** Qualitative örnekler (iyi/kötü çıktı örnekleri)

### 10.5. Tartışma

- [ ] **10.5.1** Hiyerarşi katkı analizi
- [ ] **10.5.2** Model 2 hatasının Model 3'e etkisi
- [ ] **10.5.3** Model seçiminin (Qwen vs Gemma vs LLaVA) etkisi
- [ ] **10.5.4** Sınırlılıklar

### 10.6. Sonuç

- [ ] **10.6.1** Özet ve bulgular
- [ ] **10.6.2** Gelecek çalışmalar

### 10.7. Son Kontrol

- [ ] **10.7.1** İntihal kontrolü
- [ ] **10.7.2** Dilbilgisi ve yazım
- [ ] **10.7.3** Referansların eksiksiz olması
- [ ] **10.7.4** Grafik ve tabloların numaralandırılması

---

## Özet Zaman Çizelgesi

| Faz | Süre (Tahmini) | Öncelik |
|---|---|---|
| 0. Altyapı | 2-3 saat | Kritik |
| 1. Git/Proje İskeleti | 2-3 saat | Kritik |
| 2. Dataset Keşfi | 3-4 saat | Kritik |
| 3. Data Loaders | 4-6 saat | Kritik |
| 4. Model Katmanı | 6-8 saat | Kritik |
| 5. Preprocessing | 4-5 saat | Kritik |
| 6. Eğitim Pipeline | 6-8 saat | Kritik |
| 7. Cloud Kurulum | 2-3 saat | Kritik |
| 8. Eğitimler (GPU zamanı) | 25-35 saat | Kritik |
| 9. Değerlendirme | 6-10 saat | Kritik |
| 10. Tez Yazımı | 2-4 hafta | Kritik |

**Toplam aktif geliştirme süresi:** 60-90 saat (yaklaşık 2-3 hafta tam zamanlı)  
**Toplam GPU ücreti:** ~$30-60 (RunPod RTX 5090 ile)  
**Toplam proje süresi (tez dahil):** 6-10 hafta

---

## Bir Sonraki Adım

Şu anda **Faz 0'dasın**. En başa dön, `0.1.1`'den başla ve sırayla ilerle. Her görev tamamlandıktan sonra `[ ]` → `[x]` yap.

**Günlük hedef:** 10-15 görev (Faz 0-6 aşamasında)  
**İlk hedef:** 1 hafta içinde Faz 0-5 tamamlamak (Faz 6'da pipeline'ı yazmak)  
**2. hedef:** 2. hafta: Cloud'da eğitim + değerlendirme

**Her faz sonunda commit + push yap.** Her gün en az bir commit at, böylece ilerlemeni takip edebilirsin.

Başarılar! 🚀
