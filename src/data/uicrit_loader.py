"""UICrit dataset yükleyici.

UICrit CSV dosyasını okur, yorumları parse eder.

UICrit formatı (gerçek veri incelemesinden):
    - comments sütunu: Python literal list formatında string.
      Örnek: '["Comment 1\\n...\\nBounding Box: [0.1, 0.2, 0.3, 0.4]", ...]'
    - comments_source: Aynı formatta liste. ["human", "llm", "both"]
    - Her comment şu formatta:
        "Comment N
        <yorum metni>
        Bounding Box: [x1, y1, x2, y2]"
      Koordinatlar 0-1 arası normalize.
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# Bounding Box'u çıkarmak için regex
BBOX_PATTERN = re.compile(
    r"Bounding Box:\s*\[([0-9.\-eE,\s]+)\]",
    re.IGNORECASE,
)


def parse_comment_string(raw: str) -> Dict[str, Any]:
    """Tek bir 'Comment N\\n...\\nBounding Box: [...]' stringini parse eder.

    Args:
        raw: Ham yorum metni.

    Returns:
        {'comment': str, 'bounding_box': [x1,y1,x2,y2] or None}
    """
    bbox_match = BBOX_PATTERN.search(raw)
    bounding_box = None
    if bbox_match:
        try:
            nums = [float(x.strip()) for x in bbox_match.group(1).split(",")]
            if len(nums) == 4:
                bounding_box = nums
        except ValueError:
            pass

    # "Comment 1\n" başlığını ve "Bounding Box:" satırını atla
    text = raw
    text = re.sub(r"^Comment\s+\d+\s*\n?", "", text).strip()
    text = BBOX_PATTERN.sub("", text).strip()

    return {"comment": text, "bounding_box": bounding_box}


def parse_python_list_string(raw: Any) -> List[Any]:
    """Python literal list stringini Python listesine çevir.

    json.loads tek tırnaklı veya karışık tırnaklı stringleri yiyemez.
    ast.literal_eval her iki biçimi de destekler ve güvenlidir.
    """
    if pd.isna(raw):
        return []
    if isinstance(raw, list):
        return raw
    return ast.literal_eval(raw)


class UICritLoader:
    """UICrit datasetini yükler.

    Örnek:
        >>> loader = UICritLoader("data/uicrit/uicrit_public.csv")
        >>> df = loader.load()
        >>> print(f"Toplam kayıt: {len(df)}")
    """

    def __init__(self, csv_path: str):
        """UICritLoader'ı başlat.

        Args:
            csv_path: UICrit CSV dosyasının yolu.
        """
        self.csv_path = Path(csv_path)
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """CSV'yi yükle ve comments sütunlarını parse et.

        Returns:
            Parse edilmiş pandas DataFrame.

            comments sütunu artık List[Dict] formatında:
                [{"comment": "...", "bounding_box": [x1,y1,x2,y2]}, ...]

        Raises:
            FileNotFoundError: CSV dosyası bulunamazsa.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"UICrit CSV bulunamadı: {self.csv_path}"
            )

        self.df = pd.read_csv(self.csv_path)

        # Python literal list'leri Python listesine çevir
        raw_comments = self.df["comments"].apply(parse_python_list_string)
        self.df["comments_source"] = self.df["comments_source"].apply(
            parse_python_list_string
        )

        # Her yorumu "comment" + "bounding_box"a böl
        self.df["comments"] = raw_comments.apply(
            lambda lst: [parse_comment_string(c) for c in lst]
        )
        return self.df

    def get_by_rico_id(self, rico_id: int) -> pd.DataFrame:
        """Belirli bir rico_id için tüm annotator kayıtlarını döndür."""
        if self.df is None:
            self.load()
        return self.df[self.df["rico_id"] == rico_id]

    def get_unique_rico_ids(self) -> List[int]:
        """Datasette bulunan benzersiz rico_id'lerin listesi."""
        if self.df is None:
            self.load()
        return sorted(self.df["rico_id"].unique().tolist())
