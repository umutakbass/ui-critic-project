"""UICrit Loader manuel test script."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.uicrit_loader import UICritLoader


def main():
    print("=" * 50)
    print("UICRIT LOADER CANLI TESTI")
    print("=" * 50)

    csv_path = PROJECT_ROOT / "data" / "uicrit" / "uicrit_public.csv"
    loader = UICritLoader(str(csv_path))
    df = loader.load()

    print("Toplam kayit:", len(df))
    print("Benzersiz UI:", df["rico_id"].nunique())
    print("Sutunlar:", df.columns.tolist())

    ids = loader.get_unique_rico_ids()
    first_id = ids[0]
    print("Ilk rico_id:", first_id)

    records = loader.get_by_rico_id(first_id)
    print("Bu UI icin kayit:", len(records))

    first = records.iloc[0]
    comments = first["comments"]
    print("Task:", first["task"])
    print("Aesthetics:", first["aesthetics_rating"])
    print("Yorum sayisi:", len(comments))

    if comments:
        c = comments[0]
        text = c.get("comment", "") or ""
        print("Ilk yorum (ilk 120 krk):", text[:120])
        print("Ilk yorum bbox:", c.get("bounding_box"))

    print("TEST BASARILI")


if __name__ == "__main__":
    main()
