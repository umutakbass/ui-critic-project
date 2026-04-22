"""UICrit ile RICO datasetini rico_id üzerinden eşleştiren modül."""

from typing import Dict, List, Optional

from .uicrit_loader import UICritLoader
from .rico_loader import RicoLoader


class UICritRicoAligner:
    """UICrit kritiklerini RICO görsel+VH ile eşleştirir."""

    def __init__(self, uicrit: UICritLoader, rico: RicoLoader):
        self.uicrit = uicrit
        self.rico = rico

    def get_aligned_record(self, rico_id: int) -> Optional[Dict]:
        """Bir rico_id için UICrit kritikleri + RICO görseli/VH'sini döndür.

        Returns:
            Dict veya None (RICO görseli yoksa ya da UICrit kaydı yoksa).
        """
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
        """UICrit'te olan VE RICO'da görseli bulunan tüm rico_id'ler."""
        uicrit_ids = self.uicrit.get_unique_rico_ids()
        return [rid for rid in uicrit_ids if self.rico.image_exists(rid)]

    def coverage_report(self) -> Dict:
        """UICrit ID'lerinin RICO'da ne kadarının mevcut olduğunu raporla."""
        uicrit_ids = self.uicrit.get_unique_rico_ids()
        aligned = self.get_all_aligned_ids()
        missing = [rid for rid in uicrit_ids if not self.rico.image_exists(rid)]
        return {
            "total_uicrit_ids": len(uicrit_ids),
            "aligned_count": len(aligned),
            "missing_count": len(missing),
            "coverage_pct": round(len(aligned) / len(uicrit_ids) * 100, 2) if uicrit_ids else 0.0,
            "missing_ids": missing[:20],  # ilk 20'sini göster
        }
