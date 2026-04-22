"""rico_id listesini train/val/test olarak bölen yardımcı fonksiyon."""

import random
from typing import List, Tuple


def split_ids(
    ids: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """rico_id listesini train/val/test olarak böl.

    Args:
        ids: Tüm rico_id'lerin listesi.
        train_ratio: Eğitim seti oranı.
        val_ratio: Doğrulama seti oranı.
        test_ratio: Test seti oranı.
        seed: Tekrarlanabilirlik için rastgele tohum.

    Returns:
        (train_ids, val_ids, test_ids) tuple'ı.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Oranların toplamı 1.0 olmalı"
    )

    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return train, val, test
