"""Kurulum doğrulama — setup_cloud.sh sonunda çalışır."""

import sys
from pathlib import Path

ERRORS = []
WARNINGS = []


def check(condition: bool, msg_ok: str, msg_fail: str, fatal: bool = True) -> bool:
    if condition:
        print(f"  ✓ {msg_ok}")
        return True
    else:
        (ERRORS if fatal else WARNINGS).append(msg_fail)
        print(f"  {'✗' if fatal else '!'} {msg_fail}")
        return False


def verify_gpu():
    print("\n[1/4] GPU")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        check(cuda_ok, f"CUDA mevcut: {torch.cuda.get_device_name(0)}", "CUDA bulunamadı!")
        if cuda_ok:
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            check(vram >= 20, f"VRAM: {vram:.1f} GB", f"VRAM düşük: {vram:.1f} GB (≥20 GB önerilir)", fatal=False)
    except ImportError:
        ERRORS.append("torch kurulamadı")
        print("  ✗ torch import edilemedi")


def verify_data():
    print("\n[2/4] Veri Dosyaları")
    uicrit_csv = Path("data/uicrit/uicrit_public.csv")
    check(uicrit_csv.exists(), f"UICrit CSV: {uicrit_csv}", f"UICrit CSV bulunamadı: {uicrit_csv}")

    combined = Path("data/archive/unique_uis/combined")
    check(combined.is_dir(), f"RICO combined/: {combined}", f"RICO combined/ bulunamadı: {combined}")

    if combined.is_dir():
        jpg_count = len(list(combined.glob("*.jpg")))
        check(jpg_count >= 1000, f"RICO görseller: {jpg_count} .jpg", f"Çok az RICO görseli: {jpg_count}", fatal=False)


def verify_python_packages():
    print("\n[3/4] Python Paketleri")
    required = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("accelerate", "accelerate"),
        ("PIL", "pillow"),
        ("pandas", "pandas"),
        ("yaml", "pyyaml"),
        ("pydantic", "pydantic"),
    ]
    for import_name, pkg_name in required:
        try:
            __import__(import_name)
            print(f"  ✓ {pkg_name}")
        except ImportError:
            ERRORS.append(f"Eksik paket: {pkg_name}")
            print(f"  ✗ {pkg_name} eksik")

    try:
        import unsloth  # noqa
        print("  ✓ unsloth (opsiyonel)")
    except ImportError:
        print("  ! unsloth yok (opsiyonel, atlanıyor)")


def verify_src():
    print("\n[4/4] Kaynak Kod")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from src.data.uicrit_loader import UICritLoader
        from src.models.registry import MODEL_REGISTRY
        from src.training.config_loader import load_config

        check(True, "src modülleri import edildi", "")
        check(len(MODEL_REGISTRY) >= 6, f"Model registry: {len(MODEL_REGISTRY)} model", "Registry boş")

        uicrit_csv = Path("data/uicrit/uicrit_public.csv")
        if uicrit_csv.exists():
            loader = UICritLoader(str(uicrit_csv))
            df = loader.load()
            check(len(df) > 0, f"UICrit yüklendi: {len(df)} satır", "UICrit boş!")
    except Exception as e:
        ERRORS.append(f"src import hatası: {e}")
        print(f"  ✗ {e}")


def main():
    print("=" * 50)
    print("UI Critic — Kurulum Doğrulama")
    print("=" * 50)

    verify_gpu()
    verify_data()
    verify_python_packages()
    verify_src()

    print("\n" + "=" * 50)
    if ERRORS:
        print(f"BAŞARISIZ — {len(ERRORS)} hata:")
        for e in ERRORS:
            print(f"  • {e}")
        sys.exit(1)
    elif WARNINGS:
        print(f"UYARI ile geçti — {len(WARNINGS)} uyarı:")
        for w in WARNINGS:
            print(f"  • {w}")
        print("Devam edebilirsiniz.")
    else:
        print("TÜM KONTROLLER BAŞARILI — Eğitimi başlatabilirsiniz.")


if __name__ == "__main__":
    main()
