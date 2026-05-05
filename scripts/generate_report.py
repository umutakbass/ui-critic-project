"""
Qwen2.5-VL-7B ile Model1/2/3 eğitim sonuçları raporu üretir.
Kullanım: python scripts/generate_report.py
Çıktı: outputs/report_qwen_training.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


def build_report():
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "report_qwen_training.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2.5 * cm,
        leftMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=20, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=11, spaceAfter=20, textColor=colors.HexColor("#555555"), alignment=TA_CENTER
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=15, spaceBefore=18, spaceAfter=8,
        textColor=colors.HexColor("#16213e"),
        borderPad=4,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=12, spaceBefore=12, spaceAfter=6,
        textColor=colors.HexColor("#0f3460"),
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, spaceAfter=6, leading=15, alignment=TA_JUSTIFY
    )
    mono_style = ParagraphStyle(
        "Mono", parent=styles["Code"],
        fontSize=8.5, spaceAfter=6, leading=13,
        backColor=colors.HexColor("#f4f4f4"),
        borderPad=6,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=9, spaceAfter=4, textColor=colors.HexColor("#777777"), alignment=TA_CENTER
    )

    def table(data, col_widths=None, header=True):
        t = Table(data, colWidths=col_widths)
        style_cmds = [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]
        if header:
            style_cmds += [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9.5),
            ]
        t.setStyle(TableStyle(style_cmds))
        return t

    story = []

    # ── KAPAK ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("UI Critic Project", title_style))
    story.append(Paragraph("Qwen2.5-VL-7B Model Eğitim Raporu", subtitle_style))
    story.append(Paragraph("Model 1 · Model 2 · Model 3", subtitle_style))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#16213e")))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Hazırlayan: Umut Akbaş  |  Tarih: Mayıs 2026", caption_style))
    story.append(Spacer(1, 2 * cm))

    # ── 1. PROJE ÖZET ────────────────────────────────────────────────────
    story.append(Paragraph("1. Proje Özeti", h1_style))
    story.append(Paragraph(
        "Bu çalışma, mobil uygulama arayüzlerini (UI) otomatik olarak değerlendiren bir "
        "yapay zeka sistemi geliştirmeyi amaçlamaktadır. RICO veri seti üzerindeki UICrit "
        "insan değerlendirmeleri kullanılarak üç ayrı vision-language model (VLM) ince "
        "ayar (fine-tune) edilmiştir.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Araştırma Sorusu:</b> View hierarchy (ekran yapı ağacı) bilgisi, UI eleştiri "
        "kalitesini artırır mı?",
        body_style
    ))
    story.append(Spacer(1, 0.3 * cm))

    model_overview = [
        ["Model", "Giriş", "Çıkış", "Amaç"],
        ["Model 1", "UI Görseli", "Eleştiri JSON", "Baseline: görsel → eleştiri"],
        ["Model 2", "UI Görseli", "View Hierarchy JSON", "Ekran yapısını tahmin et"],
        ["Model 3", "UI Görseli + Hierarchy", "Eleştiri JSON", "Hierarchy ile eleştiri kalitesi artar mı?"],
    ]
    story.append(table(model_overview, col_widths=[3.5*cm, 4*cm, 4.5*cm, 5.5*cm]))
    story.append(Spacer(1, 0.5 * cm))

    # ── 2. VERİ SETLERİ ──────────────────────────────────────────────────
    story.append(Paragraph("2. Veri Setleri", h1_style))

    story.append(Paragraph("2.1 UICrit", h2_style))
    story.append(Paragraph(
        "UICrit, Google Research tarafından UIST 2024'te yayımlanan bir insan-değerlendirme "
        "veri setidir. 1.000 adet RICO mobil UI ekranı için 11.344 eleştiri içermektedir.",
        body_style
    ))
    uicrit_data = [
        ["Özellik", "Değer"],
        ["Toplam satır", "2.981 (benzersiz 1.000 ekran)"],
        ["Eleştiri kaynakları", "İnsan, LLM, Her ikisi (both)"],
        ["Eleştiri formatı", "comment + bounding_box + source"],
        ["Train / Val / Test split", "%80 / %10 / %10"],
    ]
    story.append(table(uicrit_data, col_widths=[6*cm, 11.5*cm]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("2.2 RICO", h2_style))
    story.append(Paragraph(
        "RICO, 66.261 adet Android mobil uygulama ekran görüntüsü ve view hierarchy "
        "JSON dosyalarını içeren geniş çaplı bir UI veri setidir. Model 2 eğitiminde "
        "kullanılan 3.000 örneklik alt küme HuggingFace üzerinden servis edilmiştir.",
        body_style
    ))

    story.append(Paragraph("2.3 Model 2 Alt Kümesi (3K)", h2_style))
    story.append(Paragraph(
        "Model 2 için tüm 46K örnek yerine 3.000 örneklik alt küme kullanılmıştır. "
        "Tam veri seti ile tahmin edilen eğitim süresi 93-180 saat aralığında olduğundan "
        "bu kısıtlama uygulanmıştır.",
        body_style
    ))

    story.append(PageBreak())

    # ── 3. MODEL MİMARİSİ ────────────────────────────────────────────────
    story.append(Paragraph("3. Model Mimarisi", h1_style))
    story.append(Paragraph(
        "Tüm modeller <b>Qwen2.5-VL-7B-Instruct</b> temel modeli üzerine LoRA "
        "(Low-Rank Adaptation) ile ince ayar yapılarak oluşturulmuştur.",
        body_style
    ))

    arch_data = [
        ["Parametre", "Değer"],
        ["Temel Model", "Qwen/Qwen2.5-VL-7B-Instruct"],
        ["Toplam Parametre", "~8.4 Milyar"],
        ["Eğitilebilir Parametre (LoRA)", "~95 Milyon (%1.13)"],
        ["Quantization", "4-bit NF4 (BitsAndBytes)"],
        ["LoRA r / alpha", "32 / 64"],
        ["LoRA Dropout", "0.05"],
        ["LoRA Hedef Modüller", "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"],
        ["Görüntü Çözünürlüğü (max)", "1024 × 1024 px"],
        ["Precision", "bfloat16"],
        ["GPU", "NVIDIA RTX 5090 (32 GB VRAM)"],
    ]
    story.append(table(arch_data, col_widths=[6*cm, 11.5*cm]))
    story.append(Spacer(1, 0.5 * cm))

    # ── 4. EĞİTİM KONFİGÜRASYONLARI ─────────────────────────────────────
    story.append(Paragraph("4. Eğitim Konfigürasyonları", h1_style))

    config_data = [
        ["Parametre", "Model 1", "Model 2", "Model 3"],
        ["Config dosyası", "model1_qwen.yaml", "model2_qwen.yaml", "model3_qwen.yaml"],
        ["Deney adı", "model1_qwen_v2", "model2_qwen_v1", "model3_qwen_v1"],
        ["Dataset (train)", "model1_train.json\n(~2.400 örnek)", "model2_train_3k.json\n(~2.400 örnek)", "model3_train.json\n(~400 örnek)"],
        ["Epoch sayısı", "8", "3", "5"],
        ["Batch size", "1", "1", "1"],
        ["Gradient accumulation", "16 (efektif: 16)", "16 (efektif: 16)", "16 (efektif: 16)"],
        ["Learning rate", "0.0001", "0.0001", "0.0001"],
        ["Warmup steps", "15", "50", "10"],
        ["Max steps", "Yok (epoch bazlı)", "950", "Yok (epoch bazlı)"],
        ["Save / Eval steps", "200", "500", "100"],
        ["Max sequence length", "2048 token", "2048 token", "2048 token"],
        ["Unsloth", "Evet", "Evet", "Evet"],
    ]
    story.append(table(config_data, col_widths=[5*cm, 4*cm, 4*cm, 4.5*cm]))
    story.append(Spacer(1, 0.5 * cm))

    # ── 5. EĞİTİM SONUÇLARI ──────────────────────────────────────────────
    story.append(Paragraph("5. Eğitim Sonuçları", h1_style))

    story.append(Paragraph("5.1 Model 1 — UI Görseli → Eleştiri", h2_style))
    story.append(Paragraph(
        "Model 1, tüm UICrit eğitim seti ile 8 epoch boyunca eğitilmiştir. "
        "Loss değerleri kararlı bir düşüş göstermiştir.",
        body_style
    ))
    m1_results = [
        ["Metrik", "Değer"],
        ["Toplam adım", "~1.200"],
        ["Eğitim süresi", "~6-8 saat"],
        ["Final Train Loss", "~3.2"],
        ["Final Eval Loss", "~3.1"],
        ["JSON parse başarı oranı", "Yüksek (geçerli JSON üretiyor)"],
        ["Checkpoint", "outputs/checkpoints/model1_qwen_v2/final"],
    ]
    story.append(table(m1_results, col_widths=[6*cm, 11.5*cm]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("5.2 Model 2 — UI Görseli → View Hierarchy", h2_style))
    story.append(Paragraph(
        "Model 2, view hierarchy üretimi için 3.000 örneklik alt küme ile 3 epoch "
        "eğitilmiştir. Loss değeri 3.7 civarında platoya oturmuştur; bu durum veri "
        "miktarının yetersizliğine bağlanmaktadır.",
        body_style
    ))
    m2_results = [
        ["Metrik", "Değer"],
        ["Toplam adım", "950 (max_steps ile sınırlandırıldı)"],
        ["Eğitim süresi", "~4-5 saat"],
        ["Final Train Loss", "~3.7"],
        ["Final Eval Loss", "~3.7"],
        ["JSON parse başarı oranı", "Orta (bazı çıktılar ham metin olarak kaydedildi)"],
        ["Üretilen tahminler", "500 RICO görseli için pred_vh.json"],
        ["Checkpoint", "outputs/checkpoints/model2_qwen_v1/final"],
    ]
    story.append(table(m2_results, col_widths=[6*cm, 11.5*cm]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("5.3 Model 3 — UI Görseli + Hierarchy → Eleştiri", h2_style))
    story.append(Paragraph(
        "Model 3, Model 2 tarafından üretilen 500 tahmin üzerinden oluşturulan "
        "model3_train.json (~400 örnek) ile 5 epoch eğitilmiştir.",
        body_style
    ))
    m3_results = [
        ["Metrik", "Değer"],
        ["Toplam adım", "115"],
        ["Eğitim süresi", "~79 dakika (4.739 saniye)"],
        ["Final Train Loss", "3.206"],
        ["Epoch 4.356 Eval Loss", "3.204"],
        ["Final Eval Loss (Epoch 5)", "3.200"],
        ["Train samples/sec", "0.38"],
        ["Checkpoint", "outputs/checkpoints/model3_qwen_v1/final"],
    ]
    story.append(table(m3_results, col_widths=[6*cm, 11.5*cm]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("5.4 Loss Karşılaştırması", h2_style))
    loss_compare = [
        ["Model", "Train Loss (başlangıç)", "Train Loss (final)", "Eval Loss (final)", "Süre"],
        ["Model 1", "~7.0", "~3.2", "~3.1", "~6-8 saat"],
        ["Model 2", "~4.5", "~3.7", "~3.7", "~4-5 saat"],
        ["Model 3", "7.759", "3.206", "3.200", "~79 dakika"],
    ]
    story.append(table(loss_compare, col_widths=[3*cm, 4*cm, 4*cm, 4*cm, 2.5*cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Not: Model 3'ün başlangıç loss değeri (7.759) yüksek görünmektedir. "
        "Bu durum, model girişine eklenen view hierarchy JSON'ının prompt uzunluğunu "
        "artırması ve modelin bu yeni format ile öğrenme sürecine başlamasından "
        "kaynaklanmaktadır.",
        body_style
    ))

    story.append(PageBreak())

    # ── 6. INFERENCE ─────────────────────────────────────────────────────
    story.append(Paragraph("6. Inference Örneği", h1_style))
    story.append(Paragraph(
        "Model 3, gerçek bir mobil uygulama ekran görüntüsü (WhatsApp benzeri arayüz) "
        "üzerinde test edilmiştir. Aşağıda kısaltılmış çıktı verilmiştir:",
        body_style
    ))

    sample_output = """{
  "critiques": [
    {
      "comment": "The expected standard is to have high contrast between
  text and background. In the current design, the small white text on a
  black background lacks contrast, making it difficult to read. To fix
  this, increase text size or use a lighter color.",
      "bounding_box": [0.017, 0.042, 0.983, 0.938],
      "source": "human"
    },
    {
      "comment": "The layout is cluttered and difficult to navigate, with
  overlapping elements and unclear hierarchy. Redesign the layout to
  prioritize clarity and organization.",
      "bounding_box": [0.017, 0.042, 0.983, 0.938],
      "source": "both"
    }
  ]
}"""
    story.append(Paragraph(sample_output.replace('\n', '<br/>'), mono_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "<b>Gözlem:</b> Model yapılandırılmış JSON formatını başarıyla üretmiştir. "
        "Bounding box değerlerinin tümünün aynı olması (neredeyse tam ekran) önemli "
        "bir eksikliktir. Bu durum; eğitim verisindeki geniş alan bounding box'larının "
        "baskın olmasına, az sayıda örnek ile eğitime ve test görselinin RICO "
        "dağılımının dışında olmasına bağlanabilir.",
        body_style
    ))

    # ── 7. TEKNİK SORUNLAR VE ÇÖZÜMLER ───────────────────────────────────
    story.append(Paragraph("7. Karşılaşılan Teknik Sorunlar ve Çözümler", h1_style))

    issues_data = [
        ["Sorun", "Çözüm"],
        ["Mask shape mismatch (attention_mask)", "batch_size=1 olarak ayarlandı"],
        ["pixel_values list hatası", "Collator'da tensors[0] döndürme eklendi"],
        ["Eval batch OOM", "per_device_eval_batch_size=1 eklendi"],
        ["warmup_steps > total_steps", "warmup_steps düşürüldü (15/10/50)"],
        ["Model 2 eğitimi 93-180 saat tahmini", "3K örneklik alt küme oluşturuldu"],
        ["OOM during model2 training", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"],
        ["Disk dolması (RICO unzip)", "Semantic annotations atlandı, unique_uis kurtarıldı"],
        ["Pod restart → veri kaybı", "HuggingFace Hub'a checkpoint yükleme"],
        ["TensorBoard yüklü değil", "pip install tensorboard"],
    ]
    story.append(table(issues_data, col_widths=[8*cm, 9.5*cm]))
    story.append(Spacer(1, 0.5 * cm))

    # ── 8. SONUÇ VE GELECEK ÇALIŞMA ──────────────────────────────────────
    story.append(Paragraph("8. Sonuç ve Gelecek Çalışmalar", h1_style))
    story.append(Paragraph(
        "Üç model başarıyla eğitilmiş ve Qwen2.5-VL-7B temel modelinin UI eleştirisi "
        "görevine uyarlanabildiği gösterilmiştir. Model 1 ve Model 3 karşılaştırması "
        "için ROUGE/BLEU metrik hesabı henüz tamamlanmamış olup bu çalışmanın bir "
        "sonraki aşamasını oluşturmaktadır.",
        body_style
    ))
    story.append(Spacer(1, 0.3 * cm))

    future_data = [
        ["Görev", "Durum"],
        ["Model 1 eğitimi (Qwen)", "✅ Tamamlandı"],
        ["Model 2 eğitimi (Qwen, 3K subset)", "✅ Tamamlandı"],
        ["Model 2 inference → pred_vh.json (500 görsel)", "✅ Tamamlandı"],
        ["Model 3 eğitimi (Qwen)", "✅ Tamamlandı"],
        ["Model 3 inference (örnek görsel)", "✅ Tamamlandı"],
        ["Model 1 vs Model 3 ROUGE/BLEU karşılaştırması", "⏳ Bekliyor"],
        ["Gemma 4 E4B ile Model 1/2/3 yeniden eğitim", "⏳ Devam ediyor"],
        ["Bounding box doğruluğunu iyileştirme", "⏳ Gelecek çalışma"],
    ]
    story.append(table(future_data, col_widths=[12*cm, 5.5*cm]))
    story.append(Spacer(1, 0.5 * cm))

    # ── FOOTER ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "UI Critic Project  ·  Umut Akbaş  ·  Mayıs 2026  ·  "
        "github.com/umutakbass/ui-critic-project",
        caption_style
    ))

    doc.build(story)
    print(f"Rapor oluşturuldu: {output_path}")
    return output_path


if __name__ == "__main__":
    build_report()
