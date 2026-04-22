"""Eğitim ve inference için prompt şablonları."""

MODEL1_SYSTEM_PROMPT = (
    "Sen bir UI/UX tasarım uzmanısın. "
    "Sana gösterilen mobil uygulama ekran görüntüsünü analiz et ve "
    "profesyonel bir tasarım kritiği üret."
)

MODEL1_USER_INSTRUCTION = """\
Bu UI ekran görüntüsünü değerlendir. Çıktını aşağıdaki JSON formatında ver:

{
  "critiques": [
    {
      "comment": "...",
      "bounding_box": [x1, y1, x2, y2],
      "severity": "low|medium|high",
      "category": "accessibility|readability|hierarchy|consistency|aesthetics"
    }
  ],
  "overall_feedback": "..."
}"""

MODEL2_USER_INSTRUCTION = """\
Bu UI ekranının view hierarchy'sini (yapısal ağacını) çıkar. \
Her element için tip, bounds ve (varsa) text bilgisini JSON formatında ver.

Örnek:
{
  "type": "FrameLayout",
  "bounds": [0, 0, 1080, 1920],
  "children": [
    {"type": "TextView", "bounds": [...], "text": "..."},
    ...
  ]
}"""

MODEL3_USER_INSTRUCTION_TEMPLATE = """\
Bu UI ekranının yapısal hiyerarşisi:
{hierarchy_json}

Yukarıdaki hiyerarşi bilgisini kullanarak bu UI için tasarım kritiği üret. \
Çıktını Model 1 ile aynı JSON formatında ver."""
