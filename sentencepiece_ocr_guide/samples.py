"""A stratified default sample set.

The validation checklist calls for "a stratified sample from every script and from math". This
is a small built-in stand-in so the CLI does something useful with no arguments — it is *not* a
substitute for samples drawn from your own ground truth, which is the only text whose
normalization and domain actually match what you will train on.

Each group is chosen to exercise a specific trap from the guide rather than to be
representative prose.
"""

from types import MappingProxyType

# Group label -> samples. Fertility ceilings are per-group because a single global average
# hides exactly the imbalance the check is looking for.
DEFAULT_SAMPLES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "latin": (
            "The quick brown fox jumps over the lazy dog.",
            "Ligature test: office, flour, definite.",
        ),
        "cyrillic": (
            "Жебракує філософ тьмяне сяйво знань, чуючи гидоту зимових вечорів.",
            # ґ ї є і and the apostrophe are exactly what a Russian-trained tokenizer lacks.
            "Ґудзик, аґрус і п'ятдесят щиглів: об'єкти цієї шафи.",
        ),
        # Latin script, but the trap is normalization rather than script: stacked
        # tone-plus-vowel diacritics are where NFD input silently doubles fertility.
        "vietnamese": (
            "Hệ thống nhận dạng ký tự quang học cần dữ liệu cân bằng.",
            "Đường phố Hà Nội rợp bóng cây bàng mùa thu.",
        ),
        "greek": ("Ταχίστη αλώπηξ βαφής ψημένη γη.",),
        "cjk": (
            "光学字符识别系统的准确率取决于标注质量。",
            "日本語には漢字とひらがなとカタカナが混在する。",
            "한국어 문자 인식 시스템의 정확도를 측정한다.",
        ),
        "arabic": (
            "نظام التعرف الضوئي على الحروف يحتاج إلى بيانات متوازنة.",
            "می‌رود و میرود دو کلمهٔ متفاوت هستند.",  # ZWNJ — failure mode #3
        ),
        "hebrew": (
            "מערכת זיהוי תווים אופטי דורשת נתונים מאוזנים.",
            # Pointed text: niqqud roughly double the codepoints per word, so a vocabulary
            # trained on unpointed text tokenizes this at a wildly different rate.
            "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ.",
        ),
        "devanagari": (
            "प्रकाशिक वर्ण पहचान प्रणाली की सटीकता।",
            "क्ष त्र ज्ञ श्र संयुक्ताक्षर हैं।",
        ),
        # Bengali is conjunct-heavy where Tamil is not; one Indic group tests one Indic
        # behaviour, so Devanagari alone does not stand in for the others.
        "bengali": ("আলোকীয় অক্ষর শনাক্তকরণ ব্যবস্থার নির্ভুলতা।",),
        "tamil": ("ஒளி எழுத்து உணரும் அமைப்பின் துல்லியம்.",),
        "thai": ("ระบบรู้จำอักขระด้วยแสงต้องการข้อมูลที่สมดุล",),
        "math": (
            r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
            r"\int_{0}^{\infty} e^{-x^2} \, dx = \frac{\sqrt{\pi}}{2}",
            r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}",
            r"\sum_{i=1}^{n} x_i \approx \mu \pm \sigma",
        ),
        "digits": (
            "Invoice 4827193 total 1284.65 due 2026-03-14",
            "Readings: 0.00317, 98765, 1000000, 42",
        ),
        "whitespace": (
            "  leading and trailing spaces  ",
            "column\tseparated\tby\ttabs",
            "double  internal  spaces",
        ),
        "fullwidth": (
            "Ｆｕｌｌｗｉｄｔｈ　ＡＢＣ　１２３",
            "半角カナ ﾊﾝｶｸ と全角の混在",
        ),
    }
)


def all_samples() -> tuple[str, ...]:
    """Every default sample, flattened — for checks that take one undifferentiated corpus."""
    return tuple(sample for group in DEFAULT_SAMPLES.values() for sample in group)
