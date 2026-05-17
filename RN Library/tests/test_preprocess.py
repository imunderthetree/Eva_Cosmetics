import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arabic_nlp import ArabicPreprocessor, normalize_arabic, tokenize_arabic


def test_normalize_arabic_removes_diacritics_and_tatweel():
    text = "\u0640\u0627\u0644\u0645\u064E\u0643\u062A\u064E\u0628"
    normalized = normalize_arabic(text)
    print("normalized:", normalized)
    assert "\u0640" not in normalized
    assert "\u064E" not in normalized


def test_tokenize_arabic_split_clitics():
    text = "\u0648\u0627\u0644\u0643\u062A\u0627\u0628"
    tokens = tokenize_arabic(text, split_clitics=True)
    print("split_clitics tokens:", tokens)
    assert tokens[0] == "\u0648\u0627\u0644"
    assert tokens[1] == "\u0643\u062A\u0627\u0628"


def test_tokenize_arabic_literal_text():
    text = "والكتاب جديد"
    tokens = tokenize_arabic(text, split_clitics=True)
    print("literal arabic tokens:", tokens)
    assert tokens[0] == "وال"
    assert tokens[1] == "كتاب"
    assert tokens[2] == "جديد"


def test_preprocess_with_stopwords():
    pre = ArabicPreprocessor(stopwords={"\u0643\u062A\u0627\u0628"}, use_light_stemmer=False)
    tokens = pre("\u0643\u062A\u0627\u0628 \u062C\u062F\u064A\u062F")
    print("preprocess tokens:", tokens)
    assert tokens == ["\u062C\u062F\u064A\u062F"]


def run_all():
    test_normalize_arabic_removes_diacritics_and_tatweel()
    test_tokenize_arabic_split_clitics()
    test_tokenize_arabic_literal_text()
    test_preprocess_with_stopwords()


if __name__ == "__main__":
    run_all()
