import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arabic_nlp import BM25Vectorizer, CountVectorizer, HashingVectorizer, TfidfVectorizer


def test_count_vectorizer_sparse():
    corpus = [["kitab", "jadid"], ["kitab", "qadim"]]
    vec = CountVectorizer()
    vec.fit(corpus)
    output = vec.transform(["kitab", "jadid"], return_sparse=True)
    print("count sparse:", output)
    assert any(value > 0 for value in output.values())


def test_tfidf_vectorizer_dense():
    corpus = [["kitab", "jadid"], ["kitab", "qadim"]]
    vec = TfidfVectorizer()
    vec.fit(corpus)
    output = vec.transform(["kitab", "jadid"], return_sparse=False)
    print("tfidf dense:", output)
    assert len(output) == len(vec.vocab_.id_to_token)


def test_bm25_vectorizer_sparse():
    corpus = [["kitab", "jadid"], ["kitab", "qadim"]]
    vec = BM25Vectorizer()
    vec.fit(corpus)
    output = vec.transform(["kitab", "jadid"], return_sparse=True)
    print("bm25 sparse:", output)
    assert any(value != 0 for value in output.values())


def test_ir_with_arabic_tokens():
    corpus = [["كتاب", "جديد"], ["كتاب", "قديم"]]
    vec = CountVectorizer()
    vec.fit(corpus)
    output = vec.transform(["كتاب", "جديد"], return_sparse=True)
    print("arabic count sparse:", output)
    assert any(value > 0 for value in output.values())


def test_hashing_vectorizer_dense():
    vec = HashingVectorizer(n_features=32)
    output = vec.transform(["kitab", "jadid"], return_sparse=False)
    print("hashing dense:", output)
    assert len(output) == 32


def run_all():
    test_count_vectorizer_sparse()
    test_tfidf_vectorizer_dense()
    test_bm25_vectorizer_sparse()
    test_ir_with_arabic_tokens()
    test_hashing_vectorizer_dense()


if __name__ == "__main__":
    run_all()
