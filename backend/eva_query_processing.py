from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "eva_products.csv"
SEARCH_INDEX_DIR = ROOT_DIR / "search_index"
REQUIRED_COLUMNS = [
    "name of product",
    "price",
    "how many in stock",
    "description",
    "type",
    "features",
]


def clean_text(value: Any) -> str:
    """Normalize text-like CSV values for display and search."""
    if value is None or pd.isna(value):
        return ""

    text = html.unescape(str(value))
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_col(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "too",
    "was",
    "will",
    "with",
    "you",
    "your",
}


try:
    from nltk.stem import PorterStemmer

    _stemmer = PorterStemmer()

    def stem_token(token: str) -> str:
        return _stemmer.stem(token)

except Exception:

    def stem_token(token: str) -> str:
        for suffix in ("ingly", "edly", "ing", "ed", "ly", "es", "s", "ment"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]
        return token


def normalize_term(term: str) -> str:
    term = re.sub(r"[^a-z0-9]+", "", term.lower())
    if not term or term in STOPWORDS:
        return ""
    return stem_token(term)


def tokenize(text: str) -> list[str]:
    text = clean_text(text).lower()
    raw_tokens = re.findall(r"[a-z0-9]+", text)
    filtered = [token for token in raw_tokens if token and token not in STOPWORDS]
    return [stem_token(token) for token in filtered]


def load_dataframe(data_path: Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(data_path, skipinitialspace=True)
    frame.columns = [normalize_col(column) for column in frame.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing expected CSV columns: {missing_columns}")

    frame = frame[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].map(clean_text)
    return frame


def stock_label(stock: str) -> str:
    if not stock:
        return "Stock unknown"

    try:
        stock_number = int(float(stock))
    except ValueError:
        return f"Stock: {stock}"

    if stock_number <= 0:
        return "Out of stock"
    if stock_number == 1:
        return "1 in stock"
    return f"{stock_number} in stock"


def build_documents(frame: pd.DataFrame) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        name = clean_text(row.get("name of product", ""))
        description = clean_text(row.get("description", ""))
        product_type = clean_text(row.get("type", ""))
        features = clean_text(row.get("features", ""))
        price = clean_text(row.get("price", ""))
        stock = clean_text(row.get("how many in stock", ""))
        stock_status = stock_label(stock)

        metadata_parts = [
            part
            for part in [product_type, features, price, stock_status]
            if part and part.lower() != "nan"
        ]
        metadata = " | ".join(metadata_parts)
        snippet_source = description or metadata or name
        snippet = snippet_source[:180] + ("..." if len(snippet_source) > 180 else "")
        content = " ".join(
            part
            for part in [name, description, product_type, features, price, stock_status]
            if part
        )

        docs.append(
            {
                "id": str(index + 1),
                "title": name or f"Product {index + 1}",
                "category": "products",
                "snippet": snippet,
                "metadata": metadata,
                "date": stock_status,
                "content": content,
                "price": price,
                "stock": stock,
                "type": product_type,
                "features": features,
            }
        )

    return docs


def build_inverted_index(
    documents: list[dict[str, Any]],
) -> tuple[dict[str, dict[int, int]], list[Counter[str]]]:
    inverted: dict[str, dict[int, int]] = defaultdict(dict)
    doc_term_freqs: list[Counter[str]] = []

    for doc_id, doc in enumerate(documents):
        tokens = tokenize(doc["content"])
        term_freq = Counter(tokens)
        doc_term_freqs.append(term_freq)

        for term, freq in term_freq.items():
            inverted[term][doc_id] = freq

    return dict(inverted), doc_term_freqs


def compute_tfidf(
    doc_term_freqs: list[Counter[str]],
    inverted_index: dict[str, dict[int, int]],
) -> tuple[dict[str, float], list[dict[str, float]], list[float]]:
    n_docs = len(doc_term_freqs)
    doc_frequency = {term: len(postings) for term, postings in inverted_index.items()}
    idf = {
        term: math.log((n_docs + 1) / (frequency + 1)) + 1.0
        for term, frequency in doc_frequency.items()
    }

    doc_vectors: list[dict[str, float]] = []
    doc_norms: list[float] = []

    for term_freq in doc_term_freqs:
        vector = {
            term: frequency * idf[term]
            for term, frequency in term_freq.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
        doc_vectors.append(vector)
        doc_norms.append(norm)

    return idf, doc_vectors, doc_norms


def query_search(
    query: str,
    documents: list[dict[str, Any]],
    inverted_index: dict[str, dict[int, int]],
    idf: dict[str, float],
    doc_vectors: list[dict[str, float]],
    doc_norms: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    terms = tokenize(query)
    if not terms:
        return []

    query_tf = Counter(terms)
    query_vector = {
        term: frequency * idf[term]
        for term, frequency in query_tf.items()
        if term in idf
    }
    if not query_vector:
        return []

    postings = [set(inverted_index.get(term, {}).keys()) for term in query_vector]
    if any(len(posting) == 0 for posting in postings):
        return []

    candidate_ids = set.intersection(*postings)

    query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0
    results: list[tuple[float, int]] = []

    for doc_id in candidate_ids:
        doc_vector = doc_vectors[doc_id]
        score = sum(
            query_weight * doc_vector.get(term, 0.0)
            for term, query_weight in query_vector.items()
        )
        score = score / (query_norm * doc_norms[doc_id])
        if score > 0:
            results.append((score, doc_id))

    results.sort(reverse=True, key=lambda result: result[0])
    return [
        {**documents[doc_id], "score": round(score, 6)}
        for score, doc_id in results[:top_k]
    ]


SYNONYM_MAP_RAW = {
    "shampoo": ["cleanser", "wash"],
    "cleanser": ["shampoo", "wash", "soap"],
    "conditioner": ["treatment", "cream", "moisturizer"],
    "cream": ["lotion", "moisturizer", "treatment"],
    "lotion": ["cream", "moisturizer"],
    "moisturizer": ["cream", "lotion", "hydrating"],
    "gel": ["styling", "hold", "wax"],
    "wax": ["styling", "gel"],
    "oil": ["serum", "argan"],
    "serum": ["oil", "treatment"],
    "mask": ["treatment"],
    "kids": ["children", "baby", "bebe"],
    "children": ["kids", "baby", "bebe"],
    "baby": ["children", "kids", "bebe"],
    "bebe": ["baby", "children", "kids"],
    "wipes": ["wipe"],
    "wipe": ["wipes"],
    "beard": ["grooming", "shaving"],
    "shaving": ["beard", "grooming"],
    "hair": ["scalp", "curls", "styling"],
    "curl": ["curly", "wavy", "coily", "curls"],
    "curls": ["curl", "curly", "wavy", "coily"],
    "skin": ["skincare", "body"],
    "aloe": ["vera"],
    "vera": ["aloe"],
    "fragrance": ["perfume", "scent", "body splash"],
    "perfume": ["fragrance", "scent"],
    "sunscreen": ["sunblock", "sun", "spf"],
    "dandruff": ["scalp"],
}


def build_synonym_map(raw_map: dict[str, list[str]]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for term, values in raw_map.items():
        term_tokens = tokenize(term)
        if not term_tokens:
            continue

        for term_norm in term_tokens:
            expanded: list[str] = []
            seen = set()
            for value in values:
                for value_norm in tokenize(value):
                    if value_norm and value_norm != term_norm and value_norm not in seen:
                        seen.add(value_norm)
                        expanded.append(value_norm)

            if expanded:
                normalized.setdefault(term_norm, [])
                for value_norm in expanded:
                    if value_norm not in normalized[term_norm]:
                        normalized[term_norm].append(value_norm)

    return normalized


SYNONYM_MAP = build_synonym_map(SYNONYM_MAP_RAW)


def expand_with_synonyms(
    terms: list[str],
    synonym_map: dict[str, list[str]],
    max_per_term: int = 3,
) -> list[str]:
    expanded: list[str] = []
    seen = set(terms)

    for term in terms:
        for synonym in synonym_map.get(term, [])[:max_per_term]:
            if synonym not in seen:
                seen.add(synonym)
                expanded.append(synonym)

    return expanded


def build_expanded_term_groups(
    terms: list[str],
    synonym_map: dict[str, list[str]],
    max_per_term: int = 3,
) -> list[set[str]]:
    """Return OR-groups for each query concept."""
    groups: list[set[str]] = []
    for term in terms:
        group = {term}
        group.update(synonym_map.get(term, [])[:max_per_term])
        groups.append(group)
    return groups


def feedback_terms_from_docs(
    doc_ids: list[int],
    doc_vectors: list[dict[str, float]],
    exclude_terms: set[str] | None = None,
    top_n: int = 5,
) -> list[str]:
    if exclude_terms is None:
        exclude_terms = set()

    term_scores: Counter[str] = Counter()
    for doc_id in doc_ids:
        for term, weight in doc_vectors[doc_id].items():
            if term not in exclude_terms:
                term_scores[term] += weight

    return [term for term, _ in term_scores.most_common(top_n)]


def hash_embedding_encoder(texts: list[str], dimensions: int = 256) -> np.ndarray:
    """Small deterministic vector encoder used when heavyweight models are unavailable."""
    rows = np.zeros((len(texts), dimensions), dtype=np.float32)

    for row_index, text in enumerate(texts):
        terms = tokenize(text)
        features: list[str] = []
        for term in terms:
            padded = f"^{term}$"
            features.append(term)
            features.extend(padded[index : index + 3] for index in range(max(1, len(padded) - 2)))

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=4).digest()
            column = int.from_bytes(digest, "big") % dimensions
            rows[row_index, column] += 1.0

    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.clip(norms, 1e-9, None)


def get_embedding_backend(backend: str, model_path: str | None = None):
    backend = backend.lower().strip()

    if backend == "bert":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_path or "all-MiniLM-L6-v2")
        except Exception:
            return hash_embedding_encoder, ""

        def encode(texts: list[str]) -> np.ndarray:
            return model.encode(texts, normalize_embeddings=True)

        return encode, ""

    if backend == "elmo":
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            model = hub.load("https://tfhub.dev/google/elmo/3")
        except Exception as exc:
            return None, f"ELMo expansion unavailable: {exc}"

        def encode(texts: list[str]) -> np.ndarray:
            embeddings = model.signatures["default"](tf.constant(texts))["elmo"]
            vectors = tf.reduce_mean(embeddings, axis=1).numpy()
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / np.clip(norms, 1e-9, None)

        return encode, ""

    if backend == "rnn":
        if not model_path:
            return None, "Provide model_path for the RNN vector file."
        try:
            from gensim.models import KeyedVectors

            vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
        except Exception as exc:
            return None, f"RNN expansion unavailable: {exc}"

        def encode(texts: list[str]) -> np.ndarray:
            rows = []
            for text in texts:
                terms = tokenize(text)
                term_vectors = [vectors[term] for term in terms if term in vectors]
                if term_vectors:
                    row = np.mean(term_vectors, axis=0)
                else:
                    row = np.zeros(vectors.vector_size, dtype=np.float32)
                rows.append(row)
            rows = np.vstack(rows)
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            return rows / np.clip(norms, 1e-9, None)

        return encode, ""

    return None, f"Unknown backend: {backend}"


def build_embedding_index(vocab: list[str], encoder) -> np.ndarray:
    vectors = encoder(vocab)
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


EMBEDDING_STATE = {
    "backend": None,
    "model_path": None,
    "vocab": None,
    "vectors": None,
    "encoder": None,
}


def prepare_embedding_index(
    vocab: list[str],
    backend: str = "bert",
    model_path: str | None = None,
) -> dict[str, object]:
    if (
        EMBEDDING_STATE["backend"] == backend
        and EMBEDDING_STATE["model_path"] == model_path
        and EMBEDDING_STATE["vocab"] == vocab
    ):
        return EMBEDDING_STATE

    encoder, error = get_embedding_backend(backend, model_path)
    if encoder is None:
        raise RuntimeError(error)

    vectors = build_embedding_index(vocab, encoder)
    EMBEDDING_STATE.update(
        {
            "backend": backend,
            "model_path": model_path,
            "vocab": vocab,
            "vectors": vectors,
            "encoder": encoder,
        }
    )
    return EMBEDDING_STATE


def expand_with_embeddings(
    query_terms: list[str],
    vocab: list[str],
    backend: str = "bert",
    model_path: str | None = None,
    top_n: int = 5,
    min_similarity: float = 0.35,
) -> list[str]:
    state = prepare_embedding_index(vocab, backend=backend, model_path=model_path)
    encoder = state["encoder"]
    vocab_vectors = state["vectors"]

    query_text = " ".join(query_terms)
    query_vector = np.asarray(encoder([query_text])[0], dtype=np.float32)
    query_norm = np.linalg.norm(query_vector) or 1.0
    query_vector = query_vector / query_norm

    scores = vocab_vectors @ query_vector
    sorted_idx = np.argsort(-scores)
    expansions: list[str] = []

    for index in sorted_idx:
        term = vocab[index]
        if term in query_terms:
            continue
        if scores[index] < min_similarity:
            break
        expansions.append(term)
        if len(expansions) >= top_n:
            break

    return expansions


def build_query_vector(
    term_weights: dict[str, float],
    idf: dict[str, float],
) -> dict[str, float]:
    return {
        term: weight * idf[term]
        for term, weight in term_weights.items()
        if term in idf and weight > 0.0
    }


def score_candidates(
    candidate_ids: set[int],
    query_vector: dict[str, float],
    doc_vectors: list[dict[str, float]],
    doc_norms: list[float],
) -> list[tuple[float, int]]:
    if not query_vector:
        return []

    query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0
    scored: list[tuple[float, int]] = []

    for doc_id in candidate_ids:
        doc_vector = doc_vectors[doc_id]
        score = sum(
            query_weight * doc_vector.get(term, 0.0)
            for term, query_weight in query_vector.items()
        )
        score = score / (query_norm * doc_norms[doc_id])
        if score > 0:
            scored.append((score, doc_id))

    scored.sort(reverse=True, key=lambda item: item[0])
    return scored


def intersect_postings(
    terms: list[str],
    inverted_index: dict[str, dict[int, int]],
) -> set[int]:
    postings = [set(inverted_index.get(term, {}).keys()) for term in terms]
    if not postings or any(len(posting) == 0 for posting in postings):
        return set()
    return set.intersection(*postings)


def union_postings(
    terms: set[str],
    inverted_index: dict[str, dict[int, int]],
) -> set[int]:
    candidate_ids: set[int] = set()
    for term in terms:
        candidate_ids.update(inverted_index.get(term, {}).keys())
    return candidate_ids


def candidates_from_term_groups(
    term_groups: list[set[str]],
    inverted_index: dict[str, dict[int, int]],
) -> set[int]:
    """Require at least one term from every query concept group."""
    group_postings: list[set[int]] = []
    for group in term_groups:
        matches = union_postings(group, inverted_index)
        if not matches:
            return set()
        group_postings.append(matches)

    return set.intersection(*group_postings) if group_postings else set()


def unique_terms(terms: list[str]) -> list[str]:
    ordered_terms: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            ordered_terms.append(term)
    return ordered_terms


def is_displayable_expansion_term(term: str) -> bool:
    return any(character.isalpha() for character in term)


def build_suggested_queries(
    query: str,
    expanded_terms: list[str],
    max_suggestions: int = 4,
) -> list[str]:
    base_query = clean_text(query)
    if not base_query:
        return []

    suggestions: list[str] = []
    seen_queries = {base_query.lower()}
    for term in expanded_terms:
        candidate = clean_text(f"{base_query} {term}")
        if not candidate:
            continue
        candidate_lower = candidate.lower()
        if candidate_lower in seen_queries:
            continue
        seen_queries.add(candidate_lower)
        suggestions.append(candidate)
        if len(suggestions) >= max_suggestions:
            break
    return suggestions


SUGGESTION_SKIP_WORDS = {"ml", "gm", "g", "kg", "le"}


def extract_suggestion_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", clean_text(text).lower())
    return [
        word
        for word in words
        if word
        and word not in STOPWORDS
        and word not in SUGGESTION_SKIP_WORDS
        and not word.isdigit()
    ]


def add_query_suggestion(counter: Counter[str], phrase: str, weight: int = 1) -> None:
    normalized = re.sub(r"\s+", " ", clean_text(phrase).lower()).strip()
    if not normalized or len(normalized) < 2:
        return
    counter[normalized] += weight


def build_query_suggestion_counter(documents: list[dict[str, Any]]) -> Counter[str]:
    suggestions: Counter[str] = Counter()

    for query in TEST_QUERIES:
        add_query_suggestion(suggestions, query, weight=20)

    for term, related_terms in SYNONYM_MAP_RAW.items():
        add_query_suggestion(suggestions, term, weight=8)
        for related in related_terms:
            add_query_suggestion(suggestions, related, weight=4)

    for doc in documents:
        title_words = extract_suggestion_words(doc.get("title", ""))
        for length in range(1, min(5, len(title_words)) + 1):
            add_query_suggestion(suggestions, " ".join(title_words[:length]), weight=6 - min(length, 5))
        for length in range(2, min(4, len(title_words)) + 1):
            for start in range(0, len(title_words) - length + 1):
                add_query_suggestion(suggestions, " ".join(title_words[start : start + length]), weight=2)

        for feature_chunk in clean_text(doc.get("features", "")).split(";"):
            feature_words = extract_suggestion_words(feature_chunk)
            for length in range(1, min(3, len(feature_words)) + 1):
                add_query_suggestion(suggestions, " ".join(feature_words[:length]), weight=2)

        product_type_words = extract_suggestion_words(doc.get("type", ""))
        for length in range(1, min(3, len(product_type_words)) + 1):
            add_query_suggestion(suggestions, " ".join(product_type_words[:length]), weight=2)

    return suggestions


def score_query_suggestion(query: str, suggestion: str, weight: int) -> float | None:
    if suggestion == query:
        return None

    score = float(weight * 10)
    if suggestion.startswith(query):
        score += 1000
    elif f" {query}" in suggestion:
        score += 700
    elif any(word.startswith(query) for word in suggestion.split()):
        score += 450
    elif query in suggestion:
        score += 200
    else:
        return None

    score -= len(suggestion) * 0.25
    return score


def suggest_queries(query: str, limit: int = 8) -> list[str]:
    normalized_query = re.sub(r"\s+", " ", clean_text(query).lower()).strip()
    if len(normalized_query) < 2:
        return []

    ranked: list[tuple[float, str]] = []
    for suggestion, weight in QUERY_SUGGESTION_COUNTER.items():
        score = score_query_suggestion(normalized_query, suggestion, weight)
        if score is not None:
            ranked.append((score, suggestion))

    ranked.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return [suggestion for _, suggestion in ranked[:limit]]


def analyze_query_expansion(
    query: str,
    documents: list[dict[str, Any]],
    inverted_index: dict[str, dict[int, int]],
    idf: dict[str, float],
    doc_vectors: list[dict[str, float]],
    doc_norms: list[float],
    feedback_docs: int = 5,
    feedback_terms: int = 5,
    use_synonyms: bool = True,
    use_feedback: bool = True,
    use_embeddings: bool = False,
    embedding_backend: str = "bert",
    embedding_model_path: str | None = None,
    embedding_top_n: int = 5,
    embedding_min_sim: float = 0.35,
    synonym_weight: float = 0.7,
    feedback_weight: float = 0.6,
    embedding_weight: float = 0.6,
) -> dict[str, Any]:
    base_terms = tokenize(query)
    if not base_terms:
        return {
            "query": clean_text(query),
            "base_terms": [],
            "synonym_terms": [],
            "feedback_terms": [],
            "embedding_terms": [],
            "expanded_terms": [],
            "suggested_queries": [],
            "expanded_vector": {},
            "candidate_ids": set(),
        }

    base_weights = dict(Counter(base_terms))
    base_vector = build_query_vector(base_weights, idf)
    if not base_vector:
        return {
            "query": clean_text(query),
            "base_terms": base_terms,
            "synonym_terms": [],
            "feedback_terms": [],
            "embedding_terms": [],
            "expanded_terms": [],
            "suggested_queries": [],
            "expanded_vector": {},
            "candidate_ids": set(),
        }

    exact_candidate_ids = intersect_postings(base_terms, inverted_index)
    base_ranked = score_candidates(exact_candidate_ids, base_vector, doc_vectors, doc_norms)

    feedback_terms_list: list[str] = []
    if use_feedback and base_ranked:
        top_doc_ids = [doc_id for _, doc_id in base_ranked[:feedback_docs]]
        feedback_terms_list = feedback_terms_from_docs(
            top_doc_ids,
            doc_vectors,
            exclude_terms=set(base_terms),
            top_n=feedback_terms,
        )

    synonym_terms = expand_with_synonyms(base_terms, SYNONYM_MAP) if use_synonyms else []

    embedding_terms: list[str] = []
    if use_embeddings:
        try:
            embedding_terms = expand_with_embeddings(
                base_terms,
                vocab=VOCAB,
                backend=embedding_backend,
                model_path=embedding_model_path,
                top_n=embedding_top_n,
                min_similarity=embedding_min_sim,
            )
        except RuntimeError as exc:
            print(f"Embedding expansion disabled: {exc}")
            embedding_terms = []

    term_weights: dict[str, float] = dict(base_weights)
    for term in synonym_terms:
        term_weights.setdefault(term, synonym_weight)
    for term in feedback_terms_list:
        term_weights.setdefault(term, feedback_weight)
    for term in embedding_terms:
        term_weights.setdefault(term, embedding_weight)

    expanded_vector = build_query_vector(term_weights, idf)
    if not expanded_vector:
        return {
            "query": clean_text(query),
            "base_terms": base_terms,
            "synonym_terms": synonym_terms,
            "feedback_terms": feedback_terms_list,
            "embedding_terms": embedding_terms,
            "expanded_terms": [],
            "suggested_queries": [],
            "expanded_vector": {},
            "candidate_ids": set(),
        }

    if use_synonyms:
        term_groups = build_expanded_term_groups(base_terms, SYNONYM_MAP)
        candidate_ids = candidates_from_term_groups(term_groups, inverted_index)
    else:
        candidate_ids = set(exact_candidate_ids)

    candidate_ids.update(exact_candidate_ids)
    candidate_ids.update(union_postings(set(embedding_terms), inverted_index))

    if not candidate_ids:
        candidate_ids = union_postings(set(expanded_vector), inverted_index)

    expanded_terms = unique_terms(
        [
            *synonym_terms,
            *feedback_terms_list,
            *embedding_terms,
        ]
    )
    expanded_terms = [term for term in expanded_terms if is_displayable_expansion_term(term)]

    return {
        "query": clean_text(query),
        "base_terms": unique_terms(base_terms),
        "synonym_terms": [term for term in unique_terms(synonym_terms) if is_displayable_expansion_term(term)],
        "feedback_terms": [term for term in unique_terms(feedback_terms_list) if is_displayable_expansion_term(term)],
        "embedding_terms": [term for term in unique_terms(embedding_terms) if is_displayable_expansion_term(term)],
        "expanded_terms": expanded_terms,
        "suggested_queries": build_suggested_queries(query, expanded_terms),
        "expanded_vector": expanded_vector,
        "candidate_ids": candidate_ids,
    }


def query_search_expanded(
    query: str,
    documents: list[dict[str, Any]],
    inverted_index: dict[str, dict[int, int]],
    idf: dict[str, float],
    doc_vectors: list[dict[str, float]],
    doc_norms: list[float],
    top_k: int = 10,
    feedback_docs: int = 5,
    feedback_terms: int = 5,
    use_synonyms: bool = True,
    use_feedback: bool = True,
    use_embeddings: bool = False,
    embedding_backend: str = "bert",
    embedding_model_path: str | None = None,
    embedding_top_n: int = 5,
    embedding_min_sim: float = 0.35,
    synonym_weight: float = 0.7,
    feedback_weight: float = 0.6,
    embedding_weight: float = 0.6,
) -> list[dict[str, Any]]:
    analysis = analyze_query_expansion(
        query,
        documents,
        inverted_index,
        idf,
        doc_vectors,
        doc_norms,
        feedback_docs=feedback_docs,
        feedback_terms=feedback_terms,
        use_synonyms=use_synonyms,
        use_feedback=use_feedback,
        use_embeddings=use_embeddings,
        embedding_backend=embedding_backend,
        embedding_model_path=embedding_model_path,
        embedding_top_n=embedding_top_n,
        embedding_min_sim=embedding_min_sim,
        synonym_weight=synonym_weight,
        feedback_weight=feedback_weight,
        embedding_weight=embedding_weight,
    )
    expanded_vector = analysis["expanded_vector"]
    candidate_ids = analysis["candidate_ids"]

    if not expanded_vector or not candidate_ids:
        return []

    ranked = score_candidates(candidate_ids, expanded_vector, doc_vectors, doc_norms)

    return [
        {**documents[doc_id], "score": round(score, 6)}
        for score, doc_id in ranked[:top_k]
    ]


def run_expanded_search_with_analysis(
    query: str,
    top_k: int = 10,
    feedback_docs: int = 5,
    feedback_terms: int = 5,
    use_synonyms: bool = True,
    use_feedback: bool = True,
    use_embeddings: bool = False,
    embedding_backend: str = "bert",
    embedding_model_path: str | None = None,
    embedding_top_n: int = 5,
    embedding_min_sim: float = 0.35,
    synonym_weight: float = 0.7,
    feedback_weight: float = 0.6,
    embedding_weight: float = 0.6,
) -> dict[str, Any]:
    analysis = analyze_query_expansion(
        query,
        docs,
        inverted_index,
        idf,
        doc_vectors,
        doc_norms,
        feedback_docs=feedback_docs,
        feedback_terms=feedback_terms,
        use_synonyms=use_synonyms,
        use_feedback=use_feedback,
        use_embeddings=use_embeddings,
        embedding_backend=embedding_backend,
        embedding_model_path=embedding_model_path,
        embedding_top_n=embedding_top_n,
        embedding_min_sim=embedding_min_sim,
        synonym_weight=synonym_weight,
        feedback_weight=feedback_weight,
        embedding_weight=embedding_weight,
    )

    results = query_search_expanded(
        query,
        docs,
        inverted_index,
        idf,
        doc_vectors,
        doc_norms,
        top_k=top_k,
        feedback_docs=feedback_docs,
        feedback_terms=feedback_terms,
        use_synonyms=use_synonyms,
        use_feedback=use_feedback,
        use_embeddings=use_embeddings,
        embedding_backend=embedding_backend,
        embedding_model_path=embedding_model_path,
        embedding_top_n=embedding_top_n,
        embedding_min_sim=embedding_min_sim,
        synonym_weight=synonym_weight,
        feedback_weight=feedback_weight,
        embedding_weight=embedding_weight,
    )

    return {
        "results": results,
        "expansion": {
            "query": analysis["query"],
            "base_terms": analysis["base_terms"],
            "synonym_terms": analysis["synonym_terms"],
            "feedback_terms": analysis["feedback_terms"],
            "embedding_terms": analysis["embedding_terms"],
            "expanded_terms": analysis["expanded_terms"],
            "suggested_queries": analysis["suggested_queries"],
        },
    }


def run_search_ui(query: str, top_k: int = 10) -> pd.DataFrame:
    results = run_expanded_search(query, top_k=top_k)
    return pd.DataFrame(
        [
            {
                "title": item["title"],
                "score": item["score"],
                "metadata": item["metadata"],
                "snippet": item["snippet"],
            }
            for item in results
        ]
    )


def launch_search_widgets(
    initial_query: str = "aloe vera shampoo",
    top_k: int = 5,
) -> None:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    query_box = widgets.Text(
        value=initial_query,
        description="Query:",
        placeholder="Type your search...",
    )
    topk_box = widgets.IntSlider(
        value=top_k,
        min=1,
        max=20,
        step=1,
        description="Top K:",
    )
    search_button = widgets.Button(description="Search", button_style="primary")
    output = widgets.Output()

    def on_search_click(_):
        with output:
            clear_output()
            display(run_search_ui(query_box.value, top_k=topk_box.value))

    search_button.on_click(on_search_click)
    display(query_box, topk_box, search_button, output)


def build_doc_term_sets(documents: list[dict[str, Any]]) -> list[set[str]]:
    return [set(tokenize(doc["content"])) for doc in documents]


def heuristic_relevant_indices(
    query: str,
    doc_term_sets: list[set[str]],
) -> set[int]:
    terms = tokenize(query)
    if not terms:
        return set()
    return {
        index
        for index, terms_set in enumerate(doc_term_sets)
        if all(term in terms_set for term in terms)
    }


def evaluate_queries(
    queries: list[str],
    search_fn,
    documents: list[dict[str, Any]],
    doc_term_sets: list[set[str]],
    gold_labels: dict[str, list[int]] | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    doc_id_to_index = {doc["id"]: index for index, doc in enumerate(documents)}
    rows = []

    for query in queries:
        start = perf_counter()
        results = search_fn(query, top_k=top_k)
        elapsed_ms = (perf_counter() - start) * 1000

        result_indices = [
            doc_id_to_index[result["id"]]
            for result in results
            if result.get("id") in doc_id_to_index
        ]

        if gold_labels and query in gold_labels:
            relevant = set(gold_labels[query])
            label_source = "manual"
        else:
            relevant = heuristic_relevant_indices(query, doc_term_sets)
            label_source = "heuristic"

        retrieved_relevant = set(result_indices) & relevant
        precision = len(retrieved_relevant) / len(result_indices) if result_indices else 0.0
        recall = len(retrieved_relevant) / len(relevant) if relevant else 0.0

        rows.append(
            {
                "query": query,
                "results": len(result_indices),
                "latency_ms": round(elapsed_ms, 2),
                "precision@k": round(precision, 3),
                "recall@k": round(recall, 3),
                "relevant_docs": len(relevant),
                "label_source": label_source,
            }
        )

    return pd.DataFrame(rows)


def export_search_assets(
    documents: list[dict[str, Any]],
    inverted_index: dict[str, dict[int, int]],
    idf: dict[str, float],
    doc_vectors: list[dict[str, float]],
    doc_norms: list[float],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    export_docs = [
        {
            "id": doc["id"],
            "title": doc["title"],
            "category": doc["category"],
            "snippet": doc["snippet"],
            "metadata": doc["metadata"],
            "date": doc["date"],
        }
        for doc in documents
    ]

    vocab = sorted(idf.keys())
    term_to_index = {term: index for index, term in enumerate(vocab)}

    payload = {
        "version": 1,
        "docs": export_docs,
        "vocab": vocab,
        "idf": [round(idf[term], 6) for term in vocab],
        "doc_norms": [round(norm, 6) for norm in doc_norms],
        "doc_vectors": [
            [
                [term_to_index[term], round(weight, 6)]
                for term, weight in vector.items()
            ]
            for vector in doc_vectors
        ],
        "inverted_index": {
            term: [[doc_id, freq] for doc_id, freq in sorted(postings.items())]
            for term, postings in inverted_index.items()
        },
    }

    docs_path = output_dir / "eva_products_search_docs.json"
    index_path = output_dir / "eva_products_tfidf_index.json"

    with docs_path.open("w", encoding="utf-8") as file:
        json.dump(export_docs, file, ensure_ascii=True, indent=2)

    with index_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)

    return docs_path, index_path


EMBEDDING_BACKEND = "bert"
EMBEDDING_MODEL_PATH = None
USE_EMBEDDINGS = False

TEST_QUERIES = [
    "aloe vera shampoo",
    "beard oil",
    "baby wipes",
    "hair gel",
    "shaving cream",
    "curl defining shampoo",
    "children wipes",
    "anti dandruff shampoo",
]

GOLD_LABELS: dict[str, list[int]] = {
    "aloe vera shampoo": [35],
    "beard oil": [7, 10],
    "baby wipes": [30, 32],
    "hair gel": [6, 9, 13, 17, 19, 77],
    "shaving cream": [14, 18, 21, 22, 23, 24],
    "curl defining shampoo": [74],
    "children wipes": [30, 32],
    "anti dandruff shampoo": [96],
}

df = load_dataframe()
docs = build_documents(df)
inverted_index, doc_term_freqs = build_inverted_index(docs)
idf, doc_vectors, doc_norms = compute_tfidf(doc_term_freqs, inverted_index)
VOCAB = sorted(idf.keys())
DOC_TERM_SETS = build_doc_term_sets(docs)
QUERY_SUGGESTION_COUNTER = build_query_suggestion_counter(docs)


def run_expanded_search(
    query: str,
    top_k: int = 10,
    feedback_docs: int = 5,
    feedback_terms: int = 5,
    use_synonyms: bool = True,
    use_feedback: bool = True,
    use_embeddings: bool = USE_EMBEDDINGS,
    embedding_backend: str = EMBEDDING_BACKEND,
    embedding_model_path: str | None = EMBEDDING_MODEL_PATH,
    embedding_top_n: int = 5,
    embedding_min_sim: float = 0.35,
    synonym_weight: float = 0.7,
    feedback_weight: float = 0.6,
    embedding_weight: float = 0.6,
) -> list[dict[str, Any]]:
    return query_search_expanded(
        query,
        docs,
        inverted_index,
        idf,
        doc_vectors,
        doc_norms,
        top_k=top_k,
        feedback_docs=feedback_docs,
        feedback_terms=feedback_terms,
        use_synonyms=use_synonyms,
        use_feedback=use_feedback,
        use_embeddings=use_embeddings,
        embedding_backend=embedding_backend,
        embedding_model_path=embedding_model_path,
        embedding_top_n=embedding_top_n,
        embedding_min_sim=embedding_min_sim,
        synonym_weight=synonym_weight,
        feedback_weight=feedback_weight,
        embedding_weight=embedding_weight,
    )


def run_notebook_evaluation(top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_df = evaluate_queries(
        TEST_QUERIES,
        run_expanded_search,
        docs,
        DOC_TERM_SETS,
        gold_labels=GOLD_LABELS,
        top_k=top_k,
    )
    summary_df = pd.DataFrame(
        [
            {
                "queries": len(results_df),
                "avg_latency_ms": round(results_df["latency_ms"].mean(), 2),
                "mean_precision@10": round(results_df["precision@k"].mean(), 3),
                "mean_recall@10": round(results_df["recall@k"].mean(), 3),
            }
        ]
    )
    return results_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Eva product search pipeline extracted from the notebook.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Run an expanded search query.")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--no-synonyms", action="store_true")
    search_parser.add_argument("--no-feedback", action="store_true")
    search_parser.add_argument("--use-embeddings", action="store_true")
    search_parser.add_argument("--embedding-backend", default=EMBEDDING_BACKEND)
    search_parser.add_argument("--embedding-model-path")

    export_parser = subparsers.add_parser("export", help="Export notebook search assets to JSON.")
    export_parser.add_argument("--output-dir", default=str(SEARCH_INDEX_DIR))

    eval_parser = subparsers.add_parser("evaluate", help="Run the notebook evaluation queries.")
    eval_parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    if args.command == "search":
        results = run_expanded_search(
            args.query,
            top_k=max(1, args.top_k),
            use_synonyms=not args.no_synonyms,
            use_feedback=not args.no_feedback,
            use_embeddings=args.use_embeddings,
            embedding_backend=args.embedding_backend,
            embedding_model_path=args.embedding_model_path,
        )
        print(json.dumps(results, ensure_ascii=True, indent=2))
        return

    if args.command == "export":
        docs_path, index_path = export_search_assets(
            docs,
            inverted_index,
            idf,
            doc_vectors,
            doc_norms,
            Path(args.output_dir),
        )
        print(f"Exported docs to {docs_path}")
        print(f"Exported index to {index_path}")
        return

    if args.command == "evaluate":
        results_df, summary_df = run_notebook_evaluation(top_k=max(1, args.top_k))
        print(results_df.to_string(index=False))
        print()
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
