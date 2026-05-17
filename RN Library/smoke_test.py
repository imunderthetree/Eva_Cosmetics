from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path


def _add_src_to_path() -> None:
	root = Path(__file__).resolve().parent
	src = root / "src"
	if src.exists() and str(src) not in sys.path:
		sys.path.insert(0, str(src))


def _case_fold_fallback(text: str, language: str = "arabic") -> str:
	if language == "arabic":
		return normalize_arabic(text)
	return unicodedata.normalize("NFKC", text).casefold()


_add_src_to_path()

import arabic_nlp as rna
from arabic_nlp import (
	normalize_arabic,
	soundex_arabic,
	thesaurus_lookup,
	lemmatize_arabic,
	tokenize_arabic,
	tokenize_sentences,
	split_clitic_tokens,
	get_arabic_stopwords,
	load_stopwords,
	remove_stopwords,
	light_stem,
	stem_tokens,
	get_isri_stemmer,
	ArabicPreprocessor,
	PreprocessConfig,
	preprocess_text,
)
from arabic_nlp.ir import (
	term_frequencies,
	document_frequencies,
	inverse_document_frequencies,
	word_ngrams,
	char_ngrams,
	document_stats,
	CountVectorizer,
	TfidfVectorizer,
	BM25Vectorizer,
	HashingVectorizer,
)
from arabic_nlp.representations import (
	CBOW,
	SkipGram,
	Word2VecConfig,
	BertEncoder,
	ElmoEncoder,
)


case_fold = rna.case_fold if hasattr(rna, "case_fold") else _case_fold_fallback


def _header(title: str) -> None:
	print("\n== " + title + " ==")


def main() -> None:
	print("Package path:", getattr(rna, "__file__", "<unknown>"))

	text_msa = "ذهب الولد الى المدرسة صباحا."
	text_egy = "الولد راح المدرسة النهاردة والجو حلو."

	_header("Normalization")
	text_with_diacritics = "ـالمَدرَسَةُ"
	normalized = normalize_arabic(text_with_diacritics)
	assert "\u0640" not in normalized
	assert "\u064E" not in normalized
	print("Normalized:", normalized)
	print("Case fold (Arabic):", case_fold("أَحْمَد", language="arabic"))
	print("Soundex (Arabic):", soundex_arabic("كتبت"))

	_header("Tokenization")
	tokens_msa = tokenize_arabic(text_msa)
	tokens_egy = tokenize_arabic(text_egy)
	print("Tokens:", tokens_msa)
	print("Tokens with clitic split:", tokenize_arabic("والكتاب الجديد", split_clitics=True))
	print("Clitic split:", split_clitic_tokens(tokenize_arabic("وللمدرسة")))
	print("Sentence split:", tokenize_sentences("الجو جميل. هل تحب الطقس؟ انا احبه."))

	_header("Stopwords")
	sw = get_arabic_stopwords()
	filtered = remove_stopwords(tokens_egy, stopwords=sw)
	assert len(filtered) <= len(tokens_egy)
	print("Before:", tokens_egy)
	print("After:", filtered)
	custom_sw_path = Path("custom_stopwords.txt")
	custom_sw_path.write_text("الولد\nالجو\n", encoding="utf-8")
	custom_sw = load_stopwords(str(custom_sw_path))
	print("Custom:", remove_stopwords(tokens_egy, stopwords=custom_sw))
	custom_sw_path.unlink(missing_ok=True)

	_header("Stemming")
	print("Light stem:", [light_stem(t) for t in tokens_msa])
	print("Stemmed tokens:", stem_tokens(tokens_msa))
	try:
		isri = get_isri_stemmer()
		print("ISRI:", [isri(t) for t in tokens_msa])
	except RuntimeError as exc:
		print("ISRI unavailable:", exc)

	_header("Preprocess")
	print("Preprocess MSA:", preprocess_text(text_msa))
	print("Preprocess Egyptian:", preprocess_text(text_egy, split_clitics=True))
	config = PreprocessConfig(split_clitics=True, use_lemmatizer=True)
	pre = ArabicPreprocessor.from_config(config)
	try:
		print("Preprocessor:", pre(text_egy))
	except Exception as exc:
		print("Lemmatizer unavailable:", exc)

	_header("IR")
	corpus_tokens = [preprocess_text(text_msa), preprocess_text(text_egy)]
	doc_freqs = document_frequencies(corpus_tokens)
	print("Term frequencies:", term_frequencies(corpus_tokens[0]))
	print("Document frequencies:", doc_freqs)
	print("Inverse document frequencies:", inverse_document_frequencies(doc_freqs, len(corpus_tokens)))
	print("Word ngrams:", word_ngrams(corpus_tokens[0], ngram_range=(1, 2)))
	print("Char ngrams:", char_ngrams(text_egy, ngram_range=(3, 4))[:10])
	print("Document stats:", document_stats(corpus_tokens[1]))

	count_vec = CountVectorizer(ngram_range=(1, 2))
	count_vec.fit(corpus_tokens)
	print("Count:", count_vec.transform(corpus_tokens[1], return_sparse=True))

	tfidf_vec = TfidfVectorizer(ngram_range=(1, 2))
	tfidf_vec.fit(corpus_tokens)
	print("TF-IDF:", tfidf_vec.transform(corpus_tokens[1], return_sparse=False))

	bm25_vec = BM25Vectorizer()
	bm25_vec.fit(corpus_tokens)
	print("BM25:", bm25_vec.transform(corpus_tokens[1], return_sparse=True))

	hash_vec = HashingVectorizer(n_features=32)
	print("Hashing:", hash_vec.transform(corpus_tokens[1], return_sparse=True))

	_header("Word2Vec")
	try:
		w2v_config = Word2VecConfig(vector_size=20, window=2, epochs=2, min_count=1)
		cbow = CBOW(w2v_config)
		cbow.fit(corpus_tokens)
		print("CBOW vector length:", len(cbow.encode(tokens=corpus_tokens[0])))
		sg = SkipGram(w2v_config)
		sg.fit(corpus_tokens)
		print("SkipGram vector length:", len(sg.encode(tokens=corpus_tokens[1])))
	except RuntimeError as exc:
		print("Word2Vec unavailable:", exc)

	_header("BERT/ELMo (optional)")
	os.environ["RUN_HEAVY"] = "1"
	if os.environ.get("RUN_HEAVY") == "1":
		try:
			bert = BertEncoder(pooling="cls")
			print("BERT vector length:", len(bert.encode(text=text_msa)))
		except RuntimeError as exc:
			print("BERT unavailable:", exc)
		try:
			options_file = os.environ.get("ELMO_OPTIONS", "")
			weight_file = os.environ.get("ELMO_WEIGHTS", "")
			if Path(options_file).exists() and Path(weight_file).exists():
				elmo = ElmoEncoder(options_file=options_file, weight_file=weight_file, pooling="mean")
				print("ELMo vector length:", len(elmo.encode(text=text_egy)))
			else:
				print("ELMo skipped (missing files)")
		except RuntimeError as exc:
			print("ELMo unavailable:", exc)
	else:
		print("Set RUN_HEAVY=1 to run BERT/ELMo.")

	_header("Thesaurus/Lemmatization")
	try:
		print("Thesaurus (Arabic):", thesaurus_lookup("ترك", source="arabic")[:5])
		print("Arabic lemmatize:", lemmatize_arabic("مدارس"))
	except Exception as exc:
		print("Thesaurus/lemmatizer unavailable:", exc)

	print("\nSMOKE TEST OK")


if __name__ == "__main__":
	main()
