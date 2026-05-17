# RNArabic

Arabic preprocessing + IR features + embeddings with a clean, teachable API.

## Install
python -m pip install RNArabic

## Install (editable)
python -m pip install -e .

### Optional extras
- CBOW/Skip-gram: python -m pip install RNArabic[ml]
- BERT: python -m pip install RNArabic[transformers]
- ELMo: python -m pip install RNArabic[elmo]
- Tests: python -m pip install RNArabic[dev]

### Optional extras (editable)
- CBOW/Skip-gram: python -m pip install -e .[ml]
- BERT: python -m pip install -e .[transformers]
- ELMo: python -m pip install -e .[elmo]
- Tests: python -m pip install -e .[dev]

## Core preprocessing
from arabic_nlp import ArabicPreprocessor, PreprocessConfig, load_stopwords

stopwords = load_stopwords("path/to/stopwords_ar.txt")
preprocess = ArabicPreprocessor(stopwords=stopwords, use_light_stemmer=True)

tokens = preprocess("arabic_text")
print(tokens)

tokens = preprocess("والكتاب الجديد")
print(tokens)

config = PreprocessConfig(split_clitics=True, min_clitic_length=2)
preprocess = ArabicPreprocessor.from_config(config)

## Using ISRI stemmer (requires nltk)
from arabic_nlp import get_isri_stemmer

stemmer = get_isri_stemmer()
preprocess = ArabicPreprocessor(stopwords=stopwords, stemmer=stemmer.stem)

## IR features
from arabic_nlp import CountVectorizer, TfidfVectorizer, BM25Vectorizer

corpus = [
	["kitab", "jadid"],
	["kitab", "qadim"],
]

tfidf = TfidfVectorizer(ngram_range=(1, 2))
tfidf.fit(corpus)
vector = tfidf.transform(["kitab", "jadid"])

bm25 = BM25Vectorizer()
bm25.fit(corpus)
score = bm25.transform(["kitab", "jadid"])

## CBOW / Skip-gram (educational)
from arabic_nlp import CBOW, SkipGram, Word2VecConfig

model = CBOW(Word2VecConfig(vector_size=100, window=3, epochs=3))
model.fit(corpus)
doc_vec = model.encode(tokens=["kitab", "jadid"], pooling="mean")

sg = SkipGram(Word2VecConfig(vector_size=100))
sg.fit(corpus)

## BERT (wrapper)
from arabic_nlp import BertEncoder

bert = BertEncoder(model_name="bert-base-multilingual-cased", pooling="cls")
vec = bert.encode(text="arabic_text")

## ELMo (wrapper)
from arabic_nlp import ElmoEncoder

elmo = ElmoEncoder(
	options_file="path/to/options.json",
	weight_file="path/to/weights.hdf5",
)
vec = elmo.encode(text="arabic_text")
