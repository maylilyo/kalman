import nltk
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt


def stopwords():
    pass


def tf_idf(corpus, args):
    okt = Okt()
    # corpus_text = "".join(corpus)
    # corpus_nouns = " ".join(okt.nouns(corpus_text))
    for i, text in enumerate(corpus):
        corpus[i] = " ".join(okt.nouns(text))

    tfidf = TfidfVectorizer()
    sp_matrix = tfidf.fit_transform(corpus).toarray()

    word2id = defaultdict(lambda: 0)
    for idx, feature in enumerate(tfidf.get_feature_names()):
        word2id[idx] = feature

    wordlist = []
    for i, scores in enumerate(sp_matrix):
        words = sorted(scores, reverse=True)
        words = list(set(words[:3])) #docx별 TF-IDF score top3 추출
        idx = []
        for word in words:
            idx.extend(np.where(scores == word)[0])
        wordlist.extend([word2id[i] for i in set(idx)])
    return wordlist


if __name__ == "__main__":
    corpus = [
        "you know I want your love",
        "I like you",
        "what should I do ",
    ]
    tf_idf(corpus)
