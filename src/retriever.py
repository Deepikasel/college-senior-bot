from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from pathlib import Path

class TfidfRetriever:
    def __init__(self, ngram_range=(1,2), max_features=5000):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            stop_words="english",
            max_features=max_features
        )
        self.matrix = None  # document-term matrix
        self.docs: List[str] = []

    def fit(self, docs: List[str]):
        self.docs = docs
        self.matrix = self.vectorizer.fit_transform(docs)
        return self

    def save(self, dirpath: str | Path):
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, dirpath / "tfidf_vectorizer.joblib")
        joblib.dump(self.matrix, dirpath / "tfidf_matrix.joblib")
        joblib.dump(self.docs, dirpath / "docs.joblib")

    def load(self, dirpath: str | Path):
        dirpath = Path(dirpath)
        self.vectorizer = joblib.load(dirpath / "tfidf_vectorizer.joblib")
        self.matrix = joblib.load(dirpath / "tfidf_matrix.joblib")
        self.docs = joblib.load(dirpath / "docs.joblib")
        return self

    def top_k(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        idxs = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idxs]
