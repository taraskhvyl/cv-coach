from langchain_core.embeddings import Embeddings

from cv_coach.rag import Index, chunk


class KeywordEmbeddings(Embeddings):
    """Two-dim fake: counts of 'python' and 'cooking'."""

    def embed_documents(self, texts):
        return [[t.count("python") + 0.01, t.count("cooking") + 0.01] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_chunk_splits_long_text():
    parts = chunk("word " * 400, size=400, overlap=50)
    assert len(parts) > 3 and all(len(p) <= 400 for p in parts)


def test_search_ranks_by_similarity():
    idx = Index("python python python.\n\n" + "x " * 500 + "\n\ncooking cooking cooking.", embeddings=KeywordEmbeddings())
    assert "python" in idx.search("python", k=1)[0]
    assert "cooking" in idx.search("cooking", k=1)[0]
