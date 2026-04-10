import faiss
import numpy as np
from app.model import model

index = None
sentences = []

def add_documents(new_sentences):
    global index, sentences

    vectors = np.array(model.encode(new_sentences)).astype("float32")

    if index is None:
        index = faiss.IndexFlatL2(384)

    index.add(vectors)
    sentences.extend(new_sentences)


def search(query):
    global index, sentences

    query_vector = np.array(model.encode([query])).astype("float32")

    distances, indices = index.search(query_vector, 1)

    return sentences[indices[0][0]]
