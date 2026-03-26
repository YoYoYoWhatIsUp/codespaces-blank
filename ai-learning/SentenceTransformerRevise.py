from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from numpy import dot

embedding=SentenceTransformer("all-MiniLM-L6-v2")

sentence=[
        "sssssssssssssss",
        "xxxxxxxxxxxxxx",
        "ssssssswssssss"
        ]
embeddingresult=embedding.encode(sentence)

for i,emb in enumerate(embeddingresult):
    print(f"句子{i} 向量和向量长度分别是:",emb,len(emb))

def cosine(a,b):
    return dot(a,b)/(norm(a)*norm(b))

print("向量1和向量2的相似度是:",cosine(embeddingresult[0],embeddingresult[1]))
print("向量1和向量3的相似度是:",cosine(embeddingresult[0],embeddingresult[2]))
print("向量2和向量3的相似度是:",cosine(embeddingresult[1],embeddingresult[2]))
