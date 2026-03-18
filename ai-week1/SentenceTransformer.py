from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences=[
        "第一段测试embedding向量的转换",
        "The Second testing embedding向量的转换",
        "第三段测试embedding向量的转换"
        ]

embeddings = model.encode(sentences)

for i,emb in enumerate(embeddings):
    print(f"句子{i}向量长度:",len(emb))

def cosine(a,b):
    return dot(a,b)/(norm(a)*norm(b))

print("句子1 vs 句子2",
      cosine(embeddings[0],embeddings[1]))
print("句子2 vs 句子3",
      cosine(embeddings[1],embeddings[2]))
print("句子1 vs 句子3",cosine(embeddings[0],embeddings[2]))
