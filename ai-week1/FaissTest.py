from sentence_transformers import SentenceTransformer
import faiss
import numpy as np 

#build data
sentences=[
        "sssssssssssssssssss",
        "swswssswswwgwfwfwfw",
        "wefwefwefwefwefwefwe"
        ]
query=input("please type your word i will find a smilar word like you type it for u :  ")

#get objet from sentencetramsformer
sentencetransformer = SentenceTransformer("all-MiniLM-L6-v2")

#encode the data
embeddings = sentencetransformer.encode(sentences)
embeddings = np.array(embeddings).astype("float32")
query_vector=sentencetransformer.encode([query]).astype("float32")
#put embeddings into faiss

index = faiss.IndexFlatL2(384)
index.add(embeddings)

#get to which is similar 
k=1
distance,indices=index.search(query_vector,k)

#get result 
print("查询",query)
for i in indices[0]:
    print("匹配:",sentences[i])





