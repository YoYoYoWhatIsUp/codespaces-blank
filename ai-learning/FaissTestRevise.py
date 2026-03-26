from sentence_transformers import SentenceTransformer
import faiss
import numpy as np 

#make database and query

sentences=[
        "ssssssss",
        "sssswsws",
        "wwwwwsws",
        "sssswwww",
        "wwwwwwww"
        ]
query=input("please input the word i'll match the similar one for u : ")

#encode the database and query to vector form

model=SentenceTransformer("all-MiniLM-L6-v2")
sentences_vector=np.array(model.encode(sentences)).astype("float32")
query_vector=np.array(model.encode([query])).astype("float32")

#put vector into faiss
index=faiss.IndexFlatL2(384)
index.add(sentences_vector)

#use faiss to search anwer
k=2
distance,result=index.search(query_vector,k)

#print result

for i,j in enumerate(result[0]):
    print(f"top {i+1}:",sentences[j])



