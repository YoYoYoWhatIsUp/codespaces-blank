from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from zhipuai import ZhipuAI

#build the data and the query

sentences=[
        "basketball is a good game",
        "how to play ping-pong",
        "eating fruits is good for health",
        "drinking tea is better than coffe"
        ]
query = input("please input the query: ")

#encode the data 

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences_vector=np.array(model.encode(sentences).astype("float32"))
query_vector=np.array(model.encode([query]).astype("float32"))

#put the data into faiss
index = faiss.IndexFlatL2(384)
index.add(sentences_vector)

#search query from faiss

k=1
distance,result=index.search(query_vector,k)

#gather the answer

context="\n".join([sentences[j] for j in result[0]])

#get the result from LLM

client = ZhipuAI(api_key="0f5f2de27971431c84c93f39e4f44c52.AM9Ml0wPmhFayaPb")

response=client.chat.completions.create(
        model="glm-4",
        messages=[{
            "role":"system",
            "content":"你是一个问答助手"
            },
            {
            "role":"user",
            "content":f"根据以下内容来回答问题:\n{context}\n\n问题:{query}"
                }
                  ]
        )
#print the result

print("\nAI回答如下:")

print(response.choices[0].message.content)

