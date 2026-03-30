from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

#read the file  
with open("test.txt","r",encoding="utf-8") as f:
    sentences=[line.strip() for line in  f.readlines()  if line.strip()]
query=input("please input your query : ")

#embedding the data

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences_vector=np.array(model.encode(sentences)).astype("float32")
query_vector=np.array(model.encode([query])).astype("float32")

#put the data into faiss

index = faiss.IndexFlatL2(384)
index.add(sentences_vector)

#search data from faiss

distance,dataArray=index.search(query_vector,1)
#print the data
print("匹配到的数据是: ",sentences[dataArray[0][0]])

#use LLM to answer the questions

llm = ZhipuAI(api_key="0f5f2de27971431c84c93f39e4f44c52.AM9Ml0wPmhFayaPb")
sentences_data="\n".join(sentences[i] for i in dataArray[0])
prompt=f"""
你必须按照如下内容来回答问题:
    {sentences_data}
问题是:{query}

如果答案不在内容中,请回答不知道.
"""
response=llm.chat.completions.create(
        model="glm-4-air",
        messages=[
            {
                "role":"system",
                "content":"你是一个非常专业的问答助手"
            },
            {
            "role":"user",
            "content":prompt
                }
            ]
        )
print("ai 回答如下:",response.choices[0].message.content)



