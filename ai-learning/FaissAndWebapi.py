#import all tools
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from zhipuai import ZhipuAI
import os
from fastapi import FastAPI

app=FastAPI()


#build data from txt.file
with open("test.txt","r",encoding="utf-8") as f:
    sentences = [line.strip() for line in  f.readlines() if line.strip()]
#input query from webapi to webserver

@app.get("/chat")
def query(ques:str):
        question = ques
#use sentencestransformer and numpy to transform data into vector form
        model = SentenceTransformer("all-MiniLM-L6-v2")
        sentences_vector = np.array(model.encode(sentences)).astype("float32")
        question_vector=np.array(model.encode([question])).astype("float32")

#put data into faiss
        index = faiss.IndexFlatL2(384)
        index.add(sentences_vector)

#search data from faiss and use llm to answer
        distance,data=index.search(question_vector,1)
        result=sentences[data[0][0]]

        prompt=f"""
the question is {question},
you have to anwser the question through {result}                         
                         """
#return answer to webserver
        llm = ZhipuAI(api_key="0f5f2de27971431c84c93f39e4f44c52.AM9Ml0wPmhFayaPb")
        response=llm.chat.completions.create(
        model="glm-4-air",
        messages=[
            {
        "role":"system",
        "content":"you are a chatrot"
    },
    {
       "role":"user",
        "content":prompt
    }
    ]
    )
        lastresult=response.choices[0].message.content
        return {
        "question":question,
        "answer":lastresult
    }                                                  
                        
                         
                         



