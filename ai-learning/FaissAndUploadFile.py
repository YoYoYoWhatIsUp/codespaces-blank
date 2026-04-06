#import all tools

import faiss
import numpy as np
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi import UploadFile,File

#upload the file to faiss
app=FastAPI()

#get the data from the file that upload 
@app.post("/upload")
async def uploadfile(file:UploadFile=File(...)):

    content = await file.read()
    text = content.decode("utf-8")
    global sentences,sentences_vector,index,model
    model=SentenceTransformer("all-MiniLM-L6-v2")
    sentences=[line.strip() for line in text.split("\n") if line.strip()]

#embedding and transform the data in using numpy
    sentences_vector=np.array(model.encode(sentences)).astype("float32")
    index=faiss.IndexFlatL2(384)

#put the data into faiss
    index.add(sentences_vector)
    return {"message": "文件上传并处理成功",
            "条数": len(sentences),
            "sentence":sentences
            }



#answer the questions in using LLM
@app.get("/chat")
def searchdata(question:str):
    question_vector=np.array(model.encode([question])).astype("float32")
    distance,data=index.search(question_vector,1)
    result=sentences[data[0][0]]
    prompt=f"""
    the question is {question}
    you have to answer the question in using the {result} to answer the question
    """
    llm=ZhipuAI(api_key="0f5f2de27971431c84c93f39e4f44c52.AM9Ml0wPmhFayaPb")
    response=llm.chat.completions.create(
        model="glm-4-air",
        messages=[
            {
                "role":"system",
                "content":"you are a chatrot to answer the questions"
                },
            {
                "role":"user",
                "content":prompt
                }
            ]
            )
    return {
        "question":question,
        "resultfromfaiss":result,
        "answer":response.choices[0].message.content
            }

#return the answer from LLM



