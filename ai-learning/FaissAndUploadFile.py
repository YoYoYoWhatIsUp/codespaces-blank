#import all tools
import os
import faiss
import numpy as np
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi import UploadFile,File
from fastapi.responses import HTMLResponse

#upload the file to faiss
app=FastAPI()
index=None
sentences=[]
#get the data from the file that upload 
@app.post("/upload")
async def uploadfile(file:UploadFile=File(...)):

    content = await file.read()
    text = content.decode("utf-8")
    global sentences,sentences_vector,index,model
    model=SentenceTransformer("all-MiniLM-L6-v2")
    Newsentences=[line.strip() for line in text.split("\n") if line.strip()]

#embedding and transform the data in using numpy
    Newsentences_vector=np.array(model.encode(Newsentences)).astype("float32")
    if index is None:
        index=faiss.IndexFlatL2(384)
    index.add(Newsentences_vector)
    sentences.extend(Newsentences)
    return {"message": "文件上传并处理成功",
            "总条数": len(sentences),
            "增加的条数":len(Newsentences),
            "sentence":sentences,
            "Newsentence":Newsentences
            }



#answer the questions in using LLM
@app.get("/chat")
def searchdata(question:str):
    question_vector=np.array(model.encode([question])).astype("float32")
    distance,data=index.search(question_vector,1)
    result=sentences[data[0][0]]
    if distance[0][0]>1 :
        return {
                "answer":"no related content"
                }
    prompt=f"""
    the question is {question}
    you have to answer the question in using the {result} to answer the question
    """
    llm=ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
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

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return"""
    <html>
    <body>

    <h2>AI知识库系统</h2>

    <h3>上传文件</h3>
    <input type="file" id="file">
    <button onclick="upload()">上传</button>

    <h3>提问</h3>
    <input id="q" placeholder="输入问题">
    <button onclick="ask()">提问</button>

    <p id="a"></p>

    <script>
    async function upload(){
        let fileInput = document.getElementById("file");
        let formData = new FormData();
        formData.append("file", fileInput.files[0]);

        let res = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        let data = await res.json();
        alert("上传成功: " + data.message);
    }

    async function ask(){
        let q = document.getElementById("q").value;

        let res = await fetch(`/chat?question=${q}`);
        let data = await res.json();

        document.getElementById("a").innerText = data.answer;
    }
    </script>

    </body>
    </html>
    """
