from fastapi import FastAPI, UploadFile, File
from app.rag import add_documents, search
from zhipuai import ZhipuAI
import os
from fastapi.responses import FileResponse
app = FastAPI()
llm = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    new_sentences = [line.strip() for line in text.split("\n") if line.strip()]

    add_documents(new_sentences)

    return {"message": "上传成功", "count": len(new_sentences)}

@app.get("/chat")
def chat(question: str):
    context = search(question)

    prompt = f"""
根据以下内容回答：
{context}

问题：{question}
"""

    response = llm.chat.completions.create(
        model="glm-4-air",
        messages=[
            {"role": "system", "content": "你是AI助手"},
            {"role": "user", "content": prompt}
        ]
    )

    return {"answer": response.choices[0].message.content}
