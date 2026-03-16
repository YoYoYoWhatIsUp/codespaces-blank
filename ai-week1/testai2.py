from zhipuai import ZhipuAI
from fastapi import FastAPI

app=FastAPI()
client=ZhipuAI(api_key="0f5f2de27971431c84c93f39e4f44c52.AM9Ml0wPmhFayaPb")

@app.get("/")
def home():
    return {"message":"ai is still running"}

@app.get("/chat")
def chat(question:str):
    response=client.chat.completions.create(
    model="glm-4-flash",
    messages=[{
        "role":"user",
        "content":question
        }])
    answer=response.choices[0].message.content
    return{
        "question":question,
        "answer":answer}

