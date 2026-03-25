from zhipuai import ZhipuAI
from fastapi import FastAPI


client=ZhipuAI(api_key="0f5f2de27971431c84c93f39e4f44c52.AM9Ml0wPmhFayaPb")
app=FastAPI()

@app.get("/")
def home():
    return {"sorry no message return because you don't type the question"}

@app.get("/chat")
def chat(question:str):
    response=client.chat.completions.create(
            model="glm-4-flash",
            messages=[{
                "role":"user",
                "content":question
                }])
    anwser=response.choices[0].message.content
    return{
        "question":question,
        "anwser":anwser
        }
