from zhipuai import ZhipuAI

client = ZhipuAI(api_key="0f5f2de27971431c84c93f39e4f44c52.AM9Ml0wPmhFayaPb")
messages=[{
    "role":"system",
    "content":"you are a translator"
    }]
while True:
    userinput=input("please input what u wanna say to robot")
    messages.append(
        {"role":"user",
         "content":userinput
         })
    response=client.chat.completions.create(
            model="glm-4-flash",
            messages=messages
            )
    ai_reply=response.choices[0].message.content
    print("robot says : " ,ai_reply)
    messages.append({
        "role":"assistant",
        "content":ai_reply})




