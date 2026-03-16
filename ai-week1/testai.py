from zhipuai import ZhipuAI

client=ZhipuAI(api_key="0f5f2de27971431c84c93f39e4f44c52.AM9Ml0wPmhFayaPb")
messages=[{
    "role":"system",
    "content":"你是一个英语翻译家"
           }]
while True:
    testinput=input("text what u wanna say:")
    messages.append({
        "role":"user",
        "content":testinput
        })
    response=client.chat.completions.create(
            model="glm-4-flash",
            messages=messages
            )
    ai_reply=response.choices[0].message.content

    print("ai says : ",ai_reply)

    messages.append({
        "role":"assistant",
        "content":ai_reply
        })
