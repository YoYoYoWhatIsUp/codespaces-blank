from zhipuai import ZhipuAI

client = ZhipuAI(api_key="0f5f2de27971431c84c93f39e4f44c52.AM9Ml0wPmhFayaPb")

response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role": "user", "content": "用一句话解释一下为什么要学习ai编程"}
    ]
)

print(response.choices[0].message.content)
