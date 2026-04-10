 AI RAG Knowledge System

项目简介

这是一个基于 RAG（Retrieval-Augmented Generation）的AI问答系统，支持用户上传文本文件，并基于语义检索进行智能问答。

⸻

 技术栈
	•	FastAPI（后端框架）
	•	FAISS（向量检索）
	•	Sentence-Transformers（文本向量化）
	•	ZhipuAI GLM（大语言模型）

⸻

功能
	•	支持上传txt文件构建知识库
	•	基于向量相似度检索相关内容
	•	结合大模型生成答案
	•	提供简单前端页面进行交互

⸻

如何运行

1. 安装依赖

pip install -r requirements.txt

2.设置环境变量

export ZHIPU_API_KEY=你的key

3.启动服务

uvicorn app.main:app --reload


⸻

使用方法
	1.	打开浏览器访问：

http://localhost:8000
	2.	上传txt文件
	3.	输入问题进行问答

⸻

项目结构

app/
 ├── main.py      # API接口
 ├── rag.py       # 检索逻辑
 ├── model.py     # embedding模型


⸻

项目亮点
	•	实现完整RAG流程（Embedding + 检索 + LLM）
	•	使用FAISS实现高效向量搜索
	•	模块化设计，便于扩展（支持PDF/数据库）
	•	可扩展为企业级知识库系统

⸻

后续优化
	•	支持PDF/网页解析
	•	FAISS持久化存储
	•	多用户隔离
	•	部署到云服务器

⸻

requirements.txt

fastapi
uvicorn
faiss-cpu
numpy
sentence-transformers
zhipuai
python-multipart



