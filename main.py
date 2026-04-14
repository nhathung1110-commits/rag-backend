import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware

# =========================
# LOAD ENV
# =========================
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found")

# =========================
# FASTAPI
# =========================
app = FastAPI(title="Simple RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST MODEL
# =========================
class ChatRequest(BaseModel):
    question: str

# =========================
# EMBEDDING
# =========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# LOAD FAISS
# =========================
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# =========================
# LLM
# =========================
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    temperature=0
)

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "running"}

# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        question = req.question

        # =========================
        # VECTOR SEARCH
        # =========================
        results = vectorstore.similarity_search_with_score(question, k=3)

        docs = []
        threshold = 2.0  # FIX threshold

        for doc, score in results:
            print("Score:", score)
            if score < threshold:
                docs.append(doc)

        # fallback nếu vẫn rỗng
        if not docs:
            print("No relevant docs → using all results")
            docs = [doc for doc, _ in results]

        # =========================
        # BUILD PROMPT
        # =========================
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful assistant.

Use the context below to answer the question.
If the context is not relevant, answer normally.

Context:
{context}

Question:
{question}
"""

        # =========================
        # LLM CALL
        # =========================
        response = llm.invoke(prompt)

        # =========================
        # FIX CRASH (.content)
        # =========================
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        return {"answer": answer}

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )