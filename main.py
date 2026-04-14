import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# ENV
# =========================
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found")

# =========================
# FASTAPI
# =========================
app = FastAPI(title="Simple RAG API")

# 🔥 WARMUP KHI START (FIX 502)
@app.on_event("startup")
def startup_event():
    print("🔥 Warming up model...")
    init_system()

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
# GLOBAL (lazy load)
# =========================
embedding_model = None
vectorstore = None
llm = None

# =========================
# INIT SYSTEM (SAFE)
# =========================
def init_system():
    global embedding_model, vectorstore, llm

    try:
        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        if vectorstore is None:
            if os.path.exists("faiss_index"):
                vectorstore = FAISS.load_local(
                    "faiss_index",
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                print("✅ FAISS loaded")
            else:
                print("⚠️ faiss_index not found → running without RAG")

        if llm is None:
            llm = ChatOpenAI(
                model="openai/gpt-4o-mini",
                base_url="https://openrouter.ai/api/v1",
                api_key=API_KEY,
                temperature=0
            )

    except Exception as e:
        print("❌ INIT ERROR:", e)

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
        init_system()

        question = req.question
        docs = []

        # =========================
        # VECTOR SEARCH (SAFE)
        # =========================
        if vectorstore:
            results = vectorstore.similarity_search_with_score(question, k=3)

            threshold = 2.0
            for doc, score in results:
                if score < threshold:
                    docs.append(doc)

            if not docs:
                docs = [doc for doc, _ in results]

        # =========================
        # BUILD PROMPT
        # =========================
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

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

        answer = response.content if hasattr(response, "content") else str(response)

        return {"answer": answer}

    except Exception as e:
        print("❌ ERROR:", e)
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