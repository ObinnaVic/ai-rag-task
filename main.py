from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from io import BytesIO
import chromadb
import os
import dotenv
import re
import docx
import json
import uuid
from pypdf import PdfReader
from google import genai
import uvicorn
from pydantic import BaseModel
# import google.generativeai as genai



dotenv.load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME")
CHROMA_HOST = os.environ.get("CHROMA_DB_HOST")
DATA_DIRECTORY = os.environ.get("RAG_DATA_DIR")
CHUNK_LENGTH = os.environ.get("CHUNK_LENGTH")
PORT = int(os.getenv("PORT", 3000))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path="./vectorDB")


llm_client = genai.Client(api_key=GEMINI_API_KEY)
# genai.configure(api_key=GEMINI_API_KEY)
# llm_model = genai.GenerativeModel(LLM_MODEL_NAME)

collection = client.get_or_create_collection(name=CHROMA_HOST)


def semanticChunking(text: str, chunk_length: int = CHUNK_LENGTH):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunk_length = int(chunk_length)
    chunks = []
    curChunk = []
    wordCount = 0
    
    for sentence in sentences:
        curSentenceLength = len(sentence.split(" "))
        if wordCount + curSentenceLength > chunk_length and len(curChunk) > 0:
            chunks.append(" ".join(curChunk))
            curChunk = [sentence]
            wordCount = curSentenceLength
        else:
            curChunk.append(sentence)
            wordCount += curSentenceLength
    
    if len(curChunk) > 0:
        chunks.append(" ".join(curChunk))
    
    return chunks
        
def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".txt") or lower.endswith(".md"):
        try:
            return content.decode("utf-8")
        except:
            return content.decode("latin-1", errors="ignore")


    if lower.endswith(".pdf"):
        if PdfReader is None:
            raise HTTPException(500, "PyPDF2 not installed")

        reader = PdfReader(BytesIO(content))
        txt = []
        for p in reader.pages:
            try:
                txt.append(p.extract_text() or "")
            except:
                txt.append("")
        
        return "\n".join(txt)


    if lower.endswith(".docx"):
        if docx is None:
            raise HTTPException(500, "python-docx not installed")

        d = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in d.paragraphs])


    try:
        return content.decode("utf-8")
    except:
        return content.decode("latin-1", errors="ignore")

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Service"}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), context: Optional[str] = Form(None)):
    if context is None:
        context = f"ctx-{uuid.uuid4().hex[:8]}"

    files_directory = os.path.join(DATA_DIRECTORY, "files")
    os.makedirs(files_directory, exist_ok=True)

    all_vectors = []
    all_ids = []
    all_metadata = []
    all_documents = []

    for file in files:
        file_contents = await file.read()
        extracted_contents = extract_text(file.filename, file_contents)
        chunks = semanticChunking(extracted_contents)

        dest = os.path.join(files_directory, file.filename)
        with open(dest, "wb") as out:
            out.write(file_contents)
        
        for i, chunk in enumerate(chunks):
            vector = embed_model.encode(chunk)
            id = uuid.uuid4().hex

            all_vectors.append(vector)
            all_ids.append(id)
            all_metadata.append({
                "filename": file.filename,
                "chunk_id": i,
                "context": context
            })
            all_documents.append(chunk)

        
    collection.add(
        ids = all_ids,
        embeddings = all_vectors,
        documents=all_documents,
        metadatas=all_metadata
    )
    print(f"Stored files into chroma vector database")
    return {"status": "success", "context": context, "num_chunks": len(all_ids)}

    
    
class PromptPayload(BaseModel):
    query: str
    context: Optional[str] = None

@app.post("/prompt")
async def prompt(payload: PromptPayload):
    query_embedding = embed_model.encode(payload.query).tolist()
    query_params = {
        "query_embeddings": query_embedding,
        "n_results": 5
    }

    if payload.context:
        query_params["where"] = {"context": payload.context}

    query_result = collection.query(**query_params)

    result = query_result["documents"][0]
    context_texts = "\n".join(result)

    prompt = f"""
    Context:
        {context_texts}

        Question: {payload.query}
        
        Based on the context provided above, generate a succint answer to the query above.
    """
    response = llm_client.models.generate_content(model=LLM_MODEL_NAME, contents=prompt)

    return {"answer": response.text, "context": context_texts}


@app.get("/health")
async def check_health():
    return {"status": "ok", "code": 200}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)