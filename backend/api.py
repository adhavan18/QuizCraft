import os
from fastapi import FastAPI, HTTPException, UploadFile
from backend.rag import ROOT, load_pdf, chunk_text, build_faiss_index, retrieve_chunks
from backend.llm import generate_quiz
from backend.models import QuizRequest, QuizResponse

app = FastAPI(title="QuizLLM API")

DATA_DIR = os.path.join(ROOT, "data")


@app.post("/ingest/upload")
async def ingest_pdf(file: UploadFile):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, os.path.basename(file.filename))
    with open(file_path, "wb") as f:
        f.write(await file.read())
    text = load_pdf(file_path)
    if not text.strip():
        raise HTTPException(400, "No extractable text found in PDF (is it a scanned image?)")
    chunks = chunk_text(text)
    build_faiss_index(chunks)
    return {"status": "indexed", "chunks": len(chunks)}


@app.post("/quiz", response_model=QuizResponse)
async def generate_quiz_api(req: QuizRequest):
    try:
        sources = retrieve_chunks(req.topic)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))
    try:
        questions = generate_quiz(" ".join(sources), req.topic, req.num_questions, req.difficulty)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return QuizResponse(topic=req.topic, difficulty=req.difficulty, questions=questions, sources=sources)
