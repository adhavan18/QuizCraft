import os
from fastapi import FastAPI, HTTPException, UploadFile
from backend.rag import ROOT, load_pdf, chunk_text, build_faiss_index, retrieve_chunks, load_chunks, index_exists
from backend.llm import generate_quiz
from backend.nlp_generator import clean_text, generate_offline_quiz
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


@app.get("/health")
def health():
    return {"index_ready": index_exists(), "gemini_available": bool(os.getenv("GOOGLE_API_KEY"))}


@app.post("/quiz", response_model=QuizResponse)
def generate_quiz_api(req: QuizRequest):
    generator = req.generator
    if generator == "auto":
        generator = "gemini" if os.getenv("GOOGLE_API_KEY") else "offline"
    try:
        # The offline generator needs more sentences to choose from
        sources = [clean_text(c) for c in retrieve_chunks(req.topic, top_k=3 if generator == "gemini" else 8)]
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))
    if generator == "gemini":
        try:
            questions = generate_quiz(" ".join(sources), req.topic, req.num_questions, req.difficulty)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    else:
        questions = generate_offline_quiz(sources, load_chunks(), req.topic, req.num_questions, req.difficulty)
    if not questions:
        raise HTTPException(422, "Could not generate questions for this topic; try a broader topic.")
    return QuizResponse(topic=req.topic, difficulty=req.difficulty, generator=generator,
                        questions=questions, sources=sources[:3])
