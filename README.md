# QuizCraft: RAG-based Science Quiz Generator (Grade 9 NCERT)

Generates multiple-choice quizzes from a textbook PDF. The PDF is chunked and embedded into a FAISS
vector index; for a given topic the most relevant chunks are retrieved and turned into questions,
either by an LLM (Gemini) or by a fully local NLP pipeline that needs no API key.

> **Credits.** This is a fork of [rohangarg27/QuizCraft](https://github.com/rohangarg27/QuizCraft)
> by Rohan Garg and Kushagra Shukla, which provided the original RAG + Gemini prototype
> (PDF ingestion, FAISS retrieval, FastAPI + Streamlit skeleton). See
> [What this fork adds](#what-this-fork-adds) for the changes made here.

## Architecture

```
PDF ──► pypdf text ──► overlapping chunks (300 words, 50 overlap)
                              │
                              ▼
                all-MiniLM-L6-v2 embeddings ──► FAISS (cosine / inner product)
                                                        │
topic ──► embed ──► top-k chunks ◄──────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   Gemini generator              Offline NLP generator
   (JSON schema output)          sentence ranking → TF-IDF key term
                                 → cloze question → embedding distractors
          └──────────────┬──────────────┘
                         ▼
          validated Question objects ──► FastAPI /quiz ──► Streamlit quiz UI
```

## What this fork adds

| Area | Original | This fork |
|---|---|---|
| Runs out of the box | No: `PyPDF2` import missing from requirements, retired `gemini-1.5-flash` model | Fixed; model configurable via `GEMINI_MODEL` (default `gemini-3.8-flash`), retries on 429/503 |
| LLM output | Free text split on newlines | JSON schema (`question`, `options`, `answer`, `explanation`), validated |
| Difficulty | Not implemented | easy / medium / hard in both generators |
| Works without API key | No | Offline NLP generator (`backend/nlp_generator.py`) |
| Retrieval | L2 distance, disjoint 500-word chunks | Cosine similarity, overlapping chunks, PDF noise cleaning |
| Windows support | `sentence-transformers` pulls in scikit-learn, blocked by Smart App Control | MiniLM embeddings via `transformers` directly |
| Storage | Pickle (unsafe to load) | JSON |
| UI | Prints raw lines | Interactive quiz, scoring, explanations, shows RAG sources |
| Evaluation | None | `evaluate.py` with yield / validity / grounding / distractor similarity / latency |

### Offline NLP generator

1. **Text cleaning**: removes running headers/footers, re-joins hyphenated words, collapses repeats.
2. **Sentence segmentation** with NLTK Punkt; captions, exercise questions and broken sentences are dropped.
3. **Semantic ranking**: sentences sorted by cosine similarity to the topic embedding.
4. **Key-phrase extraction**: POS tagging + regexp noun-phrase chunking (`<JJ>?<NN|NNS>{1,2}`);
   each noun phrase is scored by IDF over the whole book x similarity to the topic.
5. **Cloze question**: the answer phrase is blanked out of the sentence.
6. **Distractor generation**: other noun phrases from the book ranked by embedding similarity to the
   answer, with the same length and grammatical number (singular/plural) and no shared stem.
   Difficulty picks from the most similar (hard) to less similar (easy) phrases.

## Running

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
copy .env.example .env          # optional: add GOOGLE_API_KEY for the Gemini generator

uvicorn backend.api:app --port 8000
streamlit run frontend/app.py   # in a second terminal
```

Open http://localhost:8501, upload `data/sci_9.pdf` in the first tab, then generate a quiz.

### API

| Method | Path | Body |
|---|---|---|
| POST | `/ingest/upload` | multipart `file` (PDF) |
| POST | `/quiz` | `{"topic": str, "num_questions": 1-20, "difficulty": "easy/medium/hard", "generator": "auto/gemini/offline"}` |
| GET | `/health` | none |

### Evaluation

**Offline generator**, 12 Grade 9 science topics, 5 questions each (`results/eval_offline.md`):

| difficulty | yield | valid | grounded | distractor sim | latency (s) |
|---|---|---|---|---|---|
| easy | 1.00 | 1.00 | 1.00 | 0.39 | ~0.5-1.5 |
| medium | 1.00 | 1.00 | 1.00 | 0.43 | ~0.5-1.5 |
| hard | 1.00 | 1.00 | 1.00 | 0.52 | ~0.5-1.5 |

Distractor similarity rising with difficulty shows the difficulty setting works as intended.
Grounding is 1.00 by construction for the extractive offline generator. The first request also
pays a one-time ~8 s corpus build (POS-tagging the book).

**Gemini generator (partial)**, easy difficulty, 7 of 12 topics (the rest hit free-tier limits):
yield 0.97, valid 1.00, grounded 0.96, distractor sim 0.42. Gemini writes more varied,
reasoning-style questions (e.g. numerical problems, "why" questions) but depends on API
availability: the free tier allows about 20 requests per model per day and often returns
503 "high demand". A full run needs a fresh daily quota:

```bash
python evaluate.py --generators offline            # all 12 topics
python evaluate.py --generators gemini --topics 6  # 18 calls, fits one model's daily quota
```
