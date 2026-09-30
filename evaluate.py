"""Evaluate quiz generators on the indexed textbook.

Usage:  python evaluate.py [--generators offline gemini] [--n 5] [--topics 12]

Metrics (per generator x difficulty, averaged over topics):
  yield        questions returned / questions requested
  valid        4 distinct options and the answer is one of them
  grounded     share of answer words that occur in the retrieved context
  distractor   mean cosine similarity between answer and distractors
               (higher = more confusable = harder)
  latency      seconds per quiz
  api_failed   topics skipped because the Gemini API was unavailable (503/429);
               excluded from the other metrics, which measure question quality
"""
import argparse
import json
import os
import re
import time

import numpy as np
from google.genai import errors

from backend.llm import generate_quiz
from backend.nlp_generator import clean_text, generate_offline_quiz
from backend.rag import ROOT, get_embedder, load_chunks, retrieve_chunks

TOPICS = [
    "states of matter and evaporation",
    "mixtures, solutions and separation techniques",
    "atoms, molecules and chemical formulae",
    "structure of the atom and electrons",
    "cell structure and organelles",
    "plant and animal tissues",
    "motion, speed and acceleration",
    "Newton's laws of motion",
    "gravitation and weight",
    "work, energy and power",
    "sound waves and echo",
    "improvement in food resources",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def grounded(answer: str, context: str) -> float:
    words = [w for w in tokenize(answer) if len(w) > 2]
    ctx = set(tokenize(context))
    return sum(w in ctx for w in words) / len(words) if words else 0.0


def distractor_similarity(q) -> float:
    vecs = get_embedder().encode([q.answer] + [o for o in q.options if o != q.answer], normalize_embeddings=True)
    return float(np.mean(vecs[1:] @ vecs[0]))


def evaluate(generator: str, difficulty: str, n: int, all_chunks: list, topics: list) -> dict:
    rows, api_failed = [], 0
    for topic in topics:
        sources = [clean_text(c) for c in retrieve_chunks(topic, top_k=3 if generator == "gemini" else 8)]
        start = time.perf_counter()
        if generator == "gemini":
            try:
                qs = generate_quiz(" ".join(sources), topic, n, difficulty)
            except errors.APIError as e:
                print(f"  API unavailable for '{topic}' ({e.code}), skipping")
                api_failed += 1
                continue
            finally:
                time.sleep(4)  # stay under the free-tier requests-per-minute limit
        else:
            qs = generate_offline_quiz(sources, all_chunks, topic, n, difficulty)
        latency = time.perf_counter() - start
        context = " ".join(sources)
        rows.append({
            "yield": len(qs) / n,
            "valid": np.mean([len(set(q.options)) == 4 and q.answer in q.options for q in qs]) if qs else 0,
            "grounded": np.mean([grounded(q.answer, context) for q in qs]) if qs else 0,
            "distractor": np.mean([distractor_similarity(q) for q in qs]) if qs else 0,
            "latency": latency,
        })
    if not rows:
        return {"yield": 0, "valid": 0, "grounded": 0, "distractor": 0, "latency": 0, "api_failed": api_failed}
    return {**{k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}, "api_failed": api_failed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generators", nargs="+", default=["offline"])
    parser.add_argument("--n", type=int, default=5)
    # The Gemini free tier allows ~20 requests/day per model, so evaluate fewer topics there
    parser.add_argument("--topics", type=int, default=len(TOPICS))
    args = parser.parse_args()
    topics = TOPICS[::max(1, len(TOPICS) // args.topics)][:args.topics]

    all_chunks = load_chunks()
    results = []
    for gen in args.generators:
        for diff in ["easy", "medium", "hard"]:
            metrics = evaluate(gen, diff, args.n, all_chunks, topics)
            results.append({"generator": gen, "difficulty": diff, **metrics})
            print(results[-1])

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    name = "eval_" + "_".join(args.generators)
    with open(os.path.join(ROOT, "results", name + ".json"), "w") as f:
        json.dump(results, f, indent=2)
    header = ("| generator | difficulty | yield | valid | grounded | distractor sim | latency (s) | API failures |\n"
              "|---|---|---|---|---|---|---|---|\n")
    body = "".join(f"| {r['generator']} | {r['difficulty']} | {r['yield']:.2f} | {r['valid']:.2f} | "
                   f"{r['grounded']:.2f} | {r['distractor']:.2f} | {r['latency']:.2f} | {r['api_failed']} |\n"
                   for r in results)
    with open(os.path.join(ROOT, "results", name + ".md"), "w") as f:
        models = f" (model: {os.getenv('GEMINI_MODEL', 'default')})" if "gemini" in args.generators else ""
        f.write(f"Evaluated on {len(topics)} Grade 9 science topics, {args.n} questions each{models}.\n\n" + header + body)
    print(header + body)


if __name__ == "__main__":
    main()
