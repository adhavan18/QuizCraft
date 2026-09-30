"""Evaluate quiz generators on the indexed textbook.

Usage:  python evaluate.py [--generators offline gemini] [--n 5]

Metrics (per generator x difficulty, averaged over topics):
  yield        questions returned / questions requested
  valid        4 distinct options and the answer is one of them
  grounded     share of answer words that occur in the retrieved context
  distractor   mean cosine similarity between answer and distractors
               (higher = more confusable = harder)
  latency      seconds per quiz
"""
import argparse
import json
import os
import time

import numpy as np

from backend.llm import generate_quiz
from backend.nlp_generator import generate_offline_quiz, tokenize
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


def grounded(answer: str, context: str) -> float:
    words = [w for w in tokenize(answer) if len(w) > 2]
    ctx = set(tokenize(context))
    return sum(w in ctx for w in words) / len(words) if words else 0.0


def distractor_similarity(q) -> float:
    vecs = get_embedder().encode([q.answer] + [o for o in q.options if o != q.answer], normalize_embeddings=True)
    return float(np.mean(vecs[1:] @ vecs[0]))


def evaluate(generator: str, difficulty: str, n: int, all_chunks: list) -> dict:
    rows = []
    for topic in TOPICS:
        sources = retrieve_chunks(topic, top_k=3 if generator == "gemini" else 8)
        start = time.perf_counter()
        if generator == "gemini":
            qs = generate_quiz(" ".join(sources), topic, n, difficulty)
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
    return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generators", nargs="+", default=["offline"])
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    all_chunks = load_chunks()
    results = []
    for gen in args.generators:
        for diff in ["easy", "medium", "hard"]:
            metrics = evaluate(gen, diff, args.n, all_chunks)
            results.append({"generator": gen, "difficulty": diff, **metrics})
            print(results[-1])

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    header = "| generator | difficulty | yield | valid | grounded | distractor sim | latency (s) |\n|---|---|---|---|---|---|---|\n"
    body = "".join(f"| {r['generator']} | {r['difficulty']} | {r['yield']:.2f} | {r['valid']:.2f} | "
                   f"{r['grounded']:.2f} | {r['distractor']:.2f} | {r['latency']:.2f} |\n" for r in results)
    with open(os.path.join(ROOT, "results", "eval_results.md"), "w") as f:
        f.write(f"Evaluated on {len(TOPICS)} Grade 9 science topics, {args.n} questions each.\n\n" + header + body)
    print(header + body)


if __name__ == "__main__":
    main()
