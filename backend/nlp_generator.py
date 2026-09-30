"""Offline MCQ generator: a classic NLP pipeline that needs no LLM or API key.

1. Text cleaning of PDF extraction noise (running headers, hyphenation, repeats)
2. Sentence segmentation (NLTK Punkt) and filtering
3. Rank sentences by semantic similarity to the topic (MiniLM embeddings)
4. POS tagging + noun-phrase chunking; each noun phrase is scored by
   IDF over the corpus x similarity to the topic, the best one becomes the answer
5. Blank the answer out -> cloze (fill-in-the-blank) question
6. Distractors = other corpus noun phrases nearest to the answer in embedding
   space, with the same length and grammatical number; difficulty controls
   how close they are
"""
import math
import random
import re
from collections import Counter, defaultdict

import nltk
import numpy as np

from backend.models import Question
from backend.rag import get_embedder

for _pkg, _path in [("punkt_tab", "tokenizers/punkt_tab"), ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

# Adjective(s) followed by 1-2 nouns, e.g. "cell membrane", "kinetic energy", "atoms"
CHUNKER = nltk.RegexpParser("NP: {<JJ>?<NN|NNS>{1,2}}")

GENERIC = set("""
activity activities answer answers chapter chapters example examples exercise exercises fig figure
figures table tables page question questions reprint science thing things way ways kind kinds type types
part parts number numbers time times case cases fact facts form forms help lot use uses amount
group groups result results step steps order point points place places end side sides reason reasons
effect effects observation observations student students teacher teachers class day days year years
""".split())

DIFFICULTY_RANKS = {"hard": (0, 5), "medium": (3, 12), "easy": (8, 25)}


def clean_text(text: str) -> str:
    text = text.replace("�", "").replace("’", "'")
    text = re.sub(r"[-]", " ", text)             # private-use glyphs (bullets, symbols)
    text = re.sub(r"SCIENCE\s*\d+", " ", text)                # running footer "SCIENCE96"
    text = re.sub(r"Reprint\s+\d{4}-\d{2}", " ", text)
    # Running page headers: runs of ALL-CAPS words, optionally with page numbers
    text = re.sub(r"\b(?:[A-Z]{2,}\s+){2,}[A-Z]{2,}\b(?:\s+\d+)*", " ", text)
    text = re.sub(r"\b(\d+)(?:\s*\1)+\b", " ", text)          # "93 9393 9393"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)              # "sub- atomic" -> "subatomic"
    text = re.sub(r"(.{6,40}?)\1+", r"\1", text)               # "For example:For example:"
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    sentences = []
    for s in nltk.sent_tokenize(clean_text(text)):
        words = s.split()
        if not (8 <= len(words) <= 40) or not s.endswith(".") or not s[0].isupper():
            continue
        # Skip exercise questions, figure captions, chapter summaries and extraction noise
        if "?" in s or re.match(r"^(Fig|Table|Activity|What you have learnt)", s) or re.search(r"\b[A-Z]\d\.", s):
            continue
        # Words split apart by PDF extraction ("W e will lear n") leave stray single letters
        if any(len(w) == 1 and w.isalpha() and w not in {"a", "A", "I"} for w in words):
            continue
        alpha = sum(c.isalpha() or c.isspace() for c in s) / len(s)
        if alpha < 0.93:
            continue
        sentences.append(s)
    return sentences


def noun_phrases(sentence: str) -> list[tuple[str, str]]:
    """(phrase, head POS tag) pairs, e.g. ("cell membrane", "NN")."""
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    phrases = []
    for subtree in CHUNKER.parse(tagged).subtrees(lambda t: t.label() == "NP"):
        words = [w.lower() for w, _ in subtree.leaves()]
        head_tag = subtree.leaves()[-1][1]
        if not all(w.isalpha() and len(w) >= 3 for w in words):
            continue
        if any(w in GENERIC for w in words) or len(words[-1]) < 4:
            continue
        phrases.append((" ".join(words), head_tag))
    return phrases


class Corpus:
    """Noun-phrase statistics over all indexed chunks, used for IDF weighting."""

    def __init__(self, chunks: list[str]):
        self.n_docs = len(chunks)
        self.tf, self.df = Counter(), Counter()
        tags = defaultdict(Counter)
        for chunk in chunks:
            seen = set()
            for sentence in nltk.sent_tokenize(clean_text(chunk)):
                for phrase, tag in noun_phrases(sentence):
                    self.tf[phrase] += 1
                    tags[phrase][tag] += 1
                    seen.add(phrase)
            self.df.update(seen)
        self.head_tag = {p: c.most_common(1)[0][0] for p, c in tags.items()}
        # Phrases seen at least twice are less likely to be PDF extraction noise
        self.vocab = [p for p, c in self.tf.items() if c >= 2]
        self.vocab_vecs = get_embedder().encode(self.vocab, normalize_embeddings=True)
        self.vocab_index = {p: i for i, p in enumerate(self.vocab)}

    def idf(self, phrase: str) -> float:
        return math.log((1 + self.n_docs) / (1 + self.df[phrase])) + 1


_corpus_cache: dict = {}


def get_corpus(chunks: list[str]) -> Corpus:
    # Tagging the whole book takes a few seconds; reuse it across requests
    key = (len(chunks), hash(chunks[0]) if chunks else 0)
    if key not in _corpus_cache:
        _corpus_cache.clear()
        _corpus_cache[key] = Corpus(chunks)
    return _corpus_cache[key]


def related(a: str, b: str) -> bool:
    """True if two phrases share a word stem (e.g. 'atom' / 'atoms' / 'atomic mass')."""
    return any(x[:5] == y[:5] for x in a.split() for y in b.split())


def pick_answer(sentence: str, corpus: Corpus, topic_vec: np.ndarray, used: set) -> str | None:
    best, best_score = None, 0.0
    for phrase, _ in noun_phrases(sentence):
        if phrase in used or phrase not in corpus.vocab_index:
            continue
        topical = float(corpus.vocab_vecs[corpus.vocab_index[phrase]] @ topic_vec)
        # Informative (rare in the book) and on-topic; multi-word concepts are more specific
        score = corpus.idf(phrase) * max(topical, 0.05) * (1 + 0.3 * (phrase.count(" ")))
        if score > best_score:
            best, best_score = phrase, score
    return best


def blank_out(sentence: str, phrase: str) -> str:
    pattern = r"\b" + r"\s+".join(map(re.escape, phrase.split())) + r"\b"
    return re.sub(pattern, "_____", sentence, flags=re.IGNORECASE)


def pick_distractors(answer: str, sentence: str, corpus: Corpus, difficulty: str,
                     rng: random.Random) -> list[str]:
    sims = corpus.vocab_vecs @ corpus.vocab_vecs[corpus.vocab_index[answer]]
    n_words, tag = answer.count(" "), corpus.head_tag[answer]
    ranked = []
    for i in np.argsort(-sims):
        phrase = corpus.vocab[i]
        # Same length and grammatical number so the answer doesn't stand out
        if phrase.count(" ") != n_words or corpus.head_tag[phrase] != tag:
            continue
        if sims[i] > 0.85 or related(phrase, answer) or phrase in sentence.lower():
            continue
        if any(related(phrase, r) for r in ranked):
            continue
        ranked.append(phrase)
        if len(ranked) >= 25:
            break
    lo, hi = DIFFICULTY_RANKS[difficulty]
    pool = ranked[lo:hi] if len(ranked) >= lo + 3 else ranked
    return rng.sample(pool, 3) if len(pool) >= 3 else []


def generate_offline_quiz(sources: list[str], all_chunks: list[str], topic: str,
                          num_questions: int = 5, difficulty: str = "medium",
                          seed: int = 42) -> list[Question]:
    rng = random.Random(seed)
    corpus = get_corpus(all_chunks)
    embedder = get_embedder()

    sentences = list(dict.fromkeys(s for src in sources for s in split_sentences(src)))
    if not sentences:
        return []
    topic_vec = embedder.encode([topic], normalize_embeddings=True)[0]
    sent_vecs = embedder.encode(sentences, normalize_embeddings=True)
    order = np.argsort(-(sent_vecs @ topic_vec))

    questions, used = [], set()
    for i in order:
        sentence = sentences[i]
        answer = pick_answer(sentence, corpus, topic_vec, used)
        if not answer:
            continue
        blanked = blank_out(sentence, answer)
        if blanked == sentence:
            continue
        distractors = pick_distractors(answer, sentence, corpus, difficulty, rng)
        if len(distractors) < 3:
            continue
        options = distractors + [answer]
        rng.shuffle(options)
        questions.append(Question(
            question=f"Fill in the blank: {blanked}",
            options=options,
            answer=answer,
            explanation=f'From the textbook: "{sentence}"',
        ))
        used.add(answer)
        if len(questions) == num_questions:
            break
    return questions
