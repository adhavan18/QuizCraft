import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="QuizLLM", layout="wide")

st.title("📘 QuizLLM - NCERT Quiz Generator")

tab1, tab2 = st.tabs(["📂 Upload NCERT PDF", "📝 Generate Quiz"])

with tab1:
    st.header("Upload NCERT PDF")
    uploaded_file = st.file_uploader("Upload NCERT PDF", type=["pdf"])
    if uploaded_file:
        if st.button("Ingest & Build Index"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            with st.spinner("Extracting text and building the vector index..."):
                res = requests.post(f"{API_URL}/ingest/upload", files=files)
            if res.status_code == 200:
                st.success(f"Indexed {res.json()['chunks']} chunks from {uploaded_file.name}")
            else:
                st.error(f"Error {res.status_code}: {res.text}")

with tab2:
    st.header("Generate Quiz")
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    topic = col1.text_input("Enter topic or chapter name:")
    num_q = col2.slider("Number of questions", 1, 10, 5)
    difficulty = col3.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    generator = col4.selectbox("Generator", ["auto", "gemini", "offline"],
                               help="offline = local NLP pipeline (TF-IDF + embeddings), no API key needed")

    if st.button("Generate Quiz", disabled=not topic.strip()):
        payload = {"topic": topic, "num_questions": num_q, "difficulty": difficulty, "generator": generator}
        with st.spinner("Retrieving context and generating questions..."):
            res = requests.post(f"{API_URL}/quiz", json=payload)
        if res.status_code == 200:
            st.session_state.quiz = res.json()
            st.session_state.submitted = False
        else:
            st.error(f"Error {res.status_code}: {res.json().get('detail', res.text)}")

    quiz = st.session_state.get("quiz")
    if quiz:
        st.subheader(f"Quiz: {quiz['topic']} ({quiz['difficulty']}, {quiz['generator']})")
        answers = {}
        for i, q in enumerate(quiz["questions"]):
            answers[i] = st.radio(f"**Q{i + 1}.** {q['question']}", q["options"], index=None, key=f"q{i}")
            if st.session_state.get("submitted"):
                if answers[i] == q["answer"]:
                    st.success(f"Correct! {q['explanation']}")
                else:
                    st.error(f"Answer: {q['answer']}. {q['explanation']}")

        if st.button("Submit answers"):
            st.session_state.submitted = True
            st.rerun()

        if st.session_state.get("submitted"):
            score = sum(answers[i] == q["answer"] for i, q in enumerate(quiz["questions"]))
            st.metric("Score", f"{score} / {len(quiz['questions'])}")

        with st.expander("Retrieved textbook context (RAG sources)"):
            for j, src in enumerate(quiz.get("sources", []), start=1):
                st.markdown(f"**Chunk {j}:** {src[:600]}...")
