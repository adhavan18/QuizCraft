from backend.llm import MODEL_NAME, get_client

try:
    response = get_client().models.generate_content(
        model=MODEL_NAME, contents="Hello Gemini, say hi in one short sentence."
    )
    print("✅ API Key works! Response:", response.text)
except Exception as e:
    print("API Key test failed. Error:", e)
