import sys

from backend.llm import MODEL_NAME, generate_with_retry

sys.stdout.reconfigure(encoding="utf-8")  # emoji on Windows consoles

try:
    response = generate_with_retry(contents="Hello Gemini, say hi in one short sentence.")
    print(f"✅ API Key works with {MODEL_NAME}! Response:", response.text)
except Exception as e:
    print("API Key test failed. Error:", e)
