import os
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set the environment variable GOOGLE_API_KEY first.")

genai.configure(api_key=GOOGLE_API_KEY)

print("Available models for this key:")
for m in genai.list_models():
    print("  •", m.name)
