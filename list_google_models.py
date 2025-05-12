# list_google_models.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("GOOGLE_API_KEY not found in .env file.")
else:
    try:
        genai.configure(api_key=api_key)
        print("Available models for your API key:")
        for m in genai.list_models():
            # We are interested in models that support 'generateContent' for text generation
            if 'generateContent' in m.supported_generation_methods:
                print(f"- Name: {m.name}")
                print(f"  Display Name: {m.display_name}")
                print(f"  Description: {m.description}")
                print(f"  Supported Methods: {m.supported_generation_methods}")
                print("-" * 20)
    except Exception as e:
        print(f"An error occurred: {e}")