from google import genai
import logging
import os
import textwrap

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def refine_text(text):
    logging.info(f"Received text: {text}")
    
    prompt_text = textwrap.dedent(f"""
    You are a specialized text correction and interpretation engine designed for processing raw outputs from a Bi-LSTM model trained on Nigerian Sign Language (NSL) gestures. The output may include repeated words, poor grammar, fragmented phrases, or disordered sequences due to the nature of gesture recognition.
    Your task is to intelligently clean, correct, and reconstruct the input text into a coherent, grammatically correct English sentence while preserving the intended meaning of the original gestures as much as possible.
    Follow these instructions precisely:
    Correct grammar and structure: Fix all grammatical errors and make sure the sentence reads naturally in English.
    Remove repetitions: Eliminate any unnecessary repeated words or phrases (e.g., "Thank Thank You You" → "Thank You").
    Reconstruct logical meaning: If the input is a disorganized set of partial or isolated gesture words (e.g., "Hello Name Water"), infer and rearrange them into the most logical sentence based on meaning and common usage.
    Preserve intent: Do not invent or add extra meaning beyond what the gestures imply. Only reorder or slightly rephrase to improve clarity.
    Respect vocabulary scope: Stick strictly to the vocabulary provided.
    Vocabulary Reference:
    A-Z, Notebook, Thank You, Water, I love you, Hello, Food, My, Goodbye, School, Name, Good Afternoon, Book, Friend, Country, Understand, And, Sign Language, This, Table, Final, Chair, Year, Computer, Talk, Friend and more... just make it make sense
    Input (raw Bi-LSTM output): {text}""")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model='gemini-2.0-flash', contents=prompt_text
    )
    
    refined_text = response.text
    logging.info(f"Refined text: {refined_text}")
    
    return refined_text
