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
    You are a grammar and coherence correction tool. You will receive text generated from a Bi-LSTM model trained on Nigerian Sign Language gestures. This text may include unnecessary repetitions, poor grammar, and structural inconsistencies.
    Your task is to:
    1. Correct grammatical errors and ensure the sentence is well-structured.
    2. Remove unnecessary repetitions, especially if the same phrase or word appears multiple times in a row (e.g., "I love you I love you" should become "I love you").
    3. Preserve the original intent and meaning of the sentence, even if grammar or vocabulary is limited.
    4. If the input is a sequence of unrelated or partial words (e.g., "Name Hello Water My"), reconstruct them into the most logical, natural-sounding, and grammatically correct sentence based on the available vocabulary.
    5. Do not fabricate meaning beyond what the words suggest, but feel free to reorder or slightly rephrase for better clarity.
    Vocabulary list for reference:
    book, Thank, You, Water, Hello, Food, My, Goodbye, School, Name, Good Afternoon, Book, David, Nigeria, Understand, And, Sign Language, This, Table, Final, Chair, Year, Laptop, Presentation, Friend.
    Here is the text: {text}
    """)
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt_text
    )
    logging.info(f"Refined text: {response.text}")
    return response.text

