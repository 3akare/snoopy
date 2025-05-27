import os
import sys
import logging
import textwrap
from google import genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def refine_text(text):
    logging.info(f"Received text: {text}")
    
    prompt_text = textwrap.dedent(f"""
    You are an exceptionally precise and powerful language processor.
    Your sole function is to transform fragmented or keyword-based inputs, representing detected ASL signs, into perfectly natural, grammatically correct, and coherent English sentences, statements, or short phrases.
    **Input:**
    The input will always be in uppercase. It consists of recognized ASL elements:
    - Individual English alphabet letters (A-Z)
    - Numbers (0-9)
    - A selected vocabulary of common English words (e.g., NAME, LEARN, NO, WHAT, SIGN, WHERE, SISTER, NICE, NOT, CLASSROOM, GIRL-FRIEND, YOU, STUDENT, BUY, BROTHER, MEET, TEACHER, FOOD, HAVE).
    **Core Task - Natural Language Reconstruction:**
    Your primary objective is to reconstruct these raw inputs into fluid, idiomatic English. You must intelligently infer and add necessary grammatical elements such as:
    - **Verbs** (e.g., "is," "am," "are," "go")
    - **Articles** ("a," "an," "the")
    - **Prepositions** ("to," "for," "with")
    - **Conjunctions**
    - **Proper punctuation** (commas, periods, question marks).
    The final output should sound exactly like natural human speech or writing, even if the input is sparse.
    **Output Constraint:**
    - **Success:** If a coherent and meaningful translation is possible, provide **ONLY** the translated English sentence/phrase.
    - **Failure:** If the input is genuinely nonsensical or you cannot logically form a coherent English output, provide **ONLY** the following exact phrase: "I could not translate this."
    - **NO other text, explanations, or conversational elements are permitted in your response.**
    Example:
    - **Input:** "ME NAME DAVID"
    - **Output:** "My name is David."
    Input: {text}
    """)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model='gemini-2.0-flash', contents=prompt_text
    )
    refined_text = response.text
    logging.info(f"Refined text: {refined_text}")
    
    return refined_text
