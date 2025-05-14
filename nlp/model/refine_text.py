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
    # Role: NSL Gesture Interpretation Corrector
    # Task:
    Process raw text output from a Bi-LSTM trained on Nigerian Sign Language (NSL) gestures.
    Clean, correct grammar, remove repetitions, and reconstruct the input into a coherent, grammatically correct English sentence that accurately reflects the intended meaning from the gestures.
    # Input Characteristics:
    - Raw text from ML model.
    - May contain: repeated words, poor grammar, fragmented phrases, disordered sequences.
    - Uses words, letters, and numbers ONLY from the "Vocabulary Reference" list below.
    # Output Goal:
    A single, natural-sounding English sentence.
    # Core Instructions:
    1.  **Correct Grammar & Structure:** Ensure the output is a grammatically correct and well-structured English sentence.
    2.  **Remove Repetitions:** Eliminate any redundant or repeated words/phrases.
    3.  **Reconstruct Logical Meaning:** If the input is disordered, infer the most probable logical meaning and sequence based on the available vocabulary items.
    4.  **Preserve Intent Strictly:** DO NOT add information not implied by the input words. Only reorder or rephrase to improve clarity based *only* on the provided concepts.
    5.  **Human Tone**: Do your best to make the output sound human and real.
    # Vocabulary Reference:
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, Name, Learn, Restroom, No, What, Sign, Where, Sister, Nice, Not, Classroom, Girl-friend, You, Student, Buy, Brother, Meet, Teacher, Food, Have
    # Examples:
    You name what = What is your name?
    Me name D A V I D = My name is David.
    Nice to meet you = Nice to meet you.
    You learn sign you? = Do you know how to sign?
    Me not teacher, me student = I am not a teacher, I am a student.
    Restroom where? = Where is the restroom?
    Me want food, where buy? = I am hungry. Where can I buy food?
    No me not have girlfriend = I don't have a girlfriend.
    Me have 2 sister, 1 brother = I have 2 sisters and a brother.
    # Input (Raw Bi-LSTM Output): {text}
    """)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model='gemini-2.0-flash', contents=prompt_text
    )
    
    refined_text = response.text
    logging.info(f"Refined text: {refined_text}")
    
    return refined_text
