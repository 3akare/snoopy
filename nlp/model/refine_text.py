import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

model_name = "vennify/t5-base-grammar-correction"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

def refine_text(text):
    logging.info(f"Received text: {text}")
    prompt_text = f"Grammar: {text}"
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=len(text) + 10)
    refined_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logging.info(f"Refined text: {refined_text}")
    return refined_text
