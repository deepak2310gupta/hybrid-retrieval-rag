from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

def create_llm():


    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=512,
        temperature=0.1,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm