import os

OPENAI_API_HOST = "http://127.0.0.1:5000"
OPENAI_API_KEY = "sk-111111111111111111111111111111111111111111111111"
OPENAI_API_BASE = "http://127.0.0.1:5000/v1"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_HOST"] = OPENAI_API_HOST
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI


def summarize_transcription_langchain(llm_chain, transcription):
    """
    Uses langchain to create a condensed summary of a given transcription.

    Parameters:
    llm_chain (LLMChain): The langchain instance.
    transcription (str): The transcription text to be summarized.

    Returns:
    str: The response from the langchain.
    """

    # Define the prompt
    prompt = f"Provide a structured analysis and summary of the following baseball podcast transcription, focusing on the key players discussed:\n\n{transcription}"

    # Use the LLMChain instance to generate the summary
    summary = llm_chain.run(prompt)

    return summary


# Example usage
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
llm = OpenAI()

llm_chain = LLMChain(prompt=prompt, llm=llm)

# specify the file path
file_path = "./data/transcripts/shorter_20240130-135329.json"

# open the file and load the text
with open(file_path, "r") as file:
    data = json.load(file)

# access the transcription excerpt
transcription_excerpt = data["text"]

summary = summarize_transcription_langchain(llm_chain, transcription_excerpt)
print(summary)
