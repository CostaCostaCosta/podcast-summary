import json
import os
from dotenv import load_dotenv
import requests
import ipdb

## SAMPLE CODE FROM https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
# url = "http://127.0.0.1:5000/v1/chat/completions"


def summarize_transcription(api_url, transcription):
    """
    Calls an OpenAI-like API to create a condensed summary of a given transcription.

    Parameters:
    api_url (str): The URL of the API endpoint.
    api_key (str): The API key for authentication.
    transcription (str): The transcription text to be summarized.

    Returns:
    dict: The response from the API.
    """

    # # Set the headers with the API key
    headers = {"Content-Type": "application/json"}
    history = []

    # Define the prompt
    prompt = f"Provide a structured analysis and summary of the following baseball podcast transcription, focusing on the key players discussed:\n\n{transcription}"
    payload = prompt + transcription

    # Prepare the request payload
    # payload = {
    #     # "prompt": prompt + transcription,
    #     "max_tokens": 500,  # Adjust the token limit based on your needs
    # }

    history.append({"role": "user", "content": payload})
    data = {"mode": "chat", "character": "AI", "messages": history}

    # Make the API request
    response = requests.post(api_url, headers=headers, json=data, verify=False)
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        return {"Error": response.text}


# Example usage
api_url = "http://127.0.0.1:5000/v1/chat/completions"  # Replace with the actual API URL
api_key = ""  # Replace with your actual API key

# specify the file path
file_path = "./data/transcripts/shorter_20240130-135329.json"

# open the file and load the text
with open(file_path, "r") as file:
    data = json.load(file)

# access the transcription excerpt
transcription_excerpt = data["text"]
# transcription_excerpt = "Shortened version of the transcription..."  # Replace with the actual transcription excerpt

summary = summarize_transcription(api_url, transcription_excerpt)
print(summary)
