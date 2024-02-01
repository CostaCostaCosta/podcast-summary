import requests
import json
import os


def summarize_transcription(api_url, api_key, transcription):
    """
    Calls an OpenAI-like API to create a condensed summary of a given transcription.

    Parameters:
    api_url (str): The URL of the API endpoint.
    api_key (str): The API key for authentication.
    transcription (str): The transcription text to be summarized.

    Returns:
    str: The summary of the transcription.
    """

    # Define the prompt
    prompt = f"Provide a structured analysis and summary of the following baseball podcast transcription, focusing on the key players discussed:\n\n{transcription}"

    # Prepare the request payload
    payload = {
        "prompt": prompt,
        "max_tokens": 200,  # Adjust the token limit based on your needs
    }

    # Set the headers with the API key
    headers = {"Authorization": f"Bearer {api_key}"}

    # Make the API request
    response = requests.post(api_url, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("text", "").strip()
    else:
        return f"Error: {response.text}"


# Example usage
api_url = "http://127.0.0.1:5000"  # Replace with the actual API URL
api_key = ""  # Replace with your actual API key

# specify the file path
file_path = "./data/transcripts/shorter_20240130-135329.json"

# open the file and load the text
with open(file_path, "r") as file:
    data = json.load(file)

# access the transcription excerpt
transcription_excerpt = data["text"]
# transcription_excerpt = "Shortened version of the transcription..."  # Replace with the actual transcription excerpt


summary = summarize_transcription(api_url, api_key, transcription_excerpt)
print(summary)
