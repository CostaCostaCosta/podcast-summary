import segment as segment
import json
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
import numpy as np
import os
import time
import json
import ipdb

from insanely_fast_whisper.transcribe import transcribe


class AudioPreprocess:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.transcription = ""
        self.segments = []

    def perform_transcription(self):
        if self.load_transcription():
            print("Loaded pre-saved transcription")
        else:
            print("Transcribing Audio")
            self.transcription = transcribe(self.audio_file)
            print("Saving Audio transcription")
            self.save_transcription(self.transcription, self.audio_file)
        return self.transcription
    

    def save_transcription(self, transcription_text, audio_file):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_name = os.path.splitext(os.path.basename(audio_file))[
            0
        ]  # Extract base name without extension
        file_name = f"{base_name}_{timestamp}.json"
        file_path = os.path.join("data", "transcripts", file_name)

        with open(file_path, "w") as file:
            json.dump(transcription_text, file)

    def load_transcription(self):
        transcript_directory = os.path.join("data", "transcripts")
        audio_file_base_name = os.path.splitext(os.path.basename(self.audio_file))[
            0
        ]  # Extract base name without extension
        for filename in os.listdir(transcript_directory):
            base_name = filename.split("_")[
                0
            ]  # Assuming the file name format is consistent
            if base_name == audio_file_base_name:
                file_path = os.path.join(transcript_directory, filename)
                with open(file_path, "r") as file:
                    transcription_text = json.load(file)
                self.transcription = transcription_text
                return True
        return False


class Segmenter:
    def __init__(self):
        # Initialize the BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        self.model.eval()  # Put model in evaluation mode

    def generate_embeddings(self, text):
        # Add special tokens
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        # Use the average of the last four layers as the sentence embedding
        token_vecs = torch.stack(hidden_states[-4:]).mean(0).squeeze()
        return token_vecs.mean(dim=0).numpy()

    def cluster_texts(self, texts, n_clusters):
        embeddings = np.array([self.generate_embeddings(text) for text in texts])
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)
        return kmeans.labels_

    def segment_json_by_cluster(self, json_data, n_clusters):
        texts = [chunk["text"] for chunk in json_data["chunks"]]
        clusters = self.cluster_texts(texts, n_clusters)

        segmented_data = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in segmented_data:
                segmented_data[cluster_id] = []
            segmented_data[cluster_id].append(json_data["chunks"][i]["text"])

        return segmented_data

    def segment_json_by_tokens(self, json_data, n_tokens):
        segments = []
        current_segment = []
        current_token_count = 0

        for chunk in json_data["chunks"]:
            text = chunk["text"]
            tokenized_text = self.tokenizer.tokenize(text)
            token_count = len(tokenized_text)
            current_token_count += token_count

            if current_token_count <= n_tokens:
                current_segment.append(text)
            else:
                segments.append(" ".join(current_segment))
                current_segment = [text]
                current_token_count = token_count

        # Add the last segment if it's not empty
        if current_segment:
            segments.append(" ".join(current_segment))

        return segments


# Example usage of the classes
audio_transcriber = AudioPreprocess("./data/audio/shorter.mp3")
# audio_transcriber = AudioPreprocess("./data/audio/RotoGraphs-Audio-01-15-2024.mp3")
json_transcription = audio_transcriber.perform_transcription()
segmenter = Segmenter()
n_clusters = (
    5  # Define the number of topics you expect in the transcription #USE NER FOR THIS?
)
n_tokens = 2048  # set this to the context window size - prompt token length
segmented_data = segmenter.segment_json_by_tokens(json_transcription, n_tokens=2048)
ipdb.set_trace()
