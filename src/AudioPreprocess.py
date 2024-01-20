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
            print('Loaded pre-saved transcription')
        else:
            self.transcription = transcribe(self.audio_file)
        return self.transcription
    
    def save_transcription(self, transcription_text, audio_file):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"{audio_file}_{timestamp}.json"
        file_path = os.path.join("data", "transcripts", file_name)
    
        with open(file_path, 'w') as file:
            json.dump(transcription_text, file)
    
    def load_transcription(self):
        transcript_directory = os.path.join("data", "transcripts")
        for filename in os.listdir(transcript_directory):
            if filename.startswith(self.audio_file):
                file_path = os.path.join(transcript_directory, filename)
                with open(file_path, 'r') as file:
                    transcription_text = json.load(file)
                self.transcription = transcription_text
                return True
        return False

class Segmenter:
    def __init__(self):
        # Initialize the BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
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

    def segment_json(self, json_data, n_clusters):
        texts = [chunk["text"] for chunk in json_data["chunks"]]
        clusters = self.cluster_texts(texts, n_clusters)

        segmented_data = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in segmented_data:
                segmented_data[cluster_id] = []
            segmented_data[cluster_id].append(json_data["chunks"][i]["text"])

        return segmented_data

# Example usage of the classes
audio_transcriber = AudioPreprocess("./data/audio/shorter.mp3")
json_transcription = audio_transcriber.perform_transcription()
segmenter = Segmenter()
n_clusters = 5  # Define the number of topics you expect in the transcription #USE NER FOR THIS?
segmented_data = segmenter.segment_json(json_transcription, n_clusters)
ipdb.set_trace()

# segmented_data now contains the text grouped by inferred topics

    
# class Segmenter:
#     def segment(self, text):
#         self.segments = segment.segment_text(self.transcription)
#         pass

#     def segment_json(self, json_data):
#         self.sements = segment.segment_json(self.transcription)

# # Example usage
# json_transcription = {
#     "chunks": [
#         {"timestamp": [0.0, 21.72], "text": "Hello and welcome..."},
#         # ... more chunks ...
#     ]
# }

# segmenter = Segmenter()
# segments = segmenter.segment_json(json_transcription)
# print(segments)


# # Example usage of the classes
# audio_transcriber = AudioPreprocess("path/to/audio/file.mp3")
# audio_transcriber.perform_transcription()
# audio_transcriber.segment_transcription()

# segments = audio_transcriber.get_segments()
