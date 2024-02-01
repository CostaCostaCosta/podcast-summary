from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# nltk.download('punkt')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load pre-trained model (weights)
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)


# Function to compute sentence embeddings
def sentence_embedding(sentence):
    # Add special tokens
    marked_text = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    token_vecs_sum = [torch.sum(token[-4:], dim=0) for token in token_embeddings]
    sentence_embedding = torch.mean(torch.stack(token_vecs_sum), dim=0)
    return sentence_embedding


# Function to segment the text
def segment_text(text, similarity_threshold=0.5):
    sentences = sent_tokenize(text)
    sentence_embeddings = [
        sentence_embedding(sentence).numpy() for sentence in sentences
    ]

    segments = []
    current_segment = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([sentence_embeddings[i - 1]], [sentence_embeddings[i]])[
            0
        ][0]
        if sim < similarity_threshold:
            segments.append(" ".join(current_segment))
            current_segment = [sentences[i]]
        else:
            current_segment.append(sentences[i])

    segments.append(" ".join(current_segment))
    return segments


# # Example usage
# text = "Your text goes here."

# segments = segment_text(text)
# print(segments)


def segment_json(self, json_data):
    segments = []
    current_segment = []
    current_topic = None

    for chunk in json_data["chunks"]:
        text = chunk["text"]
        detected_topic = detect_topics_spacy(text)

        if detected_topic != current_topic:
            if current_segment:
                segments.append(
                    {"topic": current_topic, "text": " ".join(current_segment)}
                )
                current_segment = []
            current_topic = detected_topic

        current_segment.append(text)

    # Add the last segment
    if current_segment:
        segments.append({"topic": current_topic, "text": " ".join(current_segment)})

    return segments


def detect_topics_spacy(text):
    print("detecting topics using spacy")
    # Load BERT

    # Process the text with spaCy
    doc = self.nlp(text)

    # Extract entities and filter for proper nouns (likely to be player names)
    # You can also customize this to look for specific entity types like PERSON
    topics = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            topics.add(ent.text)

    # For simplicity, return the first identified topic
    # In a more complex scenario, you might need a more sophisticated approach
    return next(iter(topics), None)


def detect_topics_bert(text):
    return text


def detect_topics_bert(text):
    return text
