import json
from tokenizers import Tokenizer
import unicodedata
import re

file_path = "/home/angkul/my_data/coding/agi/llms/hi.txt"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def normalize_text(text):
    return unicodedata.normalize("NFKC", text)

def remove_non_hindi(text):
    hindi_pattern = re.compile(r'[\u0900-\u097F]+')
    return " ".join(hindi_pattern.findall(text))

def clean_text(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text):
    return re.split(r'[।!?]', text)

def filter_sentences(sentences, min_length=5):
    unique_sentences = list(set(sentences))
    return [s.strip() for s in unique_sentences if len(s.split()) >= min_length]

def preprocess_corpus(lines):
    processed_lines = []
    for line in lines:
        line = normalize_text(line)
        line = remove_non_hindi(line)
        line = clean_text(line)
        sentences = split_sentences(line)
        sentences = filter_sentences(sentences)
        processed_lines.extend(line)
    return processed_lines


# Process it
processed_lines = preprocess_corpus(lines)

# Save the cleaned text
with open("hindi_corpus_cleaned.txt", "w", encoding="utf-8") as f:
    for line in processed_lines:
        f.write(line + "\n")

print(f"Preprocessing completed! {len(processed_lines)} clean sentences saved.")

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())

# Define a trainer for BPE
trainer = trainers.BpeTrainer(vocab_size=150000, min_frequency=1, special_tokens=["<unk>", "<s>", "</s>", "?", "।", "!"])

# Pre-tokenizer (breaks Hindi words correctly)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Load Hindi dataset
files = ["hindi_corpus_cleaned.txt"]  # Your dataset file
tokenizer.train(files, trainer)

# Save tokenizer
tokenizer.save("hindi_tokenizer.json")

class HindiTokenizer:
    def __init__(self, model_path="hindi_tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(model_path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)