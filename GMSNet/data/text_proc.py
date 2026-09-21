

import re
import torch
from transformers import ElectraTokenizer
from transformers import AutoTokenizer 

def clean_text(text):
    """Normalize tweet text and remove URLs, mentions, and unsupported characters."""
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"@[^\s]+", "", text)  
    text = re.sub(r"#[^\s]+", "", text)  
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)  
    text = re.sub(r"\s{2,}", " ", text)  
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"#[^\s]+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


class TextProcessor:
    def __init__(self, model_name='vinai/bertweet-base', max_len=128):
        print(f"Loading tokenizer: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.max_len = max_len

    def __call__(self, text):
        cleaned = clean_text(text)
        encoding = self.tokenizer(
            cleaned,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}