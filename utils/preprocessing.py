# utils/preprocessing.py
import torch
import numpy as np
from typing import List, Tuple
from torchvision import transforms
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def denormalize_image(image_tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    device = image_tensor.device
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)

def get_image_transforms(image_size: int = 224) -> transforms.Compose:
    """Get standard image transformations"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Text preprocessing utilities
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)