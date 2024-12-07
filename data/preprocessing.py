from typing import List, Tuple
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import torch
from torchvision import transforms

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK resources
        for resource in ['wordnet', 'stopwords']:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

class ImagePreprocessor:
    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        return self.transform(image)

def get_file_paths(directory: str) -> List[str]:
    """Get all relevant file paths from directory"""
    import os
    paths = []
    class_folders = sorted(os.listdir(directory))
    
    for class_name in class_folders:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                paths.append(file_path)
    
    return paths