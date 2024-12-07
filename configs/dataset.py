
import json
import os
import re
import torch
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
from utils.preprocessing import preprocess_text

class MultiModalDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_len: int,
        modality: str = 'text_image',  # Default modality
        caption_file: Optional[str] = None,
        similarity_file: Optional[str] = None,
        transform=None,
        filter_stopwords: bool = False
    ):
        self.modality = modality
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        self.filter_stopwords = filter_stopwords

        # Initialize samples
        self.samples = self._load_data()

        # Load captions and similarities if needed
        self.captions = {}
        self.similarities = {}

        if caption_file:
            if not os.path.exists(caption_file):
                raise FileNotFoundError(f"Caption file not found: {caption_file}")
            try:
                with open(caption_file, 'r') as f:
                    self.captions = json.load(f)
                print(f"Loaded captions: {len(self.captions)}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in caption file: {caption_file}")
        # After loading captions/similarities
        # if self.captions:
        #     print("\nCaption key examples:")
        #     print(list(self.captions.keys())[:5])
            
        # # Get sample ID examples
        # print("\nSample ID examples:")
        # print([self.samples[i]['id'] for i in range(5)])

        if similarity_file:
            if not os.path.exists(similarity_file):
                raise FileNotFoundError(f"Similarity file not found: {similarity_file}")
            try:
                with open(similarity_file, 'r') as f:
                    self.similarities = json.load(f)
                print(f"Loaded similarities: {len(self.similarities)}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in similarity file: {similarity_file}")
        
        # # Add data distribution logging
        # class_counts = {}
        # for sample in self.samples:
        #     class_name = sample['class_name']
        #     class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # print(f"\nDataset statistics for {data_dir}:")
        # print(f"Total samples: {len(self.samples)}")
        # print("Class distribution:")
        # for class_name, count in sorted(class_counts.items()):
        #     print(f"{class_name}: {count} samples ({count/len(self.samples)*100:.2f}%)")


    def _load_data(self) -> List[Dict]:
        """Load and organize dataset samples from image filenames"""
        samples = []
        class_folders = sorted(os.listdir(self.data_dir))
        label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}
        
        for class_name in class_folders:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            # counter = 0
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Look for images only
                    image_path = os.path.join(class_path, file_name)
                    file_name_no_ext, _ = os.path.splitext(file_name)
                    text = file_name_no_ext.replace('_', ' ')
                    text_without_digits = re.sub(r'\d+', '', text)
                        
                    if self.filter_stopwords:
                        text_without_digits = preprocess_text(text_without_digits)

                    # Use full image path as ID
                    sample = {
                        'image_path': image_path,
                        'text': text_without_digits,
                        'label': label_map[class_name],
                        'class_name': class_name,
                        'id': image_path  # Full path as ID
                    }
                    samples.append(sample)
                    # if counter < 5:
                    #     print(f"Sample {counter} debug info:")
                    #     print(f"Class: {class_name}")
                    #     print(f"Text: {sample['text']}")
                    #     print(f"Image path: {image_path}")
                    #     print(f"Label: {label_map[class_name]}")
                    #     print(f"ID: {image_path}")
                    #     counter += 1
                        
        
        if not samples:
            raise ValueError(f"No valid samples found in {self.data_dir}")
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text input"""
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

    def _process_image(self, image_path: str) -> torch.Tensor:
        """Process image input"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

    def _process_caption(self, caption: str) -> Dict[str, torch.Tensor]:
        """Process caption input"""
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'caption_input_ids': encoding['input_ids'].flatten(),
            'caption_attention_mask': encoding['attention_mask'].flatten()
        }

    def __getitem__(self, idx):
        sample = self.samples[idx]
         # Add debug logging for random samples (e.g., every 1000th sample)
         
        # if idx % 1000 == 0:
        #     print(f"\nSample {idx} debug info:")
        #     print(f"Class: {sample['class_name']}")
        #     print(f"Text: {sample['text'][:100]}...")
        #     if self.modality == 'image_only' or self.modality == 'text_image' or self.modality == 'image_garbage' or self.modality == 'text_image_garbage' or self.modality == 'full_model':
        #         print(f"Image path: {sample['image_path']}")
        #     if self.captions != {}:
        #         print(f"Has caption: {sample['id'] in self.captions}")
        #     if self.similarities != {}:
        #         print(f"Has similarity: {sample['id'] in self.similarities}")

        output = {'label': torch.tensor(sample['label'])}

        if (self.modality == "caption_only" or 
            self.modality == "text_caption_sep" or 
            self.modality == "text_caption_concat" or 
            self.modality == "text_caption_sim" or 
            self.modality == "caption_image" or 
            self.modality == "text_caption_image_sep" or 
            self.modality == "text_caption_image_concat" or 
            self.modality == "full_model") and self.captions[sample['id']] != {}:
            output_caption = self.captions[sample['id']]

        if self.filter_stopwords and self.captions != {}:
            output_caption = preprocess_text(output_caption)

        # Handle each modality explicitly
        if self.modality == 'text_only':
            output.update(self._process_text(sample['text']))
            
        elif self.modality == 'image_only':
            output['images'] = self._process_image(sample['image_path'])
            
        elif self.modality == 'caption_only':
            if sample['id'] in self.captions:
                output.update(self._process_caption(output_caption))
                
        elif self.modality == 'text_image':
            output.update(self._process_text(sample['text']))
            output['images'] = self._process_image(sample['image_path'])

        elif self.modality == 'text_caption_sep':
            output.update(self._process_text(sample['text']))
            if sample['id'] in self.captions:
                output.update(self._process_caption(output_caption))
                
        elif self.modality == 'text_caption_concat':
            if sample['id'] in self.captions:
                combined_text = f"{sample['text']}[SEP]{output_caption}"
            else:
                combined_text = sample['text']
            output.update(self._process_text(combined_text))
                
        elif self.modality == 'text_caption_sim':
            if sample['id'] in self.captions:
                combined_text = f"{sample['text']}[SEP]{output_caption}"
            else:
                combined_text = sample['text']
            output.update(self._process_text(combined_text))
            if sample['id'] in self.similarities:
                output['similarity_score'] = torch.tensor(self.similarities[sample['id']], dtype=torch.float)
                
        elif self.modality == 'caption_image':
            if sample['id'] in self.captions:
                output.update(self._process_caption(output_caption))
            output['images'] = self._process_image(sample['image_path'])

        elif self.modality == 'text_caption_image_sep':
            output.update(self._process_text(sample['text']))
            if sample['id'] in self.captions:
                output.update(self._process_caption(output_caption))
            output['images'] = self._process_image(sample['image_path'])

        elif self.modality == 'text_caption_image_concat':
            if sample['id'] in self.captions:
                combined_text = f"{sample['text']}[SEP]{output_caption}"
            else:
                combined_text = sample['text']
            output.update(self._process_text(combined_text))
            output['images'] = self._process_image(sample['image_path'])
            
        elif self.modality == 'image_garbage':
            output['images'] = self._process_image(sample['image_path'])
            
        elif self.modality == 'text_image_garbage':
            output.update(self._process_text(sample['text']))
            output['images'] = self._process_image(sample['image_path'])
            
        elif self.modality == 'full_model':
            # Add all features
            output['images'] = self._process_image(sample['image_path'])
            if sample['id'] in self.captions:
                combined_text = f"{sample['text']}[SEP]{output_caption}"
            else:
                combined_text = sample['text']
            output.update(self._process_text(combined_text))
            if sample['id'] in self.similarities:
                output['similarity_score'] = torch.tensor(self.similarities[sample['id']], dtype=torch.float)

        return output