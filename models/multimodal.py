from enum import Enum
from typing import Optional, List
import torch
import torch.nn as nn
from configs.config import CaptionMode, ImageModelType, TextModelType
from transformers import DistilBertModel, RobertaModel
from .feature_extractors import ResNetFeatureExtractor, MobileNetFeatureExtractor, GarbageClassifierFeatureExtractor, get_feature_extractor
from .text_models import TextModel

class CaptionMode(str, Enum):
    NONE = 'none'
    SEPARATE = 'separate'
    CONCATENATE = 'concat'

class MultiModalClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        text_model_name: str = TextModelType.DISTILBERT,
        image_model_name: str = ImageModelType.RESNET18,
        use_text: bool = True,
        use_caption: bool = False,
        caption_mode: str = CaptionMode.NONE,
        use_image: bool = True,
        use_garbage_feature: bool = False,
        use_similarity: bool = False,
        fc_hidden_sizes: List[int] = [1024, 256],
        dropout_rate: float = 0.3,
        target_size_map: Optional[dict] = None
    ):
        super().__init__()
        
        # Save configuration
        self.use_text = use_text
        self.use_caption = use_caption
        self.caption_mode = caption_mode
        self.use_image = use_image
        self.use_garbage_feature = use_garbage_feature
        self.use_similarity = use_similarity
        self.target_size_map = target_size_map

        # Single text model for both text and caption
        self.text_model = TextModel(model_name=text_model_name)

        # Image processing
        if use_image:
            self.image_processor = get_feature_extractor(image_model_name)
            if use_garbage_feature:
                self.garbage_feature_extractor = GarbageClassifierFeatureExtractor()

        # Similarity projection
        if use_similarity:
            self.similarity_projection = nn.Linear(1, target_size_map['similarity'])

        # Initialize projection layers for each feature type
        if use_text:
            self.text_projection    = nn.Linear(768, target_size_map['text'])
            self.text_bn            = nn.BatchNorm1d(target_size_map['text'])
            self.text_activation    = nn.ReLU()
        if use_image:
            self.image_projection   = nn.Linear(self.image_processor.feature_dim, target_size_map['image'])
            self.image_bn           = nn.BatchNorm1d(target_size_map['image'])
            self.image_activation   = nn.ReLU()
        if use_caption:
            self.caption_projection     = nn.Linear(768, target_size_map['caption'])
            self.caption_bn             = nn.BatchNorm1d(target_size_map['caption'])
            self.caption_activation     = nn.ReLU()
        if use_garbage_feature:
            self.garbage_projection     = nn.Linear(768, target_size_map['garbage'])
            self.garbage_bn             = nn.BatchNorm1d(target_size_map['garbage'])
            self.garbage_activation     = nn.ReLU()

        # Calculate input size and build classifier
        total_input_size = self._calculate_input_size()
        classifier_layers = []
        in_features = total_input_size
        
        for hidden_size in fc_hidden_sizes:
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.PReLU())
            classifier_layers.append(nn.BatchNorm1d(hidden_size))
            classifier_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
        classifier_layers.append(nn.Linear(in_features, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)

    def _calculate_input_size(self) -> int:
        size = 0

        if self.use_text:
            size += self.target_size_map['text']  # Projected text embedding size
        
        if self.use_image:
            size += self.target_size_map['image']  # Projected image features
            if self.use_garbage_feature:
                size += self.target_size_map['garbage']  # Projected garbage classifier features
                
                
        if self.use_caption and self.caption_mode == CaptionMode.SEPARATE:
            size += self.target_size_map['caption']  # Projected caption features
            
        if self.use_similarity:
            size += self.target_size_map['similarity']  # Similarity features
            
        return size

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        caption_input_ids: Optional[torch.Tensor] = None,
        caption_attention_mask: Optional[torch.Tensor] = None,
        similarity_score: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = []

        # Caption-only case
        if not self.use_text and self.use_caption:
            if caption_input_ids is not None and caption_attention_mask is not None:
                caption_features = self.text_model(caption_input_ids, caption_attention_mask)
                caption_features = self.caption_projection(caption_features)
                caption_features = self.caption_bn(caption_features)
                caption_features = self.caption_activation(caption_features)
                features.append(caption_features)  
                del caption_features     

        # Text + Caption cases
        elif self.use_text and input_ids is not None:
            # Process text
            text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = self.text_projection(text_features)
            text_features = self.text_bn(text_features)
            text_features = self.text_activation(text_features)
            features.append(text_features)
            del text_features
          #  torch.cuda.empty_cache()
                
            if self.caption_mode == CaptionMode.SEPARATE and self.use_caption and caption_input_ids is not None:
                # Process caption with same model
                caption_features = self.text_model(input_ids=caption_input_ids, attention_mask=caption_attention_mask)
                caption_features = self.caption_projection(caption_features)
                caption_features = self.caption_bn(caption_features)
                caption_features = self.caption_activation(caption_features)
                features.append(caption_features)
                del caption_features
        #        torch.cuda.empty_cache()

        # Process images if enabled
        if self.use_image and images is not None:
            image_features = self.image_processor(images)
            image_features = self.image_projection(image_features)
            image_features = self.image_bn(image_features)
            image_features = self.image_activation(image_features)
            features.append(image_features)
            del image_features
        #    torch.cuda.empty_cache()

            if self.use_garbage_feature:
                garbage_features = self.garbage_feature_extractor(images)
                garbage_features = self.garbage_projection(garbage_features)
                garbage_features = self.garbage_bn(garbage_features)
                garbage_features = self.garbage_activation(garbage_features)
                features.append(garbage_features)
                del garbage_features
         #       torch.cuda.empty_cache()

        # Process similarity score if enabled
        if self.use_similarity and similarity_score is not None:
            similarity_features = self.similarity_projection(similarity_score.unsqueeze(-1))
            features.append(similarity_features)
            del similarity_features
        #    torch.cuda.empty_cache()


        # Combine all features
        combined_features = torch.cat(features, dim=1)
        outputs = self.classifier(combined_features)
        
        # Clean up
        del features, combined_features
      #  torch.cuda.empty_cache()
        
        return outputs