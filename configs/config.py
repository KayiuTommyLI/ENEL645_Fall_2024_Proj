from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum

class TextModelType:
    DISTILBERT = 'distilbert-base-uncased'
    ROBERTA = 'roberta-base'
    XLNET = 'xlnet-base-cased'
    ALBERT = 'albert-base-v2'
    BERT = 'bert-base-uncased'

class ImageModelType:
    RESNET18 = 'resnet18'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    MOBILENET = 'mobilenet_v2'
    INCEPTION = 'inception_v3'
    RESNEXT = 'resnext50_32x4d'
    DENSENET = 'densenet121'
    VIT = 'vit-base-patch16-224'
    VGG16 = 'vgg16'
    VIT_H_14 = 'vit_h_14'           # ViT-H/14 got too many parameter and cannot run on my machine
    REGNET_128 = 'regnet_y_128gf'   # REGNET_128 got too many parameter and cannot run on my machine
    VIT_L_16 = 'vit_l_16'           # VIT_L_16 got too many parameter and cannot run on my machine
    REGNET_32 = 'regnet_y_32gf'
    EFFICIENTNET = 'efficientnet_v2_l'      # 3hr training time for 1600 images
    REGNET_16 = 'regnet_y_16gf'
    REGNET_32_LINEAR = 'regnet_y_32_linear'
    NONE = 'none'

class CaptionMode(str, Enum):
    NONE = 'none'
    SEPARATE = 'separate'     # For separate text models
    CONCATENATE = 'concat'    # For concatenated text + caption

@dataclass
class ModelConfig:
    # Default values
    num_classes: int = 4
    text_model_name: str = TextModelType.DISTILBERT
    image_model_name: str = ImageModelType.RESNET18
    fc_hidden_sizes: List[int] = field(default_factory=lambda: [256])
    batch_size: int = 24
    learning_rate: float = 1e-5
    num_epochs: int = 50
    max_len: int = 300
    use_text: bool = True
    use_image: bool = True
    use_caption: bool = False
    caption_mode: str = 'none'
    use_garbage_feature: bool = False
    use_similarity: bool = False
    tokenizer_type: str = 'distilbert'
    modality: str = 'text_caption_concat'
    
    target_size_map: Dict[str, int] = field(default_factory=lambda: {
        'text': 512,
      #  'text': 682,
      #  'text': 342,
        'caption': 512,
        'image': 512,
      #  'image': 342,
      #  'image': 682,
        'garbage': 512,
        'similarity': 512
    })

    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        config = cls()
        # Update config with any matching args
        for key, value in vars(args).items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        return config

@dataclass
class TrainingConfig:
    early_stopping_patience: int = 5
    dropout_rate: float = 0.3
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    label_smoothing: float = 0.0

@dataclass
class Paths:
    TRAIN_PATH: str = "/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train"
    VAL_PATH: str = "/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val"
    TEST_PATH: str = "/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test"
    
    SAVE_CAPTIONS_PATH: str = "data/captions/captions_all_new.json"
    SAVE_SIMILARITY_PATH: str = "data/captions/caption_similarity_all.json"

    # TRAIN_PATH: str = "/home/tommy.li1/Git/ENGL645_Fall_2024_Proj/Assignment2/Data/CVPR_2024_dataset_Train"
    # VAL_PATH: str = "/home/tommy.li1/Git/ENGL645_Fall_2024_Proj/Assignment2/Data/CVPR_2024_dataset_Val"
    # TEST_PATH: str = "/home/tommy.li1/Git/ENGL645_Fall_2024_Proj/Assignment2/Data/CVPR_2024_dataset_Test"

    # SAVE_CAPTIONS_PATH: str = "data/captions/captions_400_new.json"
    # SAVE_SIMILARITY_PATH: str = "data/captions/caption_similarity_400.json"

    MISCLASSIFIED_SAMPLES_PATH: str = "misclassified_samples.json"
    MISCLASSIFIED_PLOTS_PATH: str = "misclassified_examples.png"

@dataclass
class DataConfig:
    tokenizer_type: TextModelType = TextModelType.DISTILBERT
    image_size: int = 224
    max_text_length: int = 512
    max_caption_length: int = 256