from typing import Optional, Tuple
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet50, resnet101, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights, inception_v3, Inception_V3_Weights, resnext50_32x4d, densenet121, vit_b_16,
    vgg16, VGG16_Weights, regnet_y_16gf, RegNet_Y_16GF_Weights, regnet_y_32gf, RegNet_Y_32GF_Weights, vit_h_14, ViT_H_14_Weights, regnet_y_128gf, RegNet_Y_128GF_Weights,
    vit_l_16, ViT_L_16_Weights
)
from torchvision.models.efficientnet import efficientnet_v2_l, EfficientNet_V2_L_Weights
from transformers import AutoImageProcessor, AutoModelForImageClassification
from utils.preprocessing import denormalize_image
from configs.config import ImageModelType

class ImageTransforms:
    # Constants
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Model specific image sizes
    MODEL_SIZES = {
        ImageModelType.VIT_L_16: (512, 512),
        ImageModelType.VIT_H_14: (518, 518),
        ImageModelType.INCEPTION: (299, 299),
        'default': (224, 224)
    }
    
    @classmethod
    def get_transforms(cls, model_name: str) -> transforms.Compose:
        """Get image transforms based on model type"""
        img_size = cls.MODEL_SIZES.get(model_name, cls.MODEL_SIZES['default'])
        
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cls.NORMALIZE_MEAN,
                std=cls.NORMALIZE_STD
            )
        ])
    
class BaseFeatureExtractor(nn.Module):
    """Base class for all feature extractors"""
    def __init__(self):
        super().__init__()
        self.feature_dim = 512  # Default dimension

    def forward(self, x) -> torch.Tensor:
        if self.feature_extractor is None:
            raise NotImplementedError("Feature extractor not initialized")
        features = self.feature_extractor(x)
        return features.squeeze(-1).squeeze(-1)

class ResNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images).squeeze(-1).squeeze(-1)
        return features

class ResNet50FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 2048

class ResNet101FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = resnet101(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 2048

class InceptionFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        # Initialize with proper weights
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        # Remove final classifier
        model.fc = nn.Identity()
        self.feature_extractor = model
        self.feature_dim = 2048
        
        # Inception v3 requires 299x299 input
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, x):
        # Ensure proper input size
        x = self.transform(x)
        # Handle auxiliary outputs
        if self.training:
            features, _ = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)
        return features.squeeze(-1).squeeze(-1)

class ResNeXTFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = resnext50_32x4d(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 2048

class DenseNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = densenet121(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 50176

    def forward(self, x):
        features = self.feature_extractor(x)
        # Ensure 2D output: [batch_size, features]
        return features.view(features.size(0), -1)
    
class ViTFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = vit_b_16(pretrained=True)
        self.feature_extractor = model
        self.feature_dim = 1000

class VGG16FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
        self.feature_extractor = nn.Sequential(*list(model.features))
        self.feature_dim = 25088

    def forward(self, x):
        features = self.feature_extractor(x)
        # Ensure 2D output: [batch_size, features]
        return features.view(features.size(0), -1)

class ViTH14FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.feature_extractor = model
        self.feature_dim = 1280

class RegNetY128FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = regnet_y_128gf(weights=RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 7392

class ViTL16FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.feature_extractor = model
        self.feature_dim = 1024

class RegNetY32FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = regnet_y_32gf(weights=RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 3712

# class RegNetY16FeatureExtractor(BaseFeatureExtractor):
#     def __init__(self):
#         super().__init__()
#         model = regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1)
#         self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
#         self.feature_dim = 3024

class EfficientNetV2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        # Remove classifier
        model.classifier = nn.Identity()
        self.feature_extractor = model
        self.feature_dim = 1280  # EfficientNetV2-L feature dimension
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return features.squeeze(-1).squeeze(-1)
    
class RegNetY16GFFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        # Load with SWAG E2E weights
        weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
        model = regnet_y_16gf(weights=weights)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 3024  # RegNet Y-16GF feature dimension

    def forward(self, x):
        features = self.feature_extractor(x)
        return features.squeeze(-1).squeeze(-1)

class RegNetY32GFLinearFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        # Load with SWAG Linear weights
        weights = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
        model = regnet_y_32gf(weights=weights)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_dim = 3712  # RegNet Y-32GF feature dimension

    def forward(self, x):
        features = self.feature_extractor(x)
        return features.squeeze(-1).squeeze(-1)
    
class MobileNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(mobilenet.features))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1280

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        features = self.global_avg_pool(features)
        return features.view(features.size(0), -1)

class GarbageClassifierFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("yangy50/garbage-classification")
        self.model = AutoModelForImageClassification.from_pretrained("yangy50/garbage-classification")
        
    def forward(self, images: torch.Tensor, train_feature_extractor: bool = False) -> torch.Tensor:
        device = images.device
        self.model = self.model.to(device)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        denormalized_images = torch.stack([
            denormalize_image(image, mean, std) 
            for image in images
        ]).to(device)
        
        # Process images
        with torch.no_grad():
            processed_images = self.processor(
                denormalized_images.cpu().numpy(),  # Convert to numpy for processor
                return_tensors="pt",
                do_rescale=False
            )["pixel_values"].to(device)
        
        # Get features
        with torch.set_grad_enabled(train_feature_extractor):
            features = self.model(processed_images, output_hidden_states=True).hidden_states[-1]
            return features.mean(dim=1)
        

def get_feature_extractor(image_model_name):
    """Factory function for feature extractors"""
    extractors = {
        ImageModelType.RESNET18 : ResNetFeatureExtractor,
        ImageModelType.RESNET50 : ResNet50FeatureExtractor,
        ImageModelType.RESNET101: ResNet101FeatureExtractor,
        ImageModelType.MOBILENET: MobileNetFeatureExtractor,
        ImageModelType.INCEPTION: InceptionFeatureExtractor,
        ImageModelType.RESNEXT: ResNeXTFeatureExtractor,
        ImageModelType.DENSENET: DenseNetFeatureExtractor,
        ImageModelType.VIT: ViTFeatureExtractor,
        ImageModelType.VGG16: VGG16FeatureExtractor,
        ImageModelType.VIT_H_14: ViTH14FeatureExtractor,
        ImageModelType.REGNET_128: RegNetY128FeatureExtractor,
        ImageModelType.VIT_L_16: ViTL16FeatureExtractor,
        ImageModelType.REGNET_32: RegNetY32FeatureExtractor,
        ImageModelType.EFFICIENTNET: EfficientNetV2FeatureExtractor,
        ImageModelType.REGNET_16: RegNetY16GFFeatureExtractor,
        ImageModelType.REGNET_32_LINEAR: RegNetY32GFLinearFeatureExtractor
    }
    
    if image_model_name not in extractors:
        raise ValueError(f"Unsupported image model: {image_model_name}")
        
    return extractors[image_model_name]()