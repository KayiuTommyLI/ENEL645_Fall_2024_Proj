import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from configs.config import CaptionMode, ImageModelType, ModelConfig, TrainingConfig, Paths, DataConfig, TextModelType
from configs.dataset import MultiModalDataset
from models.multimodal import MultiModalClassifier
from torchvision import transforms
from training.trainer import Trainer, WeightedClassificationLoss
from training.optimization import HyperparameterOptimizer
from transformers import (
    DistilBertTokenizer, 
    RobertaTokenizer, 
    XLNetTokenizer, 
    AlbertTokenizer,
    BertTokenizer
)
from torch.optim import Adam
from utils.logger import log_with_timestamp, logger
from utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    save_classification_results,
    track_misclassified,
    evaluate_classification_metrics
)
from utils.visualization import VisualizationManager
from models.feature_extractors import ImageTransforms

def get_tokenizer(model_type: str):
    """Get tokenizer based on model type string"""
    tokenizer_map = {
        'distilbert-base-uncased': DistilBertTokenizer,
        'roberta-base': RobertaTokenizer,
        'xlnet-base-cased': XLNetTokenizer,
        'albert-base-v2': AlbertTokenizer,
        'bert-base-uncased': BertTokenizer
    }
    
    if model_type not in tokenizer_map:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    tokenizer_class = tokenizer_map[model_type]
    return tokenizer_class.from_pretrained(model_type) 

def get_model_params_from_modality(modality: str) -> dict:
    """Set model parameters based on modality choice"""
    base_params = {
        'use_text': True,
        'use_image': False,
        'use_caption': False,
        'caption_mode': CaptionMode.NONE,
        'use_garbage_feature': False,
        'use_similarity': False
    }
    
    modality_configs = {
        'text_only': base_params,
        'image_only': {**base_params, 'use_text': False, 'use_image': True},
        'caption_only': {**base_params, 'use_text': False, 'use_caption': True, 'caption_mode': CaptionMode.SEPARATE},
        'text_image': {**base_params, 'use_image': True},
        'text_caption_sep': {**base_params, 'use_caption': True, 'caption_mode': CaptionMode.SEPARATE},
        'text_caption_concat': {**base_params, 'use_caption': True, 'caption_mode': CaptionMode.CONCATENATE},
        'text_caption_sim': {**base_params, 'use_caption': True, 'caption_mode': CaptionMode.CONCATENATE, 'use_similarity': True},
        'caption_image' : {**base_params, 'use_text': False, 'use_image': True, 'use_caption': True, 'caption_mode': CaptionMode.SEPARATE},
        'text_caption_image_sep': {**base_params, 'use_image': True, 'use_caption': True, 'caption_mode': CaptionMode.SEPARATE},
        'text_caption_image_concat': {**base_params, 'use_image': True, 'use_caption': True, 'caption_mode': CaptionMode.CONCATENATE},
        'image_garbage': {**base_params, 'use_text': False, 'use_image': True, 'use_garbage_feature': True},
        'text_image_garbage': {**base_params, 'use_image': True, 'use_garbage_feature': True},
        'full_model': {
            'use_text': True,
            'use_image': True,
            'use_caption': True,
            'caption_mode': CaptionMode.CONCATENATE,
            'use_garbage_feature': True,
            'use_similarity': True
        }
    }
    
    return modality_configs[modality]

def log_configuration(args, model_params):
    """Log all configuration settings"""
    log_with_timestamp("\nConfiguration:")
    log_with_timestamp(f"Mode: {args.mode}")
    log_with_timestamp(f"Modality: {args.modality}")
    log_with_timestamp(f"Criterion: {args.criterion}")
    log_with_timestamp(f"Filter stopwords: {args.filter_stopword}")
    
    log_with_timestamp(f"\nModel Settings:")
    log_with_timestamp(f"- Number of classes: {model_params.num_classes}")
    log_with_timestamp(f"- Text model: {model_params.text_model_name}")
    log_with_timestamp(f"- Image model: {model_params.image_model_name}")
    log_with_timestamp(f"- Hidden sizes: {model_params.fc_hidden_sizes}")
    log_with_timestamp(f"- Use text: {model_params.use_text}")
    log_with_timestamp(f"- Use image: {model_params.use_image}")
    log_with_timestamp(f"- Use caption: {model_params.use_caption}")
    log_with_timestamp(f"- Caption mode: {model_params.caption_mode}")
    log_with_timestamp(f"- Use garbage feature: {model_params.use_garbage_feature}")
    log_with_timestamp(f"- Use similarity: {model_params.use_similarity}")
    
    log_with_timestamp(f"\nTraining Settings:")
    log_with_timestamp(f"- Batch size: {model_params.batch_size}")
    log_with_timestamp(f"- Learning rate: {model_params.learning_rate}")
    log_with_timestamp(f"- Number of epochs: {model_params.num_epochs}")
    log_with_timestamp(f"- Max sequence length: {model_params.max_len}")
    log_with_timestamp(f"- Dropout rate: {TrainingConfig.dropout_rate}")
    log_with_timestamp(f"- Early stopping patience: {TrainingConfig.early_stopping_patience}")
    log_with_timestamp(f"- Weight decay: {TrainingConfig.weight_decay}")
    log_with_timestamp(f"- Label smoothing: {TrainingConfig.label_smoothing}")
    log_with_timestamp("")

def get_sampler(dataset):
    labels = [sample['label'] for sample in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1. / class_counts.float()
    sample_weights = weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

# In main.py
def unfreeze_feature_extractors(model, model_params):
    """Unfreeze all feature extractors"""
    # Text model
    if hasattr(model, 'text_model'):
        for param in model.text_model.parameters():
            param.requires_grad = True
            
    # Image model    
    if hasattr(model, 'image_processor'):
        for param in model.image_processor.parameters():
            param.requires_grad = True
            
    # Caption model if used
    if hasattr(model, 'caption_model'):
        for param in model.caption_model.parameters():
            param.requires_grad = True
            
    # Garbage feature extractor
    if hasattr(model, 'garbage_feature_extractor'):
        for param in model.garbage_feature_extractor.parameters():
            param.requires_grad = True
            
    # Verify unfrozen state
    print("\nFeature Extractor States:")
    if hasattr(model, 'text_model'):
        text_model = getattr(model, 'text_model', None)
        print(f"Text model exists: {text_model is not None}")
        if text_model:
            print(f"Text model trainable: {any(p.requires_grad for p in text_model.parameters())}")

    if hasattr(model, 'image_processor'):
        image_model = getattr(model, 'image_processor', None)
        print(f"Image model exists: {image_model is not None}")
        if image_model:
            print(f"Image model trainable: {any(p.requires_grad for p in image_model.parameters())}")

    if hasattr(model, 'caption_model'):
        caption_model = getattr(model, 'caption_model', None)
        print(f"Caption model exists: {caption_model is not None}")
        if caption_model:
            print(f"Caption model trainable: {any(p.requires_grad for p in caption_model.parameters())}")

def inspect_dataloader(dataloader, num_samples=2):
    """Inspect contents of dataloader"""
    print("\nDataLoader Inspection:")
    print("-" * 50)
    
    # Get iterator
    iterator = iter(dataloader)
    
    for i in range(num_samples):
        try:
            batch = next(iterator)
            print(f"\nBatch {i+1}:")
            
            # Text inputs
            print("\nText Inputs:")
            print(f"input_ids shape: {batch['input_ids'].shape}")
            print(f"attention_mask shape: {batch['attention_mask'].shape}")
            print(f"input_ids range: ({batch['input_ids'].min()}, {batch['input_ids'].max()})")
            
            # # Image inputs
            # print("\nImage Inputs:")
            # print(f"images shape: {batch['images'].shape}")
            # print(f"image range: ({batch['images'].min():.3f}, {batch['images'].max():.3f})")
            
            # Labels
            print("\nLabels:")
            print(f"labels shape: {batch['label'].shape}")
            print(f"unique labels: {torch.unique(batch['label']).tolist()}")
            
            print("=" * 50)
            
        except StopIteration:
            print("No more batches")
            break

def main():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--mode', choices=['train', 'test', 'optimize'], required=True)    
    parser.add_argument('--modality', type=str, required=True, 
                       choices=[
                           'text_only',              # Text only
                           'image_only',             # Image only
                           'caption_only',           # Caption only
                           'text_image',             # Text + Image
                           'text_caption_sep',       # Text + caption (separate models)
                           'text_caption_concat',    # Text + caption (concatenated)
                           'text_caption_sim',       # Text + caption + similarity
                           'caption_image',          # Caption + Image
                           'text_caption_image_sep',    # Text + Image + caption (separate models)
                           'text_caption_image_concat', # Text + Image + caption (concatenated)
                           'image_garbage',          # Image + garbage feature
                           'text_image_garbage',     # Text + Image + garbage
                           'full_model'              # Text + Image + caption + similarity + garbage
                       ])
    # parser.add_argument('--tokenizer', type=str, required=True)
    
    # Model configuration arguments
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--text_model_name', type=str, default=TextModelType.DISTILBERT)
    parser.add_argument('--criterion', type=str, default="CrossEntropyLoss")
    parser.add_argument('--image_model_name', type=str, default=ImageModelType.RESNET18)
    # parser.add_argument('--fc_hidden_sizes', nargs='+', type=int, default=[1024, 256])
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--learning_rate', type=float, default=2e-5)
    # parser.add_argument('--num_epochs', type=int, default=10)
    # parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--filter_stopword', type=bool)
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for testing')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Define transform 
    transform = ImageTransforms.get_transforms(args.image_model_name)

    # Use ModelConfig.from_args to handle all model parameters
    model_params = ModelConfig.from_args(args)
    modality_params = get_model_params_from_modality(args.modality)
    for key, value in modality_params.items():
        setattr(model_params, key, value)

    # Log configuration
    log_configuration(args, model_params)

    # Define tokenizer
    tokenizer = get_tokenizer(model_params.text_model_name)
    logger.info(f"Initialized {model_params.text_model_name} tokenizer")

    # Load datasets
    # Initialize datasets with all required arguments
    log_with_timestamp(f"Loading Train data from: {Paths.TRAIN_PATH}")
    log_with_timestamp(f"Loading Val data from: {Paths.VAL_PATH}")
    log_with_timestamp(f"Loading Test data from: {Paths.TEST_PATH}")
    
    train_dataset = MultiModalDataset(
        data_dir=Paths.TRAIN_PATH,
        tokenizer=tokenizer,
        max_len=model_params.max_len,
        modality=args.modality,
        transform=transform,
        caption_file=Paths.SAVE_CAPTIONS_PATH if 'caption' in args.modality or args.modality == 'full_model' else None,
        similarity_file=Paths.SAVE_SIMILARITY_PATH if 'sim' in args.modality or args.modality == 'full_model' else None,
        filter_stopwords=args.filter_stopword
    )
    # log_with_timestamp(f"Train dataset size: {len(train_dataset)}")
    # log_with_timestamp(f"Found class folders: {os.listdir(Paths.TRAIN_PATH)}")

    val_dataset = MultiModalDataset(
        data_dir=Paths.VAL_PATH,
        tokenizer=tokenizer, 
        max_len=model_params.max_len,
        modality=args.modality,
        transform=transform,
        caption_file=Paths.SAVE_CAPTIONS_PATH if 'caption' in args.modality or args.modality == 'full_model' else None,
        similarity_file=Paths.SAVE_SIMILARITY_PATH if 'sim' in args.modality or args.modality == 'full_model' else None,
        filter_stopwords=args.filter_stopword
    )

    test_dataset = MultiModalDataset(
        data_dir=Paths.TEST_PATH,
        tokenizer=tokenizer,
        max_len=model_params.max_len,
        modality=args.modality,
        transform=transform,
        caption_file=Paths.SAVE_CAPTIONS_PATH if 'caption' in args.modality or args.modality == 'full_model' else None,
        similarity_file=Paths.SAVE_SIMILARITY_PATH if 'sim' in args.modality or args.modality == 'full_model' else None,
        filter_stopwords=args.filter_stopword
    )

    for dataset, name in [(train_dataset, 'Train'), 
                            (val_dataset, 'Val'), 
                            (test_dataset, 'Test')]:
            print(f"\n{name} Dataset Statistics:")
            print(f"Size: {len(dataset)}")
            labels = [sample['label'] for sample in dataset.samples]
            unique, counts = np.unique(labels, return_counts=True)
            print("Class distribution:", dict(zip(unique, counts)))

    if args.criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss(
        #    label_smoothing=TrainingConfig.label_smoothing
        )
    else:
        criterion = WeightedClassificationLoss()
    
    if args.mode == 'train':
        train_sampler = get_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=model_params.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=model_params.batch_size)

        model = MultiModalClassifier(
            num_classes=model_params.num_classes,
            text_model_name=model_params.text_model_name,
            image_model_name=model_params.image_model_name,
            fc_hidden_sizes=model_params.fc_hidden_sizes,
            dropout_rate=TrainingConfig.dropout_rate,  # Use from TrainingConfig
            use_text=model_params.use_text,
            use_image=model_params.use_image,
            use_caption=model_params.use_caption,
            caption_mode=model_params.caption_mode,
            use_garbage_feature=model_params.use_garbage_feature,
            use_similarity=model_params.use_similarity,
            target_size_map=model_params.target_size_map
        ).to(device)

        # Phase 1: Train with frozen feature extractors
        log_with_timestamp("Phase 1: Training classifier with frozen feature extractors")
    
        optimizer = Adam(
            model.parameters(), 
            lr=model_params.learning_rate
    #        weight_decay=TrainingConfig.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )


        # Setup trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=model_params.num_epochs,
            patience = TrainingConfig.early_stopping_patience,
            training_history_file_name = 'training_history_phase1.png',
            scheduler=scheduler
        )

        # Load best model from Phase 1 before fine-tuning
        log_with_timestamp("Loading best model from Phase 1")
        checkpoint = torch.load('checkpoints/best_model.pt')
        model.load_state_dict(checkpoint)

        # Phase 2: Fine-tune with unfrozen feature extractors
        log_with_timestamp("Phase 2: Fine-tuning feature extractors")
        
        # Unfreeze feature extractors
        unfreeze_feature_extractors(model, model_params)

        # Phase 2 optimizer with lower learning rate
        optimizer = Adam(
            model.parameters(),
            lr=model_params.learning_rate * 0.1  # Lower learning rate for fine-tuning
  #          weight_decay=TrainingConfig.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )

        # Phase 2 training
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        fine_tune_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=model_params.num_epochs,
            patience = TrainingConfig.early_stopping_patience,
            training_history_file_name = 'training_history_phase2.png',
            scheduler=scheduler
        )

        # Combine histories
        history.extend(fine_tune_history)

        # Plot training history
        VisualizationManager.plot_training_history(
            history,
            save_path='training_history.png'
        )

    elif args.mode == 'optimize':
        optimizer = HyperparameterOptimizer(
            model_class=MultiModalClassifier,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device
        )
        best_params = optimizer.optimize()
        logger.info(f"Best parameters: {best_params}")

    elif args.mode == 'test':
        # Load model
        
        model = MultiModalClassifier(
            num_classes=model_params.num_classes,
            text_model_name=model_params.text_model_name,
            image_model_name=model_params.image_model_name,
            fc_hidden_sizes=model_params.fc_hidden_sizes,
            dropout_rate=TrainingConfig.dropout_rate,  # Use from TrainingConfig
            use_text=model_params.use_text,
            use_image=model_params.use_image,
            use_caption=model_params.use_caption,
            caption_mode=model_params.caption_mode,
            use_garbage_feature=model_params.use_garbage_feature,
            use_similarity=model_params.use_similarity,
            target_size_map=model_params.target_size_map
        ).to(device)

    #    print(model)

        # model = load_model_for_testing(model, args.checkpoint, device)                    
        checkpoint = torch.load('checkpoints/best_model.pt')
        model.load_state_dict(checkpoint)
        # Test model
        test_loader = DataLoader(test_dataset, batch_size=model_params.batch_size, shuffle=False)
     #   inspect_dataloader(test_loader)

        optimizer = Adam(
            model.parameters(),
            lr=model_params.learning_rate
  #          weight_decay=TrainingConfig.weight_decay
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        test_results = trainer.evaluate(test_loader)
        print(f"test_results['accuracy'] {test_results['accuracy']}")
        print(f"test_results['precision'] {test_results['precision']}")
        print(f"test_results['recall'] {test_results['recall']}")   
        print(f"test_results['f1'] {test_results['f1']}")
        print(f"test_results['loss'] {test_results['loss']}")
        
        plot_confusion_matrix(test_results)
        
        save_classification_results(
            true_labels=test_results['all_labels'],
            predictions=test_results['all_preds'],
            samples=test_loader.dataset.samples,
            output_path='test_classification_results.json'
        )

if __name__ == "__main__":
    main()