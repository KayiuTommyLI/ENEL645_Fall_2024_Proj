from typing import Optional, Dict, Any
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
from utils.logger import log_with_timestamp
from utils.metrics import evaluate_classification_metrics, plot_training_history, save_classification_results

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        save_dir: str = 'checkpoints',
        use_text: bool = True,
        use_image: bool = True,
        use_wandb: bool = False,
        gradient_clip: float = 0.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.use_text = use_text
        self.use_image = use_image
        self.use_wandb = use_wandb
        self.gradient_clip = gradient_clip
        self.global_epoch = 0
        self.training_history = []

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> float:

        if len(train_loader) == 0:
            raise ValueError("Training loader is empty")
            
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move data to device - basic inputs
            labels = batch.pop('label').to(self.device)

            # Handle optional inputs                      
            input_ids = batch['input_ids'].to(self.device) if 'input_ids' in batch else None
            attention_mask = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None
            images = batch['images'].to(self.device) if 'images' in batch else None
            caption_input_ids = batch['caption_input_ids'].to(self.device) if 'caption_input_ids' in batch else None
            caption_attention_mask = batch['caption_attention_mask'].to(self.device) if 'caption_attention_mask' in batch else None
            similarity_score = batch['similarity_score'].to(self.device) if 'similarity_score' in batch else None
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                caption_input_ids=caption_input_ids,
                caption_attention_mask=caption_attention_mask,
                similarity_score=similarity_score
            )
            
            # Calculate loss and backward pass
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping and optimization step
            if self.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            if self.use_wandb:
                wandb.log({'train_batch_loss': loss.item()})
                    
        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        print(f"Model mode: {self.model.training}")
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            # for batch in tqdm(val_loader, desc='Validating'): # display the progress bar
            for batch in val_loader:
                # print("Batch keys:", batch.keys())
                # print("Input shapes:", {k: v.shape for k,v in batch.items() if isinstance(v, torch.Tensor)})
                
                # Move data to device - basic inputs
                labels = batch['label'].to(self.device)
                
                # Handle optional inputs
                input_ids = batch['input_ids'].to(self.device) if 'input_ids' in batch else None
                attention_mask = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None
                images = batch['images'].to(self.device) if 'images' in batch else None
                caption_input_ids = batch['caption_input_ids'].to(self.device) if 'caption_input_ids' in batch else None
                caption_attention_mask = batch['caption_attention_mask'].to(self.device) if 'caption_attention_mask' in batch else None
                similarity_score = batch['similarity_score'].to(self.device) if 'similarity_score' in batch else None
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    caption_input_ids=caption_input_ids,
                    caption_attention_mask=caption_attention_mask,
                    similarity_score=similarity_score
                )
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                
                # Store predictions and labels for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = evaluate_classification_metrics(all_labels, all_preds)

        if self.use_wandb:
            wandb.log({f'val_{k}': v for k, v in metrics.items()})

        metrics.update({
            'loss': total_loss / len(val_loader),
            'all_labels': all_labels,
            'all_preds': all_preds
        })
        # # Print detailed metrics
        # print(f"\n Test Set Results:")
        # print(classification_report(all_labels, all_preds))
        # print("\n Confusion Matrix:")
        # print(confusion_matrix(all_labels, all_preds))

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        patience: int = 5,
        training_history_file_name: str = 'training_history.png',
        scheduler = None
    ) -> Dict[str, Any]:
        best_val_loss = float('inf')
        best_val_accuracy = 0
        patience_counter = 0
        training_history = []

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    current_lr = scheduler.optimizer.param_groups[0]['lr']
                    scheduler.step(val_metrics['loss'])
                    # Log if LR changed
                    new_lr = scheduler.optimizer.param_groups[0]['lr']
                    if new_lr != current_lr:
                        log_with_timestamp(f"Learning rate updated to: {new_lr}")
                else:
                    scheduler.step()

            # Log metrics
            log_with_timestamp(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Track with global epoch
            self.training_history.append({
                'epoch': self.global_epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            })
            self.global_epoch += 1

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_val_accuracy = val_metrics['accuracy']
                patience_counter = 0
                self.save_checkpoint(model=self.model)
                log_with_timestamp(f"Save best_model.pt for Loss at epochs {epoch+1} ")
            elif val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                patience_counter = 0
                self.save_checkpoint(model=self.model)
                log_with_timestamp(f"Save best_model.pt for Accuracy at epochs {epoch+1} ")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                log_with_timestamp(f"Early stopping triggered after epochs {epoch+1} ")
                break

            # Save history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
            })

        plot_training_history(history = self.training_history, save_path = training_history_file_name)
        
        return training_history


    def save_checkpoint(self, model, save_dir='checkpoints', filename='best_model.pt'):
        """Save model and optimizer state to checkpoint file"""
        if not isinstance(save_dir, (str, bytes, os.PathLike)):
            raise TypeError(f"save_dir must be a path-like object, not {type(save_dir)}")
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
                
        # Save checkpoint
        save_path = os.path.join(save_dir, filename)
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, model, save_dir='checkpoints', filename='best_model.pt'):
        """Load model and optimizer state from checkpoint file"""
        load_path = os.path.join(save_dir, filename)
        
        # Load checkpoint
        checkpoint = torch.load(load_path)
        
        # Restore states
        model.load_state_dict(checkpoint)
        print(f"Checkpoint loaded from {load_path}")
        
        return model

       
class WeightedClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Convert severity matrix to weight tensor
        severity_matrix = {
            'Black': {'Black': 0, 'Blue': 4, 'Green': 5, 'TTR': 2},  # Landfill misclassified as others (high penalty for Blue/Green)
            'Blue': {'Black': 5, 'Blue': 0, 'Green': 3, 'TTR': 1},   # Recyclable wrongly to landfill (very high), lesser for Green/TTR
            'Green': {'Black': 5, 'Blue': 3, 'Green': 0, 'TTR': 2},  # Biodegradable to landfill (very high), moderate to Blue/TTR
            'TTR': {'Black': 3, 'Blue': 2, 'Green': 2, 'TTR': 0}     # TTR misclassified as others (moderate penalty)
        }
        
        # Convert to tensor first
        class_to_idx = {'Black': 0, 'Blue': 1, 'Green': 2, 'TTR': 3}
        weights = torch.zeros(4, 4)
        for true_class, pred_dict in severity_matrix.items():
            for pred_class, weight in pred_dict.items():
                i, j = class_to_idx[true_class], class_to_idx[pred_class]
                weights[i][j] = weight
                
        # Now register as Parameter
        self.register_buffer('weights', weights)
        
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs (torch.Tensor): Model predictions [batch_size, num_classes]
            targets (torch.Tensor): Ground truth labels [batch_size]
        """
        # Apply softmax for probability distribution
        probs = F.softmax(outputs, dim=1)
        
        # Get predicted classes
        _, preds = torch.max(probs, 1)
        
        # Calculate weighted loss
        batch_size = outputs.size(0)
        device = outputs.device
        
        # Move weights to correct device
        if self.weights.device != device:
            self.weights = self.weights.to(device)
        
        # Calculate loss using severity weights
        loss = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            loss[i] = self.weights[targets[i]][preds[i]]
        
        # Add cross entropy for better gradient flow
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # Combine losses
        combined_loss = (loss + ce_loss).mean()
        
        return combined_loss

# Usage in training loop:
"""
criterion = WeightedClassificationLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""