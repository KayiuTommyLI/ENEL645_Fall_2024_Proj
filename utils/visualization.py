import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid

class VisualizationManager:
    @staticmethod
    def plot_training_history(
        history: List[Dict],
        metrics: List[str] = ['loss', 'accuracy'],
        save_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            try:
                # Try to get training metric
                train_metric = [h.get(f'train_{metric}', None) for h in history]
                val_metric = [h.get(f'val_{metric}', None) for h in history]
                epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
                
                # Filter out None values
                valid_points = [(e, t, v) for e, t, v in zip(epochs, train_metric, val_metric) 
                            if t is not None or v is not None]
                
                if valid_points:
                    epochs, train_metric, val_metric = zip(*valid_points)
                    
                    if any(t is not None for t in train_metric):
                        ax.plot(epochs, train_metric, 'b-', label=f'Training {metric}')
                    if any(v is not None for v in val_metric):
                        ax.plot(epochs, val_metric, 'r-', label=f'Validation {metric}')
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric.capitalize())
                    ax.set_title(f'{metric.capitalize()} vs. Epoch')
                    ax.legend()
                    ax.grid(True)
                else:
                    print(f"Warning: No valid data points for metric '{metric}'")
                    
            except Exception as e:
                print(f"Error plotting metric '{metric}': {str(e)}")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_image_grid(
        images: List[torch.Tensor],
        labels: List[str],
        predictions: Optional[List[str]] = None,
        nrow: int = 5,
        save_path: Optional[str] = None
    ):
        # Create image grid
        grid = make_grid(images, nrow=nrow, normalize=True, padding=2)
        
        # Plot
        plt.figure(figsize=(15, 15))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis('off')
        
        # Add labels
        for idx, label in enumerate(labels):
            row = idx // nrow
            col = idx % nrow
            text = f"True: {label}"
            if predictions:
                text += f"\nPred: {predictions[idx]}"
            plt.text(col * (grid.shape[2]//nrow), row * (grid.shape[1]//len(images)*nrow), 
                    text, color='white', backgroundcolor='black')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_distribution(
        values: List[float],
        title: str = "Distribution",
        xlabel: str = "Value",
        ylabel: str = "Count",
        save_path: Optional[str] = None
    ):
        plt.figure(figsize=(10, 6))
        sns.histplot(values, kde=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save_path:
            plt.savefig(save_path)
        plt.close()