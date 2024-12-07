# utils/metrics.py
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def calculate_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics with proper handling of zero division"""
    return {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(
            true_labels, 
            predictions, 
            average='weighted',
            zero_division=0  # Explicitly handle zero division
        ),
        'recall': recall_score(
            true_labels, 
            predictions, 
            average='weighted',
            zero_division=0
        ),
        'f1': f1_score(
            true_labels, 
            predictions, 
            average='weighted',
            zero_division=0
        )
    }

def plot_confusion_matrix_png(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    filename: str = "confusion_matrix.png"
) -> None:
    """Generate and save confusion matrix visualization"""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_class_distribution(
    labels: List[int],
    class_names: List[str],
    title: str = "Class Distribution",
    filename: str = "class_distribution.png"
) -> None:
    """Plot class distribution"""
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), class_names, rotation=45)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_classification_results(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    samples: List[Dict],
    output_path: str = 'classification_results.json'
) -> None:
    """Save classification results to JSON file"""
    results = []
    
    # Collect results for each sample
    for idx, (true, pred) in enumerate(zip(true_labels, predictions)):
        sample = samples[idx]
        result = {
            'sample_id': sample.get('id', str(idx)),
            'image_path': sample.get('image_path', ''),
            'text': sample.get('text', ''),
            'true_label': int(true),
            'predicted_label': int(pred),
            'correct': true == pred,
            'class_name': sample.get('class_name', ''),
        }
        results.append(result)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_json(output_path)
    print(f"Classification results saved to {output_path}")


def track_misclassified(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    samples: List[Dict],
    max_samples: int = 100
) -> List[Dict]:
    """Track misclassified examples"""
    misclassified = []
    for idx, (true, pred) in enumerate(zip(true_labels, predictions)):
        if true != pred and len(misclassified) < max_samples:
            misclassified.append({
                'sample_id': samples[idx].get('id', str(idx)),
                'true_label': int(true),
                'predicted_label': int(pred),
                'image_path': samples[idx].get('image_path', ''),
                'text_path': samples[idx].get('text_path', '')
            })
    return misclassified

def evaluate_classification_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str] = None
) -> Dict[str, float]:
    """Calculate and optionally print classification metrics"""
    metrics = calculate_metrics(true_labels, predictions)
    
    if class_names:
        print("\nClassification Report:")
        print(classification_report(
            true_labels,
            predictions,
            target_names=class_names,
            digits=4
        ))
    
    return metrics

def plot_confusion_matrix(test_results):
    # Get predictions and true labels from results
    y_true = test_results.get('all_labels', [])  # Match key from evaluate()
    y_pred = test_results.get('all_preds', [])   # Match key from evaluate()
    
    if not y_true or not y_pred:
        print("Warning: No labels/predictions found in results")
        return
        
    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Black', 'Blue', 'Green', 'TTR'],
                yticklabels=['Black', 'Blue', 'Green', 'TTR'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print text version
    print("\nConfusion Matrix:")
    print("-" * 50)
    print(cm)

def plot_training_history(history, save_path='training_history_good.png'):
    """Plot training history with loss and accuracy curves.
    
    Args:
        history: List of dicts containing training metrics per epoch
        save_path: Path to save the plot
    """
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_acc = [h['val_accuracy'] for h in history]
    
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Global Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc, label='Val Accuracy', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Global Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
