# training/optimization.py
from typing import Dict, Type, Optional
import optuna
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import gc
from utils.logger import log_with_timestamp

class HyperparameterOptimizer:
    def __init__(
        self,
        model_class: Type,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: torch.device,
        n_trials: int = 100,
        study_name: str = "multimodal_optimization"
    ):
        self.model_class = model_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.n_trials = n_trials
        self.study_name = study_name

    def objective(self, trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'hidden_sizes': [
                trial.suggest_int(f'hidden_size_{i}', 64, 1024, log=True)
                for i in range(trial.suggest_int('n_layers', 1, 3))
            ],
            'gradient_clip': trial.suggest_float('gradient_clip', 0.1, 1.0)
        }

        # Create dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=params['batch_size']
        )

        # Initialize model and training components
        model = self.model_class(**params).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate']
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(20):  # Max epochs per trial
            # Train
            model.train()
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = criterion(outputs, batch['label'])
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    params['gradient_clip']
                )
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += criterion(outputs, batch['label']).item()

            val_loss /= len(val_loader)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            # Report to Optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Cleanup
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        return best_val_loss

    def optimize(self) -> Dict:
        """Run hyperparameter optimization"""
        study = optuna.create_study(
            direction="minimize",
            study_name=self.study_name,
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(self.objective, n_trials=self.n_trials)
        log_with_timestamp(f"Best trial: {study.best_trial.params}")
        
        return study.best_trial.params