"""
Training Pipeline for IPL Score Prediction
==========================================
This module provides a complete training pipeline including:
- Data loading and preparation
- Model training with cross-validation
- Hyperparameter tuning
- Model evaluation
- Training visualization

Author: IPL Score Prediction Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import json
import os
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import argparse
import warnings

warnings.filterwarnings('ignore')

# Import local modules
from model_architectures import ModelFactory, get_callbacks, DNNModel, LSTMModel, TransformerModel
from feature_engineering import FeatureEngineer, prepare_model_data


class TrainingConfig:
    """
    Configuration class for training parameters.
    """
    
    def __init__(self,
                 model_type: str = 'dnn',
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 validation_split: float = 0.1,
                 early_stopping_patience: int = 15,
                 reduce_lr_patience: int = 5,
                 min_lr: float = 1e-7,
                 l1_reg: float = 0.0001,
                 l2_reg: float = 0.001,
                 dropout_rate: float = 0.3,
                 random_state: int = 42):
        """
        Initialize training configuration.
        
        Args:
            model_type: Type of model ('dnn', 'lstm', 'transformer', etc.)
            epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            validation_split: Fraction for validation
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for reducing learning rate
            min_lr: Minimum learning rate
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            dropout_rate: Dropout rate
            random_state: Random seed
        """
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.min_lr = min_lr
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.__dict__
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class DataPreparer:
    """
    Prepare data for model training.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize data preparer.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else None
        self.feature_columns = None
        self.target_column = 'final_score'
        
    def load_feature_data(self, data_path: str = "data/features/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load feature-engineered data.
        
        Args:
            data_path: Path to features directory
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = pd.read_csv(os.path.join(data_path, "train_features.csv"))
        val_df = pd.read_csv(os.path.join(data_path, "val_features.csv"))
        test_df = pd.read_csv(os.path.join(data_path, "test_features.csv"))
        
        # Load feature columns
        with open(os.path.join(data_path, "feature_columns.json"), 'r') as f:
            feature_info = json.load(f)
            self.feature_columns = feature_info['all_features']
        
        print(f"✅ Loaded data - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        return train_df, val_df, test_df
    
    def prepare_data(self, 
                     train_df: pd.DataFrame,
                     val_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     fit_scaler: bool = True) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            fit_scaler: Whether to fit scaler on training data
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Get available features
        available_features = [col for col in self.feature_columns if col in train_df.columns]
        
        print(f"📊 Using {len(available_features)} features")
        
        # Extract features and targets
        X_train = train_df[available_features].values
        y_train = train_df[self.target_column].values
        
        X_val = val_df[available_features].values
        y_val = val_df[self.target_column].values
        
        X_test = test_df[available_features].values
        y_test = test_df[self.target_column].values
        
        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        y_train = np.nan_to_num(y_train, nan=150.0)
        y_val = np.nan_to_num(y_val, nan=150.0)
        y_test = np.nan_to_num(y_test, nan=150.0)
        
        # Scale features
        if fit_scaler:
            X_train = self.scaler.fit_transform(X_train)
        else:
            X_train = self.scaler.transform(X_train)
        
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"   X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def prepare_sequence_data(self,
                              X: np.ndarray,
                              sequence_length: int = 1) -> np.ndarray:
        """
        Reshape data for sequence models (LSTM, GRU, Transformer).
        
        Args:
            X: Feature array
            sequence_length: Length of sequences
            
        Returns:
            Reshaped array
        """
        if sequence_length == 1:
            # Simple reshape: (samples, features) -> (samples, 1, features)
            return X.reshape((X.shape[0], 1, X.shape[1]))
        else:
            # Create actual sequences (for over-by-over data)
            # This would require grouping by match_id
            return X.reshape((X.shape[0], sequence_length, X.shape[1] // sequence_length))
    
    def save_scaler(self, path: str = "models/encoders/scaler.joblib"):
        """Save the fitted scaler."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"✅ Saved scaler to {path}")
    
    def load_scaler(self, path: str = "models/encoders/scaler.joblib"):
        """Load a fitted scaler."""
        self.scaler = joblib.load(path)
        print(f"✅ Loaded scaler from {path}")


class ModelTrainer:
    """
    Train and evaluate models.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.history = None
        self.training_metrics = {}
        
    def build_model(self, input_dim: int, sequence_length: int = 1) -> None:
        """
        Build the model.
        
        Args:
            input_dim: Number of input features
            sequence_length: Sequence length for RNN models
        """
        model_type = self.config.model_type.lower()
        
        if model_type == 'dnn':
            dnn = DNNModel(
                input_dim=input_dim,
                hidden_layers=[256, 128, 64, 32],
                dropout_rate=self.config.dropout_rate,
                l1_reg=self.config.l1_reg,
                l2_reg=self.config.l2_reg
            )
            self.model = dnn.build()
            dnn.compile(learning_rate=self.config.learning_rate)
            
        elif model_type == 'lstm':
            lstm = LSTMModel(
                input_dim=input_dim,
                sequence_length=sequence_length,
                lstm_units=[128, 64],
                dense_units=[64, 32],
                dropout_rate=self.config.dropout_rate,
                use_attention=True
            )
            self.model = lstm.build()
            lstm.compile(learning_rate=self.config.learning_rate)
            
        elif model_type == 'transformer':
            transformer = TransformerModel(
                input_dim=input_dim,
                sequence_length=sequence_length,
                embed_dim=64,
                num_heads=4,
                ff_dim=128,
                num_transformer_blocks=2,
                dropout_rate=self.config.dropout_rate
            )
            self.model = transformer.build()
            transformer.compile(learning_rate=self.config.learning_rate)
            
        else:
            # Use sklearn model
            self.model = ModelFactory.create_model(model_type)
        
        print(f"✅ Built {model_type.upper()} model")
        
    def train_deep_learning(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray) -> keras.callbacks.History:
        """
        Train a deep learning model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training history
        """
        print(f"\n🚀 Training {self.config.model_type.upper()} model...")
        print("=" * 50)
        
        # Get callbacks
        callbacks = get_callbacks(
            model_name=self.config.model_type,
            checkpoint_dir='models/checkpoints/',
            log_dir='models/logs/',
            patience=self.config.early_stopping_patience
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def train_sklearn(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray) -> None:
        """
        Train a sklearn model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        print(f"\n🚀 Training {self.config.model_type.upper()} model...")
        print("=" * 50)
        
        self.model.fit(X_train, y_train)
        
        print("✅ Training complete!")
    
    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        print("\n📊 Evaluating model...")
        
        # Get predictions
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X_test)
            if isinstance(y_pred, np.ndarray) and len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        self.training_metrics = metrics
        
        print(f"\n📈 Test Results:")
        print(f"   MAE:  {mae:.2f} runs")
        print(f"   RMSE: {rmse:.2f} runs")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return metrics
    
    def cross_validate(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       n_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation (for sklearn models).
        
        Args:
            X: Features
            y: Targets
            n_folds: Number of folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        if self.config.model_type.lower() in ['dnn', 'lstm', 'transformer']:
            print("⚠️ Cross-validation for deep learning models is computationally expensive.")
            print("   Using holdout validation instead.")
            return {}
        
        print(f"\n🔄 Performing {n_folds}-fold cross-validation...")
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_state)
        
        # Cross-validate with different metrics
        mae_scores = cross_val_score(self.model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(self.model, X, y, cv=kfold, scoring='r2')
        
        cv_results = {
            'mae_scores': (-mae_scores).tolist(),
            'r2_scores': r2_scores.tolist(),
            'mean_mae': float(-mae_scores.mean()),
            'std_mae': float(mae_scores.std()),
            'mean_r2': float(r2_scores.mean()),
            'std_r2': float(r2_scores.std())
        }
        
        print(f"   MAE: {cv_results['mean_mae']:.2f} (+/- {cv_results['std_mae']:.2f})")
        print(f"   R²:  {cv_results['mean_r2']:.4f} (+/- {cv_results['std_r2']:.4f})")
        
        return cv_results
    
    def save_model(self, path: str = "models/saved_models/") -> str:
        """
        Save the trained model.
        
        Args:
            path: Directory to save model
            
        Returns:
            Path to saved model
        """
        os.makedirs(path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.config.model_type}_{timestamp}"
        
        if self.config.model_type.lower() in ['dnn', 'lstm', 'transformer', 'gru']:
            # Save Keras model
            model_path = os.path.join(path, f"{model_name}.keras")
            self.model.save(model_path)
        else:
            # Save sklearn model
            model_path = os.path.join(path, f"{model_name}.joblib")
            joblib.dump(self.model, model_path)
        
        # Save config
        config_path = os.path.join(path, f"{model_name}_config.json")
        self.config.save(config_path)
        
        # Save metrics
        metrics_path = os.path.join(path, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        print(f"✅ Saved model to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> None:
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
        """
        if model_path.endswith('.keras') or model_path.endswith('.h5'):
            self.model = keras.models.load_model(model_path)
        else:
            self.model = joblib.load(model_path)
        
        print(f"✅ Loaded model from {model_path}")


class TrainingVisualizer:
    """
    Visualize training results.
    """
    
    @staticmethod
    def plot_training_history(history: keras.callbacks.History,
                              save_path: str = "models/figures/training_history.png"):
        """
        Plot training history (loss and metrics).
        
        Args:
            history: Keras training history
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        if 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
            axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('MAE (runs)', fontsize=12)
            axes[1].set_title('Training and Validation MAE', fontsize=14)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved training history plot to {save_path}")
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         save_path: str = "models/figures/predictions.png"):
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Score', fontsize=12)
        axes[0].set_ylabel('Predicted Score', fontsize=12)
        axes[0].set_title('Actual vs Predicted Scores', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Prediction Error (runs)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Distribution of Prediction Errors', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved predictions plot to {save_path}")
    
    @staticmethod
    def plot_model_comparison(results: Dict[str, Dict],
                              save_path: str = "models/figures/model_comparison.png"):
        """
        Plot comparison of different models.
        
        Args:
            results: Dictionary of model results
            save_path: Path to save figure
        """
        models = list(results.keys())
        mae_values = [results[m].get('MAE', 0) for m in models]
        rmse_values = [results[m].get('RMSE', 0) for m in models]
        r2_values = [results[m].get('R2', 0) for m in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MAE comparison
        bars1 = axes[0].bar(models, mae_values, color='steelblue', edgecolor='black')
        axes[0].set_ylabel('MAE (runs)', fontsize=12)
        axes[0].set_title('Mean Absolute Error', fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars1, mae_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # RMSE comparison
        bars2 = axes[1].bar(models, rmse_values, color='coral', edgecolor='black')
        axes[1].set_ylabel('RMSE (runs)', fontsize=12)
        axes[1].set_title('Root Mean Squared Error', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars2, rmse_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # R² comparison
        bars3 = axes[2].bar(models, r2_values, color='forestgreen', edgecolor='black')
        axes[2].set_ylabel('R² Score', fontsize=12)
        axes[2].set_title('R² Score (higher is better)', fontsize=14)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim(0, 1)
        for bar, val in zip(bars3, r2_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved model comparison plot to {save_path}")


def run_training_pipeline(model_type: str = 'dnn',
                          data_path: str = 'data/',
                          epochs: int = 100,
                          batch_size: int = 32,
                          learning_rate: float = 0.001) -> Dict:
    """
    Run the complete training pipeline.
    
    Args:
        model_type: Type of model to train
        data_path: Path to data directory
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Dictionary with training results
    """
    print("=" * 70)
    print("🏏 IPL Score Prediction - Training Pipeline")
    print("=" * 70)
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 1. Initialize configuration
    config = TrainingConfig(
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    print(f"\n📋 Configuration:")
    print(f"   Model: {config.model_type.upper()}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    
    # 2. Load and prepare data
    print("\n" + "=" * 70)
    print("📂 Loading and Preparing Data")
    print("=" * 70)
    
    # Check if feature data exists
    features_path = os.path.join(data_path, "features")
    
    if not os.path.exists(features_path):
        print("⚠️ Feature data not found. Running preprocessing and feature engineering...")
        
        # Run preprocessing
        from data_preprocessing import preprocess_ipl_data
        preprocess_ipl_data(data_path=data_path)
        
        # Run feature engineering
        from feature_engineering import engineer_features
        engineer_features(data_path=os.path.join(data_path, "processed/"))
    
    # Load data
    preparer = DataPreparer()
    train_df, val_df, test_df = preparer.load_feature_data(features_path)
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = preparer.prepare_data(
        train_df, val_df, test_df
    )
    
    # Save scaler
    preparer.save_scaler()
    
    # Reshape for sequence models if needed
    if model_type.lower() in ['lstm', 'gru', 'transformer']:
        X_train = preparer.prepare_sequence_data(X_train)
        X_val = preparer.prepare_sequence_data(X_val)
        X_test = preparer.prepare_sequence_data(X_test)
        sequence_length = X_train.shape[1]
        input_dim = X_train.shape[2]
    else:
        sequence_length = 1
        input_dim = X_train.shape[1]
    
    # 3. Build and train model
    print("\n" + "=" * 70)
    print("🔨 Building and Training Model")
    print("=" * 70)
    
    trainer = ModelTrainer(config)
    trainer.build_model(input_dim=input_dim, sequence_length=sequence_length)
    
    if model_type.lower() in ['dnn', 'lstm', 'gru', 'transformer']:
        # Deep learning training
        history = trainer.train_deep_learning(X_train, y_train, X_val, y_val)
        
        # Plot training history
        TrainingVisualizer.plot_training_history(history)
    else:
        # sklearn training
        # For sklearn, combine train and val
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        trainer.train_sklearn(X_train_full, y_train_full)
        
        # Cross-validation
        trainer.cross_validate(X_train_full, y_train_full)
    
    # 4. Evaluate model
    print("\n" + "=" * 70)
    print("📊 Evaluating Model")
    print("=" * 70)
    
    metrics = trainer.evaluate(X_test, y_test)
    
    # Get predictions for visualization
    y_pred = trainer.model.predict(X_test)
    if isinstance(y_pred, np.ndarray) and len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    # Plot predictions
    TrainingVisualizer.plot_predictions(y_test, y_pred)
    
    # 5. Save model
    print("\n" + "=" * 70)
    print("💾 Saving Model")
    print("=" * 70)
    
    model_path = trainer.save_model()
    
    # Save training config
    config.save(os.path.join('models/saved_models/', 'training_config.json'))
    
    print("\n" + "=" * 70)
    print("✅ Training Pipeline Complete!")
    print("=" * 70)
    
    return {
        'model_path': model_path,
        'metrics': metrics,
        'config': config.to_dict()
    }


def train_multiple_models(data_path: str = 'data/') -> Dict:
    """
    Train and compare multiple models.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Dictionary with comparison results
    """
    print("=" * 70)
    print("🏏 Training Multiple Models for Comparison")
    print("=" * 70)
    
    models_to_train = ['linear', 'random_forest', 'gradient_boosting', 'dnn']
    results = {}
    
    for model_type in models_to_train:
        try:
            print(f"\n{'='*50}")
            print(f"Training {model_type.upper()}")
            print('='*50)
            
            result = run_training_pipeline(
                model_type=model_type,
                data_path=data_path,
                epochs=50 if model_type == 'dnn' else 100
            )
            
            results[model_type] = result['metrics']
            
        except Exception as e:
            print(f"❌ Error training {model_type}: {str(e)}")
            results[model_type] = {'error': str(e)}
    
    # Plot comparison
    TrainingVisualizer.plot_model_comparison(results)
    
    # Save comparison results
    with open('models/saved_models/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Model Comparison Summary")
    print("=" * 70)
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-" * 56)
    for model, metrics in results.items():
        if 'error' not in metrics:
            print(f"{model:<20} {metrics['MAE']:<12.2f} {metrics['RMSE']:<12.2f} {metrics['R2']:<12.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train IPL Score Prediction Model')
    parser.add_argument('--model', type=str, default='dnn',
                       choices=['dnn', 'lstm', 'transformer', 'linear', 'random_forest', 'gradient_boosting', 'all'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to data directory')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        # Train all models for comparison
        results = train_multiple_models(args.data_path)
    else:
        # Train single model
        results = run_training_pipeline(
            model_type=args.model,
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
