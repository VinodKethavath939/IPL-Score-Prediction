"""
Deep Learning Model Architectures for IPL Score Prediction
==========================================================
This module contains various neural network architectures:
1. Deep Neural Network (DNN)
2. LSTM Network with Attention
3. Transformer-based Model
4. Baseline Models (Linear Regression, Random Forest)

Author: IPL Score Prediction Team
Date: 2024
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LSTM, GRU, Bidirectional,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Embedding, Concatenate, Reshape, Flatten, Add, Activation
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from typing import Dict, List, Tuple, Optional, Union
import os


# ============================================================================
# 1. BASELINE MODELS
# ============================================================================

class BaselineModels:
    """
    Baseline models for comparison.
    Includes Linear Regression and Random Forest.
    """
    
    @staticmethod
    def create_linear_regression():
        """
        Create a simple linear regression model using sklearn.
        
        Returns:
            LinearRegression model
        """
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    
    @staticmethod
    def create_ridge_regression(alpha: float = 1.0):
        """
        Create a Ridge regression model.
        
        Args:
            alpha: Regularization strength
            
        Returns:
            Ridge model
        """
        from sklearn.linear_model import Ridge
        return Ridge(alpha=alpha)
    
    @staticmethod
    def create_random_forest(n_estimators: int = 100, max_depth: int = 15):
        """
        Create a Random Forest regressor.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            
        Returns:
            RandomForestRegressor model
        """
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    @staticmethod
    def create_gradient_boosting(n_estimators: int = 100, learning_rate: float = 0.1):
        """
        Create a Gradient Boosting regressor.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            
        Returns:
            GradientBoostingRegressor model
        """
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=42
        )
    
    @staticmethod
    def create_xgboost(n_estimators: int = 100, learning_rate: float = 0.1):
        """
        Create an XGBoost regressor.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            
        Returns:
            XGBRegressor model
        """
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        except ImportError:
            print("⚠️ XGBoost not installed. Using Gradient Boosting instead.")
            return BaselineModels.create_gradient_boosting(n_estimators, learning_rate)


# ============================================================================
# 2. DEEP NEURAL NETWORK (DNN)
# ============================================================================

class DNNModel:
    """
    Deep Neural Network for IPL Score Prediction.
    
    Architecture:
    - Input layer
    - Multiple hidden layers with batch normalization and dropout
    - Output layer for score prediction
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_layers: List[int] = [256, 128, 64, 32],
                 dropout_rate: float = 0.3,
                 l1_reg: float = 0.0001,
                 l2_reg: float = 0.001):
        """
        Initialize DNN model.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.model = None
        
    def build(self) -> Model:
        """
        Build the DNN model.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.input_dim,), name='input_features')
        
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            x = Dense(
                units,
                kernel_regularizer=l1_l2(self.l1_reg, self.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = BatchNormalization(name=f'bn_{i+1}')(x)
            x = Activation('relu', name=f'relu_{i+1}')(x)
            x = Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        outputs = Dense(1, name='output')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='DNN_IPL_Score_Predictor')
        
        return self.model
    
    def compile(self, learning_rate: float = 0.001):
        """
        Compile the model.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
    def summary(self):
        """Print model summary."""
        self.model.summary()


# ============================================================================
# 3. LSTM MODEL WITH ATTENTION
# ============================================================================

class LSTMModel:
    """
    LSTM Network with Attention for sequence-based IPL Score Prediction.
    
    This model is designed for over-by-over prediction where the sequence
    of events matters.
    
    Architecture:
    - Input embedding/dense layer
    - Bidirectional LSTM layers
    - Attention mechanism
    - Dense output layers
    """
    
    def __init__(self,
                 input_dim: int,
                 sequence_length: int = 120,  # 20 overs * 6 balls
                 lstm_units: List[int] = [128, 64],
                 dense_units: List[int] = [64, 32],
                 dropout_rate: float = 0.3,
                 use_attention: bool = True):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of features per timestep
            sequence_length: Length of input sequence
            lstm_units: List of LSTM layer sizes
            dense_units: List of dense layer sizes
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.model = None
        
    def _attention_layer(self, inputs):
        """
        Custom attention layer.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Attended output tensor
        """
        # Self-attention
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            name='self_attention'
        )(inputs, inputs)
        
        # Add & Norm
        x = Add()([inputs, attention])
        x = LayerNormalization()(x)
        
        return x
    
    def build(self) -> Model:
        """
        Build the LSTM model.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.input_dim), name='sequence_input')
        
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1) or self.use_attention
            
            x = Bidirectional(
                LSTM(units, return_sequences=return_sequences, dropout=self.dropout_rate),
                name=f'bilstm_{i+1}'
            )(x)
            
        # Attention layer
        if self.use_attention:
            x = self._attention_layer(x)
            x = GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer
        outputs = Dense(1, name='output')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Attention_IPL_Predictor')
        
        return self.model
    
    def compile(self, learning_rate: float = 0.001):
        """
        Compile the model.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
    def summary(self):
        """Print model summary."""
        self.model.summary()


# ============================================================================
# 4. GRU MODEL (Alternative to LSTM)
# ============================================================================

class GRUModel:
    """
    GRU Network for IPL Score Prediction.
    
    GRU is computationally more efficient than LSTM while achieving
    comparable performance.
    """
    
    def __init__(self,
                 input_dim: int,
                 sequence_length: int = 120,
                 gru_units: List[int] = [128, 64],
                 dense_units: List[int] = [64, 32],
                 dropout_rate: float = 0.3):
        """
        Initialize GRU model.
        
        Args:
            input_dim: Number of features per timestep
            sequence_length: Length of input sequence
            gru_units: List of GRU layer sizes
            dense_units: List of dense layer sizes
            dropout_rate: Dropout rate
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build(self) -> Model:
        """
        Build the GRU model.
        
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=(self.sequence_length, self.input_dim), name='sequence_input')
        
        x = inputs
        
        # GRU layers
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1
            
            x = Bidirectional(
                GRU(units, return_sequences=return_sequences, dropout=self.dropout_rate),
                name=f'bigru_{i+1}'
            )(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output
        outputs = Dense(1, name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='GRU_IPL_Predictor')
        
        return self.model
    
    def compile(self, learning_rate: float = 0.001):
        """Compile the model."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
    def summary(self):
        """Print model summary."""
        self.model.summary()


# ============================================================================
# 5. TRANSFORMER MODEL
# ============================================================================

class TransformerBlock(layers.Layer):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        """
        Initialize Transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout_rate: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(layers.Layer):
    """
    Positional encoding for Transformer model.
    """
    
    def __init__(self, sequence_length: int, embed_dim: int):
        """
        Initialize positional encoding.
        
        Args:
            sequence_length: Maximum sequence length
            embed_dim: Embedding dimension
        """
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        # Create positional encoding matrix
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        
        pe = np.zeros((self.sequence_length, self.embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.constant(pe, dtype=tf.float32)
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len, :]


class TransformerModel:
    """
    Transformer-based model for IPL Score Prediction.
    
    Architecture:
    - Input projection layer
    - Positional encoding
    - Multiple transformer blocks
    - Global average pooling
    - Dense output layers
    """
    
    def __init__(self,
                 input_dim: int,
                 sequence_length: int = 120,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 ff_dim: int = 128,
                 num_transformer_blocks: int = 2,
                 dropout_rate: float = 0.2):
        """
        Initialize Transformer model.
        
        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequence
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_transformer_blocks: Number of transformer blocks
            dropout_rate: Dropout rate
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build(self) -> Model:
        """
        Build the Transformer model.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.input_dim), name='sequence_input')
        
        # Project input to embed_dim
        x = Dense(self.embed_dim, name='input_projection')(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(self.sequence_length, self.embed_dim)(x)
        
        # Transformer blocks
        for i in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate
            )(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers
        x = Dense(64, activation='relu', name='dense_1')(x)
        x = Dropout(self.dropout_rate, name='dropout_1')(x)
        x = Dense(32, activation='relu', name='dense_2')(x)
        x = Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Output
        outputs = Dense(1, name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='Transformer_IPL_Predictor')
        
        return self.model
    
    def compile(self, learning_rate: float = 0.001):
        """Compile the model."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
    def summary(self):
        """Print model summary."""
        self.model.summary()


# ============================================================================
# 6. HYBRID MODEL (DNN + LSTM)
# ============================================================================

class HybridModel:
    """
    Hybrid model combining DNN for static features and LSTM for sequential features.
    
    This model processes:
    - Static features (team, venue, etc.) through DNN branch
    - Sequential features (ball-by-ball data) through LSTM branch
    - Concatenates both branches for final prediction
    """
    
    def __init__(self,
                 static_input_dim: int,
                 sequence_input_dim: int,
                 sequence_length: int = 120,
                 dnn_units: List[int] = [128, 64],
                 lstm_units: List[int] = [64, 32],
                 dropout_rate: float = 0.3):
        """
        Initialize Hybrid model.
        
        Args:
            static_input_dim: Number of static features
            sequence_input_dim: Number of features per timestep
            sequence_length: Length of sequence
            dnn_units: DNN layer sizes
            lstm_units: LSTM layer sizes
            dropout_rate: Dropout rate
        """
        self.static_input_dim = static_input_dim
        self.sequence_input_dim = sequence_input_dim
        self.sequence_length = sequence_length
        self.dnn_units = dnn_units
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build(self) -> Model:
        """
        Build the Hybrid model.
        
        Returns:
            Compiled Keras model
        """
        # Static input branch (DNN)
        static_input = Input(shape=(self.static_input_dim,), name='static_input')
        
        x_static = static_input
        for i, units in enumerate(self.dnn_units):
            x_static = Dense(units, activation='relu', name=f'static_dense_{i+1}')(x_static)
            x_static = BatchNormalization(name=f'static_bn_{i+1}')(x_static)
            x_static = Dropout(self.dropout_rate, name=f'static_dropout_{i+1}')(x_static)
        
        # Sequence input branch (LSTM)
        sequence_input = Input(shape=(self.sequence_length, self.sequence_input_dim), name='sequence_input')
        
        x_seq = sequence_input
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x_seq = LSTM(units, return_sequences=return_sequences, dropout=self.dropout_rate,
                        name=f'lstm_{i+1}')(x_seq)
        
        # Concatenate branches
        concatenated = Concatenate(name='concatenate')([x_static, x_seq])
        
        # Final dense layers
        x = Dense(64, activation='relu', name='final_dense_1')(concatenated)
        x = Dropout(self.dropout_rate, name='final_dropout_1')(x)
        x = Dense(32, activation='relu', name='final_dense_2')(x)
        
        # Output
        outputs = Dense(1, name='output')(x)
        
        self.model = Model(
            inputs=[static_input, sequence_input],
            outputs=outputs,
            name='Hybrid_IPL_Predictor'
        )
        
        return self.model
    
    def compile(self, learning_rate: float = 0.001):
        """Compile the model."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
    def summary(self):
        """Print model summary."""
        self.model.summary()


# ============================================================================
# 7. MODEL FACTORY
# ============================================================================

class ModelFactory:
    """
    Factory class to create different model types.
    """
    
    SUPPORTED_MODELS = ['dnn', 'lstm', 'gru', 'transformer', 'hybrid', 
                        'linear', 'ridge', 'random_forest', 'gradient_boosting', 'xgboost']
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> Union[Model, object]:
        """
        Create a model of specified type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Model-specific parameters
            
        Returns:
            Created model
        """
        model_type = model_type.lower()
        
        if model_type not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported: {ModelFactory.SUPPORTED_MODELS}")
        
        # Deep Learning models
        if model_type == 'dnn':
            model = DNNModel(**kwargs)
            model.build()
            model.compile(kwargs.get('learning_rate', 0.001))
            return model.model
            
        elif model_type == 'lstm':
            model = LSTMModel(**kwargs)
            model.build()
            model.compile(kwargs.get('learning_rate', 0.001))
            return model.model
            
        elif model_type == 'gru':
            model = GRUModel(**kwargs)
            model.build()
            model.compile(kwargs.get('learning_rate', 0.001))
            return model.model
            
        elif model_type == 'transformer':
            model = TransformerModel(**kwargs)
            model.build()
            model.compile(kwargs.get('learning_rate', 0.001))
            return model.model
            
        elif model_type == 'hybrid':
            model = HybridModel(**kwargs)
            model.build()
            model.compile(kwargs.get('learning_rate', 0.001))
            return model.model
        
        # Baseline models
        elif model_type == 'linear':
            return BaselineModels.create_linear_regression()
            
        elif model_type == 'ridge':
            return BaselineModels.create_ridge_regression(kwargs.get('alpha', 1.0))
            
        elif model_type == 'random_forest':
            return BaselineModels.create_random_forest(
                kwargs.get('n_estimators', 100),
                kwargs.get('max_depth', 15)
            )
            
        elif model_type == 'gradient_boosting':
            return BaselineModels.create_gradient_boosting(
                kwargs.get('n_estimators', 100),
                kwargs.get('learning_rate', 0.1)
            )
            
        elif model_type == 'xgboost':
            return BaselineModels.create_xgboost(
                kwargs.get('n_estimators', 100),
                kwargs.get('learning_rate', 0.1)
            )


# ============================================================================
# 8. CALLBACKS
# ============================================================================

def get_callbacks(model_name: str = 'model',
                  checkpoint_dir: str = 'models/checkpoints/',
                  log_dir: str = 'models/logs/',
                  patience: int = 10) -> List:
    """
    Get training callbacks.
    
    Args:
        model_name: Name for saving model
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for TensorBoard logs
        patience: Patience for early stopping
        
    Returns:
        List of callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'{model_name}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=os.path.join(log_dir, model_name),
            histogram_freq=1
        )
    ]
    
    return callbacks


# ============================================================================
# 9. MODEL COMPARISON
# ============================================================================

def compare_models(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   models: List[str] = ['linear', 'random_forest', 'dnn']) -> Dict:
    """
    Compare multiple models on the same data.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        models: List of model types to compare
        
    Returns:
        Dictionary with comparison results
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()} model...")
        print('='*50)
        
        try:
            if model_name in ['dnn', 'lstm', 'gru', 'transformer']:
                # Deep learning model
                input_dim = X_train.shape[1]
                model = ModelFactory.create_model(model_name, input_dim=input_dim)
                
                # Reshape for sequence models if needed
                if model_name in ['lstm', 'gru', 'transformer']:
                    X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                    X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                else:
                    X_train_seq = X_train
                    X_test_seq = X_test
                
                # Train
                model.fit(
                    X_train_seq, y_train,
                    validation_split=0.1,
                    epochs=50,
                    batch_size=32,
                    callbacks=get_callbacks(model_name, patience=5),
                    verbose=0
                )
                
                y_pred = model.predict(X_test_seq).flatten()
                
            else:
                # Sklearn model
                model = ModelFactory.create_model(model_name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
            
        except Exception as e:
            print(f"  Error training {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")
    
    # DNN
    print("\n1. DNN Model:")
    dnn = DNNModel(input_dim=30)
    dnn.build()
    dnn.compile()
    dnn.summary()
    
    # LSTM
    print("\n2. LSTM Model:")
    lstm = LSTMModel(input_dim=30, sequence_length=20)
    lstm.build()
    lstm.compile()
    lstm.summary()
    
    # Transformer
    print("\n3. Transformer Model:")
    transformer = TransformerModel(input_dim=30, sequence_length=20)
    transformer.build()
    transformer.compile()
    transformer.summary()
    
    print("\n✅ All models built successfully!")
