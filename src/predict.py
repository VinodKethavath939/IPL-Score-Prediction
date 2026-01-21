"""
Prediction Module for IPL Score Prediction
==========================================
This module handles:
- Loading trained models
- Making predictions
- Confidence interval estimation
- Score range prediction

Author: IPL Score Prediction Team
Date: 2024
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class IPLScorePredictor:
    """
    Main prediction class for IPL Score Prediction.
    
    This class handles:
    - Loading trained models and preprocessors
    - Processing input data
    - Making predictions with confidence intervals
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_columns = []
        self.model_type = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to saved model
        """
        if model_path.endswith('.keras') or model_path.endswith('.h5'):
            self.model = keras.models.load_model(model_path)
            self.model_type = 'deep_learning'
        else:
            self.model = joblib.load(model_path)
            self.model_type = 'sklearn'
        
        print(f"✅ Loaded model from {model_path}")
    
    def load_preprocessors(self, 
                           scaler_path: str = "models/encoders/scaler.joblib",
                           encoders_path: str = "models/encoders/",
                           features_path: str = "data/features/feature_columns.json") -> None:
        """
        Load preprocessing objects (scaler, encoders).
        
        Args:
            scaler_path: Path to scaler file
            encoders_path: Path to encoders directory
            features_path: Path to feature columns JSON
        """
        # Load scaler
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded scaler from {scaler_path}")
        
        # Load label encoders
        if os.path.exists(encoders_path):
            import glob
            encoder_files = glob.glob(os.path.join(encoders_path, 'label_encoder_*.joblib'))
            for file in encoder_files:
                col = file.split('label_encoder_')[1].replace('.joblib', '')
                self.encoders[col] = joblib.load(file)
            print(f"✅ Loaded {len(self.encoders)} encoders")
        
        # Load feature columns
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                feature_info = json.load(f)
                self.feature_columns = feature_info.get('all_features', [])
            print(f"✅ Loaded {len(self.feature_columns)} feature columns")
    
    def prepare_input(self, input_data: Dict) -> np.ndarray:
        """
        Prepare input data for prediction.
        
        Args:
            input_data: Dictionary with input features
            
        Returns:
            Processed feature array
        """
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Calculate derived features
        df = self._calculate_derived_features(df)
        
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in df.columns:
                try:
                    df[f'{col}_encoded'] = encoder.transform(df[col].astype(str))
                except:
                    df[f'{col}_encoded'] = -1  # Unknown category
        
        # Get feature values
        features = []
        for col in self.feature_columns:
            if col in df.columns:
                features.append(df[col].values[0])
            else:
                features.append(0)  # Default value for missing features
        
        # Convert to numpy array
        X = np.array(features).reshape(1, -1)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Reshape for sequence models if needed
        if self.model_type == 'deep_learning':
            # Check if model expects 3D input
            input_shape = self.model.input_shape
            if len(input_shape) == 3:
                X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        return X
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features from basic inputs.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features
        """
        df = df.copy()
        
        # Calculate basic features
        if 'current_score' not in df.columns:
            df['current_score'] = df.get('score', 0)
        
        if 'overs' in df.columns:
            df['overs_completed'] = df['overs']
            df['overs_remaining'] = 20 - df['overs']
            df['ball_number'] = df['overs'] * 6
            df['balls_remaining'] = 120 - df['ball_number']
        
        if 'wickets' in df.columns:
            df['wickets_fallen'] = df['wickets']
            df['wickets_remaining'] = 10 - df['wickets']
        
        # Run rate calculations
        if 'current_score' in df.columns and 'overs_completed' in df.columns:
            overs = df['overs_completed'].replace(0, 0.1)
            df['current_run_rate'] = df['current_score'] / overs
        
        if 'target' in df.columns and 'overs_remaining' in df.columns:
            runs_needed = df['target'] - df.get('current_score', 0)
            overs_left = df['overs_remaining'].replace(0, 0.1)
            df['required_run_rate'] = runs_needed / overs_left
        else:
            df['required_run_rate'] = 0
        
        # Projected score
        if 'current_run_rate' in df.columns and 'overs_remaining' in df.columns:
            df['projected_score'] = df['current_score'] + (df['current_run_rate'] * df['overs_remaining'])
        
        # Phase indicators
        if 'overs' in df.columns:
            df['is_powerplay'] = (df['overs'] <= 6).astype(int)
            df['is_middle_overs'] = ((df['overs'] > 6) & (df['overs'] <= 15)).astype(int)
            df['is_death_overs'] = (df['overs'] > 15).astype(int)
            df['match_phase'] = np.select(
                [df['overs'] <= 6, (df['overs'] > 6) & (df['overs'] <= 15), df['overs'] > 15],
                [1, 2, 3],
                default=2
            )
            df['is_last_5_overs'] = (df['overs'] > 15).astype(int)
        
        # Team strength (default values)
        team_strength = {
            'Mumbai Indians': 85,
            'Chennai Super Kings': 84,
            'Royal Challengers Bangalore': 80,
            'Kolkata Knight Riders': 78,
            'Delhi Capitals': 77,
            'Punjab Kings': 75,
            'Rajasthan Royals': 79,
            'Sunrisers Hyderabad': 76,
            'Gujarat Titans': 82,
            'Lucknow Super Giants': 78
        }
        
        df['batting_team_strength'] = df.get('batting_team', 'Unknown').map(team_strength).fillna(75)
        df['bowling_team_strength'] = df.get('bowling_team', 'Unknown').map(team_strength).fillna(75)
        df['team_strength_diff'] = df['batting_team_strength'] - df['bowling_team_strength']
        
        # Venue features
        venue_stats = {
            'Wankhede Stadium': {'avg_score': 175, 'pace_advantage': 1.1, 'spin_advantage': 0.9},
            'M. A. Chidambaram Stadium': {'avg_score': 162, 'pace_advantage': 0.9, 'spin_advantage': 1.2},
            'Eden Gardens': {'avg_score': 168, 'pace_advantage': 1.0, 'spin_advantage': 1.1},
            'Arun Jaitley Stadium': {'avg_score': 170, 'pace_advantage': 0.95, 'spin_advantage': 1.0},
            'M. Chinnaswamy Stadium': {'avg_score': 182, 'pace_advantage': 1.15, 'spin_advantage': 0.85},
            'Narendra Modi Stadium': {'avg_score': 172, 'pace_advantage': 1.05, 'spin_advantage': 0.95}
        }
        
        default_venue = {'avg_score': 168, 'pace_advantage': 1.0, 'spin_advantage': 1.0}
        
        venue = df.get('venue', pd.Series(['Unknown'])).iloc[0]
        venue_info = venue_stats.get(venue, default_venue)
        
        df['venue_avg_score'] = venue_info['avg_score']
        df['venue_pace_advantage'] = venue_info['pace_advantage']
        df['venue_spin_advantage'] = venue_info['spin_advantage']
        df['is_high_scoring_venue'] = int(venue_info['avg_score'] > 175)
        
        # Player features (default values)
        df['striker_strike_rate'] = 130.0
        df['striker_avg_runs'] = 7.0
        df['striker_boundary_pct'] = 15.0
        df['bowler_economy'] = 8.0
        df['bowler_wicket_pct'] = 4.0
        df['expected_runs_per_ball'] = 1.3
        df['matchup_advantage'] = 0.0
        df['head_to_head_advantage'] = 0.0
        
        # Additional features
        df['recent_runs'] = df.get('last_5_overs_runs', df['current_score'] * 0.3)
        df['recent_run_rate'] = df.get('current_run_rate', 8.0)
        df['run_rate_momentum'] = 0.0
        df['is_post_timeout'] = 0
        df['is_batting_powerplay_phase'] = 0
        
        return df
    
    def predict(self, input_data: Dict) -> float:
        """
        Make a single prediction.
        
        Args:
            input_data: Dictionary with input features
            
        Returns:
            Predicted score
        """
        X = self.prepare_input(input_data)
        
        prediction = self.model.predict(X, verbose=0)
        
        if isinstance(prediction, np.ndarray):
            prediction = prediction.flatten()[0]
        
        return float(prediction)
    
    def predict_with_confidence(self, 
                                input_data: Dict,
                                confidence_level: float = 0.95) -> Dict:
        """
        Make prediction with confidence interval.
        
        Args:
            input_data: Dictionary with input features
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Dictionary with prediction and confidence interval
        """
        # Get point prediction
        predicted_score = self.predict(input_data)
        
        # Estimate confidence interval based on model uncertainty
        # Using a simple heuristic based on overs remaining
        overs_remaining = 20 - input_data.get('overs', 10)
        
        # Uncertainty increases with more overs remaining
        base_uncertainty = 5  # Base uncertainty in runs
        uncertainty_per_over = 2  # Additional uncertainty per over remaining
        
        total_uncertainty = base_uncertainty + (uncertainty_per_over * overs_remaining)
        
        # Calculate confidence interval
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = predicted_score - z_score * total_uncertainty
        upper_bound = predicted_score + z_score * total_uncertainty
        
        # Ensure bounds are reasonable
        lower_bound = max(input_data.get('current_score', 0), lower_bound)
        upper_bound = min(300, upper_bound)  # Maximum reasonable T20 score
        
        return {
            'predicted_score': round(predicted_score),
            'lower_bound': round(lower_bound),
            'upper_bound': round(upper_bound),
            'confidence_interval': [round(lower_bound), round(upper_bound)],
            'prediction_range': f"{round(lower_bound)}-{round(upper_bound)}",
            'confidence_level': confidence_level,
            'uncertainty': round(total_uncertainty, 1)
        }
    
    def predict_over_by_over(self, 
                             base_input: Dict,
                             current_over: int,
                             final_over: int = 20) -> List[Dict]:
        """
        Predict scores over-by-over from current over to end.
        
        Args:
            base_input: Base match state
            current_over: Current over number
            final_over: Final over (default 20)
            
        Returns:
            List of predictions for each over
        """
        predictions = []
        
        current_score = base_input.get('current_score', 0)
        wickets = base_input.get('wickets', 0)
        
        for over in range(current_over, final_over + 1):
            input_data = base_input.copy()
            input_data['overs'] = over
            input_data['current_score'] = current_score
            input_data['wickets'] = wickets
            
            prediction = self.predict_with_confidence(input_data)
            prediction['over'] = over
            predictions.append(prediction)
            
            # Estimate score progression (simple linear interpolation)
            if over < final_over:
                remaining_overs = final_over - over
                runs_needed = prediction['predicted_score'] - current_score
                runs_this_over = runs_needed / remaining_overs
                current_score += runs_this_over
        
        return predictions
    
    def predict_chase(self, input_data: Dict) -> Dict:
        """
        Predict outcome for second innings chase.
        
        Args:
            input_data: Dictionary with chase scenario details
            
        Returns:
            Chase prediction with win probability
        """
        prediction = self.predict_with_confidence(input_data)
        
        target = input_data.get('target', 180)
        current_score = input_data.get('current_score', 0)
        overs = input_data.get('overs', 0)
        wickets = input_data.get('wickets', 0)
        
        runs_needed = target - current_score
        balls_remaining = (20 - overs) * 6
        wickets_remaining = 10 - wickets
        
        required_run_rate = (runs_needed / (balls_remaining / 6)) if balls_remaining > 0 else float('inf')
        
        # Simple win probability estimation
        # Based on required run rate and wickets remaining
        base_prob = 0.5
        
        # Adjust for required run rate
        if required_run_rate <= 6:
            rr_factor = 0.2
        elif required_run_rate <= 8:
            rr_factor = 0.1
        elif required_run_rate <= 10:
            rr_factor = 0.0
        elif required_run_rate <= 12:
            rr_factor = -0.1
        elif required_run_rate <= 15:
            rr_factor = -0.2
        else:
            rr_factor = -0.3
        
        # Adjust for wickets
        wicket_factor = (wickets_remaining - 5) * 0.05
        
        win_probability = min(0.95, max(0.05, base_prob + rr_factor + wicket_factor))
        
        return {
            **prediction,
            'target': target,
            'runs_needed': runs_needed,
            'balls_remaining': balls_remaining,
            'required_run_rate': round(required_run_rate, 2),
            'win_probability': round(win_probability, 2),
            'chase_status': 'favorable' if win_probability > 0.5 else 'challenging'
        }


class BatchPredictor:
    """
    Make predictions on multiple samples efficiently.
    """
    
    def __init__(self, predictor: IPLScorePredictor):
        """
        Initialize batch predictor.
        
        Args:
            predictor: IPLScorePredictor instance
        """
        self.predictor = predictor
    
    def predict_batch(self, input_list: List[Dict]) -> List[Dict]:
        """
        Make predictions for a batch of inputs.
        
        Args:
            input_list: List of input dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for input_data in input_list:
            try:
                pred = self.predictor.predict_with_confidence(input_data)
                pred['status'] = 'success'
            except Exception as e:
                pred = {
                    'status': 'error',
                    'error': str(e)
                }
            predictions.append(pred)
        
        return predictions
    
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with predictions
        """
        predictions = []
        
        for _, row in df.iterrows():
            input_data = row.to_dict()
            pred = self.predictor.predict_with_confidence(input_data)
            predictions.append(pred)
        
        pred_df = pd.DataFrame(predictions)
        result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
        
        return result_df


def load_best_model() -> IPLScorePredictor:
    """
    Load the best available trained model.
    
    Returns:
        Loaded IPLScorePredictor
    """
    model_dir = "models/saved_models/"
    
    # Look for keras models first (deep learning)
    keras_models = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    
    if keras_models:
        # Get most recent model
        keras_models.sort(reverse=True)
        model_path = os.path.join(model_dir, keras_models[0])
    else:
        # Look for joblib models
        joblib_models = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        if joblib_models:
            joblib_models.sort(reverse=True)
            model_path = os.path.join(model_dir, joblib_models[0])
        else:
            raise FileNotFoundError("No trained models found!")
    
    predictor = IPLScorePredictor(model_path)
    predictor.load_preprocessors()
    
    return predictor


def main():
    """
    Example usage of the prediction module.
    """
    print("=" * 60)
    print("🏏 IPL Score Prediction - Prediction Module")
    print("=" * 60)
    
    # Example input
    sample_input = {
        'batting_team': 'Mumbai Indians',
        'bowling_team': 'Chennai Super Kings',
        'venue': 'Wankhede Stadium',
        'current_score': 85,
        'overs': 10.0,
        'wickets': 2,
        'striker': 'Rohit Sharma',
        'non_striker': 'Suryakumar Yadav',
        'bowler': 'Ravindra Jadeja'
    }
    
    print("\n📥 Input Match State:")
    for key, value in sample_input.items():
        print(f"   {key}: {value}")
    
    try:
        # Load model
        predictor = load_best_model()
        
        # Make prediction
        result = predictor.predict_with_confidence(sample_input)
        
        print("\n📊 Prediction Result:")
        print(f"   Predicted Score: {result['predicted_score']}")
        print(f"   Prediction Range: {result['prediction_range']}")
        print(f"   Confidence Level: {result['confidence_level']}")
        
        # Over-by-over prediction
        print("\n📈 Over-by-Over Predictions:")
        over_predictions = predictor.predict_over_by_over(sample_input, 10, 20)
        for pred in over_predictions[::2]:  # Show every 2nd over
            print(f"   After Over {pred['over']}: {pred['predicted_score']} ({pred['prediction_range']})")
        
    except FileNotFoundError:
        print("\n⚠️ No trained model found. Please train a model first:")
        print("   python src/train.py --model dnn --epochs 100")


if __name__ == "__main__":
    main()
