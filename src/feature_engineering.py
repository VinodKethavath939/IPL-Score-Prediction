"""
Feature Engineering Module for IPL Score Prediction
====================================================
This module creates advanced features for the IPL score prediction model.

Features Created:
- Match state features (score, overs, wickets)
- Run rate features (current, required)
- Phase indicators (powerplay, middle, death)
- Player impact features
- Team strength features
- Historical features

Author: IPL Score Prediction Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
import os
import joblib

warnings.filterwarnings('ignore')


class MatchStateFeatures:
    """
    Create features representing the current state of the match.
    """
    
    @staticmethod
    def calculate_current_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate running total score at each ball.
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with current_score feature
        """
        df = df.copy()
        
        # Current score is already in total_runs column
        if 'total_runs' in df.columns:
            df['current_score'] = df['total_runs']
        else:
            # Calculate cumulative runs per innings
            df['current_score'] = df.groupby(['match_id', 'innings'])['runs_off_bat'].cumsum()
            if 'extras' in df.columns:
                df['current_score'] += df.groupby(['match_id', 'innings'])['extras'].cumsum()
        
        return df
    
    @staticmethod
    def calculate_balls_remaining(df: pd.DataFrame, total_overs: int = 20) -> pd.DataFrame:
        """
        Calculate balls remaining in the innings.
        
        Args:
            df: Ball-by-ball DataFrame
            total_overs: Total overs in innings (default 20 for T20)
            
        Returns:
            DataFrame with balls_remaining feature
        """
        df = df.copy()
        
        total_balls = total_overs * 6
        
        # Calculate ball number in innings
        df['ball_number'] = (df['over'] - 1) * 6 + df['ball']
        df['balls_remaining'] = total_balls - df['ball_number']
        df['balls_remaining'] = df['balls_remaining'].clip(lower=0)
        
        # Overs as decimal
        df['overs_completed'] = df['over'] - 1 + df['ball'] / 6
        df['overs_remaining'] = total_overs - df['overs_completed']
        df['overs_remaining'] = df['overs_remaining'].clip(lower=0)
        
        return df
    
    @staticmethod
    def calculate_wickets_remaining(df: pd.DataFrame, total_wickets: int = 10) -> pd.DataFrame:
        """
        Calculate wickets remaining.
        
        Args:
            df: Ball-by-ball DataFrame
            total_wickets: Total wickets (default 10)
            
        Returns:
            DataFrame with wickets_remaining feature
        """
        df = df.copy()
        
        if 'wickets_fallen' in df.columns:
            df['wickets_remaining'] = total_wickets - df['wickets_fallen']
        else:
            # Calculate cumulative wickets
            df['wickets_fallen'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
            df['wickets_remaining'] = total_wickets - df['wickets_fallen']
        
        df['wickets_remaining'] = df['wickets_remaining'].clip(lower=0)
        
        return df


class RunRateFeatures:
    """
    Create run rate related features.
    """
    
    @staticmethod
    def calculate_current_run_rate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate current run rate.
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with current_run_rate feature
        """
        df = df.copy()
        
        # Avoid division by zero
        df['overs_completed_safe'] = df['overs_completed'].replace(0, 0.1)
        df['current_run_rate'] = df['current_score'] / df['overs_completed_safe']
        
        # Cap extreme values
        df['current_run_rate'] = df['current_run_rate'].clip(0, 36)  # Max possible run rate
        
        # Drop helper column
        df = df.drop(columns=['overs_completed_safe'], errors='ignore')
        
        return df
    
    @staticmethod
    def calculate_required_run_rate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate required run rate (for second innings).
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with required_run_rate feature
        """
        df = df.copy()
        
        # Only applicable for second innings
        df['required_run_rate'] = 0.0
        
        if 'target' in df.columns and 'innings' in df.columns:
            second_innings_mask = df['innings'] == 2
            
            if second_innings_mask.any():
                runs_needed = df.loc[second_innings_mask, 'target'] - df.loc[second_innings_mask, 'current_score']
                overs_left = df.loc[second_innings_mask, 'overs_remaining'].replace(0, 0.1)
                
                df.loc[second_innings_mask, 'required_run_rate'] = runs_needed / overs_left
                df['required_run_rate'] = df['required_run_rate'].clip(0, 36)
        
        return df
    
    @staticmethod
    def calculate_projected_score(df: pd.DataFrame, total_overs: int = 20) -> pd.DataFrame:
        """
        Calculate projected final score based on current run rate.
        
        Args:
            df: Ball-by-ball DataFrame
            total_overs: Total overs in innings
            
        Returns:
            DataFrame with projected_score feature
        """
        df = df.copy()
        
        df['projected_score'] = df['current_score'] + (df['current_run_rate'] * df['overs_remaining'])
        df['projected_score'] = df['projected_score'].clip(0, 300)  # Reasonable max score
        
        return df
    
    @staticmethod
    def calculate_run_rate_momentum(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
        """
        Calculate run rate momentum (recent vs overall).
        
        Args:
            df: Ball-by-ball DataFrame
            window: Number of balls to consider for recent run rate
            
        Returns:
            DataFrame with run_rate_momentum feature
        """
        df = df.copy()
        
        # Calculate runs in last 'window' balls
        df['recent_runs'] = df.groupby(['match_id', 'innings'])['runs_off_bat'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )
        
        if 'extras' in df.columns:
            df['recent_runs'] += df.groupby(['match_id', 'innings'])['extras'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
        
        # Recent run rate (per over equivalent)
        df['recent_run_rate'] = (df['recent_runs'] / window) * 6
        
        # Momentum: recent run rate vs overall run rate
        df['run_rate_momentum'] = df['recent_run_rate'] - df['current_run_rate']
        
        return df


class PhaseFeatures:
    """
    Create match phase indicator features.
    """
    
    @staticmethod
    def add_phase_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add phase indicators (powerplay, middle, death overs).
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with phase indicator features
        """
        df = df.copy()
        
        # Powerplay: Overs 1-6
        df['is_powerplay'] = (df['over'] <= 6).astype(int)
        
        # Middle overs: Overs 7-15
        df['is_middle_overs'] = ((df['over'] > 6) & (df['over'] <= 15)).astype(int)
        
        # Death overs: Overs 16-20
        df['is_death_overs'] = (df['over'] > 15).astype(int)
        
        # Phase number (1, 2, or 3)
        df['match_phase'] = np.select(
            [df['over'] <= 6, (df['over'] > 6) & (df['over'] <= 15), df['over'] > 15],
            [1, 2, 3],
            default=2
        )
        
        return df
    
    @staticmethod
    def add_critical_phase_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add critical phase indicators.
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with critical phase features
        """
        df = df.copy()
        
        # Last 5 overs indicator
        df['is_last_5_overs'] = (df['over'] > 15).astype(int)
        
        # Strategic timeout phase (after 6th and 13th over)
        df['is_post_timeout'] = ((df['over'] == 7) | (df['over'] == 14)).astype(int)
        
        # Batting powerplay phase (can be taken once between overs 11-16)
        df['is_batting_powerplay_phase'] = ((df['over'] >= 11) & (df['over'] <= 16)).astype(int)
        
        return df


class PlayerFeatures:
    """
    Create player-specific features.
    """
    
    def __init__(self, historical_df: Optional[pd.DataFrame] = None):
        """
        Initialize with historical data for player stats.
        
        Args:
            historical_df: Historical ball-by-ball data for player stats
        """
        self.historical_df = historical_df
        self.batsman_stats = {}
        self.bowler_stats = {}
        
        if historical_df is not None:
            self._compute_player_stats()
    
    def _compute_player_stats(self):
        """Compute player statistics from historical data."""
        print("Computing player statistics...")
        
        df = self.historical_df.copy()
        
        # Batsman statistics
        batsman_groups = df.groupby('striker')
        
        self.batsman_stats = {
            'strike_rate': batsman_groups['runs_off_bat'].sum() / batsman_groups.size() * 100,
            'average_runs': batsman_groups['runs_off_bat'].mean() * 6,  # Per over equivalent
            'boundary_percentage': (
                df[df['runs_off_bat'].isin([4, 6])].groupby('striker').size() / 
                batsman_groups.size() * 100
            ).fillna(0)
        }
        
        # Bowler statistics
        bowler_groups = df.groupby('bowler')
        
        self.bowler_stats = {
            'economy_rate': bowler_groups['runs_off_bat'].sum() / (bowler_groups.size() / 6),
            'wicket_percentage': (
                df[df['is_wicket'] == 1].groupby('bowler').size() / 
                bowler_groups.size() * 100
            ).fillna(0)
        }
    
    def add_batsman_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add batsman-specific features.
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with batsman features
        """
        df = df.copy()
        
        if len(self.batsman_stats) == 0:
            # Use default values if no historical data
            df['striker_strike_rate'] = 130.0
            df['striker_avg_runs'] = 7.0
            df['striker_boundary_pct'] = 15.0
            return df
        
        # Map striker statistics
        df['striker_strike_rate'] = df['striker'].map(
            self.batsman_stats.get('strike_rate', {})
        ).fillna(130.0)
        
        df['striker_avg_runs'] = df['striker'].map(
            self.batsman_stats.get('average_runs', {})
        ).fillna(7.0)
        
        df['striker_boundary_pct'] = df['striker'].map(
            self.batsman_stats.get('boundary_percentage', {})
        ).fillna(15.0)
        
        return df
    
    def add_bowler_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add bowler-specific features.
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with bowler features
        """
        df = df.copy()
        
        if len(self.bowler_stats) == 0:
            df['bowler_economy'] = 8.0
            df['bowler_wicket_pct'] = 4.0
            return df
        
        # Map bowler statistics
        df['bowler_economy'] = df['bowler'].map(
            self.bowler_stats.get('economy_rate', {})
        ).fillna(8.0)
        
        df['bowler_wicket_pct'] = df['bowler'].map(
            self.bowler_stats.get('wicket_percentage', {})
        ).fillna(4.0)
        
        return df
    
    def add_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add batsman vs bowler matchup features.
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with matchup features
        """
        df = df.copy()
        
        # Expected runs based on striker and bowler
        df['expected_runs_per_ball'] = (
            df['striker_strike_rate'] / 100 * (1 - df['bowler_economy'] / 36)
        ).clip(0, 3)
        
        # Matchup advantage (positive = batsman favored)
        df['matchup_advantage'] = (df['striker_strike_rate'] - 130) - (df['bowler_economy'] - 8) * 10
        
        return df


class TeamFeatures:
    """
    Create team-level features.
    """
    
    def __init__(self, historical_df: Optional[pd.DataFrame] = None):
        """
        Initialize with historical data.
        
        Args:
            historical_df: Historical match/ball data
        """
        self.historical_df = historical_df
        self.team_stats = {}
        
        if historical_df is not None:
            self._compute_team_stats()
    
    def _compute_team_stats(self):
        """Compute team statistics from historical data."""
        print("Computing team statistics...")
        
        df = self.historical_df.copy()
        
        # Get innings totals
        innings_totals = df.groupby(['match_id', 'innings', 'batting_team']).agg({
            'total_runs': 'max',
            'wickets_fallen': 'max'
        }).reset_index()
        
        # Team batting statistics
        team_bat_stats = innings_totals.groupby('batting_team').agg({
            'total_runs': ['mean', 'std', 'max']
        })
        team_bat_stats.columns = ['avg_score', 'score_std', 'max_score']
        
        self.team_stats['batting'] = team_bat_stats.to_dict('index')
    
    def add_team_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team strength features.
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with team strength features
        """
        df = df.copy()
        
        # Team strength ratings (can be updated with actual data)
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
        
        df['batting_team_strength'] = df['batting_team'].map(team_strength).fillna(75)
        df['bowling_team_strength'] = df['bowling_team'].map(team_strength).fillna(75)
        
        # Team strength difference
        df['team_strength_diff'] = df['batting_team_strength'] - df['bowling_team_strength']
        
        return df
    
    def add_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add head-to-head features between teams.
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with head-to-head features
        """
        df = df.copy()
        
        # Create matchup column
        df['matchup'] = df.apply(
            lambda x: tuple(sorted([x['batting_team'], x['bowling_team']])),
            axis=1
        )
        
        # Head-to-head historical data would be computed here
        # For now, use neutral value
        df['head_to_head_advantage'] = 0.0
        
        df = df.drop(columns=['matchup'], errors='ignore')
        
        return df


class VenueFeatures:
    """
    Create venue-specific features.
    """
    
    def __init__(self):
        """Initialize venue features."""
        # Venue characteristics (based on historical data)
        self.venue_stats = {
            'Wankhede Stadium': {'avg_score': 175, 'pace_advantage': 1.1, 'spin_advantage': 0.9},
            'M. A. Chidambaram Stadium': {'avg_score': 162, 'pace_advantage': 0.9, 'spin_advantage': 1.2},
            'Eden Gardens': {'avg_score': 168, 'pace_advantage': 1.0, 'spin_advantage': 1.1},
            'Arun Jaitley Stadium': {'avg_score': 170, 'pace_advantage': 0.95, 'spin_advantage': 1.0},
            'M. Chinnaswamy Stadium': {'avg_score': 182, 'pace_advantage': 1.15, 'spin_advantage': 0.85},
            'Narendra Modi Stadium': {'avg_score': 172, 'pace_advantage': 1.05, 'spin_advantage': 0.95},
            'Rajiv Gandhi International Stadium': {'avg_score': 165, 'pace_advantage': 1.0, 'spin_advantage': 1.0},
            'Punjab Cricket Association Stadium': {'avg_score': 178, 'pace_advantage': 1.1, 'spin_advantage': 0.9}
        }
        
        self.default_stats = {'avg_score': 168, 'pace_advantage': 1.0, 'spin_advantage': 1.0}
    
    def add_venue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add venue-specific features.
        
        Args:
            df: Ball-by-ball DataFrame
            
        Returns:
            DataFrame with venue features
        """
        df = df.copy()
        
        # Map venue statistics
        df['venue_avg_score'] = df['venue'].apply(
            lambda x: self.venue_stats.get(x, self.default_stats)['avg_score']
        )
        
        df['venue_pace_advantage'] = df['venue'].apply(
            lambda x: self.venue_stats.get(x, self.default_stats)['pace_advantage']
        )
        
        df['venue_spin_advantage'] = df['venue'].apply(
            lambda x: self.venue_stats.get(x, self.default_stats)['spin_advantage']
        )
        
        # Venue is high-scoring or not
        df['is_high_scoring_venue'] = (df['venue_avg_score'] > 175).astype(int)
        
        return df


class EncodingFeatures:
    """
    Encode categorical features for model input.
    """
    
    def __init__(self):
        """Initialize encoders."""
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.fitted = False
    
    def fit_label_encoders(self, df: pd.DataFrame, columns: List[str]) -> 'EncodingFeatures':
        """
        Fit label encoders for categorical columns.
        
        Args:
            df: Training DataFrame
            columns: Columns to encode
            
        Returns:
            Self for chaining
        """
        for col in columns:
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(df[col].astype(str))
            print(f"   ✓ Fitted label encoder for '{col}' ({len(self.label_encoders[col].classes_)} classes)")
        
        self.fitted = True
        return self
    
    def transform_label_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns using fitted label encoders.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        
        return df
    
    def fit_transform_label_encoders(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fit and transform label encoders.
        
        Args:
            df: Training DataFrame
            columns: Columns to encode
            
        Returns:
            Transformed DataFrame
        """
        self.fit_label_encoders(df, columns)
        return self.transform_label_encoders(df)
    
    def save_encoders(self, path: str = "models/encoders/"):
        """
        Save encoders to disk.
        
        Args:
            path: Directory to save encoders
        """
        os.makedirs(path, exist_ok=True)
        
        for col, encoder in self.label_encoders.items():
            joblib.dump(encoder, os.path.join(path, f'label_encoder_{col}.joblib'))
        
        print(f"✅ Saved {len(self.label_encoders)} encoders to {path}")
    
    def load_encoders(self, path: str = "models/encoders/"):
        """
        Load encoders from disk.
        
        Args:
            path: Directory containing saved encoders
        """
        import glob
        
        encoder_files = glob.glob(os.path.join(path, 'label_encoder_*.joblib'))
        
        for file in encoder_files:
            col = file.split('label_encoder_')[1].replace('.joblib', '')
            self.label_encoders[col] = joblib.load(file)
        
        self.fitted = True
        print(f"✅ Loaded {len(self.label_encoders)} encoders from {path}")


class FeatureEngineer:
    """
    Main feature engineering class that combines all feature generators.
    """
    
    def __init__(self, historical_df: Optional[pd.DataFrame] = None):
        """
        Initialize feature engineer.
        
        Args:
            historical_df: Historical data for computing statistics
        """
        self.match_state = MatchStateFeatures()
        self.run_rate = RunRateFeatures()
        self.phase = PhaseFeatures()
        self.player = PlayerFeatures(historical_df)
        self.team = TeamFeatures(historical_df)
        self.venue = VenueFeatures()
        self.encoder = EncodingFeatures()
        
        self.feature_columns = []
        self.categorical_columns = ['batting_team', 'bowling_team', 'venue', 'striker', 'non_striker', 'bowler']
        self.numerical_columns = []
    
    def create_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            df: Ball-by-ball DataFrame
            fit_encoders: Whether to fit encoders (True for training data)
            
        Returns:
            DataFrame with all engineered features
        """
        print("\n🔧 Feature Engineering Pipeline")
        print("=" * 50)
        
        # Make a copy
        df_features = df.copy()
        
        # 1. Match state features
        print("Creating match state features...")
        df_features = self.match_state.calculate_current_score(df_features)
        df_features = self.match_state.calculate_balls_remaining(df_features)
        df_features = self.match_state.calculate_wickets_remaining(df_features)
        
        # 2. Run rate features
        print("Creating run rate features...")
        df_features = self.run_rate.calculate_current_run_rate(df_features)
        df_features = self.run_rate.calculate_required_run_rate(df_features)
        df_features = self.run_rate.calculate_projected_score(df_features)
        df_features = self.run_rate.calculate_run_rate_momentum(df_features)
        
        # 3. Phase features
        print("Creating phase features...")
        df_features = self.phase.add_phase_indicators(df_features)
        df_features = self.phase.add_critical_phase_indicators(df_features)
        
        # 4. Player features
        print("Creating player features...")
        df_features = self.player.add_batsman_features(df_features)
        df_features = self.player.add_bowler_features(df_features)
        df_features = self.player.add_matchup_features(df_features)
        
        # 5. Team features
        print("Creating team features...")
        df_features = self.team.add_team_strength_features(df_features)
        df_features = self.team.add_head_to_head_features(df_features)
        
        # 6. Venue features
        print("Creating venue features...")
        df_features = self.venue.add_venue_features(df_features)
        
        # 7. Encode categorical features
        print("Encoding categorical features...")
        if fit_encoders:
            df_features = self.encoder.fit_transform_label_encoders(
                df_features, 
                self.categorical_columns
            )
        else:
            df_features = self.encoder.transform_label_encoders(df_features)
        
        # Define feature columns
        self.numerical_columns = [
            'current_score', 'overs_completed', 'overs_remaining', 
            'balls_remaining', 'wickets_remaining', 'wickets_fallen',
            'current_run_rate', 'required_run_rate', 'projected_score',
            'recent_runs', 'recent_run_rate', 'run_rate_momentum',
            'is_powerplay', 'is_middle_overs', 'is_death_overs', 'match_phase',
            'is_last_5_overs', 'is_post_timeout', 'is_batting_powerplay_phase',
            'striker_strike_rate', 'striker_avg_runs', 'striker_boundary_pct',
            'bowler_economy', 'bowler_wicket_pct',
            'expected_runs_per_ball', 'matchup_advantage',
            'batting_team_strength', 'bowling_team_strength', 'team_strength_diff',
            'head_to_head_advantage',
            'venue_avg_score', 'venue_pace_advantage', 'venue_spin_advantage',
            'is_high_scoring_venue'
        ]
        
        encoded_columns = [f'{col}_encoded' for col in self.categorical_columns 
                          if f'{col}_encoded' in df_features.columns]
        
        self.feature_columns = self.numerical_columns + encoded_columns
        
        print(f"\n✅ Created {len(self.feature_columns)} features")
        print("=" * 50)
        
        return df_features
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns
    
    def get_numerical_columns(self) -> List[str]:
        """Get list of numerical column names."""
        return self.numerical_columns
    
    def save_encoder(self, path: str = "models/encoders/"):
        """Save encoder to disk."""
        self.encoder.save_encoders(path)
    
    def load_encoder(self, path: str = "models/encoders/"):
        """Load encoder from disk."""
        self.encoder.load_encoders(path)


def create_target_variable(df: pd.DataFrame, target_type: str = 'final_score') -> pd.DataFrame:
    """
    Create target variable for prediction.
    
    Args:
        df: Feature DataFrame
        target_type: Type of target ('final_score', 'runs_next_over', 'win_probability')
        
    Returns:
        DataFrame with target variable
    """
    df = df.copy()
    
    if target_type == 'final_score':
        # Final score of the innings
        final_scores = df.groupby(['match_id', 'innings'])['total_runs'].max().reset_index()
        final_scores.columns = ['match_id', 'innings', 'final_score']
        
        df = df.merge(final_scores, on=['match_id', 'innings'], how='left')
        
    elif target_type == 'runs_next_over':
        # Runs scored in the next over
        df['runs_this_ball'] = df['runs_off_bat'] + df.get('extras', 0)
        df['runs_next_over'] = df.groupby(['match_id', 'innings'])['runs_this_ball'].shift(-6).rolling(6).sum()
        
    elif target_type == 'win_probability':
        # Win probability (for second innings)
        if 'target' in df.columns:
            df['will_win'] = (df['total_runs'] >= df['target']).astype(int)
    
    return df


def prepare_model_data(df: pd.DataFrame, 
                       feature_columns: List[str], 
                       target_column: str = 'final_score') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for model training.
    
    Args:
        df: Feature DataFrame
        feature_columns: List of feature column names
        target_column: Target column name
        
    Returns:
        Tuple of (X, y) arrays
    """
    # Filter to available columns
    available_features = [col for col in feature_columns if col in df.columns]
    
    if len(available_features) < len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        print(f"⚠️ Missing features: {missing}")
    
    X = df[available_features].values
    y = df[target_column].values
    
    # Handle any remaining NaN values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    return X, y


# Main feature engineering pipeline
def engineer_features(data_path: str = "data/processed/",
                      save_features: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Main feature engineering pipeline.
    
    Args:
        data_path: Path to processed data
        save_features: Whether to save engineered features
        
    Returns:
        Dictionary containing feature DataFrames
    """
    print("=" * 60)
    print("🔧 IPL Feature Engineering Pipeline")
    print("=" * 60)
    
    # Load processed data
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    
    print(f"✅ Loaded data - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    
    # Initialize feature engineer with historical data
    feature_engineer = FeatureEngineer(historical_df=train_df)
    
    # Create features
    print("\n📊 Creating features for training data...")
    train_features = feature_engineer.create_features(train_df, fit_encoders=True)
    
    print("\n📊 Creating features for validation data...")
    val_features = feature_engineer.create_features(val_df, fit_encoders=False)
    
    print("\n📊 Creating features for test data...")
    test_features = feature_engineer.create_features(test_df, fit_encoders=False)
    
    # Create target variable
    train_features = create_target_variable(train_features)
    val_features = create_target_variable(val_features)
    test_features = create_target_variable(test_features)
    
    # Save features
    if save_features:
        features_path = os.path.join(os.path.dirname(data_path), "features")
        os.makedirs(features_path, exist_ok=True)
        
        train_features.to_csv(os.path.join(features_path, "train_features.csv"), index=False)
        val_features.to_csv(os.path.join(features_path, "val_features.csv"), index=False)
        test_features.to_csv(os.path.join(features_path, "test_features.csv"), index=False)
        
        # Save encoder
        feature_engineer.save_encoder()
        
        # Save feature column names
        import json
        with open(os.path.join(features_path, "feature_columns.json"), 'w') as f:
            json.dump({
                'all_features': feature_engineer.get_feature_columns(),
                'numerical_features': feature_engineer.get_numerical_columns()
            }, f, indent=2)
        
        print(f"\n💾 Saved features to {features_path}")
    
    print("\n" + "=" * 60)
    print("✅ Feature engineering complete!")
    print("=" * 60)
    
    return {
        'train': train_features,
        'val': val_features,
        'test': test_features,
        'feature_engineer': feature_engineer
    }


if __name__ == "__main__":
    # Run feature engineering pipeline
    feature_data = engineer_features(data_path="data/processed/")
    
    # Print feature summary
    print("\n📈 Feature Summary:")
    for name, data in feature_data.items():
        if isinstance(data, pd.DataFrame):
            print(f"   {name}: {data.shape}")
