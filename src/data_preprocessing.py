"""
Data Preprocessing Module for IPL Score Prediction
===================================================
This module handles all data loading, cleaning, and preprocessing tasks
for the IPL score prediction project.

Author: IPL Score Prediction Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import warnings
import os
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')


class IPLDataLoader:
    """
    Class to load and preprocess IPL datasets from various sources.
    
    Attributes:
        data_path (str): Path to the data directory
        ball_by_ball_df (pd.DataFrame): Ball-by-ball match data
        match_df (pd.DataFrame): Match summary data
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        self.ball_by_ball_df = None
        self.match_df = None
        self.combined_df = None
        
    def load_ball_by_ball_data(self, file_name: str = "IPL_Ball_by_Ball_2008_2022.csv") -> pd.DataFrame:
        """
        Load ball-by-ball match data.
        
        Args:
            file_name: Name of the ball-by-ball data file
            
        Returns:
            DataFrame with ball-by-ball data
        """
        file_path = os.path.join(self.data_path, "raw", file_name)
        
        if os.path.exists(file_path):
            self.ball_by_ball_df = pd.read_csv(file_path)
            print(f"✅ Loaded ball-by-ball data: {self.ball_by_ball_df.shape}")
        else:
            print(f"⚠️ File not found: {file_path}")
            print("Creating sample ball-by-ball data...")
            self.ball_by_ball_df = self._create_sample_ball_by_ball_data()
            
        return self.ball_by_ball_df
    
    def load_match_data(self, file_name: str = "IPL_Matches_2008_2022.csv") -> pd.DataFrame:
        """
        Load match summary data.
        
        Args:
            file_name: Name of the match data file
            
        Returns:
            DataFrame with match summary data
        """
        file_path = os.path.join(self.data_path, "raw", file_name)
        
        if os.path.exists(file_path):
            self.match_df = pd.read_csv(file_path)
            print(f"✅ Loaded match data: {self.match_df.shape}")
        else:
            print(f"⚠️ File not found: {file_path}")
            print("Creating sample match data...")
            self.match_df = self._create_sample_match_data()
            
        return self.match_df
    
    def _create_sample_ball_by_ball_data(self) -> pd.DataFrame:
        """
        Create sample ball-by-ball data for demonstration purposes.
        
        Returns:
            DataFrame with sample ball-by-ball data
        """
        # IPL Teams
        teams = [
            'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
            'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans',
            'Lucknow Super Giants'
        ]
        
        # Venues
        venues = [
            'Wankhede Stadium', 'M. A. Chidambaram Stadium', 'Eden Gardens',
            'Arun Jaitley Stadium', 'M. Chinnaswamy Stadium', 'Narendra Modi Stadium',
            'Rajiv Gandhi International Stadium', 'Punjab Cricket Association Stadium'
        ]
        
        # Sample batsmen and bowlers
        batsmen = [
            'Virat Kohli', 'Rohit Sharma', 'KL Rahul', 'Shikhar Dhawan',
            'Rishabh Pant', 'Hardik Pandya', 'Suryakumar Yadav', 'Faf du Plessis',
            'Jos Buttler', 'Shubman Gill', 'Ruturaj Gaikwad', 'Devon Conway'
        ]
        
        bowlers = [
            'Jasprit Bumrah', 'Mohammed Shami', 'Rashid Khan', 'Yuzvendra Chahal',
            'Ravindra Jadeja', 'Kuldeep Yadav', 'Trent Boult', 'Kagiso Rabada',
            'Pat Cummins', 'Arshdeep Singh', 'Mohammed Siraj', 'Shardul Thakur'
        ]
        
        np.random.seed(42)
        
        # Generate sample data
        n_matches = 100
        n_balls_per_innings = 120  # 20 overs
        
        data = []
        match_id = 1
        
        for _ in range(n_matches):
            batting_team = np.random.choice(teams)
            bowling_team = np.random.choice([t for t in teams if t != batting_team])
            venue = np.random.choice(venues)
            
            # First innings
            total_runs = 0
            wickets = 0
            
            for ball_num in range(1, n_balls_per_innings + 1):
                if wickets >= 10:
                    break
                    
                over = (ball_num - 1) // 6 + 1
                ball_in_over = (ball_num - 1) % 6 + 1
                
                # Simulate runs
                run_probs = [0.3, 0.35, 0.1, 0.05, 0.12, 0.03, 0.05]
                runs = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=run_probs)
                
                # Simulate wicket (5% chance per ball)
                is_wicket = np.random.random() < 0.05
                if is_wicket:
                    wickets += 1
                    runs = 0
                
                total_runs += runs
                
                # Add extras occasionally
                extras = np.random.choice([0, 1, 2], p=[0.9, 0.08, 0.02])
                total_runs += extras
                
                data.append({
                    'match_id': match_id,
                    'innings': 1,
                    'over': over,
                    'ball': ball_in_over,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'striker': np.random.choice(batsmen),
                    'non_striker': np.random.choice(batsmen),
                    'bowler': np.random.choice(bowlers),
                    'runs_off_bat': runs,
                    'extras': extras,
                    'total_runs': total_runs,
                    'wickets_fallen': wickets,
                    'is_wicket': int(is_wicket),
                    'venue': venue,
                    'date': f'2023-{np.random.randint(4, 6):02d}-{np.random.randint(1, 28):02d}'
                })
            
            # Store first innings total for second innings
            first_innings_total = total_runs
            
            # Second innings
            total_runs = 0
            wickets = 0
            
            for ball_num in range(1, n_balls_per_innings + 1):
                if wickets >= 10 or total_runs > first_innings_total:
                    break
                    
                over = (ball_num - 1) // 6 + 1
                ball_in_over = (ball_num - 1) % 6 + 1
                
                run_probs = [0.28, 0.35, 0.12, 0.05, 0.12, 0.03, 0.05]
                runs = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=run_probs)
                
                is_wicket = np.random.random() < 0.05
                if is_wicket:
                    wickets += 1
                    runs = 0
                
                total_runs += runs
                extras = np.random.choice([0, 1, 2], p=[0.9, 0.08, 0.02])
                total_runs += extras
                
                data.append({
                    'match_id': match_id,
                    'innings': 2,
                    'over': over,
                    'ball': ball_in_over,
                    'batting_team': bowling_team,  # Teams swap
                    'bowling_team': batting_team,
                    'striker': np.random.choice(batsmen),
                    'non_striker': np.random.choice(batsmen),
                    'bowler': np.random.choice(bowlers),
                    'runs_off_bat': runs,
                    'extras': extras,
                    'total_runs': total_runs,
                    'wickets_fallen': wickets,
                    'is_wicket': int(is_wicket),
                    'venue': venue,
                    'target': first_innings_total + 1,
                    'date': f'2023-{np.random.randint(4, 6):02d}-{np.random.randint(1, 28):02d}'
                })
            
            match_id += 1
        
        df = pd.DataFrame(data)
        
        # Save sample data
        os.makedirs(os.path.join(self.data_path, "raw"), exist_ok=True)
        df.to_csv(os.path.join(self.data_path, "raw", "sample_ball_by_ball.csv"), index=False)
        
        print(f"✅ Created sample ball-by-ball data: {df.shape}")
        return df
    
    def _create_sample_match_data(self) -> pd.DataFrame:
        """
        Create sample match summary data.
        
        Returns:
            DataFrame with sample match data
        """
        if self.ball_by_ball_df is None:
            self.load_ball_by_ball_data()
        
        # Aggregate ball-by-ball data to match level
        match_data = []
        
        for match_id in self.ball_by_ball_df['match_id'].unique():
            match_balls = self.ball_by_ball_df[self.ball_by_ball_df['match_id'] == match_id]
            
            # First innings
            first_innings = match_balls[match_balls['innings'] == 1]
            if len(first_innings) > 0:
                first_innings_score = first_innings['total_runs'].max()
                first_innings_wickets = first_innings['wickets_fallen'].max()
                team1 = first_innings['batting_team'].iloc[0]
                team2 = first_innings['bowling_team'].iloc[0]
                venue = first_innings['venue'].iloc[0]
                date = first_innings['date'].iloc[0]
            
            # Second innings
            second_innings = match_balls[match_balls['innings'] == 2]
            if len(second_innings) > 0:
                second_innings_score = second_innings['total_runs'].max()
                second_innings_wickets = second_innings['wickets_fallen'].max()
            else:
                second_innings_score = 0
                second_innings_wickets = 0
            
            # Determine winner
            if second_innings_score > first_innings_score:
                winner = team2
                win_type = 'wickets'
                win_margin = 10 - second_innings_wickets
            else:
                winner = team1
                win_type = 'runs'
                win_margin = first_innings_score - second_innings_score
            
            match_data.append({
                'match_id': match_id,
                'team1': team1,
                'team2': team2,
                'venue': venue,
                'date': date,
                'toss_winner': np.random.choice([team1, team2]),
                'toss_decision': np.random.choice(['bat', 'field']),
                'team1_score': first_innings_score,
                'team1_wickets': first_innings_wickets,
                'team2_score': second_innings_score,
                'team2_wickets': second_innings_wickets,
                'winner': winner,
                'win_type': win_type,
                'win_margin': win_margin
            })
        
        df = pd.DataFrame(match_data)
        
        # Save sample data
        os.makedirs(os.path.join(self.data_path, "raw"), exist_ok=True)
        df.to_csv(os.path.join(self.data_path, "raw", "sample_matches.csv"), index=False)
        
        print(f"✅ Created sample match data: {df.shape}")
        return df


class IPLDataCleaner:
    """
    Class to clean and preprocess IPL data.
    
    This class handles:
    - Missing value imputation
    - Outlier detection and handling
    - Data type conversions
    - Duplicate removal
    """
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.cleaning_stats = {}
    
    def clean_ball_by_ball_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean ball-by-ball data.
        
        Args:
            df: Raw ball-by-ball DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("\n🧹 Cleaning ball-by-ball data...")
        original_shape = df.shape
        
        # Make a copy
        df_clean = df.copy()
        
        # 1. Remove duplicates
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = original_shape[0] - df_clean.shape[0]
        print(f"   ✓ Removed {duplicates_removed} duplicate rows")
        
        # 2. Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        
        # Fill numerical columns with appropriate values
        numerical_cols = ['runs_off_bat', 'extras', 'total_runs', 'wickets_fallen', 'is_wicket']
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0).astype(int)
        
        # Fill categorical columns with mode or 'Unknown'
        categorical_cols = ['batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler', 'venue']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"   ✓ Handled {missing_before - missing_after} missing values")
        
        # 3. Validate and fix data ranges
        # Runs per ball should be between 0 and 7 (including no-ball)
        if 'runs_off_bat' in df_clean.columns:
            df_clean['runs_off_bat'] = df_clean['runs_off_bat'].clip(0, 7)
        
        # Overs should be between 1 and 20
        if 'over' in df_clean.columns:
            df_clean['over'] = df_clean['over'].clip(1, 20)
        
        # Ball in over should be between 1 and 6
        if 'ball' in df_clean.columns:
            df_clean['ball'] = df_clean['ball'].clip(1, 6)
        
        # Wickets should be between 0 and 10
        if 'wickets_fallen' in df_clean.columns:
            df_clean['wickets_fallen'] = df_clean['wickets_fallen'].clip(0, 10)
        
        print(f"   ✓ Validated data ranges")
        
        # 4. Convert date column
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        
        # 5. Standardize team names
        team_name_mapping = {
            'Rising Pune Supergiant': 'Rising Pune Supergiants',
            'Delhi Daredevils': 'Delhi Capitals',
            'Kings XI Punjab': 'Punjab Kings',
            'Deccan Chargers': 'Sunrisers Hyderabad'
        }
        
        for col in ['batting_team', 'bowling_team']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace(team_name_mapping)
        
        print(f"   ✓ Standardized team names")
        
        # Store cleaning stats
        self.cleaning_stats['ball_by_ball'] = {
            'original_rows': original_shape[0],
            'cleaned_rows': df_clean.shape[0],
            'duplicates_removed': duplicates_removed,
            'missing_handled': missing_before - missing_after
        }
        
        print(f"   ✅ Cleaning complete: {original_shape} → {df_clean.shape}")
        
        return df_clean
    
    def clean_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean match summary data.
        
        Args:
            df: Raw match DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("\n🧹 Cleaning match data...")
        original_shape = df.shape
        
        # Make a copy
        df_clean = df.copy()
        
        # 1. Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['match_id'])
        duplicates_removed = original_shape[0] - df_clean.shape[0]
        print(f"   ✓ Removed {duplicates_removed} duplicate matches")
        
        # 2. Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        
        # Fill scores with 0
        score_cols = ['team1_score', 'team2_score', 'team1_wickets', 'team2_wickets', 'win_margin']
        for col in score_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0).astype(int)
        
        # Fill categorical with mode
        for col in ['winner', 'toss_winner', 'toss_decision', 'win_type']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"   ✓ Handled {missing_before - missing_after} missing values")
        
        # 3. Convert date column
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        
        # 4. Standardize team names
        team_name_mapping = {
            'Rising Pune Supergiant': 'Rising Pune Supergiants',
            'Delhi Daredevils': 'Delhi Capitals',
            'Kings XI Punjab': 'Punjab Kings',
            'Deccan Chargers': 'Sunrisers Hyderabad'
        }
        
        for col in ['team1', 'team2', 'winner', 'toss_winner']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace(team_name_mapping)
        
        print(f"   ✓ Standardized team names")
        
        # Store cleaning stats
        self.cleaning_stats['match'] = {
            'original_rows': original_shape[0],
            'cleaned_rows': df_clean.shape[0],
            'duplicates_removed': duplicates_removed
        }
        
        print(f"   ✅ Cleaning complete: {original_shape} → {df_clean.shape}")
        
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers in numerical columns.
        
        Args:
            df: Input DataFrame
            column: Column name to handle outliers
            method: 'iqr' (Interquartile Range) or 'zscore'
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
            
        elif method == 'zscore':
            mean = df_clean[column].mean()
            std = df_clean[column].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
        
        return df_clean


class DataSplitter:
    """
    Class to split data for training and evaluation.
    
    Implements match-wise splitting to avoid data leakage.
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split_by_match(self, df: pd.DataFrame, match_id_col: str = 'match_id') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by match to avoid data leakage.
        
        All balls from a match will be in the same split.
        
        Args:
            df: Input DataFrame
            match_id_col: Column name for match identifier
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("\n📊 Splitting data by match...")
        
        # Get unique match IDs
        unique_matches = df[match_id_col].unique()
        n_matches = len(unique_matches)
        
        # First split: train+val and test
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        
        train_val_idx, test_idx = next(gss.split(df, groups=df[match_id_col]))
        
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]
        
        # Second split: train and val
        gss_val = GroupShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.random_state)
        
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[match_id_col]))
        
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
        
        print(f"   ✓ Total matches: {n_matches}")
        print(f"   ✓ Train matches: {train_df[match_id_col].nunique()} ({len(train_df)} rows)")
        print(f"   ✓ Validation matches: {val_df[match_id_col].nunique()} ({len(val_df)} rows)")
        print(f"   ✓ Test matches: {test_df[match_id_col].nunique()} ({len(test_df)} rows)")
        
        return train_df, val_df, test_df
    
    def split_by_time(self, df: pd.DataFrame, date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (time-based split).
        
        This ensures that the model is tested on future matches.
        
        Args:
            df: Input DataFrame
            date_col: Column name for date
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("\n📊 Splitting data by time...")
        
        # Sort by date
        df_sorted = df.sort_values(date_col)
        
        n_rows = len(df_sorted)
        train_end = int(n_rows * (1 - self.test_size - self.val_size))
        val_end = int(n_rows * (1 - self.test_size))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        print(f"   ✓ Train: {len(train_df)} rows (up to {train_df[date_col].max()})")
        print(f"   ✓ Validation: {len(val_df)} rows")
        print(f"   ✓ Test: {len(test_df)} rows (from {test_df[date_col].min()})")
        
        return train_df, val_df, test_df


class DataNormalizer:
    """
    Class to normalize/standardize features.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize the normalizer.
        
        Args:
            method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        self.method = method
        self.scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self.fitted = False
        self.feature_names = None
    
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'DataNormalizer':
        """
        Fit the scaler on training data.
        
        Args:
            df: Training DataFrame
            columns: Columns to normalize
            
        Returns:
            Self for chaining
        """
        self.feature_names = columns
        self.scaler.fit(df[columns])
        self.fitted = True
        print(f"✅ Fitted {self.method} scaler on {len(columns)} features")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        df_transformed = df.copy()
        df_transformed[self.feature_names] = self.scaler.transform(df[self.feature_names])
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fit and transform data.
        
        Args:
            df: Training DataFrame
            columns: Columns to normalize
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, columns)
        return self.transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform normalized data.
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Original scale DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        df_inverse = df.copy()
        df_inverse[self.feature_names] = self.scaler.inverse_transform(df[self.feature_names])
        
        return df_inverse


# Main preprocessing pipeline
def preprocess_ipl_data(data_path: str = "data/", 
                        save_processed: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Main preprocessing pipeline for IPL data.
    
    Args:
        data_path: Path to data directory
        save_processed: Whether to save processed data
        
    Returns:
        Dictionary containing processed DataFrames
    """
    print("=" * 60)
    print("🏏 IPL Data Preprocessing Pipeline")
    print("=" * 60)
    
    # 1. Load data
    loader = IPLDataLoader(data_path)
    ball_df = loader.load_ball_by_ball_data()
    match_df = loader.load_match_data()
    
    # 2. Clean data
    cleaner = IPLDataCleaner()
    ball_df_clean = cleaner.clean_ball_by_ball_data(ball_df)
    match_df_clean = cleaner.clean_match_data(match_df)
    
    # 3. Split data
    splitter = DataSplitter(test_size=0.2, val_size=0.1)
    train_df, val_df, test_df = splitter.split_by_match(ball_df_clean)
    
    # 4. Save processed data
    if save_processed:
        processed_path = os.path.join(data_path, "processed")
        os.makedirs(processed_path, exist_ok=True)
        
        train_df.to_csv(os.path.join(processed_path, "train.csv"), index=False)
        val_df.to_csv(os.path.join(processed_path, "val.csv"), index=False)
        test_df.to_csv(os.path.join(processed_path, "test.csv"), index=False)
        match_df_clean.to_csv(os.path.join(processed_path, "matches.csv"), index=False)
        
        print(f"\n💾 Saved processed data to {processed_path}")
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print("=" * 60)
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'matches': match_df_clean,
        'full_ball_data': ball_df_clean
    }


if __name__ == "__main__":
    # Run preprocessing pipeline
    processed_data = preprocess_ipl_data(data_path="data/")
    
    # Print summary
    print("\n📈 Data Summary:")
    for name, df in processed_data.items():
        print(f"   {name}: {df.shape}")
