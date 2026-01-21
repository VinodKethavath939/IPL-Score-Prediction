"""
Utility Functions for IPL Score Prediction
==========================================
Common utility functions used across the project.

Author: IPL Score Prediction Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger('IPL_Score_Prediction')
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_json(file_path: str) -> Dict:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary from JSON
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save to
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def get_ipl_teams() -> List[str]:
    """
    Get list of IPL teams.
    
    Returns:
        List of team names
    """
    return [
        'Mumbai Indians',
        'Chennai Super Kings',
        'Royal Challengers Bangalore',
        'Kolkata Knight Riders',
        'Delhi Capitals',
        'Punjab Kings',
        'Rajasthan Royals',
        'Sunrisers Hyderabad',
        'Gujarat Titans',
        'Lucknow Super Giants'
    ]


def get_ipl_venues() -> List[str]:
    """
    Get list of IPL venues.
    
    Returns:
        List of venue names
    """
    return [
        'Wankhede Stadium',
        'M. A. Chidambaram Stadium',
        'Eden Gardens',
        'Arun Jaitley Stadium',
        'M. Chinnaswamy Stadium',
        'Narendra Modi Stadium',
        'Rajiv Gandhi International Stadium',
        'Punjab Cricket Association Stadium',
        'Sawai Mansingh Stadium',
        'DY Patil Stadium'
    ]


def get_sample_players() -> Dict[str, List[str]]:
    """
    Get sample player names by team.
    
    Returns:
        Dictionary of team -> players
    """
    return {
        'Mumbai Indians': ['Rohit Sharma', 'Suryakumar Yadav', 'Ishan Kishan', 'Jasprit Bumrah', 'Hardik Pandya'],
        'Chennai Super Kings': ['MS Dhoni', 'Ruturaj Gaikwad', 'Devon Conway', 'Ravindra Jadeja', 'Deepak Chahar'],
        'Royal Challengers Bangalore': ['Virat Kohli', 'Faf du Plessis', 'Glenn Maxwell', 'Mohammed Siraj', 'Wanindu Hasaranga'],
        'Kolkata Knight Riders': ['Shreyas Iyer', 'Rinku Singh', 'Andre Russell', 'Sunil Narine', 'Varun Chakravarthy'],
        'Delhi Capitals': ['Rishabh Pant', 'David Warner', 'Axar Patel', 'Anrich Nortje', 'Kuldeep Yadav'],
        'Punjab Kings': ['Shikhar Dhawan', 'Liam Livingstone', 'Kagiso Rabada', 'Arshdeep Singh', 'Rahul Chahar'],
        'Rajasthan Royals': ['Sanju Samson', 'Jos Buttler', 'Yashasvi Jaiswal', 'Yuzvendra Chahal', 'Trent Boult'],
        'Sunrisers Hyderabad': ['Aiden Markram', 'Heinrich Klaasen', 'Rahul Tripathi', 'Bhuvneshwar Kumar', 'T Natarajan'],
        'Gujarat Titans': ['Shubman Gill', 'Rashid Khan', 'Mohammed Shami', 'Sai Sudharsan', 'Rahul Tewatia'],
        'Lucknow Super Giants': ['KL Rahul', 'Quinton de Kock', 'Marcus Stoinis', 'Ravi Bishnoi', 'Avesh Khan']
    }


# ============================================================================
# METRICS UTILITIES
# ============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n📊 {title}")
    print("-" * 40)
    for name, value in metrics.items():
        if name in ['MAE', 'MSE', 'RMSE']:
            print(f"   {name}: {value:.2f} runs")
        elif name in ['R2']:
            print(f"   {name}: {value:.4f}")
        elif name in ['MAPE']:
            print(f"   {name}: {value:.2f}%")
        else:
            print(f"   {name}: {value}")


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_feature_importance(feature_names: List[str],
                            importances: np.ndarray,
                            top_n: int = 20,
                            save_path: str = None):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    set_plot_style()
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_score_distribution(scores: np.ndarray,
                            title: str = "Score Distribution",
                            save_path: str = None):
    """
    Plot score distribution.
    
    Args:
        scores: Array of scores
        title: Plot title
        save_path: Path to save figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.1f}')
    axes[0].axvline(x=np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.1f}')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(title)
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(scores, vert=True)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Score Box Plot')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame,
                            columns: List[str] = None,
                            save_path: str = None):
    """
    Plot correlation matrix.
    
    Args:
        df: DataFrame
        columns: Columns to include (default: all numerical)
        save_path: Path to save figure
    """
    set_plot_style()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5)
    
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_input(input_data: Dict, required_fields: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate input data.
    
    Args:
        input_data: Input dictionary
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing = [field for field in required_fields if field not in input_data]
    return len(missing) == 0, missing


def validate_score(score: float, min_val: float = 0, max_val: float = 300) -> bool:
    """
    Validate if score is within reasonable range.
    
    Args:
        score: Score to validate
        min_val: Minimum valid score
        max_val: Maximum valid score
        
    Returns:
        True if valid, False otherwise
    """
    return min_val <= score <= max_val


def validate_overs(overs: float) -> bool:
    """
    Validate overs value.
    
    Args:
        overs: Overs to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not 0 <= overs <= 20:
        return False
    
    # Check decimal part (should be 0-5 for ball number)
    decimal_part = overs - int(overs)
    ball_number = round(decimal_part * 10)
    
    return ball_number <= 5


# ============================================================================
# TIME UTILITIES
# ============================================================================

def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


# ============================================================================
# FILE UTILITIES
# ============================================================================

def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        The same path
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_project_root() -> str:
    """
    Get project root directory.
    
    Returns:
        Project root path
    """
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    return os.path.dirname(src_dir)


def get_latest_file(directory: str, pattern: str = "*") -> Optional[str]:
    """
    Get the most recently modified file matching pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern
        
    Returns:
        Path to most recent file, or None
    """
    import glob
    
    files = glob.glob(os.path.join(directory, pattern))
    
    if not files:
        return None
    
    return max(files, key=os.path.getmtime)


# ============================================================================
# CRICKET-SPECIFIC UTILITIES
# ============================================================================

def balls_to_overs(balls: int) -> float:
    """
    Convert balls to overs.
    
    Args:
        balls: Number of balls
        
    Returns:
        Overs in decimal format (e.g., 10.3 = 10 overs 3 balls)
    """
    complete_overs = balls // 6
    remaining_balls = balls % 6
    return complete_overs + remaining_balls / 10


def overs_to_balls(overs: float) -> int:
    """
    Convert overs to balls.
    
    Args:
        overs: Overs in decimal format
        
    Returns:
        Total number of balls
    """
    complete_overs = int(overs)
    remaining_balls = round((overs - complete_overs) * 10)
    return complete_overs * 6 + remaining_balls


def calculate_run_rate(runs: int, overs: float) -> float:
    """
    Calculate run rate.
    
    Args:
        runs: Total runs
        overs: Overs bowled
        
    Returns:
        Run rate
    """
    if overs <= 0:
        return 0.0
    return runs / overs


def calculate_required_run_rate(target: int, current_score: int, overs_remaining: float) -> float:
    """
    Calculate required run rate.
    
    Args:
        target: Target score
        current_score: Current score
        overs_remaining: Overs remaining
        
    Returns:
        Required run rate
    """
    runs_needed = target - current_score
    
    if overs_remaining <= 0:
        return float('inf')
    
    return runs_needed / overs_remaining


def get_match_phase(overs: float) -> str:
    """
    Get match phase based on overs.
    
    Args:
        overs: Current overs
        
    Returns:
        Phase name ('powerplay', 'middle', 'death')
    """
    if overs <= 6:
        return 'powerplay'
    elif overs <= 15:
        return 'middle'
    else:
        return 'death'


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test cricket utilities
    print(f"\nBalls to overs: 63 balls = {balls_to_overs(63)} overs")
    print(f"Overs to balls: 10.3 overs = {overs_to_balls(10.3)} balls")
    print(f"Run rate: 85 runs in 10 overs = {calculate_run_rate(85, 10):.2f}")
    print(f"Required RR: Need 100 from 10 overs = {calculate_required_run_rate(180, 80, 10):.2f}")
    print(f"Match phase at 8 overs: {get_match_phase(8)}")
    
    # Test teams and venues
    print(f"\nIPL Teams: {len(get_ipl_teams())}")
    print(f"IPL Venues: {len(get_ipl_venues())}")
    
    print("\n✅ All utility tests passed!")
