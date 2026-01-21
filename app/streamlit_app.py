"""
Streamlit Web Application for IPL Score Prediction
==================================================
Interactive web application for predicting IPL match scores.

Features:
- Team and venue selection
- Real-time score prediction
- Over-by-over visualization
- Confidence intervals

Run: streamlit run app/streamlit_app.py

Author: IPL Score Prediction Team
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from predict import IPLScorePredictor, load_best_model
    from utils import get_ipl_teams, get_ipl_venues, get_sample_players
except ImportError:
    # Fallback definitions if modules not found
    def get_ipl_teams():
        return [
            'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
            'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants'
        ]
    
    def get_ipl_venues():
        return [
            'Wankhede Stadium', 'M. A. Chidambaram Stadium', 'Eden Gardens',
            'Arun Jaitley Stadium', 'M. Chinnaswamy Stadium', 'Narendra Modi Stadium',
            'Rajiv Gandhi International Stadium', 'Punjab Cricket Association Stadium'
        ]
    
    def get_sample_players():
        return {team: [f'Player {i}' for i in range(1, 12)] for team in get_ipl_teams()}


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f9ff, #e0f2fe);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-score {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .prediction-range {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
    }
    .team-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.25rem;
    }
    .stSelectbox > div > div {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_predictor():
    """Load the prediction model (cached)."""
    try:
        predictor = load_best_model()
        return predictor, None
    except Exception as e:
        return None, str(e)


def create_score_gauge(predicted_score, lower_bound, upper_bound):
    """Create a gauge chart for predicted score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Score", 'font': {'size': 24}},
        delta={'reference': (lower_bound + upper_bound) / 2, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [100, 250], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [100, 150], 'color': '#fee2e2'},
                {'range': [150, 180], 'color': '#fef3c7'},
                {'range': [180, 220], 'color': '#d1fae5'},
                {'range': [220, 250], 'color': '#cffafe'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': predicted_score
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_over_progression_chart(predictions):
    """Create over-by-over progression chart."""
    df = pd.DataFrame(predictions)
    
    fig = go.Figure()
    
    # Confidence interval band
    fig.add_trace(go.Scatter(
        x=df['over'].tolist() + df['over'].tolist()[::-1],
        y=df['upper_bound'].tolist() + df['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    # Predicted score line
    fig.add_trace(go.Scatter(
        x=df['over'],
        y=df['predicted_score'],
        mode='lines+markers',
        name='Predicted Score',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Over-by-Over Score Progression',
        xaxis_title='Over',
        yaxis_title='Predicted Score',
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig


def create_run_rate_chart(current_rr, projected_rr, required_rr=None):
    """Create run rate comparison chart."""
    categories = ['Current RR', 'Projected RR']
    values = [current_rr, projected_rr]
    colors = ['#3b82f6', '#10b981']
    
    if required_rr and required_rr > 0:
        categories.append('Required RR')
        values.append(required_rr)
        colors.append('#ef4444')
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Run Rate Analysis',
        yaxis_title='Runs per Over',
        showlegend=False,
        height=300
    )
    
    return fig


def simulate_prediction(input_data):
    """Simulate prediction when model is not available."""
    current_score = input_data.get('current_score', 0)
    overs = input_data.get('overs', 0)
    wickets = input_data.get('wickets', 0)
    
    # Simple projection formula
    if overs > 0:
        current_rr = current_score / overs
        # Adjust based on wickets and phase
        if overs <= 6:  # Powerplay
            factor = 1.2
        elif overs <= 15:  # Middle
            factor = 1.1
        else:  # Death
            factor = 1.3
        
        # Wicket penalty
        wicket_factor = 1 - (wickets * 0.03)
        
        remaining_overs = 20 - overs
        projected_additional = current_rr * factor * wicket_factor * remaining_overs
        predicted_score = int(current_score + projected_additional)
    else:
        predicted_score = 170  # Default
    
    # Add uncertainty
    uncertainty = max(5, int((20 - overs) * 3))
    
    return {
        'predicted_score': predicted_score,
        'lower_bound': predicted_score - uncertainty,
        'upper_bound': predicted_score + uncertainty,
        'confidence_interval': [predicted_score - uncertainty, predicted_score + uncertainty],
        'prediction_range': f"{predicted_score - uncertainty}-{predicted_score + uncertainty}",
        'confidence_level': 0.95,
        'uncertainty': uncertainty
    }


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🏏 IPL Score Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load predictor
    predictor, error = load_predictor()
    
    if error:
        st.warning(f"⚠️ Model not loaded. Using simulation mode. Error: {error}")
        use_simulation = True
    else:
        use_simulation = False
    
    # Sidebar - Match Settings
    with st.sidebar:
        st.header("⚙️ Match Settings")
        
        # Teams
        st.subheader("Teams")
        batting_team = st.selectbox(
            "Batting Team",
            options=get_ipl_teams(),
            index=0
        )
        
        bowling_team = st.selectbox(
            "Bowling Team",
            options=[t for t in get_ipl_teams() if t != batting_team],
            index=0
        )
        
        # Venue
        st.subheader("Venue")
        venue = st.selectbox(
            "Match Venue",
            options=get_ipl_venues(),
            index=0
        )
        
        # Innings
        st.subheader("Innings")
        innings = st.radio(
            "Current Innings",
            options=["First Innings", "Second Innings"],
            index=0
        )
        
        if innings == "Second Innings":
            target = st.number_input(
                "Target Score",
                min_value=100,
                max_value=300,
                value=180,
                step=1
            )
        else:
            target = None
    
    # Main content - Two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Current Match State")
        
        # Score input
        score_col1, score_col2, score_col3 = st.columns(3)
        
        with score_col1:
            current_score = st.number_input(
                "Current Score",
                min_value=0,
                max_value=300,
                value=85,
                step=1,
                help="Current total runs scored"
            )
        
        with score_col2:
            overs = st.number_input(
                "Overs Completed",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.1,
                format="%.1f",
                help="Overs completed (e.g., 10.3 = 10 overs and 3 balls)"
            )
        
        with score_col3:
            wickets = st.number_input(
                "Wickets Fallen",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
                help="Number of wickets lost"
            )
        
        # Players
        st.subheader("🏃 Current Players")
        players = get_sample_players()
        
        player_col1, player_col2, player_col3 = st.columns(3)
        
        with player_col1:
            striker = st.selectbox(
                "Striker",
                options=players.get(batting_team, ["Player 1"]),
                index=0
            )
        
        with player_col2:
            non_striker = st.selectbox(
                "Non-Striker",
                options=[p for p in players.get(batting_team, ["Player 2"]) if p != striker],
                index=0
            )
        
        with player_col3:
            bowler = st.selectbox(
                "Bowler",
                options=players.get(bowling_team, ["Bowler 1"]),
                index=0
            )
    
    with col2:
        st.header("📈 Current Stats")
        
        # Calculate current stats
        if overs > 0:
            current_rr = current_score / overs
        else:
            current_rr = 0
        
        balls_remaining = int((20 - overs) * 6)
        wickets_remaining = 10 - wickets
        
        # Display metrics
        st.metric("Current Run Rate", f"{current_rr:.2f}")
        st.metric("Balls Remaining", balls_remaining)
        st.metric("Wickets in Hand", wickets_remaining)
        
        if target:
            runs_needed = target - current_score
            required_rr = runs_needed / (20 - overs) if overs < 20 else float('inf')
            st.metric("Runs Needed", runs_needed)
            st.metric("Required Run Rate", f"{required_rr:.2f}")
    
    st.markdown("---")
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button("🎯 Predict Score", use_container_width=True, type="primary")
    
    if predict_button:
        # Prepare input
        input_data = {
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'venue': venue,
            'current_score': current_score,
            'overs': overs,
            'wickets': wickets,
            'striker': striker,
            'non_striker': non_striker,
            'bowler': bowler
        }
        
        if target:
            input_data['target'] = target
        
        with st.spinner("🔮 Predicting..."):
            # Make prediction
            if use_simulation or predictor is None:
                result = simulate_prediction(input_data)
            else:
                try:
                    result = predictor.predict_with_confidence(input_data)
                except:
                    result = simulate_prediction(input_data)
        
        st.markdown("---")
        
        # Display prediction
        st.header("🎯 Prediction Results")
        
        pred_col1, pred_col2 = st.columns([1, 1])
        
        with pred_col1:
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Final Score</h3>
                <div class="prediction-score">{result['predicted_score']}</div>
                <div class="prediction-range">Range: {result['prediction_range']}</div>
                <p>Confidence: {result.get('confidence_level', 0.95)*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_col2:
            # Gauge chart
            fig_gauge = create_score_gauge(
                result['predicted_score'],
                result['lower_bound'],
                result['upper_bound']
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Additional charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Over progression
            # Simulate over-by-over predictions
            over_predictions = []
            for over in range(int(overs), 21):
                temp_input = input_data.copy()
                temp_input['overs'] = over
                temp_input['current_score'] = current_score + int((over - overs) * current_rr * 1.1)
                
                if use_simulation or predictor is None:
                    pred = simulate_prediction(temp_input)
                else:
                    try:
                        pred = predictor.predict_with_confidence(temp_input)
                    except:
                        pred = simulate_prediction(temp_input)
                
                pred['over'] = over
                over_predictions.append(pred)
            
            fig_progression = create_over_progression_chart(over_predictions)
            st.plotly_chart(fig_progression, use_container_width=True)
        
        with chart_col2:
            # Run rate comparison
            projected_rr = result['predicted_score'] / 20
            required_rr_val = required_rr if target else None
            
            fig_rr = create_run_rate_chart(current_rr, projected_rr, required_rr_val)
            st.plotly_chart(fig_rr, use_container_width=True)
        
        # Match analysis
        st.header("📋 Match Analysis")
        
        analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
        
        with analysis_col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Predicted Runs Remaining</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(result['predicted_score'] - current_score), unsafe_allow_html=True)
        
        with analysis_col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Projected Run Rate (Remaining)</div>
                <div class="metric-value">{:.2f}</div>
            </div>
            """.format((result['predicted_score'] - current_score) / max(20 - overs, 0.1)), unsafe_allow_html=True)
        
        with analysis_col3:
            # Match phase
            if overs <= 6:
                phase = "Powerplay 🚀"
            elif overs <= 15:
                phase = "Middle Overs ⚡"
            else:
                phase = "Death Overs 🔥"
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Current Phase</div>
                <div class="metric-value" style="font-size: 1.5rem;">{}</div>
            </div>
            """.format(phase), unsafe_allow_html=True)
        
        # Chase scenario (if second innings)
        if target:
            st.header("🏃 Chase Analysis")
            
            chase_col1, chase_col2 = st.columns(2)
            
            with chase_col1:
                win_probability = 0.5
                if required_rr_val:
                    if required_rr_val <= 6:
                        win_probability = 0.85
                    elif required_rr_val <= 8:
                        win_probability = 0.70
                    elif required_rr_val <= 10:
                        win_probability = 0.50
                    elif required_rr_val <= 12:
                        win_probability = 0.35
                    elif required_rr_val <= 15:
                        win_probability = 0.20
                    else:
                        win_probability = 0.10
                
                # Adjust for wickets
                win_probability *= (1 - wickets * 0.05)
                
                fig_win = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=win_probability * 100,
                    title={'text': "Win Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#10b981" if win_probability > 0.5 else "#ef4444"},
                        'steps': [
                            {'range': [0, 30], 'color': '#fee2e2'},
                            {'range': [30, 50], 'color': '#fef3c7'},
                            {'range': [50, 70], 'color': '#d1fae5'},
                            {'range': [70, 100], 'color': '#cffafe'}
                        ]
                    },
                    number={'suffix': '%'}
                ))
                fig_win.update_layout(height=250)
                st.plotly_chart(fig_win, use_container_width=True)
            
            with chase_col2:
                status = "Favorable" if win_probability > 0.5 else "Challenging"
                status_color = "green" if win_probability > 0.5 else "red"
                
                st.markdown(f"""
                ### Chase Status: <span style="color: {status_color}">{status}</span>
                
                - **Runs Needed:** {runs_needed}
                - **Balls Remaining:** {balls_remaining}
                - **Required Rate:** {required_rr_val:.2f} per over
                - **Wickets in Hand:** {wickets_remaining}
                
                **Strategy Recommendation:**
                """, unsafe_allow_html=True)
                
                if required_rr_val <= 8:
                    st.info("🟢 Comfortable chase. Keep wickets in hand and accelerate in death overs.")
                elif required_rr_val <= 10:
                    st.warning("🟡 Moderate chase. Look for boundaries while keeping wickets.")
                elif required_rr_val <= 12:
                    st.warning("🟠 Challenging chase. Need consistent boundaries. Minimize dots.")
                else:
                    st.error("🔴 Difficult chase. Require big hits every over. High-risk approach needed.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p>🏏 IPL Score Predictor | Built with Deep Learning</p>
        <p>Disclaimer: Predictions are for educational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
