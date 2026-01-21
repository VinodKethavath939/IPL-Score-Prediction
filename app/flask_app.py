"""
Flask REST API for IPL Score Prediction
=======================================
RESTful API for integrating IPL score predictions.

Endpoints:
- GET /: Health check
- POST /predict: Make prediction
- POST /predict/batch: Batch predictions
- GET /teams: Get team list
- GET /venues: Get venue list

Run: python app/flask_app.py

Author: IPL Score Prediction Team
Date: 2024
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from predict import IPLScorePredictor, load_best_model
    from utils import get_ipl_teams, get_ipl_venues, get_sample_players, validate_input
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import prediction module: {e}")
    MODEL_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_predictor():
    """Get or initialize the predictor."""
    global predictor
    
    if predictor is None and MODEL_AVAILABLE:
        try:
            predictor = load_best_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    return predictor


def simulate_prediction(input_data):
    """Simulate prediction when model is not available."""
    current_score = input_data.get('current_score', 0)
    overs = input_data.get('overs', 0)
    wickets = input_data.get('wickets', 0)
    
    # Simple projection formula
    if overs > 0:
        current_rr = current_score / overs
        # Adjust based on wickets and phase
        if overs <= 6:
            factor = 1.2
        elif overs <= 15:
            factor = 1.1
        else:
            factor = 1.3
        
        # Wicket penalty
        wicket_factor = 1 - (wickets * 0.03)
        
        remaining_overs = 20 - overs
        projected_additional = current_rr * factor * wicket_factor * remaining_overs
        predicted_score = int(current_score + projected_additional)
    else:
        predicted_score = 170
    
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


def get_ipl_teams_fallback():
    """Fallback function for getting teams."""
    if MODEL_AVAILABLE:
        return get_ipl_teams()
    return [
        'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
        'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants'
    ]


def get_ipl_venues_fallback():
    """Fallback function for getting venues."""
    if MODEL_AVAILABLE:
        return get_ipl_venues()
    return [
        'Wankhede Stadium', 'M. A. Chidambaram Stadium', 'Eden Gardens',
        'Arun Jaitley Stadium', 'M. Chinnaswamy Stadium', 'Narendra Modi Stadium'
    ]


# ============================================================================
# HTML TEMPLATES
# ============================================================================

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>IPL Score Prediction API</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #1e3a8a;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #374151;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .endpoint {
            background: #f3f4f6;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }
        .method {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            color: white;
            margin-right: 10px;
        }
        .get { background: #10b981; }
        .post { background: #3b82f6; }
        code {
            background: #1f2937;
            color: #10b981;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
        }
        pre {
            background: #1f2937;
            color: #e5e7eb;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
        }
        .status {
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .status.online { background: #d1fae5; color: #065f46; }
        .status.offline { background: #fee2e2; color: #991b1b; }
        .emoji { font-size: 1.5em; }
    </style>
</head>
<body>
    <div class="container">
        <h1><span class="emoji">🏏</span> IPL Score Prediction API</h1>
        
        <div class="status {{ 'online' if model_status else 'offline' }}">
            <strong>Model Status:</strong> {{ 'Online ✅' if model_status else 'Simulation Mode ⚠️' }}
        </div>
        
        <h2>📡 API Endpoints</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/</code>
            <p>API health check and documentation</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/predict</code>
            <p>Make a score prediction</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/predict/batch</code>
            <p>Make batch predictions</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/teams</code>
            <p>Get list of IPL teams</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/venues</code>
            <p>Get list of IPL venues</p>
        </div>
        
        <h2>📝 Example Request</h2>
        
        <pre>
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "batting_team": "Mumbai Indians",
    "bowling_team": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "current_score": 85,
    "overs": 10.0,
    "wickets": 2,
    "striker": "Rohit Sharma",
    "non_striker": "Suryakumar Yadav",
    "bowler": "Ravindra Jadeja"
  }'
        </pre>
        
        <h2>📤 Example Response</h2>
        
        <pre>
{
    "success": true,
    "prediction": {
        "predicted_score": 175,
        "lower_bound": 165,
        "upper_bound": 185,
        "confidence_interval": [165, 185],
        "prediction_range": "165-185",
        "confidence_level": 0.95
    }
}
        </pre>
        
        <h2>📊 Input Parameters</h2>
        
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #f3f4f6;">
                <th style="padding: 10px; text-align: left;">Parameter</th>
                <th style="padding: 10px; text-align: left;">Type</th>
                <th style="padding: 10px; text-align: left;">Description</th>
            </tr>
            <tr>
                <td style="padding: 10px;"><code>batting_team</code></td>
                <td style="padding: 10px;">string</td>
                <td style="padding: 10px;">Name of batting team</td>
            </tr>
            <tr style="background: #f9fafb;">
                <td style="padding: 10px;"><code>bowling_team</code></td>
                <td style="padding: 10px;">string</td>
                <td style="padding: 10px;">Name of bowling team</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><code>venue</code></td>
                <td style="padding: 10px;">string</td>
                <td style="padding: 10px;">Match venue</td>
            </tr>
            <tr style="background: #f9fafb;">
                <td style="padding: 10px;"><code>current_score</code></td>
                <td style="padding: 10px;">integer</td>
                <td style="padding: 10px;">Current total score</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><code>overs</code></td>
                <td style="padding: 10px;">float</td>
                <td style="padding: 10px;">Overs completed (e.g., 10.3)</td>
            </tr>
            <tr style="background: #f9fafb;">
                <td style="padding: 10px;"><code>wickets</code></td>
                <td style="padding: 10px;">integer</td>
                <td style="padding: 10px;">Wickets fallen (0-10)</td>
            </tr>
            <tr>
                <td style="padding: 10px;"><code>target</code></td>
                <td style="padding: 10px;">integer (optional)</td>
                <td style="padding: 10px;">Target score (for 2nd innings)</td>
            </tr>
        </table>
        
        <p style="text-align: center; color: #6b7280; margin-top: 30px;">
            Built with ❤️ for IPL fans | Deep Learning Powered
        </p>
    </div>
</body>
</html>
"""


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """API home page with documentation."""
    pred = get_predictor()
    model_status = pred is not None
    return render_template_string(HOME_TEMPLATE, model_status=model_status)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    pred = get_predictor()
    return jsonify({
        'status': 'healthy',
        'model_loaded': pred is not None,
        'version': '1.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a score prediction.
    
    Expected JSON payload:
    {
        "batting_team": "Mumbai Indians",
        "bowling_team": "Chennai Super Kings",
        "venue": "Wankhede Stadium",
        "current_score": 85,
        "overs": 10.0,
        "wickets": 2,
        "striker": "Rohit Sharma",
        "non_striker": "Suryakumar Yadav",
        "bowler": "Ravindra Jadeja",
        "target": null  // Optional, for second innings
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['batting_team', 'bowling_team', 'venue', 'current_score', 'overs', 'wickets']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Validate data types and ranges
        if not isinstance(data['current_score'], (int, float)):
            return jsonify({
                'success': False,
                'error': 'current_score must be a number'
            }), 400
        
        if not 0 <= data['overs'] <= 20:
            return jsonify({
                'success': False,
                'error': 'overs must be between 0 and 20'
            }), 400
        
        if not 0 <= data['wickets'] <= 10:
            return jsonify({
                'success': False,
                'error': 'wickets must be between 0 and 10'
            }), 400
        
        # Get predictor
        pred = get_predictor()
        
        # Make prediction
        if pred is not None:
            try:
                if data.get('target'):
                    result = pred.predict_chase(data)
                else:
                    result = pred.predict_with_confidence(data)
            except Exception as e:
                logger.warning(f"Prediction failed, using simulation: {e}")
                result = simulate_prediction(data)
        else:
            result = simulate_prediction(data)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'input': {
                'batting_team': data['batting_team'],
                'bowling_team': data['bowling_team'],
                'venue': data['venue'],
                'current_score': data['current_score'],
                'overs': data['overs'],
                'wickets': data['wickets']
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make batch predictions.
    
    Expected JSON payload:
    {
        "predictions": [
            { ... prediction 1 data ... },
            { ... prediction 2 data ... }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid payload. Expected {"predictions": [...]}'
            }), 400
        
        predictions_data = data['predictions']
        
        if not isinstance(predictions_data, list):
            return jsonify({
                'success': False,
                'error': 'predictions must be an array'
            }), 400
        
        results = []
        pred = get_predictor()
        
        for i, input_data in enumerate(predictions_data):
            try:
                if pred is not None:
                    result = pred.predict_with_confidence(input_data)
                else:
                    result = simulate_prediction(input_data)
                
                results.append({
                    'index': i,
                    'success': True,
                    'prediction': result
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(predictions_data),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/over-by-over', methods=['POST'])
def predict_over_by_over():
    """
    Get over-by-over predictions.
    
    Expected JSON payload:
    {
        "batting_team": "Mumbai Indians",
        "bowling_team": "Chennai Super Kings",
        "venue": "Wankhede Stadium",
        "current_score": 85,
        "overs": 10.0,
        "wickets": 2,
        "final_over": 20
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        current_over = int(data.get('overs', 10))
        final_over = int(data.get('final_over', 20))
        
        pred = get_predictor()
        
        predictions = []
        current_score = data.get('current_score', 0)
        current_rr = current_score / max(current_over, 1)
        
        for over in range(current_over, final_over + 1):
            temp_data = data.copy()
            temp_data['overs'] = over
            temp_data['current_score'] = current_score + int((over - current_over) * current_rr * 1.1)
            
            if pred is not None:
                try:
                    result = pred.predict_with_confidence(temp_data)
                except:
                    result = simulate_prediction(temp_data)
            else:
                result = simulate_prediction(temp_data)
            
            result['over'] = over
            predictions.append(result)
        
        return jsonify({
            'success': True,
            'over_predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Over-by-over prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/teams', methods=['GET'])
def get_teams():
    """Get list of IPL teams."""
    return jsonify({
        'success': True,
        'teams': get_ipl_teams_fallback()
    })


@app.route('/venues', methods=['GET'])
def get_venues():
    """Get list of IPL venues."""
    return jsonify({
        'success': True,
        'venues': get_ipl_venues_fallback()
    })


@app.route('/players/<team>', methods=['GET'])
def get_players(team):
    """Get players for a specific team."""
    if MODEL_AVAILABLE:
        players = get_sample_players()
    else:
        players = {t: [f'Player {i}' for i in range(1, 12)] for t in get_ipl_teams_fallback()}
    
    if team in players:
        return jsonify({
            'success': True,
            'team': team,
            'players': players[team]
        })
    else:
        return jsonify({
            'success': False,
            'error': f'Team "{team}" not found'
        }), 404


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    pred = get_predictor()
    
    return jsonify({
        'success': True,
        'model_loaded': pred is not None,
        'model_type': 'Deep Neural Network' if pred else 'Simulation',
        'features': {
            'first_innings_prediction': True,
            'second_innings_chase': True,
            'over_by_over_prediction': True,
            'confidence_intervals': True
        }
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        'success': False,
        'error': 'Bad request'
    }), 400


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🏏 IPL Score Prediction API")
    print("=" * 60)
    print(f"\nStarting server...")
    print(f"Model available: {MODEL_AVAILABLE}")
    
    # Initialize predictor at startup
    get_predictor()
    
    print(f"\nAPI running at: http://localhost:5000")
    print(f"Documentation: http://localhost:5000/")
    print("\nEndpoints:")
    print("  POST /predict         - Make prediction")
    print("  POST /predict/batch   - Batch predictions")
    print("  GET  /teams           - List teams")
    print("  GET  /venues          - List venues")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
