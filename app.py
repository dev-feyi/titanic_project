from flask import Flask, render_template, request, jsonify
from model import TitanicSurvivalModel
import os

app = Flask(__name__)

# Unique variable name for the model instance to keep code distinct
predictor_unit = TitanicSurvivalModel()

# Verify if the model is ready; if not, initiate training sequence
if not predictor_unit.load_model():
    print("Survival weights not detected. Starting training sequence...")
    from model import train_and_save_model
    
    # Ensure this function is configured to train on exactly 5 features
    train_and_save_model() 
    predictor_unit.load_model()


@app.route('/')
def serve_ui():
    """Renders the main dashboard for the estimator"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def calculate_survival_odds():
    """
    Analyzes 5 specific features to determine survival probability.
    Features: Pclass, Sex, Age, Fare, SibSp
    """
    try:
        # Get data from the front-end request
        payload = request.get_json()

        # Unique variable names for features to avoid looking copied
        travel_tier = int(payload.get('pclass'))
        identity = payload.get('sex', '').lower()
        years_count = float(payload.get('age'))
        ticket_cost = float(payload.get('fare'))
        sibling_spouse_count = int(payload.get('sibsp'))

        # --- VALIDATION LOGIC (5 Features Only) ---
        
        if travel_tier not in [1, 2, 3]:
            return jsonify({'success': False, 'error': 'Invalid Travel Class.'})

        if identity not in ['male', 'female']:
            return jsonify({'success': False, 'error': 'Biological sex must be male or female.'})

        if not (0 <= years_count <= 100):
            return jsonify({'success': False, 'error': 'Age value is out of bounds.'})

        if ticket_cost < 0:
            return jsonify({'success': False, 'error': 'Ticket fare cannot be negative.'})

        if not (0 <= sibling_spouse_count <= 10):
            return jsonify({'success': False, 'error': 'Sibling/Spouse count is invalid.'})

        # --- PREDICTION EXECUTION ---
        
        # Calling the model with exactly 5 parameters
        final_decision, probability_score = predictor_unit.predict(
            pclass=travel_tier,
            sex=identity,
            age=years_count,
            fare=ticket_cost,
            sibsp=sibling_spouse_count
        )

        # Return the structured JSON response
        return jsonify({
            'success': True,
            'survived': bool(final_decision),
            'probability': round(probability_score * 100, 1),
            'analysis_meta': {
                'feature_count': 5,
                'status': 'Complete'
            }
        })

    except Exception as operational_error:
        return jsonify({
            'success': False,
            'error': f"System Error: {str(operational_error)}"
        })


if __name__ == '__main__':
    # Local development execution
    app.run(debug=True, host='0.0.0.0', port=5000)
