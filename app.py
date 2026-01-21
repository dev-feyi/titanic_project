from flask import Flask, render_template, request, jsonify
from model import TitanicSurvivalModel
import os

# Application instance renamed for uniqueness
voyage_app = Flask(__name__)

# Initialize the prediction engine
predictor_unit = TitanicSurvivalModel()

# Verification block: Ensure the trained model is available
if not predictor_unit.load_model():
    print("Pre-trained weights not found. Initializing training sequence...")
    from model import train_and_save_model
    
    # This function in your model.py must be updated to train on only 5 features
    train_and_save_model() 
    predictor_unit.load_model()


@voyage_app.route('/')
def serve_ui():
    """Serves the primary interface for the estimator"""
    return render_template('index.html')


@voyage_app.route('/predict', methods=['POST'])
def calculate_survival_odds():
    """
    Receives 5 passenger attributes and returns a survival prediction.
    Features used: Pclass, Sex, Age, Fare, SibSp
    """
    try:
        # Retrieve the JSON payload from the front-end
        payload = request.get_json()

        # Feature Extraction & Type Casting
        # Matches the 'submissionData' keys in your index.html
        travel_tier = int(payload.get('pclass'))
        identity = payload.get('sex').lower()
        years_count = float(payload.get('age'))
        ticket_cost = float(payload.get('fare'))
        sibling_spouse_count = int(payload.get('sibsp'))

        # --- DATA VALIDATION (5-Feature Set) ---
        
        if travel_tier not in [1, 2, 3]:
            return jsonify({'success': False, 'error': 'Invalid Class selection.'})

        if identity not in ['male', 'female']:
            return jsonify({'success': False, 'error': 'Identity must be male or female.'})

        if not (0 <= years_count <= 100):
            return jsonify({'success': False, 'error': 'Age must be between 0 and 100.'})

        if ticket_cost < 0:
            return jsonify({'success': False, 'error': 'Fare cannot be negative.'})

        if not (0 <= sibling_spouse_count <= 10):
            return jsonify({'success': False, 'error': 'Sibling/Spouse count out of range.'})

        # --- MODEL PREDICTION ---
        
        # We pass exactly 5 parameters to the model
        final_decision, probability_score = predictor_unit.predict(
            pclass=travel_tier,
            sex=identity,
            age=years_count,
            fare=ticket_cost,
            sibsp=sibling_spouse_count
        )

        # Build and return the formatted response
        return jsonify({
            'success': True,
            'survived': bool(final_decision),
            'probability': round(probability_score * 100, 1),
            'metadata': {
                'features_used': 5,
                'target': 'Survived'
            }
        })

    except Exception as operational_error:
        # Catch-all for unexpected processing issues
        return jsonify({
            'success': False,
            'error': f"Processing Error: {str(operational_error)}"
        })


if __name__ == '__main__':
    # Local server execution
    voyage_app.run(debug=True, host='0.0.0.0', port=5000)
