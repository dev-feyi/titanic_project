from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Path to the saved model
MODEL_PATH = os.path.join('model', 'house_price_model.pkl')

def load_house_model():
    """Helper to load the model safely"""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Load the trained model
house_model = load_house_model()

@app.route('/')
def home():
    """Render the main house price prediction page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle house price prediction requests"""
    try:
        # Get data from request (supports both JSON and Form data)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # 1. Extract House Features
        overall_qual = int(data['OverallQual'])
        gr_liv_area = float(data['GrLivArea'])
        total_bsmt_sf = float(data['TotalBsmtSF'])
        garage_cars = int(data['GarageCars'])
        year_built = int(data['YearBuilt'])
        full_bath = int(data['FullBath'])

        # 2. Validate inputs
        if not (1 <= overall_qual <= 10):
            return jsonify({'success': False, 'error': 'Overall Quality must be between 1 and 10'})

        if gr_liv_area <= 0 or total_bsmt_sf < 0:
            return jsonify({'success': False, 'error': 'Area measurements must be positive numbers'})

        if not (0 <= garage_cars <= 5):
            return jsonify({'success': False, 'error': 'Garage capacity must be between 0 and 5'})

        if not (1800 <= year_built <= 2026):
            return jsonify({'success': False, 'error': 'Year Built must be between 1800 and 2026'})

        # 3. Prepare data for model
        input_df = pd.DataFrame([{
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'TotalBsmtSF': total_bsmt_sf,
            'GarageCars': garage_cars,
            'YearBuilt': year_built,
            'FullBath': full_bath
        }])

        # 4. Make prediction
        if house_model is None:
            return jsonify({'success': False, 'error': 'Model file not found on server'})
            
        prediction = house_model.predict(input_df)[0]

        # 5. Return result
        return jsonify({
            'success': True,
            'price': round(float(prediction), 2),
            'formatted_price': f"${prediction:,.2f}",
            'message': 'Prediction successful',
            'inputs': {
                'OverallQual': overall_qual,
                'GrLivArea': gr_liv_area,
                'TotalBsmtSF': total_bsmt_sf,
                'GarageCars': garage_cars,
                'YearBuilt': year_built,
                'FullBath': full_bath
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Invalid input data: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)