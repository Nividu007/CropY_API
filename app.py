from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load('CropyModel.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        input_data = {
            'Crop': request.args.get('Crop'),
            'Season': request.args.get('Season'),
            'Area': float(request.args.get('Area')),
            'Annual_Rainfall': float(request.args.get('Annual_Rainfall')),
            'Fertilizer': float(request.args.get('Fertilizer')),
            'Pesticide': float(request.args.get('Pesticide'))
        }

        if input_data['Area'] <= 0:
            raise ValueError("Area must be greater than 0.")
        if input_data['Fertilizer'] < 0 or input_data['Pesticide'] < 0 or input_data['Annual_Rainfall'] < 0:
            raise ValueError("Fertilizer, Pesticide, and Annual Rainfall must be non-negative.")
        
        scaler = joblib.load('scaler.pkl')
        input_df = pd.DataFrame([input_data])
        scale_columns = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        input_df[scale_columns] = scaler.transform(input_df[scale_columns])
        input_df = pd.get_dummies(input_df, columns=['Crop', 'Season'], drop_first=True)

        input_df = pd.get_dummies(input_df, columns=['Crop', 'Season'])

        model_features = model.feature_names_in_
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
