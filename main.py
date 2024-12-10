
from flask import Flask, request
from flask_cors import CORS
from predict.garbage import predict as garbage_predict
from predict.carbons import predict as carbons_predict
import os
import numpy as np
import requests
from handler import response as rp
import pickle
from flask import jsonify
import joblib
import pandas as pd


model = joblib.load(os.path.join(os.getcwd(), '..', 'model', 'linear_model.pkl'))  # Model XGBoost yang sudah dilatih
label_encoder = joblib.load(os.path.join(os.getcwd(), '..', 'model', 'label_encoder.pkl'))  # LabelEncoder untuk fitur 'item'
features = joblib.load(os.path.join(os.getcwd(), '..', 'model', 'features.pkl'))  # Fitur dari pelatihan



with open('../model/carbon_emission.pkl', 'rb') as model_file:
    model_carbons = pickle.load(model_file)

app = Flask(__name__)
CORS(app)

@app.route('/predict-sampah', methods=['POST'])
def sampah():
    try:
        img_param = request.form['img']
        res = garbage_predict(img_param)
        print("Typenya adalah ", type(img_param))
        print(res)
        return rp.success(res[0])
    except requests.exceptions.HTTPError as err:
        return rp.error(err)
    

@app.route('/predict-carbons', methods=['POST'])
def carbons():
    data = request.json
    features = np.array([[data['electriccity'], data['gas'], data['transportation'], 
                          data['food'], data['organic_waste'], data['inorganic_waste']]])
    
    prediction = model_carbons.predict(features)
    return jsonify({'predicted_carbon_footprint': prediction[0]})


@app.route('/predict-price', methods=['POST'])
def price():
    try:
        # Ambil data JSON dari request
        data = request.json

        # Validasi input data
        if 'item' not in data:
            return jsonify({'error': 'Field "item" is required.'}), 400

        # Encode fitur 'item'
        item = data['item']
        try:
            item_encoded = label_encoder.transform([item])[0]
        except ValueError:
            return jsonify({'error': f'Item "{item}" not recognized in the dataset.'}), 400

        # Buat DataFrame input dengan one-hot encoding
        input_data = pd.DataFrame(0, index=[0], columns=features)  # Template DataFrame
        input_data[f'item_{item}'] = 1  # One-hot encode kolom item

        # Prediksi harga
        predicted_price = model.predict(input_data)[0]

        # Kembalikan hasil prediksi
        return jsonify({
            'item': item,
            'predicted_price': round(predicted_price, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)
