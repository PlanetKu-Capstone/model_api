from flask import jsonify, make_response
import requests

def success(res):
    data = {
            'code': 200,
            'message': 'Successfully Predict!!',
            'prediksi': res
        }
    return make_response(jsonify(data)), 200


def error(res):
    data = {
            'code': 400,
            'message': 'Failed to predict the data!!!',
            'error': res
        }
    return make_response(jsonify(data)), 400