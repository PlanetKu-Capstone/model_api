import pickle



with open('../model/carbon_emission.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict(features):
    prediction = model.predict(features)
    return prediction