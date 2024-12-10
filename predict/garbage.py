import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO

model = tf.keras.models.load_model('../model/model98.h5')

class_names  = ['battery', 'cardboard', 'clothes', 'glass', 'metal', 'organic', 'paper', 'plastic', 'shoes', 'styrofoam']
def predict(url):

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((150, 150))
    
   
    img_array = np.array(img) / 255.0
    
   
    img_array = np.expand_dims(img_array, axis=0)
    
   
    predictions = model.predict(img_array)
    
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class]
    confidence = predictions[0][predicted_class]
    
    return predicted_class_name, confidence
