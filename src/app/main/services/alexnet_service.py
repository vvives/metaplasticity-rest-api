"""
MIT License

Copyright (c) 2020 Víctor Vives Boix

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json 
from keras.models import load_model
import numpy as np
import os
from PIL import Image
from skimage import transform

class AlexNetService():
    """
    The AlexNet service.

    Loads the Keras Boston house pricing regression model
    and predicts the median value of a house.
    """

    def __init__(self):
        """
        Initializes the Boston house pricing model.
        """

        # Load model paths.
        self.folder_path = os.path.dirname(os.path.abspath(__file__))

    def predict(self, request):  
        """
        Predicts the MNIST value.
        """
        
        # Load model
        model_path = os.path.join(self.folder_path, '../keras/mnist_am_alexnet.h5')
        self.model = load_model(model_path, compile=False)

        # Convert JSON request to dictionary.
        dictionary = json.loads(request)

        # Get the path from the dictionary.
        path = list(dictionary.values())[0]

        # Create input data for Keras model.
        np_image = Image.open(path)
        np_image = np.array(np_image).astype('float32')/255
        np_image = transform.resize(np_image, (28, 28, 1))
        np_image = np.expand_dims(np_image, axis=0)

        # Predict data and get result.
        prediction = self.model.predict(np_image)
        results = np.round(prediction[0], 2)
        value = np.where(results == np.amax(results))[0]

        return {"value": value}