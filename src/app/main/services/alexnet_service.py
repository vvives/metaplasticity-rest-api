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
        folder_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(folder_path, '../keras/mnist_am_alexnet.h5')

        # Load model
        self.model = load_model(model_path)

    def predict(self, request):  
        """
        Predicts the MNIST value.
        """

        # Convert JSON request to dictionary.
        dictionary = json.loads(request)

        # Get the values from the dictionary.
        values = np.fromiter(dictionary.values(), dtype=float)

        # Create input data for Keras model.
        data = np.array([values])

        # Predict data and get result.
        prediction = self.model.predict(data)
        value = prediction[0][0]

        return {"value": value}