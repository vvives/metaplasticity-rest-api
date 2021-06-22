"""
MIT License

Copyright (c) 2021 Víctor Vives Boix

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

class MnistService():
    """
    The MNIST service.

    Loads the MNIST classification model
    and predicts the value of an image.
    """

    def __init__(self):
        """
        Initializes the MNIST classification model.
        """

        # Load model paths.
        self.folder_path = os.path.dirname(os.path.abspath(__file__))

    def get_path(self, request):
        """
        Gets the MNIST image path.
        """

        dictionary = json.loads(request)

        # Get the path from the dictionary.
        path = list(dictionary.values())[0]

        return path

    def load_image(self, path):
        """
        Gets the MNIST image.
        """

        image = Image.open(path)
        image = np.array(image).astype('float32')/255
        image = transform.resize(image, (28, 28, 1))
        image = np.expand_dims(image, axis=0)

        return image

    def classify(self, image):
        """
        Classifies the MNIST image.
        """

        prediction = self.model.predict(image)
        results = np.round(prediction[0], 2)
        value = np.where(results == np.amax(results))[0]

        return value

    def alexnet(self, request):  
        """
        Classifies the MNIST image using AlexNet with synaptic metaplasticity.
        """
        
        # Load model
        model_path = os.path.join(self.folder_path, '../keras/mnist_am_alexnet.h5')
        self.model = load_model(model_path, compile=False)

        # Convert JSON request to dictionary.
        path = self.get_path(request)

        # Create input data for Keras model.
        image = self.load_image(path)

        # Predict data and get result.
        value = self.classify(image)

        return {"value": value}

    def lenet(self, request):  
        """
        Classifies the MNIST image using LeNet5 with synaptic metaplasticity.
        """
        
        # Load model
        model_path = os.path.join(self.folder_path, '../keras/mnist_am_lenet5.h5')
        self.model = load_model(model_path, compile=False)

        # Convert JSON request to dictionary.
        path = self.get_path(request)

        # Create input data for Keras model.
        image = self.load_image(path)

        # Predict data and get result.
        value = self.classify(image)

        return {"value": value}