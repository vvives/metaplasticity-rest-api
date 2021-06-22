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

from flask_restplus import Namespace, fields

api = Namespace('MNIST Image classification', description='The MNIST image classification models with synaptic metaplasticity.')

class MnistImageRequest:
    """
    The MNIST image request.

    Includes the input data for MNIST image classification with convolutional neural networks with synaptic metaplasticity:
        - image: The image to be predicted.
    """

    request = api.model('ImageRequest', {
        'image': fields.String(required=True, description='The image path to be predicted.', example="C:/Users/vvives/Downloads/eight.jpg"),
    })

class MnistImageResponse:
    """
    The MNIST image request.

    Includes the output data for MNIST image classification with convolutional neural networks with synaptic metaplasticity:
        - value: The MNIST prediction value.
    """

    response = api.model('MnistImageResponse', {
        'value': fields.Integer,
    })