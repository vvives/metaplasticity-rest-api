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

from flask import request
from flask_restplus import Resource
import json

from ..dtos.mnist_dto import api, MnistImageRequest, MnistImageResponse
from ..services.mnist_service import MnistService

mnist_service = MnistService()

request_dto = MnistImageRequest.request
response_dto = MnistImageResponse.response

@api.route('/alexnet')
@api.expect(request_dto, validate=True)
class AlexNetController(Resource):
    """
    The AlexNet controller.
    """

    @api.marshal_with(response_dto, code=201)
    def post(self):
        """
        MNIST image classification using AlexNet with synaptic metaplasticity. 
        """

        prediction = mnist_service.alexnet(json.dumps(api.payload))

        return prediction, 201

@api.route('/lenet')
@api.expect(request_dto, validate=True)
class LeNetController(Resource):
    """
    The LeNet5 controller.
    """

    @api.marshal_with(response_dto, code=201)
    def post(self):
        """
        MNIST image classification using LeNet5 with synaptic metaplasticity. 
        """

        prediction = mnist_service.alexnet(json.dumps(api.payload))

        return prediction, 201

@api.route('/googlenet')
@api.expect(request_dto, validate=True)
class GoogLeNetController(Resource):
    """
    The GoogLeNet controller.
    """

    @api.marshal_with(response_dto, code=201)
    def post(self):
        """
        MNIST image classification using GoogLeNet with synaptic metaplasticity. 
        """

        prediction = mnist_service.alexnet(json.dumps(api.payload))

        return prediction, 201