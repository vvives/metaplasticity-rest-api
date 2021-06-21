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

from flask import Blueprint
from flask_restplus import Api

from .main.controllers.house_pricing_controller import api as house_pricing_controller_ns
from .main.controllers.alexnet_controller import api as alexnet_controller_ns

blueprint = Blueprint('keras-api', __name__)

api = Api(blueprint, version='1.0', title='Keras REST API', description='A Keras REST API for Convolutional Neural Networks with synaptic metaplasticity.')
api.add_namespace(house_pricing_controller_ns, path='/api/housepricing')
api.add_namespace(alexnet_controller_ns, path='/api/alexnet')
