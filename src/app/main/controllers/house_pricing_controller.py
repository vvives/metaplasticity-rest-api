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

from flask import request
from flask_restplus import Resource, Api, Namespace, fields, reqparse
import json

from ..dtos.house_pricing_dto import api, HousePricingRequest, HousePricingResponse
from ..services.house_pricing_service import HousePricingService

house_pricing_service = HousePricingService()

request_dto = HousePricingRequest.request
response_dto = HousePricingResponse.response

@api.route('/prediction')
@api.expect(request_dto, validate=True)
class HousePricingController(Resource):
    """
    The house pricing controller.
    """

    @api.marshal_with(response_dto, code=201)
    def post(self):
        """
        Post request for the prediction of a Boston house price. 
        """

        prediction = house_pricing_service.predict(json.dumps(api.payload))

        return prediction, 201