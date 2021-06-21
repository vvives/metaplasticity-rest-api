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

api = Namespace('House pricing', description='The Boston house pricing regression model.')

class HousePricingRequest:
    """
    The house pricing request.

    Includes all needed input data for a request to the Keras Boston house pricing regression model:
        - crim: Per capita crime rate by town.
        - zn: Proportion of residential land zoned for lots over 25,000 sq.ft.
        - indus: Proportion of non-retail business acres per town.
        - chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
        - nox: Nitric oxides concentration (parts per 10 million).'
        - rm: Average number of rooms per dwelling.
        - age: Proportion of owner-occupied units built prior to 1940.
        - dis: Weighted distances to five Boston employment centres.
        - rad: Index of accessibility to radial highways.
        - tax: Full-value property-tax rate per $10,000.
        - ptratio: Pupil-teacher ratio by town.
        - b: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
        - lstat: Percentage of lower status of the population.
    """

    request = api.model('HousePricingRequest', {
        'crim': fields.Float(required=True, description='Per capita crime rate by town.', example=0.00632),
        'zn': fields.Float(required=True, description='Proportion of residential land zoned for lots over 25,000 sq.ft.', example=18.00),
        'indus': fields.Float(required=True, description='Proportion of non-retail business acres per town.', example=2.310),
        'chas': fields.Float(required=True, description='Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).', example=0.0),
        'nox': fields.Float(required=True, description='Nitric oxides concentration (parts per 10 million).', example=0.2380),
        'rm': fields.Float(required=True, description='Average number of rooms per dwelling.', example=8.5750),
        'age': fields.Float(required=True, description='Proportion of owner-occupied units built prior to 1940.', example=35.20),
        'dis': fields.Float(required=True, description='Weighted distances to five Boston employment centres.', example=9.0900),
        'rad': fields.Float(required=True, description='Index of accessibility to radial highways.', example=2.0),
        'tax': fields.Float(required=True, description='Full-value property-tax rate per $10,000.', example=122.0),
        'ptratio': fields.Float(required=True, description='Pupil-teacher ratio by town.', example=17.80),
        'b': fields.Float(required=True, description='1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.', example=94.63),
        'lstat': fields.Float(required=True, description='Percentage of lower status of the population.', example=2.94),
    })

class HousePricingResponse:
    """
    The house pricing request.

    Includes the output data for a prediction with the Keras Boston house pricing regression model:
        - medv: Median value of owner-occupied homes in $1000's.
        - description: The median value description.
    """

    response = api.model('HousePricingResponse', {
        'medv': fields.Float,
        'description': fields.String
    })