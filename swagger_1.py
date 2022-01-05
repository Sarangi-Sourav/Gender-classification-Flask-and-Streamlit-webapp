from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('R_F_Classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome All and Happy New Year 2022'

@app.route('/predict',methods = ['Get'])
def predict_gender():
    
    """Let's predict the Gender of people given their facial dimensions
    ---
    parameters:
      - name: long_hair
        in: query
        type: number 
        required: true
      - name: nose_wide
        in: query
        type: number
        required: true
      - name: nose_long
        in: query
        type: number
        required: true
      - name: lips_thin
        in: query
        type: number
        required: true
      - name: distance_nose_to_lip_long
        in: query
        type: number
        required: true
      - name: forehead_width_height_ratio
        in: query
        type: number
        required: true
    responses:
        200:
            description: The Output Values
            
    """
    long_hair = request.args.get('long_hair')
    nose_wide = request.args.get('nose_wide')
    nose_long = request.args.get('nose_long')
    lips_thin = request.args.get('lips_thin')
    distance_nose_to_lip_long = request.args.get('distance_nose_to_lip_long')
    forehead_width_height_ratio = request.args.get('forehead_width_height_ratio')
    prediction = classifier.predict([[long_hair,
                                      nose_wide,	
                                      nose_long,	
                                      lips_thin,	
                                      distance_nose_to_lip_long,	
                                      forehead_width_height_ratio]])
    print(prediction)
    return 'The predicted value is '+str(prediction)

if __name__ == '__main__':
    app.run()
    
    
    