# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 18:35:21 2022

@author: HP
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

pickle_in = open('R_F_Classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome All and Happy New Year 2022'

@app.route('/predict')
def predict_gender():
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
    return 'The predicted value is '+str(prediction)
if __name__ == '__main__':
    app.run()
    
    
    