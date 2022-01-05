# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:09:12 2022

@author: Sourav
"""

#from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import streamlit as st
#import flasgger
#from flasgger import Swagger

#app = Flask(__name__)
#Swagger(app)

result = {1:'Female',0:'Male'}

pickle_in = open('R_F_Classifier.pkl','rb')
classifier = pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return 'Welcome All and Happy New Year 2022'

#@app.route('/predict',methods = ['Get'])
def predict_gender(long_hair,nose_wide,nose_long,	
                  lips_thin,distance_nose_to_lip_long,
                  forehead_width_height_ratio):
    
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
    #long_hair = request.args.get('long_hair')
    #nose_wide = request.args.get('nose_wide')
    #nose_long = request.args.get('nose_long')
    #lips_thin = request.args.get('lips_thin')
    #distance_nose_to_lip_long = request.args.get('distance_nose_to_lip_long')
    #forehead_width_height_ratio = request.args.get('forehead_width_height_ratio')
    prediction = classifier.predict([[long_hair,
                                      nose_wide,	
                                      nose_long,	
                                      lips_thin,	
                                      distance_nose_to_lip_long,	
                                      forehead_width_height_ratio]])
    print(prediction)
    return prediction


def main():
    st.title("Gender Classification")
    html_temp = """
    <div style = "background-color:tomato;padding:10px">
    <h2 style = "color: white;text-align:center;">StreamLit Gender Classification ML App
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    long_hair = st.text_input('long_hair',"Type Here")
    nose_wide = st.text_input('nose_wide',"Type Here")
    nose_long = st.text_input('nose_long',"Type Here")
    lips_thin = st.text_input('lips_thin',"Type Here")
    distance_nose_to_lip_long = st.text_input('distance_nose_to_lip_long',"Type Here")
    forehead_width_height_ratio = st.text_input('forehead_width_height_ratio',"Type Here")
    
    res = ""
    if st.button("Predict"):
        res = predict_gender(long_hair, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long, forehead_width_height_ratio)
    st.success("The person is a {}".format(res))
    if st.button("About"):
        st.text("Lets LeArN")
        st.text("Built With StreamLit")
if __name__ == '__main__':
    main()