# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 13:50:56 2022

@author: Sourav Kumar Sarangi
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

if __name__ == '__main__':
    app.run()