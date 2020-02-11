# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:44:37 2020

@author: 703172796
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_feature = [x for x in request.form.values()]
    #final_result = [np.array(int_features)]
    #predictions = model.predict(final_result)
    prediction = model.most_similar(input_feature)[:10]
    
    return render_template('index.html', prediction_text= prediction)

if __name__=="__main__":
    app.run(debug=True)
    
    