# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template
app = Flask(__name__)
df = pd.read_csv('Cleaned_pizza_data.csv')
model = joblib.load(open('pizza_price_prediction.pkl','rb'))
@app.route('/')
def index():
    company = sorted(df['company'].unique())
    topping = sorted(df['topping'].unique())
    variant = sorted(df['variant'].unique())
    size = sorted(df['size'].unique())
    return render_template('index.html',company=company,topping=topping,variant=variant,size=size)
@app.route('/predict', methods = ['POST'])
def predict():
    d1 = int(request.form['company'])
    d2 = request.form['diameter']
    d3 = int(request.form['topping'])
    d4 = int(request.form['variant'])
    d5 = request.form['size']
    d6 = request.form['extra_sauce']
    d7 = request.form['extra_cheese']
    arr = np.array([[d1,d2,d3,d4,d5,d6,d7]])
    result = model.predict(arr)
    return render_template('index.html',prediction_text=result)
    pass
if __name__=='__main__':
    app.run()

