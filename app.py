#app.py
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
       
    prediction=model.predict(final_features)
    
   
    return render_template('index.html', pred='Height predicted is :  {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=False)