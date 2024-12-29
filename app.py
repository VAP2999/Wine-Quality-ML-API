import pickle
from flask import Flask,jsonify,render_template,request
import numpy as np
import pandas as pd

app=Flask(__name__, static_url_path='/static')

#Load model
modelEn=pickle.load(open('modelEn.pkl','rb'))


@app.route('/')
def home():
    
        
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    
    d1=np.array(list(data.values())).reshape(1,-1)
    output=modelEn.predict(d1)
    output1=output.tolist()
    print(output1)
    return jsonify(output1)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    d1=np.array(data).reshape(1,-1)
    print(d1)
    output=modelEn.predict(d1).tolist()[0]
    
    
    return render_template("index.html",prediction_text="The red wine prediction  is {}".format(output))







if __name__=='__main__':

    app.run(debug=True)