import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle


app = Flask(__name__)
model = pickle.load(open('rf.pkl','rb'))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if (prediction[0]==0):
        prediction = "Non-Hazardous"
    else:
        prediction = "Hazardous"

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="The nature of asteroid is found to be {}".format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    if (prediction[0]==0):
        output = "Non-Hazardous"
    else:
        output = "Hazardous"
        
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
