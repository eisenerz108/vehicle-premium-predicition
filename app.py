from flask import Flask, render_template,request
import pickle
import numpy as np
import sklearn


model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_premium():
    driverAge = int(request.form.get("driverAge"))
    vehicleAge = int(request.form.get("vehicleAge"))

    #Prediction
    result = model.predict(np.array([driverAge, vehicleAge]).reshape(1, -1))
    return str(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
