pip install flask gunicorn

from flask import Flask, request, render_template
import pickle
import numpy as np

model = pickle.load(open('/content/drive/MyDrive/DATA602-FP/lr_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(i) for i in request.form.values()]
    array_features = [np.array(features)]
    prediction = model.predict(array_features)

    output = prediction

    return render_template('home.html', prediction_text='The predicted value is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
