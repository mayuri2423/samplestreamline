import numpy as np
from collections.abc import Mapping
from flask import Flask, request, jsonify, render_template
import pickle

# flask app
app = Flask(__name__)
# loading model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict' ,methods = ['POST'])
def predict():
    final_features = [float(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)
    if int(prediction)== 1:
         prediction ='Sample is reached on time'
    else:
         prediction ='Sorry the sample is late'   

    return render_template('index.html', output='{}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)