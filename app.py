import numpy as np
from flask import Flask, request, jsonify, render_template,json
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    print(data)
    result = []
    for i in range(len(data)):
        print(data[i])
        result.append(np.array(list(data[i].values())))
    
    prediction = model.predict(result)
    print(prediction)
    output = []
    for i in range(len(prediction)):
        print(prediction[i])
        output.append(prediction[i])
    print(output)

    return json.dumps(output, cls=NpEncoder)


if __name__ == "__main__":
    app.run(debug=True)